"""Neural forecasters: EventGate-TS plus matched baselines.

All models consume the same tensors and predict a single value in
*standardised log space* (`log1p(volume)`, z-scored with training statistics),
so any difference in the results comes from architecture, not from inputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., 2022)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:      # [B, L]
        self.mu = x.mean(1, keepdim=True)
        self.sd = x.std(1, keepdim=True).clamp_min(self.eps)
        return (x - self.mu) / self.sd

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.sd + self.mu


class FreqBranch(nn.Module):
    """Split a window into low-frequency trend and high-frequency residual.

    Only the first `k` FFT coefficients are kept for the low band; the residual
    is what remains. Both bands are encoded with independent 1-D convolutions.
    With ``n_channels > 1`` the same filter bank is applied to every input
    channel, so the frequency expert sees the same information as the temporal
    expert and only the inductive bias differs.
    """

    def __init__(self, seq_len: int, d_model: int, k: int = 8, width: int = 32,
                 n_channels: int = 1):
        super().__init__()
        self.k = k
        self.c = n_channels
        self.n_channels = n_channels
        self.conv_low = nn.Conv1d(n_channels, width, kernel_size=7, padding=3)
        self.conv_high = nn.Conv1d(n_channels, width, kernel_size=7, padding=3)
        self.proj = nn.Linear(2 * width, d_model)

    def band_split(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """v : [B, L] or [B, C, L] -> low / high with the same shape."""
        spec = torch.fft.rfft(v, dim=-1)
        k = min(self.k, spec.shape[-1])
        low_spec = torch.zeros_like(spec)
        low_spec[..., :k] = spec[..., :k]
        low = torch.fft.irfft(low_spec, n=v.shape[-1], dim=-1)
        return low, v - low

    def forward(self, v: torch.Tensor) -> torch.Tensor:        # [B, L] / [B, C, L] -> [B, d]
        if v.dim() == 2:
            v = v.unsqueeze(1)
        low, high = self.band_split(v)
        hl = F.gelu(self.conv_low(low)).mean(-1)
        hh = F.gelu(self.conv_high(high)).mean(-1)
        return self.proj(torch.cat([hl, hh], dim=-1))


class PatchEmbed(nn.Module):
    def __init__(self, seq_len: int, n_channels: int, d_model: int,
                 patch_len: int = 24, stride: int = 12):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patch = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * n_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, L, C] -> [B, N, d]
        # [B, N, patch_len * C]
        p = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        p = p.reshape(x.shape[0], self.n_patch, -1)
        return self.proj(p) + self.pos


class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.norm = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(F.gelu(self.norm(self.conv(x))))


# --------------------------------------------------------------------------- #
# main model
# --------------------------------------------------------------------------- #
class VolStats(nn.Module):
    """Cheap, hand-designed shape descriptors of the volume channel.

    These are the same statistics a GBDT gets for free when the modeller builds
    lag/rolling features by hand.  Appending them to a deep model tests whether
    the tree advantage is *representational* (missing features) or
    *architectural* (better inductive bias).  Everything here is computed with
    plain indexing, so it survives corrupted / partially missing inputs without
    any change to the training loop.
    """

    LAGS = 7
    WINDOWS = (6, 24, 168)

    def __init__(self, seq_len: int, out_dim: int = 32):
        super().__init__()
        f = self.LAGS + 2 * len(self.WINDOWS) + 2 + 3
        self.proj = nn.Sequential(nn.Linear(f, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, vol: torch.Tensor) -> torch.Tensor:          # [B, L] -> [B, out_dim]
        feats = [vol[:, -i] for i in range(1, self.LAGS + 1)]
        for w in self.WINDOWS:
            seg = vol[:, -w:]
            feats += [seg.mean(1), seg.std(1)]
        seg24 = vol[:, -24:]
        feats += [seg24.min(1).values, seg24.max(1).values]
        feats += [vol[:, -1] - vol[:, -2],
                  vol[:, -1] - vol[:, -25],
                  vol[:, -1] - vol[:, 0]]
        return self.proj(torch.stack(feats, dim=-1))


class EventGateTS(nn.Module):
    """Event-gated mixture of a frequency expert and a temporal expert.

    Ex1 (freq expert)   : dual-band rFFT recon of the target channel -> conv encoder
                          -> delta prediction p_f
    Ex2 (temporal expert): patch transformer over all covariates -> delta p_t
    Gate                 : g = sigmoid(MLP([p_f, p_t, |p_f-p_t|, event_descriptor]))
                           delta = g * p_f + (1-g) * p_t,  out = vol[-1] + delta

    The gate is a *scalar* per horizon step, so it is a genuine expert selector
    whose value can be read off and related to the event context afterwards.
    """

    def __init__(
        self,
        seq_len: int = 168,
        c_num: int = 6,
        c_cal: int = 7,
        c_cat: int = 7,
        ev_dim: int = 18,
        d_model: int = 128,
        patch_len: int = 24,
        stride: int = 12,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        k_freq: int = 8,
        n_q: int = 1,
        use_freq: bool = True,
        use_gate: bool = True,
        use_ev: bool = True,
        use_revin: bool = True,
        freq_n_channels: int | None = None,
        use_volstats: bool = False,
    ):
        super().__init__()
        self.use_freq = use_freq
        self.use_gate = use_gate
        self.use_ev = use_ev
        self.use_revin = use_revin
        self.use_volstats = use_volstats
        self.n_q = n_q
        c_in = c_num + c_cal + c_cat

        self.revin = RevIN() if use_revin else None
        self.embed = PatchEmbed(seq_len, c_in, d_model, patch_len, stride)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(2 * d_model, d_model)

        self.head_time = nn.Linear(d_model + (32 if use_volstats else 0), n_q)
        self.vs = VolStats(seq_len, 32) if use_volstats else None
        self.freq = (FreqBranch(seq_len, d_model, k=k_freq,
                                n_channels=freq_n_channels or c_in)
                     if use_freq else None)
        if use_freq:
            self.head_freq = nn.Linear(d_model, n_q)
        if use_gate and use_freq:
            gate_in = ev_dim + 3 * n_q + (32 if use_volstats else 0)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, n_q),
            )
        self.drop = nn.Dropout(dropout)

    def forward(self, x_num, x_cal, x_cat, x_ev, return_gate: bool = False):
        vol = x_num[..., 0]
        if self.revin is not None:
            vol_n = self.revin.normalize(vol)
            x_num = torch.cat([vol_n.unsqueeze(-1), x_num[..., 1:]], dim=-1)
        else:
            vol_n = vol

        x = torch.cat([x_num, x_cal, x_cat], dim=-1)
        seq = self.encoder(self.embed(x))
        h_time = self.norm(self.pool(torch.cat([seq.mean(1), seq[:, -1]], dim=-1)))
        vs = self.vs(vol_n) if self.vs is not None else None
        if vs is not None:
            h_time = torch.cat([h_time, vs], dim=-1)
        p_t = self.head_time(self.drop(h_time))                      # [B, n_q]

        if self.freq is not None:
            # The frequency expert normally sees the full covariate stack,
            # exactly like the temporal expert, so only the inductive bias
            # differs.  v1 checkpoints were trained on the target channel only;
            # freq_n_channels=1 reproduces that variant for the record.
            if self.freq.n_channels == 1:
                p_f = self.head_freq(self.drop(self.freq(vol_n.unsqueeze(1))))
            else:
                p_f = self.head_freq(self.drop(self.freq(x.transpose(1, 2))))
            self.last_experts = (p_f.detach(), p_t.detach())
            if self.use_gate:
                ev = x_ev if self.use_ev else torch.zeros_like(x_ev)
                gate_in = [p_f, p_t, (p_f - p_t).abs(), ev]
                if vs is not None:
                    gate_in.append(vs)
                g = torch.sigmoid(self.gate_mlp(torch.cat(gate_in, dim=-1)))  # [B, n_q]
                delta = g * p_f + (1.0 - g) * p_t
                self.last_gate = g.detach()
            else:
                delta = 0.5 * (p_f + p_t)
                self.last_gate = torch.full_like(p_t, 0.5).detach()
        else:
            delta = p_t
            self.last_gate = torch.full_like(p_t, 0.0).detach()

        out = vol_n[:, -1:] + delta                                  # [B, n_q]
        if self.revin is not None:
            out = self.revin.denormalize(out)
        if return_gate:
            return out, self.last_gate
        return out


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    def __init__(self, c_num=6, c_cal=7, c_cat=7, hidden=128, layers=2, dropout=0.2, n_q=1):
        super().__init__()
        self.revin = RevIN()
        self.lstm = nn.LSTM(c_num + c_cal + c_cat, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, n_q)

    def forward(self, x_num, x_cal, x_cat, x_ev):
        vol = self.revin.normalize(x_num[..., 0])
        x_num = torch.cat([vol.unsqueeze(-1), x_num[..., 1:]], dim=-1)
        x = torch.cat([x_num, x_cal, x_cat], dim=-1)
        out, _ = self.lstm(x)
        return self.revin.denormalize(vol[:, -1:] + self.head(out[:, -1]))


class TCNForecaster(nn.Module):
    def __init__(self, c_num=6, c_cal=7, c_cat=7, channels=128, dilations=(1, 2, 4, 8), n_q=1):
        super().__init__()
        self.revin = RevIN()
        c_in = c_num + c_cal + c_cat
        blocks, prev = [], c_in
        for d in dilations:
            blocks.append(TCNBlock(prev, channels, d))
            prev = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_q)

    def forward(self, x_num, x_cal, x_cat, x_ev):
        vol = self.revin.normalize(x_num[..., 0])
        x_num = torch.cat([vol.unsqueeze(-1), x_num[..., 1:]], dim=-1)
        x = torch.cat([x_num, x_cal, x_cat], dim=-1).transpose(1, 2)
        h = self.net(x).mean(-1)
        return self.revin.denormalize(vol[:, -1:] + self.head(h))


class PatchTSTLite(nn.Module):
    """Same patch backbone as EventGate-TS, without the frequency/gate module."""

    def __init__(self, seq_len=168, c_num=6, c_cal=7, c_cat=7, ev_dim=18,
                 d_model=128, patch_len=24, stride=12, n_layers=2, n_heads=4, n_q=1):
        super().__init__()
        self.revin = RevIN()
        self.embed = PatchEmbed(seq_len, c_num + c_cal + c_cat, d_model, patch_len, stride)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 2 * d_model, dropout=0.1, batch_first=True,
            activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(2 * d_model, d_model)
        self.head = nn.Linear(d_model, n_q)

    def forward(self, x_num, x_cal, x_cat, x_ev):
        vol = self.revin.normalize(x_num[..., 0])
        x_num = torch.cat([vol.unsqueeze(-1), x_num[..., 1:]], dim=-1)
        x = torch.cat([x_num, x_cal, x_cat], dim=-1)
        seq = self.encoder(self.embed(x))
        h = self.norm(self.pool(torch.cat([seq.mean(1), seq[:, -1]], dim=-1)))
        return self.revin.denormalize(vol[:, -1:] + self.head(h))


class PatchMLP(nn.Module):
    """Cheap control: flatten the window into an MLP (no sequence modelling)."""

    def __init__(self, seq_len=168, c_num=6, c_cal=7, c_cat=7, ev_dim=18,
                 hidden=256, dropout=0.2, n_q=1):
        super().__init__()
        self.revin = RevIN()
        d_in = seq_len * (c_num + c_cal + c_cat) + ev_dim
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_q),
        )

    def forward(self, x_num, x_cal, x_cat, x_ev):
        vol = self.revin.normalize(x_num[..., 0])
        x_num = torch.cat([vol.unsqueeze(-1), x_num[..., 1:]], dim=-1)
        x = torch.cat([x_num, x_cal, x_cat], dim=-1).flatten(1)
        return self.revin.denormalize(vol[:, -1:] + self.net(torch.cat([x, x_ev], dim=-1)))


MODEL_ZOO = {
    "eventgate": EventGateTS,
    "patchtst": PatchTSTLite,
    "lstm": LSTMForecaster,
    "tcn": TCNForecaster,
    "mlp": PatchMLP,
}


def build_model(name: str, **kw) -> nn.Module:
    return MODEL_ZOO[name](**kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
