"""Inspect what the event gate actually learned.

The gate g is the mixture weight of the *frequency* expert, so g -> 1 means
"trust the periodic/low-frequency expert" and g -> 0 means "trust the temporal
patch encoder".  We check whether g varies with the event context (weather,
holidays, regime volatility) and whether it tracks the disagreement between the
two experts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.models.nets import build_model

PROC = Path("data/processed")
RUNS = Path("runs")
H = int(os.environ.get("EG_HORIZON", "24"))
WEATHER_CLASSES = ["Clear", "Clouds", "Rain", "Snow", "LowVis", "Thunder", "Squall"]
EV_CHANNELS = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "holiday_kernel"]
    + [f"wx_{c}" for c in WEATHER_CLASSES]
    + ["rain_6h_log", "snow_6h_log", "vol_delta_6h", "vol_zscore_abs"]
)


def load_model(run_dir: Path, device="cpu"):
    res = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = res["config"]
    H_ = cfg["horizon"]
    n_q = 3 * H_ if cfg["n_q"] == 3 else H_
    sd = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    kw = dict(d_model=cfg["d_model"], n_layers=cfg["layers"],
              patch_len=cfg["patch_len"], stride=cfg["stride"])
    if cfg["model"] == "eventgate":
        # older checkpoints (v1) fed the frequency branch the target channel
        # only; the number of input channels is read back from the weights so
        # both generations can be reconstructed without extra bookkeeping.
        fch = sd["freq.conv_low.weight"].shape[1] if "freq.conv_low.weight" in sd else None
        kw.update(n_q=n_q, k_freq=cfg.get("k_freq", 8),
                  use_volstats=bool(cfg.get("volstats", False)),
                  use_freq=not cfg.get("no_freq", False),
                  use_gate=not cfg.get("no_gate", False),
                  use_ev=not cfg.get("no_ev", False),
                  use_revin=not cfg.get("no_revin", False),
                  freq_n_channels=fch)
    elif cfg["model"] == "patchtst":
        kw.update(n_q=n_q)
    elif cfg["model"] == "lstm":
        kw = dict(hidden=cfg["d_model"], n_q=n_q)
    elif cfg["model"] == "tcn":
        kw = dict(channels=cfg["d_model"], dilations=(1, 2, 4, 8, 16, 32), n_q=n_q)
    elif cfg["model"] == "mlp":
        kw = dict(n_q=n_q)
    model = build_model(cfg["model"], **kw).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


@torch.no_grad()
def collect(run_dir: Path, device="cpu", bs: int = 256) -> dict:
    """Forward the test split and undo RevIN for the mixture and both experts.

    The network emits a value in the *global* standardised log space, because
    ``RevIN.denormalize`` is applied inside ``forward``.  The two experts,
    however, emit a **delta** in the *per-window* normalised space, so undoing
    them requires the per-window location/scale that RevIN stored during the very
    same forward pass.  Getting this wrong silently inflates the error by an
    order of magnitude, so both paths are computed explicitly here.
    """
    model, cfg = load_model(run_dir, device)
    H_ = cfg["horizon"]
    w = np.load(PROC / f"windows_h{H_}.npz")
    split = np.load(PROC / f"split_h{H_}.npy", allow_pickle=True)
    te = np.where(split == "test")[0]
    xs = [torch.from_numpy(w[k][te]) for k in ["X_num", "X_cal", "X_cat", "X_ev"]]
    mu, sd = np.atleast_1d(w["mu"]), np.atleast_1d(w["sd"])

    def to_raw(z):
        """global standardised log1p-volume -> vehicles/hour"""
        return np.expm1(np.clip(np.asarray(z, np.float64) * sd[0] + mu[0], -2.0, 12.0))

    gates, pf_all, pt_all, pred_all = [], [], [], []
    for s in range(0, len(te), bs):
        xb = [x[s:s + bs].to(device) for x in xs]
        out = model(*xb)
        pred_all.append(out.float().cpu().numpy())
        g = getattr(model, "last_gate", None)
        if g is not None:
            gates.append(g.float().cpu().numpy())
        exp = getattr(model, "last_experts", None)
        if exp is not None and model.revin is not None:
            # forward does  denormalize(vol_n[-1] + delta) = vol[-1] + delta * sd_w
            # so an expert's absolute z-space forecast is  vol[-1] + p_expert * sd_w
            pf, pt = exp
            vol_last = xb[0][..., 0][:, -1:]                       # global z space
            sd_w = model.revin.sd                                  # per-window scale
            pf_all.append((vol_last + pf * sd_w).float().cpu().numpy())
            pt_all.append((vol_last + pt * sd_w).float().cpu().numpy())

    res = {
        "pred": to_raw(np.concatenate(pred_all)),
        "y": w["y"][te],
        "y_hour": w["y_hour"][te],
        "X_ev": w["X_ev"][te],
        "y_time": w["y_time"][te],
    }
    if gates:
        res["gate"] = np.concatenate(gates)
    if pf_all:
        res["p_freq"] = to_raw(np.concatenate(pf_all))
        res["p_time"] = to_raw(np.concatenate(pt_all))
    return res


def analyse(d: dict, tag: str) -> dict:
    out = {}
    ev = d["X_ev"]
    idx = {c: i for i, c in enumerate(EV_CHANNELS)}
    y = d["y"].astype(np.float64)
    g = d.get("gate")
    if g is None:
        return {"note": "model has no gate"}

    # per-horizon mean gate
    out["gate_by_horizon"] = [float(v) for v in g.mean(0)]

    regimes = {
        "weather": (ev[:, [idx[f"wx_{c}"] for c in WEATHER_CLASSES]].argmax(1), WEATHER_CLASSES),
        "holiday": ((ev[:, idx["is_holiday"]] > 0.5).astype(int), ["normal", "holiday"]),
        "weekend": ((ev[:, idx["is_weekend"]] > 0.5).astype(int), ["weekday", "weekend"]),
        "vol_zscore": (np.digitize(ev[:, idx["vol_zscore_abs"]], [1, 2, 3]), ["<1", "1-2", "2-3", ">3"]),
        "rain_6h": (np.digitize(np.expm1(ev[:, idx["rain_6h_log"]]), [0.01, 1, 5]),
                    ["none", "<1mm", "1-5mm", ">5mm"]),
        "snow_6h": (np.digitize(np.expm1(ev[:, idx["snow_6h_log"]]), [0.01, 0.5]),
                    ["none", "<0.5mm", ">=0.5mm"]),
        "trend": (np.digitize(ev[:, idx["vol_delta_6h"]],
                              np.quantile(ev[:, idx["vol_delta_6h"]], [0.25, 0.5, 0.75])),
                  ["falling_fast", "falling", "rising", "rising_fast"]),
    }
    rows = []
    for name, (codes, labels) in regimes.items():
        for c, lab in enumerate(labels):
            sel = codes == c
            if sel.sum() < 20:
                continue
            rows.append({
                "regime": name, "level": lab, "n": int(sel.sum()),
                "gate_mean": float(g[sel].mean()),
                "gate_std": float(g[sel].std()),
                "mae": float(np.abs(d["pred"][sel] - y[sel]).mean()),
            })
    out["gate_by_regime"] = rows

    # ---- can the gate actually route between two usable experts? --------
    # NOTE: the two heads are trained *jointly* to be blended, so neither is a
    # standalone forecaster and neither is a valid counterfactual on its own.
    # We report their standalone error for diagnosis only, and rely on the
    # retrained no_freq / no_gate ablations for the causal claim.
    if "p_freq" in d:
        dis = np.abs(d["p_freq"] - d["p_time"])
        out["spearman_gate_vs_disagreement"] = {
            "rho": float(stats.spearmanr(g.ravel(), dis.ravel()).statistic),
            "p": float(stats.spearmanr(g.ravel(), dis.ravel()).pvalue),
        }
        e_f = np.abs(d["p_freq"] - y)
        e_t = np.abs(d["p_time"] - y)
        out["expert_standalone_mae"] = {
            "freq": float(e_f.mean()), "time": float(e_t.mean()),
            "mixture": float(np.abs(d["pred"] - y).mean()),
            "note": "diagnostic only - heads are trained jointly, not separately",
        }
        out["corr_gate_vs_abs_time_expert"] = float(
            np.corrcoef(g.ravel(), np.abs(d["p_time"]).ravel())[0, 1])

        # A "freeze the gate at its mean" counterfactual is NOT reported: the
        # heads are trained jointly with a per-sample gate, so replacing g by a
        # constant afterwards is out of distribution and the resulting error is
        # meaningless.  The causal counterfactual is the retrained --no_gate run.

        # --- what explains g? experts vs event descriptor ------------------
        # window-level means, because the event descriptor is per window
        def r2(X, target):
            X = np.column_stack([X, np.ones(len(X))])
            coef, *_ = np.linalg.lstsq(X, target, rcond=None)
            pred = X @ coef
            return float(1 - ((target - pred) ** 2).sum()
                         / ((target - target.mean()) ** 2).sum())
        g_w = g.mean(1)
        exp_f = d["p_freq"].mean(1)
        exp_t = d["p_time"].mean(1)
        dis_w = dis.mean(1)
        out["gate_r2"] = {
            "experts_only": r2(np.column_stack([exp_f, exp_t, dis_w]), g_w),
            "events_only": r2(ev, g_w),
            "both": r2(np.column_stack([exp_f, exp_t, dis_w, ev]), g_w),
        }
        # gate regret of the learned gate vs the oracle
        out["gate_regret"] = None

    (RUNS / f"gate_analysis_h{H}_{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    cands = sorted(RUNS.glob(f"eventgate*-h{H}-s0"))
    for d in cands:
        if not (d / "model.pt").exists():
            continue
        cfg = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["config"]
        if cfg.get("n_q") not in (0, 1, None):
            print(f"skip {d.name} (quantile model, analysed by calibration.py)")
            continue
        if d.name.startswith("eventgate_v1"):
            print(f"skip {d.name} (single-channel frequency prototype; kept as a "
                  f"record of the v1 architecture in comparison_h24)")
            continue
        print(f"== {d.name}", flush=True)
        res = collect(d)
        a = analyse(res, d.name)
        if "gate_by_regime" in a:
            print(pd.DataFrame(a["gate_by_regime"]).to_string(index=False))
        if "expert_standalone_mae" in a:
            print("standalone expert MAE (diagnostic):",
                  json.dumps(a["expert_standalone_mae"], ensure_ascii=False))
            print("gate R2:", {k: round(v, 3) for k, v in a["gate_r2"].items()})
            print("corr(g, |time expert|): %.3f" % a["corr_gate_vs_abs_time_expert"])
            print("mean gate per horizon:",
                  np.round(a["gate_by_horizon"], 3).tolist())


if __name__ == "__main__":
    main()
