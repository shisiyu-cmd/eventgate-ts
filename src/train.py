"""Train one neural forecaster on the traffic windows.

Example
-------
python -m src.train --model eventgate --seed 0 --epochs 80
python -m src.train --model eventgate --n_q 3 --tag quantile
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.nets import build_model, count_params

PROC = Path("data/processed")
RUNS = Path("runs")


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(horizon: int):
    w = np.load(PROC / f"windows_h{horizon}.npz")
    split = np.load(PROC / f"split_h{horizon}.npy", allow_pickle=True)
    tensors = {k: torch.from_numpy(w[k]) for k in ["X_num", "X_cal", "X_cat", "X_ev", "y"]}
    return tensors, split, w["mu"], w["sd"], w["y_time"]


def to_raw(pred_z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """standardised log space -> original volume scale (target channel only)"""
    logv = pred_z * sd[0] + mu[0]
    logv = logv.clamp(-2.0, 12.0)
    return torch.expm1(logv)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    err = p - y
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err ** 2))
    mask = y >= 50
    mape = np.mean(np.abs(err[mask]) / y[mask]) * 100 if mask.any() else float("nan")
    smape = np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(p) + 1e-8)) * 100
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape),
            "smape": float(smape), "r2": float(r2), "n": int(len(y))}


def metrics_multi(y: np.ndarray, p: np.ndarray) -> dict:
    """Per-horizon metrics plus the horizon average (the headline number)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim == 1:
        return {"mean": metrics(y, p), "per_h": {}}
    per_h = {f"h{h + 1}": metrics(y[:, h], p[:, h]) for h in range(y.shape[1])}
    keys = ["mae", "rmse", "mape", "smape", "r2"]
    agg = {k: float(np.nanmean([per_h[h][k] for h in per_h])) for k in keys}
    agg["n"] = int(y.size)
    return {"mean": agg, "per_h": per_h}


# --------------------------------------------------------------------------- #
def run_epoch(model, tensors, idx, mu, sd, opt=None, bs=128, quantiles=None,
              train=True, horizon=1):
    model.train(train)
    dev = mu.device
    order = np.random.permutation(idx) if train else idx
    total, n = 0.0, 0
    out_pred = np.zeros((len(idx), horizon), dtype=np.float32)
    nq = quantiles.shape[0] if quantiles is not None else 1
    for start in range(0, len(order), bs):
        sel = order[start:start + bs]
        xb = [tensors["X_num"][sel].to(dev), tensors["X_cal"][sel].to(dev),
              tensors["X_cat"][sel].to(dev), tensors["X_ev"][sel].to(dev)]
        yb = tensors["y"][sel].to(dev)
        if yb.ndim == 1:
            yb = yb.unsqueeze(-1)
        with torch.set_grad_enabled(train):
            out = model(*xb)                                    # [B, nq*H] or [B, H]
            if out.shape[1] == 3 * horizon and nq == 3:
                out = out.view(-1, 3, horizon)
                raw = to_raw(out, mu, sd)
                e = yb.unsqueeze(1) - raw                        # [B,3,H]
                q = quantiles.view(1, -1, 1)
                pin = torch.maximum(q * e, (q - 1) * e).mean()
                loss = pin + 0.5 * F.l1_loss(raw[:, 1, :], yb)
                p = raw[:, 1, :]
            else:
                raw = to_raw(out, mu, sd)
                if raw.shape[1] != horizon:
                    raw = raw[:, :horizon]
                loss = F.smooth_l1_loss(raw, yb, beta=100.0)
                p = raw
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        total += float(loss.detach()) * len(sel)
        n += len(sel)
        out_pred[start:start + len(sel)] = p.detach().cpu().numpy()
    return total / max(n, 1), out_pred


def collect_gates(model, tensors, idx, mu, sd, bs=256, horizon=1):
    """Mean gate value per sample (needs a model exposing `last_gate`)."""
    model.eval()
    dev = mu.device
    gates = []
    with torch.no_grad():
        for start in range(0, len(idx), bs):
            sel = idx[start:start + bs]
            xb = [tensors["X_num"][sel].to(dev), tensors["X_cal"][sel].to(dev),
                  tensors["X_cat"][sel].to(dev), tensors["X_ev"][sel].to(dev)]
            try:
                model(*xb, return_gate=True)
            except TypeError:
                model(*xb)
            g = getattr(model, "last_gate", None)
            if g is None:
                return None
            gates.append(g.float().cpu().numpy())
    return np.concatenate(gates, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eventgate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("EG_HORIZON", "24")))
    ap.add_argument("--n_q", type=int, default=0, help="0 -> one output per horizon step")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patch_len", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k_freq", type=int, default=8)
    ap.add_argument("--no_freq", action="store_true")
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--no_ev", action="store_true")
    ap.add_argument("--no_revin", action="store_true")
    ap.add_argument("--volstats", action="store_true",
                    help="append hand-designed lag/rolling descriptors to the deep model")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="smoke test on 2 epochs")
    args = ap.parse_args()

    H = args.horizon
    # one output per horizon step; with --n_q 3 the head emits 3 quantiles per step
    if args.n_q == 3:
        n_q = 3 * H
    elif args.n_q in (0, 1):
        n_q = H
    else:
        n_q = args.n_q
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors, split, mu, sd, y_time = load_data(H)
    mu_t = torch.tensor(mu, device=device)
    sd_t = torch.tensor(sd, device=device)

    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]

    kw = dict(d_model=args.d_model, n_layers=args.layers,
              patch_len=args.patch_len, stride=args.stride)
    if args.model == "eventgate":
        kw.update(n_q=n_q, k_freq=args.k_freq, use_volstats=args.volstats,
                  use_freq=not args.no_freq, use_gate=not args.no_gate,
                  use_ev=not args.no_ev, use_revin=not args.no_revin)
    elif args.model == "patchtst":
        kw.update(n_q=n_q)
    elif args.model == "lstm":
        kw = dict(hidden=args.d_model, n_q=n_q)
    elif args.model == "tcn":
        kw = dict(channels=args.d_model, dilations=(1, 2, 4, 8, 16, 32), n_q=n_q)
    elif args.model == "mlp":
        kw = dict(n_q=n_q)

    model = build_model(args.model, **kw).to(device)
    n_params = count_params(model)

    quantiles = None
    if args.n_q == 3:
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    epochs = 2 if args.quick else args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=epochs * math.ceil(len(tr) / args.bs),
        pct_start=0.15, anneal_strategy="cos")

    name = f"{args.model}{'_vs' if args.volstats else ''}{'_q' if args.n_q == 3 else ''}"
    for flag, suf in [("no_freq", "-nofreq"), ("no_gate", "-nogate"),
                      ("no_ev", "-noev"), ("no_revin", "-norevin")]:
        if getattr(args, flag):
            name += suf
    if args.tag:
        name += f"-{args.tag}"
    name += f"-h{H}"
    run_dir = RUNS / f"{name}-s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = {"val_mae": float("inf"), "epoch": -1}
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, _ = run_epoch(model, tensors, tr, mu_t, sd_t, opt, args.bs,
                               quantiles, True, H)
        sched.step()
        va_loss, va_pred = run_epoch(model, tensors, va, mu_t, sd_t, None, args.bs,
                                     quantiles, False, H)
        vm = metrics_multi(tensors["y"][va].numpy(), va_pred)["mean"]
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, **vm})
        if vm["mae"] < best["val_mae"]:
            best = {"val_mae": vm["mae"], "epoch": ep}
            torch.save(model.state_dict(), run_dir / "model.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"[{name}] ep{ep:02d} tr={tr_loss:.2f} va_loss={va_loss:.2f} "
                  f"va_MAE={vm['mae']:.1f} (best {best['val_mae']:.1f}@{best['epoch']})",
                  flush=True)
        if ep - best["epoch"] >= args.patience:
            print(f"[{name}] early stop at ep{ep}", flush=True)
            break

    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    _, val_pred = run_epoch(model, tensors, va, mu_t, sd_t, None, args.bs, quantiles, False, H)
    _, test_pred = run_epoch(model, tensors, te, mu_t, sd_t, None, args.bs, quantiles, False, H)
    gate_test = collect_gates(model, tensors, te, mu_t, sd_t, horizon=H)

    res = {
        "config": vars(args),
        "n_params": n_params,
        "best_epoch": best["epoch"],
        "val": metrics_multi(tensors["y"][va].numpy(), val_pred),
        "test": metrics_multi(tensors["y"][te].numpy(), test_pred),
        "train_seconds": round(time.time() - t0, 1),
        "device": str(device),
    }
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    payload = dict(val_pred=val_pred, test_pred=test_pred,
                   y_val=tensors["y"][va].numpy(), y_test=tensors["y"][te].numpy(),
                   t_val=y_time[va], t_test=y_time[te], idx_test=te, idx_val=va)
    if gate_test is not None:
        payload["gate_test"] = gate_test
    np.savez_compressed(run_dir / "preds.npz", **payload)
    print(json.dumps({k: res[k] for k in ["val", "test", "n_params", "best_epoch"]},
                     indent=2, ensure_ascii=False))
    print("test mean:", json.dumps(res["test"]["mean"]))


if __name__ == "__main__":
    main()
