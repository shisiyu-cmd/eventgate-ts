"""Probabilistic evaluation: interval coverage, width, pinball loss and CRPS.

Three probabilistic forecasters are compared on the same test windows:
  * EventGate-Q  : native quantile head (0.1 / 0.5 / 0.9 trained jointly)
  * LGBM + residual quantiles : point forecast from the window LightGBM plus
    empirical quantiles of its own validation residuals (per horizon step)
  * seasonal + residual quantiles : same calibration idea on the seasonal mean
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from src.analysis.gates import load_model

PROC = Path("data/processed")
RUNS = Path("runs")
H = int(os.environ.get("EG_HORIZON", "24"))
QS = [0.1, 0.5, 0.9]


def pinball(y: np.ndarray, q_hat: np.ndarray, q: float) -> float:
    e = y - q_hat
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def interval_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "picp": float(inside.mean()),
        "pinaw": float(width.mean()),
        "pinaw_norm": float(width.mean() / (y.max() - y.min())),
        "mean_width": float(width.mean()),
    }


def crps_from_quantiles(y: np.ndarray, qs: np.ndarray, levels=QS) -> float:
    """Quantile-based CRPS approximation (mean of the pinball losses x 2)."""
    tot = 0.0
    for i, q in enumerate(levels):
        tot += pinball(y, qs[:, i], q)
    return float(2 * tot / len(levels))


@torch.no_grad()
def predict_quantiles(run_dir: Path, device="cpu", bs=256):
    import torch
    model, cfg = load_model(run_dir, device)
    H_ = cfg["horizon"]
    w = np.load(PROC / f"windows_h{H_}.npz")
    split = np.load(PROC / f"split_h{H_}.npy", allow_pickle=True)
    te = np.where(split == "test")[0]
    xs = [torch.from_numpy(w[k][te]) for k in ["X_num", "X_cal", "X_cat", "X_ev"]]
    mu, sd = w["mu"][0], w["sd"][0]
    outs = []
    for s in range(0, xs[0].shape[0], bs):
        xb = [x[s:s + bs].to(device) for x in xs]
        outs.append(model(*xb).float().cpu().numpy())
    o = np.concatenate(outs).astype(np.float64)                 # [N, 3*H]
    N = o.shape[0]
    o = o.reshape(N, 3, H_)
    # forward already applies RevIN.denormalize, so each channel is a value in
    # the global standardised log1p space - the anchor is inside `o`, it must
    # not be added a second time here.
    q = np.expm1(np.clip(o * sd + mu, -2, 12))                    # [N,3,H]
    return q, w["y"][te], te


def main() -> None:
    w = np.load(PROC / f"windows_h{H}.npz")
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    te = np.where(split == "test")[0]
    va = np.where(split == "val")[0]
    y = w["y"][te].astype(np.float64)
    y_va = w["y"][va].astype(np.float64)

    rows, detail = [], {}

    # ---- 1. native quantile head -----------------------------------------
    qruns = [d for d in sorted(RUNS.glob(f"eventgate_q*-h{H}-s0")) if (d / "model.pt").exists()]
    if qruns:
        q, _, _ = predict_quantiles(qruns[0])
        lo, mid, hi = q[:, 0], q[:, 1], q[:, 2]
        s = interval_scores(y, lo, hi)
        s.update({"model": "EventGate-Q", "pinball_10": pinball(y, lo, 0.1),
                  "pinball_50": pinball(y, mid, 0.5), "pinball_90": pinball(y, hi, 0.9),
                  "crps": crps_from_quantiles(y, q),
                  "mae_median": float(np.abs(mid - y).mean())})
        rows.append(s)
        detail["eventgate_q"] = {"picp_by_h": [float(((y[:, h] >= lo[:, h]) &
                                                      (y[:, h] <= hi[:, h])).mean())
                                               for h in range(H)]}
    else:
        print("no quantile run found - train with --n_q 3 first")

    # ---- 2/3. point forecast + residual quantiles ------------------------
    def residual_quantiles(pred_va, pred_te, tag):
        lo = np.zeros_like(pred_te)
        mid = pred_te.copy()
        hi = np.zeros_like(pred_te)
        for h in range(H):
            r = y_va[:, h] - pred_va[:, h]
            ql, qm, qh = np.quantile(r, [0.1, 0.5, 0.9])
            lo[:, h] = pred_te[:, h] + ql
            mid[:, h] = pred_te[:, h] + qm
            hi[:, h] = pred_te[:, h] + qh
        s = interval_scores(y, lo, hi)
        s.update({"model": tag, "pinball_10": pinball(y, lo, 0.1),
                  "pinball_50": pinball(y, mid, 0.5), "pinball_90": pinball(y, hi, 0.9),
                  "crps": crps_from_quantiles(y, np.stack([lo, mid, hi], 1)),
                  "mae_median": float(np.abs(mid - y).mean())})
        rows.append(s)
        detail[tag] = {"picp_by_h": [float(((y[:, h] >= lo[:, h]) &
                                            (y[:, h] <= hi[:, h])).mean()) for h in range(H)]}

    lg = RUNS / f"window_lgbm_h{H}.npz"
    if lg.exists():
        d = np.load(lg)
        residual_quantiles(d["pred_val"], d["pred_test"], "LGBM+resid-q")

    sn = RUNS / f"seasonal_preds_h{H}.npz"
    if sn.exists():
        d = np.load(sn)
        seasonal_va = d["seasonal_avg_24_168"]
        # rebuild validation predictions of the seasonal baseline
        Xv = w["X_num"][va]
        mu, sd = w["mu"][0], w["sd"][0]
        vol_v = np.expm1(Xv[:, :, 0] * sd + mu)
        L = vol_v.shape[1]
        idx24 = np.array([L - 24 + h for h in range(H)])
        idx168 = np.array([L - 168 + (h % 168) for h in range(H)])
        seas_va = (vol_v[:, idx24] + vol_v[:, idx168]) / 2
        residual_quantiles(seas_va, d["seasonal_avg_24_168"], "seasonal+resid-q")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / f"probabilistic_h{H}.csv", index=False)
    (RUNS / f"probabilistic_h{H}.json").write_text(
        json.dumps({"summary": rows, "detail": detail}, indent=2))
    print(df.round(3).to_string(index=False))
    for k, v in detail.items():
        print(k, "PICP per horizon:", [round(x, 2) for x in v["picp_by_h"]])


if __name__ == "__main__":
    main()
