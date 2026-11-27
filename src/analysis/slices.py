"""Slice the test error by calendar / weather / event / demand regime.

Every slice reports n, MAE, RMSE, bias (mean signed error) and the mean target,
so a systematic over/under-estimation can be read directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
RUNS = Path("runs")
H = int(os.environ.get("EG_HORIZON", "24"))

WEATHER_CLASSES = ["Clear", "Clouds", "Rain", "Snow", "LowVis", "Thunder", "Squall"]
EV_CHANNELS = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "holiday_kernel"]
    + [f"wx_{c}" for c in WEATHER_CLASSES]
    + ["rain_6h_log", "snow_6h_log", "vol_delta_6h", "vol_zscore_abs"]
)


def load_meta() -> dict:
    w = np.load(PROC / f"windows_h{H}.npz")
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    te = np.where(split == "test")[0]
    ev = w["X_ev"][te]
    idx = {c: i for i, c in enumerate(EV_CHANNELS)}
    wx = ev[:, [idx[f"wx_{c}"] for c in WEATHER_CLASSES]]
    return {
        "idx": te,
        "y_hour": w["y_hour"][te],                      # [N, H] hour of day of each target step
        "y": w["y"][te],
        "wx": wx.argmax(1),
        "is_holiday": ev[:, idx["is_holiday"]],
        "holiday_kernel": ev[:, idx["holiday_kernel"]],
        "is_weekend": ev[:, idx["is_weekend"]],
        "rain_6h": np.expm1(ev[:, idx["rain_6h_log"]]),
        "snow_6h": np.expm1(ev[:, idx["snow_6h_log"]]),
        "vol_delta_6h": ev[:, idx["vol_delta_6h"]],
        "vol_zscore_abs": ev[:, idx["vol_zscore_abs"]],
        "t_test": w["y_time"][te],
    }


def slice_stats(y: np.ndarray, p: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """y, p : [N, H]; group : [N] -> one row per group value."""
    rows = []
    for g in np.unique(group):
        sel = group == g
        e = p[sel] - y[sel]
        rows.append({
            "group": g, "n": int(sel.sum()),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()),
            "mape": float((np.abs(e) / np.maximum(y[sel], 1)).mean() * 100),
            "y_mean": float(y[sel].mean()),
        })
    return pd.DataFrame(rows)


def build_slices(preds: dict[str, np.ndarray], meta: dict) -> dict:
    """preds: model -> [N, H] test predictions (aligned with meta['y'])."""
    y = meta["y"].astype(np.float64)
    regimes = {
        "window_hour_end": pd.to_datetime(meta["t_test"], unit="h").hour,
        "wday_type": np.where(meta["is_holiday"] > 0.5, "holiday",
                              np.where(meta["is_weekend"] > 0.5, "weekend", "workday")),
        "weather": np.array(WEATHER_CLASSES)[meta["wx"]],
        "demand_level": pd.qcut(y.mean(1), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(str),
        "vol_zscore": pd.cut(meta["vol_zscore_abs"], [-.01, 1, 2, 3, 99],
                             labels=["<1", "1-2", "2-3", ">3"]).astype(str),
        "rain_6h": pd.cut(meta["rain_6h"], [-.01, 0.01, 1, 5, 9999],
                          labels=["none", "<1mm", "1-5mm", ">5mm"]).astype(str),
        "snow_6h": pd.cut(meta["snow_6h"], [-.01, 0.01, 0.5, 9999],
                          labels=["none", "<0.5mm", ">=0.5mm"]).astype(str),
        "trend_6h": pd.qcut(meta["vol_delta_6h"], 4,
                            labels=["falling_fast", "falling", "rising", "rising_fast"]).astype(str),
        "h_step_bucket": np.full(len(y), "all"),
    }
    out: dict[str, pd.DataFrame] = {}
    for model, p in preds.items():
        for name, grp in regimes.items():
            df = slice_stats(y, np.asarray(p, dtype=np.float64), grp)
            df.insert(0, "model", model)
            out.setdefault(name, pd.DataFrame())
            out[name] = pd.concat([out[name], df], ignore_index=True)

    # per horizon step and per (horizon, hour-of-day)
    per_h = []
    for model, p in preds.items():
        p = np.asarray(p, dtype=np.float64)
        for h in range(y.shape[1]):
            e = p[:, h] - y[:, h]
            per_h.append({"model": model, "h": h + 1, "mae": float(np.abs(e).mean()),
                          "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())})
    out["per_horizon"] = pd.DataFrame(per_h)

    hod = []
    for model, p in preds.items():
        p = np.asarray(p, dtype=np.float64)
        for hh in range(24):
            mask = meta["y_hour"] == hh
            if mask.sum() == 0:
                continue
            e = p[mask] - y[mask]
            hod.append({"model": model, "hour": hh, "n": int(mask.sum()),
                        "mae": float(np.abs(e).mean()),
                        "rmse": float(np.sqrt((e ** 2).mean())),
                        "bias": float(e.mean())})
    out["horizon_hour"] = pd.DataFrame(hod)
    return out


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    meta = load_meta()
    preds = {}
    sn_path = RUNS / f"seasonal_preds_h{H}.npz"
    if sn_path.exists():
        sn = np.load(sn_path)
        preds["seasonal_avg_24_168"] = sn["seasonal_avg_24_168"]
        preds["seasonal_24"] = sn["seasonal_24"]
    lg_path = RUNS / f"lgbm_preds_h{H}.npz"
    if lg_path.exists():
        preds["lightgbm"] = np.load(lg_path)["pred_test"]
    for d in sorted(RUNS.glob(f"*-h{H}-s0")):
        if not (d / "preds.npz").exists():
            continue
        cfg = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["config"]
        if cfg.get("n_q") not in (0, 1, None):
            continue                      # quantile heads are scored by calibration.py
        if d.name.startswith("eventgate_v1"):
            continue                      # superseded prototype, kept for the ablation table
        preds[d.name.replace(f"-h{H}-s0", "")] = np.load(d / "preds.npz")["test_pred"]

    print("models:", list(preds))
    out = build_slices(preds, meta)
    outdir = RUNS / f"slices_h{H}"
    outdir.mkdir(exist_ok=True)
    for name, df in out.items():
        df.to_csv(outdir / f"{name}.csv", index=False)
    summary = {name: df.to_dict("records") for name, df in out.items()}
    (outdir / "slices.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(out["per_horizon"].pivot(index="h", columns="model", values="mae").round(1).to_string())
    print()
    print(out["wday_type"].pivot(index="group", columns="model", values="mae").round(1).to_string())
    print()
    print(out["weather"].pivot(index="group", columns="model", values="mae").round(1).to_string())


if __name__ == "__main__":
    main()
