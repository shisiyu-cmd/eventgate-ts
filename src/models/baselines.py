"""Classical baselines: seasonal naive and gradient-boosted trees with causal features.

Every covariate attached to window end `t` uses information available at `t` or
earlier; the forecast targets are t+1 .. t+H.
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


def feature_matrix() -> tuple[np.ndarray, list[str], dict]:
    t = pd.read_csv(PROC / "table.csv", parse_dates=["timestamp"])
    w = np.load(PROC / f"windows_h{H}.npz")
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    end = w["x_end"]

    vol = np.log1p(t["traffic_volume"].to_numpy(dtype=np.float64))
    vol_s = pd.Series(vol)

    lags = [1, 2, 3, 4, 5, 6, 12, 24, 25, 48, 72, 168]
    feats, names = [], []

    def take(series: np.ndarray) -> np.ndarray:
        return series[end]

    for L in lags:
        # window ends at t, target starts at t+1 -> lag_1 is shift(0)
        feats.append(take(vol_s.shift(L - 1).to_numpy())); names.append(f"lag_{L}")

    for win_len, tag in [(6, "6h"), (24, "24h"), (168, "168h")]:
        r = vol_s.rolling(win_len, min_periods=max(2, win_len // 2))
        for stat in ["mean", "std", "min", "max"]:
            feats.append(take(getattr(r, stat)().to_numpy())); names.append(f"roll_{tag}_{stat}")

    for d in [1, 24, 168]:
        feats.append(take(vol_s.diff(d).to_numpy())); names.append(f"diff_{d}")

    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
              "is_weekend", "is_holiday", "holiday_kernel",
              "temp_c", "rain_1h", "snow_1h", "clouds_all",
              "rain_6h", "snow_6h", "vol_zscore_abs"]:
        feats.append(take(t[c].to_numpy(dtype=np.float64))); names.append(c)

    for c in ["wx_Clear", "wx_Clouds", "wx_Rain", "wx_Snow", "wx_LowVis", "wx_Thunder", "wx_Squall"]:
        feats.append(take(t[c].to_numpy(dtype=np.float64))); names.append(c)

    X = np.stack(feats, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names, {"split": split, "y": w["y"], "x_end": end}


def seasonal_naive() -> dict:
    w = np.load(PROC / f"windows_h{H}.npz")
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    X = w["X_num"]
    mu, sd = w["mu"][0], w["sd"][0]
    vol = np.expm1(X[:, :, 0] * sd + mu)          # [N, L]
    L = vol.shape[1]
    y = w["y"]                                    # [N, H]
    out = {"y": y, "split": split}

    # persistence: every future step predicted with the last observed value
    out["naive_t-1"] = np.repeat(vol[:, -1:], H, axis=1)
    # seasonal: the value observed k*24h before the target step
    idx24 = np.array([L - 24 + h for h in range(H)])
    out["seasonal_24"] = vol[:, idx24]
    idx168 = np.array([L - 168 + (h % 168) for h in range(H)])
    out["seasonal_168"] = vol[:, idx168]
    out["seasonal_avg_24_168"] = (out["seasonal_24"] + out["seasonal_168"]) / 2
    # drift: repeat the last value plus the mean 24h slope
    slope = (vol[:, -1] - vol[:, -25]) / 24.0
    out["drift_24"] = vol[:, -1:] + slope[:, None] * np.arange(1, H + 1)[None, :]
    return out


def train_lgbm(seed: int = 0, rounds: int = 1500) -> dict:
    """One LightGBM per horizon step (direct multi-step)."""
    import lightgbm as lgb

    X, names, d = feature_matrix()
    split, y = d["split"], d["y"]
    tr, va, te = split == "train", split == "val", split == "test"

    pred_va = np.zeros((va.sum(), H), dtype=np.float32)
    pred_te = np.zeros((te.sum(), H), dtype=np.float32)
    iters, imp = [], {}
    for h in range(H):
        params = dict(objective="regression", metric="l2",
                      learning_rate=0.05, num_leaves=127, min_data_in_leaf=20,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbose=-1, seed=seed, num_threads=8)
        dtr = lgb.Dataset(X[tr], label=y[tr, h], feature_name=names)
        dva = lgb.Dataset(X[va], label=y[va, h], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        pred_va[:, h] = m.predict(X[va], num_iteration=m.best_iteration)
        pred_te[:, h] = m.predict(X[te], num_iteration=m.best_iteration)
        iters.append(int(m.best_iteration))
        for k, v in zip(names, m.feature_importance("gain")):
            imp[k] = imp.get(k, 0.0) + float(v)
        print(f"  h{h + 1:02d} best_iter={m.best_iteration}", flush=True)
    return {
        "pred_val": pred_va, "pred_test": pred_te,
        "y_val": y[va], "y_test": y[te],
        "best_iters": iters,
        "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    from src.train import metrics_multi

    RUNS.mkdir(exist_ok=True)
    sn = seasonal_naive()
    te_mask = sn["split"] == "test"
    rows = {}
    for name in ["naive_t-1", "seasonal_24", "seasonal_168", "seasonal_avg_24_168", "drift_24"]:
        m = metrics_multi(sn["y"][te_mask], sn[name][te_mask])
        rows[name] = {"test": m["mean"], "n_params": 0}
        print(name, json.dumps(m["mean"]), flush=True)
    np.savez_compressed(RUNS / f"seasonal_preds_h{H}.npz",
                        y_test=sn["y"][te_mask],
                        **{n: sn[n][te_mask] for n in
                           ["naive_t-1", "seasonal_24", "seasonal_168",
                            "seasonal_avg_24_168", "drift_24"]})

    print("training lightgbm (direct multi-step) ...", flush=True)
    res = train_lgbm()
    m = metrics_multi(res["y_test"], res["pred_test"])
    rows["lightgbm"] = {"test": m["mean"], "n_params": int(sum(res["best_iters"])),
                        "best_iters": res["best_iters"]}
    print("lightgbm", json.dumps(m["mean"]), flush=True)
    np.savez_compressed(RUNS / f"lgbm_preds_h{H}.npz", **res)
    (RUNS / f"lgbm_importance_h{H}.json").write_text(
        json.dumps(res["importance"], indent=2))
    (RUNS / f"baselines_summary_h{H}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
