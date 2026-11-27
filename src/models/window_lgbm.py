"""LightGBM that reads exactly the same window as the neural models.

Used for the missing-data robustness study: both families of models then see the
identical information set, and a corrupted window can be fed to both without
rebuilding the feature pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROC = Path("data/processed")
RUNS = Path("runs")
H = int(os.environ.get("EG_HORIZON", "24"))


def window_features(w: dict) -> tuple[np.ndarray, list[str]]:
    """[N, L, C] window -> [N, F] causal feature block at the window end."""
    vol = np.expm1(np.clip(w["X_num"][:, :, 0] * w["sd"][0] + w["mu"][0], -2, 12)
                   ).astype(np.float64)                       # [N, L]
    feats, names = [], []
    for lag in range(1, 25):
        feats.append(vol[:, -lag]); names.append(f"lag_{lag}")
    for win in (6, 24, 168):
        seg = vol[:, -win:]
        feats += [seg.mean(1), seg.std(1), seg.min(1), seg.max(1)]
        names += [f"roll_{win}_mean", f"roll_{win}_std", f"roll_{win}_min", f"roll_{win}_max"]
    feats += [vol[:, -1] - vol[:, -2], vol[:, -1] - vol[:, -25], vol[:, -1] - vol[:, -168]]
    names += ["diff_1", "diff_24", "diff_168"]

    for i, c in enumerate(["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                           "doy_sin", "doy_cos", "is_weekend"]):
        feats.append(w["X_cal"][:, -1, i]); names.append(c)
    for i, c in enumerate(["wx_Clear", "wx_Clouds", "wx_Rain", "wx_Snow",
                           "wx_LowVis", "wx_Thunder", "wx_Squall"]):
        feats.append(w["X_cat"][:, -1, i]); names.append(c)
    ev_names = (["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                 "is_holiday", "holiday_kernel",
                 "wx_Clear", "wx_Clouds", "wx_Rain", "wx_Snow", "wx_LowVis",
                 "wx_Thunder", "wx_Squall",
                 "rain_6h_log", "snow_6h_log", "vol_delta_6h", "vol_zscore_abs"])
    for i, c in enumerate(ev_names):
        feats.append(w["X_ev"][:, i]); names.append(f"ev_{c}")

    X = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def train_window_lgbm(w: dict, split: np.ndarray, seed: int = 0, rounds: int = 1200,
                      quiet: bool = False) -> dict:
    import lightgbm as lgb

    X, names = window_features(w)
    y = w["y"]
    tr, va = split == "train", split == "val"
    pred_va = np.zeros(((split == "val").sum(), y.shape[1]), dtype=np.float32)
    pred_te = np.zeros(((split == "test").sum(), y.shape[1]), dtype=np.float32)
    iters = []
    for h in range(y.shape[1]):
        params = dict(objective="regression", metric="l2", learning_rate=0.05,
                      num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, seed=seed, num_threads=8)
        dtr = lgb.Dataset(X[tr], label=y[tr, h])
        dva = lgb.Dataset(X[va], label=y[va, h], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        pred_va[:, h] = m.predict(X[va], num_iteration=m.best_iteration)
        pred_te[:, h] = m.predict(X[split == "test"], num_iteration=m.best_iteration)
        iters.append(int(m.best_iteration))
        if not quiet:
            print(f"  h{h + 1:02d}/{y.shape[1]} it={m.best_iteration}", flush=True)
    return {"pred_val": pred_va, "pred_test": pred_te, "best_iters": iters,
            "y_val": y[va], "y_test": y[split == "test"], "names": names}


def main() -> None:
    from src.train import metrics_multi

    w = dict(np.load(PROC / f"windows_h{H}.npz"))
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    res = train_window_lgbm(w, split)
    print("window-lgbm test:", metrics_multi(res["y_test"], res["pred_test"])["mean"])
    np.savez_compressed(RUNS / f"window_lgbm_h{H}.npz",
                        pred_val=res["pred_val"], pred_test=res["pred_test"],
                        y_val=res["y_val"], y_test=res["y_test"])


if __name__ == "__main__":
    main()
