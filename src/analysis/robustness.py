"""Missing-input robustness: corrupt the tail of the input window and measure
how each model degrades.

Two corruption mechanisms:
  * ``gap``   : the k most recent hours are unavailable (sensor outage at the
                end of the window - the realistic failure for a loop detector).
  * ``block`` : a contiguous block of length k at a random position disappears.

Missing values are filled with the *training mean* of the log-volume channel
(0 in standardised space) and the observation mask is zeroed, for every model.
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


def corrupt(w: dict, idx: np.ndarray, mode: str, k: int, seed: int = 0) -> dict:
    """Return a copy of the window arrays with the volume channel masked."""
    out = {key: w[key][idx].copy() for key in ["X_num", "X_cal", "X_cat", "X_ev"]}
    X = out["X_num"]
    N, L, _ = X.shape
    rng = np.random.default_rng(seed)
    if mode == "gap":
        X[:, L - k:, 0] = 0.0
        X[:, L - k:, 1] = 0.0
    elif mode == "block":
        starts = rng.integers(0, L - k, size=N)
        for i, s in enumerate(starts):
            X[i, s:s + k, 0] = 0.0
            X[i, s:s + k, 1] = 0.0
    else:
        raise ValueError(mode)
    return out


@torch.no_grad()
def predict_nn(run_dir: Path, arrays: dict, device="cpu", bs=256) -> np.ndarray:
    model, cfg = load_model(run_dir, device)
    w = np.load(PROC / f"windows_h{H}.npz")
    mu, sd = float(w["mu"][0]), float(w["sd"][0])
    xs = [torch.from_numpy(arrays[k]) for k in ["X_num", "X_cal", "X_cat", "X_ev"]]
    preds = []
    for s in range(0, xs[0].shape[0], bs):
        xb = [x[s:s + bs].to(device) for x in xs]
        preds.append(model(*xb).float().cpu().numpy())
    # forward already applied RevIN.denormalize, so the output is a value in the
    # global standardised log1p space and the anchor is already inside it
    o = np.concatenate(preds).astype(np.float64)
    return np.expm1(np.clip(o * sd + mu, -2, 12))


def predict_lgbm(X: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Placeholder kept for symmetry - the window LightGBM is retrained per condition."""
    raise NotImplementedError


def main() -> None:
    from src.train import metrics_multi
    from src.models.window_lgbm import train_window_lgbm

    w = dict(np.load(PROC / f"windows_h{H}.npz"))
    split = np.load(PROC / f"split_h{H}.npy", allow_pickle=True)
    te = np.where(split == "test")[0]
    y = w["y"][te]

    nn_runs = {}
    for d in sorted(RUNS.glob(f"*-h{H}-s0")):
        if not (d / "model.pt").exists():
            continue
        cfg = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["config"]
        if cfg.get("n_q") not in (0, 1, None):
            continue                      # quantile heads are analysed separately
        nn_runs[d.name.replace(f"-h{H}-s0", "")] = d
    print("neural models:", list(nn_runs))

    results = []
    conditions = [("clean", "gap", 0)] + \
        [("gap", "gap", k) for k in (1, 3, 6, 12, 24, 48, 72)] + \
        [("block", "block", k) for k in (6, 12, 24, 48)]

    for tag, mode, k in conditions:
        arrays = corrupt(w, te, mode, k) if k else {key: w[key][te] for key in
                                                    ["X_num", "X_cal", "X_cat", "X_ev"]}
        for name, run in nn_runs.items():
            p = predict_nn(run, arrays)
            m = metrics_multi(y, p)["mean"]
            results.append({"condition": tag, "k": k, "model": name,
                            "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
            print(f"{tag:6s} k={k:02d} {name:22s} MAE={m['mae']:8.1f}", flush=True)

        # retrain the window LightGBM on the corrupted training split
        wtr = {kk: w[kk].copy() for kk in ["X_num", "X_cal", "X_cat", "X_ev", "y"]}
        wtr["sd"], wtr["mu"] = w["sd"], w["mu"]
        if k:
            full = corrupt(w, np.arange(len(w["X_num"])), mode, k)
            for kk in ["X_num", "X_cal", "X_cat", "X_ev"]:
                wtr[kk] = full[kk]
        from src.models.window_lgbm import window_features
        Xw, names = window_features(wtr)
        import lightgbm as lgb

        tr, va = split == "train", split == "val"
        pred = np.zeros((len(te), y.shape[1]), dtype=np.float32)
        for h in range(y.shape[1]):
            params = dict(objective="regression", metric="l2", learning_rate=0.05,
                          num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                          verbose=-1, seed=0, num_threads=8)
            dtr = lgb.Dataset(Xw[tr], label=w["y"][tr, h])
            dva = lgb.Dataset(Xw[va], label=w["y"][va, h], reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=1200, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            pred[:, h] = m.predict(Xw[te], num_iteration=m.best_iteration)
        mm = metrics_multi(y, pred)["mean"]
        results.append({"condition": tag, "k": k, "model": "window_lgbm",
                        "mae": mm["mae"], "rmse": mm["rmse"], "r2": mm["r2"]})
        print(f"{tag:6s} k={k:02d} {'window_lgbm':22s} MAE={mm['mae']:8.1f}", flush=True)

    (RUNS / f"robustness_h{H}.json").write_text(json.dumps(results, indent=2))
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RUNS / f"robustness_h{H}.csv", index=False)
    print(df.pivot_table(index=["condition", "k"], columns="model", values="mae").round(1).to_string())


if __name__ == "__main__":
    main()
