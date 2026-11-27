"""Aggregate every run into one comparison table with paired significance tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import dm_test, wilcoxon_test, block_bootstrap_ci

RUNS = Path("runs")
H = int(os.environ.get("EG_HORIZON", "24"))
BASELINE_FILES = {
    "seasonal_24": f"seasonal_preds_h{H}.npz",
    "seasonal_avg_24_168": f"seasonal_preds_h{H}.npz",
    "lightgbm": f"lgbm_preds_h{H}.npz",
    "window_lgbm": f"window_lgbm_h{H}.npz",
}


def collect() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Return (metrics table, {model: flat prediction}, flat truth)."""
    rows, preds, truth = [], {}, None

    for d in sorted(RUNS.glob(f"*-h{H}-s*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        name = d.name.rsplit("-s", 1)[0].replace(f"-h{H}", "")
        seed = int(d.name.rsplit("-s", 1)[1])
        rows.append({"model": name, "seed": seed, "params": m.get("n_params"),
                     "best_epoch": m.get("best_epoch"),
                     **{f"test_{k}": v for k, v in m["test"]["mean"].items()},
                     **{f"val_{k}": v for k, v in m["val"]["mean"].items()}})
        z = np.load(d / "preds.npz")
        preds[f"{name}|s{seed}"] = z["test_pred"].reshape(-1)
        if truth is None:
            truth = z["y_test"].reshape(-1)

    # seasonal baselines live in one archive, lgbm ones in another
    sp = RUNS / BASELINE_FILES["seasonal_24"]
    if sp.exists():
        z = np.load(sp)
        for k in ("seasonal_24", "seasonal_168", "seasonal_avg_24_168",
                  "naive_t-1", "drift_24"):
            if k in z:
                rows.append({"model": k, "seed": -1, "params": 0, "best_epoch": 0,
                             **{f"test_{m}": v for m, v in _metrics(z["y_test"], z[k]).items()}})
                preds[f"{k}|s-1"] = z[k].reshape(-1)
    lp = RUNS / BASELINE_FILES["lightgbm"]
    if lp.exists():
        z = np.load(lp)
        key = "pred_test" if "pred_test" in z else "lightgbm"
        rows.append({"model": "lightgbm", "seed": -1, "params": 0, "best_epoch": 0,
                     **{f"test_{m}": v for m, v in _metrics(z["y_test"], z[key]).items()}})
        preds["lightgbm|s-1"] = z[key].reshape(-1)
    wp = RUNS / BASELINE_FILES["window_lgbm"]
    if wp.exists():
        z = np.load(wp)
        key = "pred_test" if "pred_test" in z else z.files[0]
        rows.append({"model": "window_lgbm", "seed": -1, "params": 0, "best_epoch": 0,
                     **{f"test_{m}": v for m, v in _metrics(z["y_test"], z[key]).items()}})
        preds["window_lgbm|s-1"] = z[key].reshape(-1)

    return pd.DataFrame(rows), preds, truth


def _metrics(y, p) -> dict:
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    e = p - y
    denom = np.abs(y) + np.abs(p)
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mape": float(np.mean(100 * np.abs(e) / np.clip(np.abs(y), 1.0, None))),
        "smape": float(np.mean(200 * np.abs(e) / np.clip(denom, 1e-8, None))),
        "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "n": int(e.size),
    }


def main() -> None:
    df, preds, truth = collect()
    if df.empty:
        print("no runs found")
        return
    out = RUNS / f"comparison_h{H}"
    out.mkdir(exist_ok=True)

    # ---- summary: mean over seeds ---------------------------------------- #
    agg = (df.groupby("model")
             .agg(mae=("test_mae", "mean"), mae_std=("test_mae", "std"),
                  rmse=("test_rmse", "mean"), mape=("test_mape", "mean"),
                  r2=("test_r2", "mean"), r2_std=("test_r2", "std"),
                  n_seeds=("seed", "nunique"), params=("params", "max"))
             .sort_values("mae"))
    agg["mae_std"] = agg["mae_std"].fillna(0.0)
    agg.to_csv(out / "summary.csv")
    print(agg.round(3).to_string())

    # ---- paired tests on the horizon-flattened point forecasts ----------- #
    pairs = [("eventgate", "patchtst"), ("eventgate", "lightgbm"),
             ("window_lgbm", "lightgbm"), ("patchtst", "lightgbm"),
             ("eventgate", "seasonal_avg_24_168"),
             ("eventgate_vs", "patchtst"), ("eventgate_vs", "lightgbm"),
             ("eventgate", "eventgate_vs"), ("eventgate", "eventgate-nofreq"),
             ("eventgate", "eventgate-noev"), ("eventgate", "eventgate-nogate")]
    tests = []
    for a, b in pairs:
        ka = next((k for k in preds if k.startswith(a + "|")), None)
        kb = next((k for k in preds if k.startswith(b + "|")), None)
        if not ka or not kb:
            continue
        pa, pb = preds[ka], preds[kb]
        if pa.shape != pb.shape:
            continue
        dm = dm_test(truth, pa, pb, h=24)          # H0: equal expected loss
        w = wilcoxon_test(truth, pa, pb)
        bs = block_bootstrap_ci(truth, pa, pb, block=24, n_boot=1000)
        # one sign convention everywhere: delta = MAE(a) - MAE(b), so a negative
        # delta always means model_a is better.  block_bootstrap_ci returns the
        # CI of (mae2 - mae1), hence the negation and the swapped bounds.
        tests.append({
            "model_a": ka.split("|")[0], "model_b": kb.split("|")[0],
            "mae_a": bs["mae1"], "mae_b": bs["mae2"],
            "mae_a_ci": bs["mae1_ci"], "mae_b_ci": bs["mae2_ci"],
            "delta_mae": bs["mae1"] - bs["mae2"],
            "delta_ci_lo": -bs["diff_ci"][1], "delta_ci_hi": -bs["diff_ci"][0],
            "bootstrap_p": bs["p_two_sided"],
            "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"],
            "wilcoxon_p": w["p_value"],
        })
    tdf = pd.DataFrame(tests)
    tdf.to_csv(out / "tests.csv", index=False)
    if not tdf.empty:
        print("\npaired tests (negative delta = model_a better)\n", tdf.round(4).to_string())
    (out / "tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
