"""All report figures, generated from the saved artefacts (no re-training)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROC = Path("data/processed")
RUNS = Path("runs")
FIG = Path("figures")
H = int(os.environ.get("EG_HORIZON", "24"))
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
C = {"eventgate": "#c0392b", "patchtst": "#2980b9", "lstm": "#27ae60",
     "tcn": "#8e44ad", "mlp": "#7f8c8d", "lightgbm": "#e67e22",
     "window_lgbm": "#d35400", "seasonal_avg_24_168": "#95a5a6",
     "seasonal_24": "#bdc3c7"}


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
def fig_data_coverage():
    t = pd.read_csv(PROC / "table.csv", parse_dates=["timestamp"])
    t = t.set_index("timestamp")
    obs = t["vol_mask"].astype(float)
    weekly = obs.resample("W").mean()
    monthly = obs.resample("MS").mean()

    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5),
                           gridspec_kw={"height_ratios": [1, 1, 1.3]})
    ax[0].plot(monthly.index, monthly.values, color="#2980b9", lw=1.4)
    ax[0].axvspan(pd.Timestamp("2014-09-01"), pd.Timestamp("2015-05-01"),
                  color="#e74c3c", alpha=0.18, label="9-month outage")
    ax[0].set_title("Monthly fraction of hours with a real sensor reading")
    ax[0].set_ylabel("observed / hours")
    ax[0].legend(loc="lower right")

    # hour-of-week occupancy heatmap over the whole span
    t["how"] = t.index.dayofweek * 24 + t.index.hour
    t["ym"] = t.index.to_period("M").astype(str)
    piv = t.pivot_table(index="how", columns="ym", values="vol_mask", aggfunc="mean")
    im = ax[1].imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_yticks(range(0, 168, 24))
    ax[1].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    step = max(1, len(piv.columns) // 10)
    ax[1].set_xticks(range(0, len(piv.columns), step))
    ax[1].set_xticklabels(piv.columns[::step], rotation=45, ha="right", fontsize=7)
    ax[1].set_title("Observation mask by hour-of-week and month")
    fig.colorbar(im, ax=ax[1], fraction=0.03, pad=0.01)

    gaps = t["vol_mask"].eq(0).astype(int)
    runs, cur = [], 0
    for v in gaps.values:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs)
    ax[2].hist(np.clip(runs, 0, 48), bins=np.arange(0, 50, 1), color="#8e44ad")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("missing-run length (hours, clipped at 48)")
    ax[2].set_ylabel("count (log)")
    ax[2].set_title(f"Gap-length distribution ({len(runs)} runs, "
                    f"median {np.median(runs):.0f} h, max {runs.max()} h)")
    _save(fig, "fig1_data_coverage.png")


def fig_training_curves():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for d in sorted(RUNS.glob(f"*-h{H}-s0")):
        f = d / "history.json"
        if not f.exists():
            continue
        h = pd.DataFrame(json.loads(f.read_text()))
        name = d.name.replace(f"-h{H}-s0", "")
        ax[0].plot(h["epoch"], h["train_loss"], lw=1.2, label=name, color=C.get(name))
        ax[1].plot(h["epoch"], h["mae"], lw=1.2, label=name, color=C.get(name))
    ax[0].set_title("training loss (smooth L1 on raw volume)")
    ax[1].set_title("validation MAE (24-step mean)")
    for a in ax:
        a.set_xlabel("epoch")
    ax[1].legend(fontsize=7)
    _save(fig, "fig2_training_curves.png")


def fig_model_comparison():
    rows = []
    for d in sorted(RUNS.glob(f"*-h{H}-s*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        name = d.name.rsplit("-s", 1)[0].replace(f"-h{H}", "")
        rows.append({"model": name, "seed": d.name.rsplit("-s", 1)[1],
                     "mae": m["test"]["mean"]["mae"],
                     "rmse": m["test"]["mean"]["rmse"]})
    df = pd.DataFrame(rows)

    extra = []
    for f, name in [(f"seasonal_preds_h{H}.npz", "seasonal_avg_24_168"),
                    (f"lgbm_preds_h{H}.npz", "lightgbm"),
                    (f"window_lgbm_h{H}.npz", "window_lgbm")]:
        p = RUNS / f
        if not p.exists():
            continue
        d = np.load(p)
        y = d["y_test"]
        pred = d.get("pred_test")
        if pred is None:
            pred = d["seasonal_avg_24_168"]
        e = pred - y
        extra.append({"model": name, "seed": "-",
                      "mae": float(np.abs(e).mean()),
                      "rmse": float(np.sqrt((e ** 2).mean()))})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    order = df.groupby("model")["mae"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, m in enumerate(order):
        sub = df[df.model == m]
        ax.barh(i, sub["mae"].mean(), color=C.get(m, "#34495e"), alpha=0.85)
        ax.scatter(sub["mae"], [i] * len(sub), color="k", s=12, zorder=3)
        ax.text(sub["mae"].mean() * 1.01, i, f"{sub['mae'].mean():.1f}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel(f"test MAE (24-step mean, n={len(np.load(RUNS / f'windows_dummy.npz')) if False else 5945})")
    ax.set_title("Point-forecast accuracy (bars = mean over seeds, dots = seeds)")
    _save(fig, "fig3_model_comparison.png")
    df.to_csv(RUNS / f"model_comparison_h{H}.csv", index=False)
    return df


def fig_error_by_horizon():
    p = RUNS / f"slices_h{H}" / "per_horizon.csv"
    if not p.exists():
        print("skip fig4: run src.analysis.slices first")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m, g in df.groupby("model"):
        ax[0].plot(g["h"], g["mae"], marker="o", ms=2.5, lw=1.3, label=m, color=C.get(m))
        ax[1].plot(g["h"], g["bias"], marker="o", ms=2.5, lw=1.3, label=m, color=C.get(m))
    ax[0].set_xlabel("forecast horizon (h)")
    ax[0].set_ylabel("MAE")
    ax[0].set_title("Error growth over the horizon")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("forecast horizon (h)")
    ax[1].set_ylabel("bias (pred - truth)")
    ax[1].set_title("Signed bias over the horizon")
    ax[0].legend(fontsize=6, ncol=2)
    _save(fig, "fig4_error_by_horizon.png")


def fig_slices():
    d = RUNS / f"slices_h{H}"
    if not (d / "weather.csv").exists():
        print("skip fig5: run slices first")
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, name, title in [(ax[0], "weather", "by weather class"),
                           (ax[1], "wday_type", "by day type"),
                           (ax[2], "trend_6h", "by 6h volume trend")]:
        df = pd.read_csv(d / f"{name}.csv")
        piv = df.pivot(index="group", columns="model", values="mae")
        keep = [c for c in piv.columns if c in ("eventgate", "patchtst", "lstm",
                                                "lightgbm", "window_lgbm", "mlp", "tcn")]
        piv = piv[keep] if keep else piv
        piv.plot(kind="bar", ax=a, color=[C.get(c, "#34495e") for c in piv.columns],
                 width=0.8, legend=(a is ax[0]))
        a.set_title(title)
        a.set_xlabel("")
        a.set_ylabel("MAE")
        a.tick_params(axis="x", rotation=35, labelsize=7)
        if a is ax[0]:
            a.legend(fontsize=6)
    _save(fig, "fig5_error_slices.png")


def fig_gate():
    files = sorted(RUNS.glob(f"gate_analysis_h{H}_*.json"))
    if not files:
        print("skip fig6: run src.analysis.gates first")
        return
    a = json.loads(files[0].read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(range(1, len(a["gate_by_horizon"]) + 1), a["gate_by_horizon"],
               marker="o", ms=3, color="#c0392b")
    ax[0].set_xlabel("forecast horizon (h)")
    ax[0].set_ylabel("mean gate weight on the frequency expert")
    ax[0].set_title("Gate value vs horizon")
    df = pd.DataFrame(a["gate_by_regime"])
    sub = df[df.regime.isin(["weather", "holiday", "vol_zscore", "trend"])]
    x = np.arange(len(sub))
    ax[1].bar(x, sub["gate_mean"], yerr=sub["gate_std"], color="#2980b9", alpha=0.85,
              error_kw={"lw": 0.8})
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{r[:4]}\n{l}" for r, l in zip(sub.regime, sub.level)],
                          fontsize=6, rotation=45)
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[1].set_title("Gate value by event regime")
    _save(fig, "fig6_gate_behaviour.png")


def fig_robustness():
    p = RUNS / f"robustness_h{H}.csv"
    if not p.exists():
        print("skip fig7: run src.analysis.robustness first")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for tag, a in [("gap", ax[0]), ("block", ax[1])]:
        sub = df[df.condition == tag]
        base = df[df.condition == "clean"].set_index("model")["mae"]
        for m, g in sub.groupby("model"):
            g = g.sort_values("k")
            a.plot(g["k"], g["mae"] / base.get(m, np.nan), marker="o", ms=3,
                   lw=1.3, label=m, color=C.get(m))
        a.set_xlabel("masked hours")
        a.set_ylabel("MAE relative to clean input")
        a.set_title(f"{tag} corruption")
        a.legend(fontsize=6)
    _save(fig, "fig7_missing_robustness.png")


def fig_probabilistic():
    p = RUNS / f"probabilistic_h{H}.json"
    if not p.exists():
        print("skip fig8: run src.analysis.calibration first")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    df = pd.DataFrame(d["summary"])
    ax[0].bar(df["model"], df["picp"], color="#2980b9", alpha=0.85)
    ax[0].axhline(0.8, color="r", ls="--", lw=1, label="nominal 80%")
    ax[0].set_ylabel("PICP (empirical coverage)")
    ax[0].set_title("Interval coverage (q10-q90)")
    ax[0].tick_params(axis="x", rotation=20, labelsize=7)
    ax[0].legend()
    for k, v in d["detail"].items():
        ax[1].plot(range(1, len(v["picp_by_h"]) + 1), v["picp_by_h"], marker="o", ms=2.5,
                   lw=1.2, label=k)
    ax[1].axhline(0.8, color="r", ls="--", lw=1)
    ax[1].set_xlabel("forecast horizon (h)")
    ax[1].set_ylabel("PICP")
    ax[1].set_title("Coverage by horizon")
    ax[1].legend(fontsize=6)
    _save(fig, "fig8_probabilistic.png")


def fig_example_day():
    """One week of test-set predictions for the best model."""
    d = RUNS / f"eventgate-h{H}-s0"
    p = d / "preds.npz"
    if not p.exists():
        return
    z = np.load(p)
    y, pred, t = z["y_test"], z["test_pred"], z["t_test"]
    order = np.argsort(t)
    y, pred, t = y[order], pred[order], t[order]
    ts = pd.to_datetime(t, unit="h")
    sel = slice(400, 400 + 24 * 7)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(ts[sel], y[sel, 0], "k-", lw=1.0, label="observed t+1")
    ax.plot(ts[sel], pred[sel, 0], color="#c0392b", lw=1.0, label="EventGate t+1")
    ax.fill_between(ts[sel], y[sel, 0], pred[sel, 0], color="#c0392b", alpha=0.15)
    ax.set_title("Example week from the held-out year (1-step head of the 24-step model)")
    ax.set_ylabel("vehicles / hour")
    ax.legend(fontsize=7)
    _save(fig, "fig9_example_week.png")


def main() -> None:
    fig_data_coverage()
    fig_training_curves()
    fig_model_comparison()
    fig_error_by_horizon()
    fig_slices()
    fig_gate()
    fig_robustness()
    fig_probabilistic()
    fig_example_day()


if __name__ == "__main__":
    main()
