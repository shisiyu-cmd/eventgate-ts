# EventGate-TS: event-gated mixture-of-experts for 24-hour traffic-volume forecasting

**Repo:** <https://github.com/shisiyu-cmd/eventgate-ts>

A small, fully reproducible machine-learning study on the UCI *Metro Interstate Traffic
Volume* dataset. It is organised as **data layer → method layer → analysis layer**, and it
reports what the experiments actually produced, including two hypotheses that failed.

📄 Full write-up (Chinese): [`REPORT.md`](REPORT.md) · data card: [`docs/data_card.md`](docs/data_card.md) ·
method notes: [`docs/method.md`](docs/method.md)

---

## Result at a glance

24-hour-ahead direct multi-step forecasting, test split = **2018-01-01 → 2018-09-30**
(5,945 windows × 24 steps = 142,680 points). Metrics are averages over the 24 horizons.

| Model | MAE | RMSE | MAPE % | R² | seeds |
|---|---|---|---|---|---|
| **LightGBM (per-horizon, 71 hand features)** | 229.0 | 380.3 | 11.19 | 0.963 | – |
| Window-LightGBM (features from the same 168 h window the NNs see) | **227.1** | 384.6 | 10.97 | 0.962 | – |
| PatchTST-lite | 258.1 ± 2.9 | 439.1 | 11.06 | 0.950 | 3 |
| **EventGate-TS + vol-stats** | 258.3 ± 2.7 | 434.2 | 11.11 | 0.951 | 3 |
| EventGate-TS (no frequency expert) | 260.2 | 440.1 | 11.58 | 0.949 | 1 |
| EventGate-TS | 262.5 ± 3.2 | 434.1 | 11.20 | 0.951 | 3 |
| EventGate-TS (fixed 0.5 gate) | 267.4 | 443.7 | 11.96 | 0.949 | 1 |
| EventGate-TS (no event descriptor) | 269.0 | 442.0 | 11.33 | 0.949 | 1 |
| EventGate-TS (no RevIN) | 271.4 | 446.0 | 11.80 | 0.948 | 1 |
| MLP | 341.5 | 529.5 | 13.93 | 0.926 | 1 |
| Seasonal naive (t−168 h) | 346.6 | 664.6 | 14.06 | 0.887 | – |
| LSTM | 354.6 | 624.6 | 14.43 | 0.896 | 1 |
| TCN | 400.1 | 636.3 | 16.14 | 0.894 | 1 |
| Seasonal naive (t−24 h) | 570.6 | 1036.7 | 25.34 | 0.724 | – |
| Persistence (t−1 h) | 2174.6 | 2721.4 | 183.4 | −0.899 | – |

Paired tests (Diebold-Mariano + Wilcoxon + 24 h block bootstrap) are in
`runs/comparison_h24/tests.csv`; every reported gap below is significant at p < 0.001
unless stated otherwise.

## Five findings

1. **Gradient-boosted trees win by ~29 MAE, and it is not a feature-engineering artifact.**
   Giving the neural net hand-made lag/rolling statistics (`--volstats`) recovers only
   4.2 MAE (262.5 → 258.3). The remaining gap survives, so the tree advantage comes from
   axis-aligned splits on lagged values, not from the feature set.
2. **The gate is a shrinkage controller, not an expert selector.** Removing the frequency
   expert costs 0.15 MAE (260.2 vs 260.4) — the frequency branch is redundant. But removing
   the *gate* costs 7.1 MAE and removing the *event descriptor* costs 8.6 MAE. The gate's value
   is "adaptively rescale the temporal expert's increment", not "route between two good experts".
   A linear model explains the gate from the 18-d event descriptor with R² = 0.953.
3. **The event descriptor's value is concentrated on holidays.** Dropping it costs
   +41.4 MAE on holiday windows (406.8 → 448.2) but only +6.9 on workdays.
4. **Native quantile heads under-sharpen.** EventGate-Q reaches PICP = 0.862 against a
   nominal 0.80 (interval width 999 veh/h, CRPS 184.9), while quantiles taken from
   LightGBM's validation residuals give PICP = 0.799 at width 670 and CRPS 165.8.
   Better point forecast ⇒ better interval, even with a much simpler calibration rule.
5. **Neural nets are far more fragile to a missing tail than trees.**
   Blanking the most recent hour raises PatchTST-lite's MAE from 255.7 to 889.5, while the
   window-LightGBM only moves from 227.1 to 229.4. The residual head anchors on
   `vol[-1]`, so a single missing hour destroys it — trees simply fall back on the other
   lags. See `figures/fig7_missing_robustness.png`.

## Data layer, in one paragraph

The raw file looks like 48k hourly rows but is not: 7,629 timestamps are duplicated
(volume identical, only weather labels conflict), 11,976 of the 52,551 grid hours are absent,
and **2014-09 → 2015-05 is a nine-month hole**. Windows (168 h) and targets (24 h) are
therefore only built inside contiguous observation segments, the target must be an actually
observed value, and interpolation is limited to gaps of ≤ 2 h with an explicit `vol_mask`
channel. This leaves 25,210 legal samples. Full evidence: [`docs/data_card.md`](docs/data_card.md).

## Repository layout

```
src/data/       audit.py (audit report) · build.py (hourly grid, windows, splits)
src/models/     nets.py (RevIN, FreqBranch, PatchTST-lite, LSTM/TCN/MLP, EventGate-TS)
                baselines.py (seasonal naives, per-horizon LightGBM)
                window_lgbm.py (LightGBM on features extracted from the 168 h window)
src/train.py    training loop, multi-step metrics, checkpoint + predictions
src/analysis/   compare.py (leaderboard + paired tests) · stats.py (DM / Wilcoxon / bootstrap)
                slices.py · gates.py · calibration.py · robustness.py · figures.py
scripts/        download_data.py · run_matrix.py · run_analysis.py
docs/           data_card.md · method.md
results/audit/  audit.json, audit.md (data-layer evidence, checked in)
```

`data/`, `runs/` and `figures/*.png` are regenerable and not versioned; every number in
`REPORT.md` is reproduced by the commands below.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py          # UCI zip -> data/raw/
python -m src.data.audit                 # -> results/audit/
python -m src.data.build                 # -> data/processed/windows_h24.npz
python -m src.models.baselines           # seasonal naives + LightGBM
python -m scripts.run_matrix --stage main       # neural models (3 seeds)
python -m scripts.run_matrix --stage ablation   # no_freq / no_gate / no_ev / no_revin / volstats
python -m scripts.run_matrix --stage quantile   # 3-quantile head
python -m scripts.run_analysis                  # slices, gates, calibration, robustness, figures
```

Set `EG_HORIZON=24` (default) and run from the repository root with `PYTHONPATH=.`.
A single neural model trains in 130–210 s on an RTX 4070 Laptop; the whole matrix is
about 40 minutes.

## Environment

Python 3.13, PyTorch 2.14 (CUDA 12.6), LightGBM 4.7, numpy / pandas / scipy / matplotlib.
No GPU is required (everything runs on CPU too, ~4× slower).

## Data

UCI Machine Learning Repository, *Metro Interstate Traffic Volume* (ID 492), CC BY 4.0.
`scripts/download_data.py` fetches it from `archive.ics.uci.edu`; the raw archive is not
committed.
