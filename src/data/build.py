"""Build the modelling table: dedup -> hour grid -> cleaning -> causal features -> splits.

Design decisions (all recorded in results/audit/data_card.md):
  * 7,629 duplicate timestamps are collapsed (traffic_volume is provably identical
    inside a duplicated hour; only weather_description differs).
  * Structural outages (gap > 2h) split the series into independent segments;
    short holes (<= 2h) are linearly interpolated and flagged by `vol_mask`.
  * US federal holidays are generated from rules instead of trusting the sparse
    `holiday` column, so every year gets the same 11 holidays.
  * Every covariate at step t uses information available at t or earlier.

Output: data/processed/table.csv, data/processed/windows.npz, data/processed/meta.json
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/Metro_Interstate_Traffic_volume.csv.gz")
PROC = Path("data/processed")

L_WINDOW = 168          # 7 days of history
HORIZON = int(__import__("os").environ.get("EG_HORIZON", "24"))
TRAIN_END = pd.Timestamp("2017-06-30 23:00:00")
VAL_END = pd.Timestamp("2017-12-31 23:00:00")
TEST_END = pd.Timestamp("2018-09-30 23:00:00")

WEATHER_MAP = {
    "Clear": "Clear",
    "Clouds": "Clouds",
    "Rain": "Rain", "Drizzle": "Rain",
    "Snow": "Snow",
    "Mist": "LowVis", "Fog": "LowVis", "Haze": "LowVis", "Smoke": "LowVis",
    "Thunderstorm": "Thunder",
    "Squall": "Squall",
}
WEATHER_CLASSES = ["Clear", "Clouds", "Rain", "Snow", "LowVis", "Thunder", "Squall"]

NUM_CHANNELS = ["vol_z", "vol_mask", "temp_c", "rain_log", "snow_log", "clouds"]
CAL_CHANNELS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
EV_CHANNELS = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "holiday_kernel"]
    + [f"wx_{c}" for c in WEATHER_CLASSES]
    + ["rain_6h_log", "snow_6h_log", "vol_delta_6h", "vol_zscore_abs"]
)


# --------------------------------------------------------------------------- #
# holidays
# --------------------------------------------------------------------------- #
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday of a month; n=-1 means last."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    d = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def us_federal_holidays(year: int) -> list[date]:
    return [
        date(year, 1, 1),                      # New Year
        _nth_weekday(year, 1, 0, 3),           # MLK
        _nth_weekday(year, 2, 0, 3),           # Presidents
        _nth_weekday(year, 5, 0, -1),          # Memorial
        date(year, 7, 4),                      # Independence
        _nth_weekday(year, 9, 0, 1),           # Labor
        _nth_weekday(year, 10, 0, 2),          # Columbus
        date(year, 11, 11),                    # Veterans
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving
        date(year, 12, 25),                    # Christmas
    ]


def holiday_kernel(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Binary flag for the holiday itself plus a gaussian proximity kernel (+-3 days)."""
    days = {d: True for y in range(2011, 2020) for d in us_federal_holidays(y)}
    day_keys = index.normalize()
    is_h = np.array([1.0 if k.date() in days else 0.0 for k in day_keys], dtype=np.float32)
    # distance in days to the nearest holiday day
    all_days = np.array(sorted(days.keys()), dtype="datetime64[D]")
    idx = index.normalize().values.astype("datetime64[D]")
    pos = np.searchsorted(all_days, idx)
    dist = np.full(len(index), 99.0)
    for cand in (pos - 1, pos):
        valid = (cand >= 0) & (cand < len(all_days))
        d = np.abs((all_days[np.clip(cand, 0, len(all_days) - 1)] - idx).astype("timedelta64[D]").astype(float))
        dist = np.where(valid, np.minimum(dist, d), dist)
    kern = np.exp(-(dist / 1.5) ** 2).astype(np.float32)
    return is_h, kern


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def load_dedup() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW)
    df["date_time"] = pd.to_datetime(df["date_time"])
    n_raw = len(df)
    n_dup_ts = int(df.duplicated(subset="date_time").sum())

    # volume must be constant inside a duplicated timestamp (verified in audit)
    spread = df.groupby("date_time")["traffic_volume"].agg(lambda s: s.max() - s.min())
    assert spread.max() == 0, "duplicated timestamps disagree on traffic_volume"

    agg = (
        df.sort_values(["date_time", "weather_main"])
        .groupby("date_time", as_index=True)
        .agg(
            traffic_volume=("traffic_volume", "first"),
            temp=("temp", "first"),
            rain_1h=("rain_1h", "first"),
            snow_1h=("snow_1h", "first"),
            clouds_all=("clouds_all", "first"),
            weather_main=("weather_main", lambda s: s.mode().iat[0]),
            holiday=("holiday", "first"),
            n_dup=("traffic_volume", "size"),
        )
    )
    return agg, {"n_raw": n_raw, "n_dup_ts": n_dup_ts, "n_unique": len(agg)}


def build_table() -> tuple[pd.DataFrame, dict]:
    agg, info = load_dedup()

    # --- put on a strict hourly grid -------------------------------------- #
    grid = pd.date_range(agg.index.min(), agg.index.max(), freq="h")
    t = agg.reindex(grid)
    info["n_grid"] = len(grid)

    # --- flag & clean impossible / corrupted values ----------------------- #
    n_zero = int((t.traffic_volume == 0).sum())
    t.loc[t.traffic_volume == 0, "traffic_volume"] = np.nan

    n_temp0 = int((t.temp == 0).sum())
    t.loc[t.temp == 0, "temp"] = np.nan

    n_rain_bad = int((t.rain_1h > 60).sum())   # >60 mm/h in Minnesota: unit/entry error
    t.loc[t.rain_1h > 60, "rain_1h"] = np.nan

    # --- interpolate short holes (<= 2 consecutive hours) ---------------- #
    observed = t.traffic_volume.notna()
    cols_num = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
    filled = t[cols_num].interpolate(method="time", limit=2, limit_area="inside")
    # 1 = real sensor reading, 0 = interpolated (or structurally missing)
    vol_mask = observed.astype(np.float32)
    info["n_interp"] = int(filled.traffic_volume.notna().sum() - observed.sum())
    info["n_observed"] = int(observed.sum())

    t["vol_mask"] = vol_mask
    for c in cols_num:
        t[c] = filled[c]

    # --- weather classes --------------------------------------------------- #
    t["weather_main"] = t["weather_main"].fillna("Clear")
    t["wx"] = t["weather_main"].map(WEATHER_MAP).fillna("Clouds")
    for c in WEATHER_CLASSES:
        t[f"wx_{c}"] = (t["wx"] == c).astype(np.float32)

    # --- calendar ---------------------------------------------------------- #
    idx = t.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    t["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    t["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    t["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    t["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["is_weekend"] = (dow >= 5).astype(np.float32)
    is_h, kern = holiday_kernel(idx)
    t["is_holiday"] = is_h
    t["holiday_kernel"] = kern

    # --- short-horizon weather aggregates (causal: trailing 6h) ----------- #
    t["rain_6h"] = t["rain_1h"].fillna(0).rolling(6, min_periods=1).sum()
    t["snow_6h"] = t["snow_1h"].fillna(0).rolling(6, min_periods=1).sum()
    t["rain_6h_log"] = np.log1p(t["rain_6h"])
    t["snow_6h_log"] = np.log1p(t["snow_6h"])

    # --- volume derived (causal) ------------------------------------------ #
    t["vol_log"] = np.log1p(t["traffic_volume"])
    t["vol_delta_6h"] = t["vol_log"] - t["vol_log"].shift(6)
    roll_mean = t["vol_log"].rolling(168, min_periods=24).mean()
    roll_std = t["vol_log"].rolling(168, min_periods=24).std()
    t["vol_zscore_abs"] = ((t["vol_log"] - roll_mean) / (roll_std + 1e-6)).abs().clip(0, 10)

    t["temp_c"] = t["temp"] - 273.15

    # model-facing channels (vol_z is filled with log1p then z-scored on train stats)
    t["vol_z"] = t["vol_log"]
    t["rain_log"] = np.log1p(t["rain_1h"].fillna(0))
    t["snow_log"] = np.log1p(t["snow_1h"].fillna(0))
    t["clouds"] = t["clouds_all"].fillna(0) / 100.0

    # --- segment id: one id per contiguous observed block ----------------- #
    seg = (t["traffic_volume"].isna() != t["traffic_volume"].isna().shift()).cumsum()
    t["seg"] = np.where(t["traffic_volume"].notna(), seg, -1)

    return t, info


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #
def make_windows(t: pd.DataFrame, info: dict) -> dict:
    num = t[NUM_CHANNELS].to_numpy(dtype=np.float32)
    cal = t[CAL_CHANNELS].to_numpy(dtype=np.float32)
    cat = t[[f"wx_{c}" for c in WEATHER_CLASSES]].to_numpy(dtype=np.float32)
    ev = t[EV_CHANNELS].to_numpy(dtype=np.float32)
    y_raw = t["traffic_volume"].to_numpy(dtype=np.float32)
    ts = t.index
    has_value = t["traffic_volume"].notna().to_numpy()
    observed = t["vol_mask"].to_numpy() > 0.5

    # standardisation uses TRAIN rows only (no leakage)
    tr = ts <= TRAIN_END
    ref = num[tr & has_value]
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    sd[sd < 1e-6] = 1.0
    num_z = (num - mu) / sd
    num_z = np.nan_to_num(num_z, nan=0.0, posinf=0.0, neginf=0.0)
    # keep the mask channel binary
    num_z[:, NUM_CHANNELS.index("vol_mask")] = num[:, NUM_CHANNELS.index("vol_mask")]

    n = len(t)
    seg_arr = t["seg"].to_numpy()
    # c[i] = number of segment changes in [0, i); a span is contiguous iff c[b]-c[a]==0
    change = (seg_arr[1:] != seg_arr[:-1]).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(change)])
    cum_obs = np.concatenate([[0], np.cumsum(observed.astype(np.int64))])

    end_pos = np.arange(L_WINDOW - 1, n - HORIZON)

    # history window constant-segment and starting on an observed value
    keep = (c[end_pos] - c[end_pos - L_WINDOW + 1] == 0) & has_value[end_pos]
    # horizon constant-segment, fully observed
    keep &= (c[end_pos + HORIZON] - c[end_pos] == 0)
    keep &= (cum_obs[end_pos + HORIZON + 1] - cum_obs[end_pos + 1] == HORIZON)
    end_pos = end_pos[keep]

    tgt_ts = ts[end_pos + HORIZON]
    split = np.where(tgt_ts <= TRAIN_END, "train",
                     np.where(tgt_ts <= VAL_END, "val", "test"))
    # keep only the declared test end
    within = tgt_ts <= TEST_END

    pos = end_pos[within]
    split = split[within]
    starts = pos - L_WINDOW + 1

    X_num = np.stack([num_z[s:s + L_WINDOW] for s in starts]).astype(np.float32)
    X_cal = np.stack([cal[s:s + L_WINDOW] for s in starts]).astype(np.float32)
    X_cat = np.stack([cat[s:s + L_WINDOW] for s in starts]).astype(np.float32)
    X_ev = ev[pos].astype(np.float32)
    cy = np.concatenate([[0], np.cumsum(y_raw.astype(np.float64))])
    y = np.stack([y_raw[pos + h] for h in range(1, HORIZON + 1)], axis=1).astype(np.float32)
    y_time = tgt_ts.values.astype("datetime64[h]").astype(np.int64)
    # calendar of each forecast step (used by the seasonal-naive baseline / slicing)
    y_hour = np.stack([(tgt_ts - pd.Timedelta(hours=HORIZON - h)).hour.to_numpy()
                       for h in range(1, HORIZON + 1)], axis=1).astype(np.int8)

    out = dict(
        X_num=X_num, X_cal=X_cal, X_cat=X_cat, X_ev=X_ev, y=y, y_time=y_time,
        y_hour=y_hour,
        split=split, x_end=pos, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
    )
    info.update({
        "n_windows": int(len(y)),
        "n_train": int((split == "train").sum()),
        "n_val": int((split == "val").sum()),
        "n_test": int((split == "test").sum()),
        "train_start": str(tgt_ts[within][split == "train"].min()),
        "train_end": str(tgt_ts[within][split == "train"].max()),
        "val_start": str(tgt_ts[within][split == "val"].min()),
        "val_end": str(tgt_ts[within][split == "val"].max()),
        "test_start": str(tgt_ts[within][split == "test"].min()),
        "test_end": str(tgt_ts[within][split == "test"].max()),
    })
    return out


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    t, info = build_table()
    w = make_windows(t, info)

    t.reset_index(names="timestamp").to_csv(PROC / "table.csv", index=False)
    np.savez_compressed(PROC / f"windows_h{HORIZON}.npz",
                        **{k: v for k, v in w.items() if k != "split"})
    np.save(PROC / f"split_h{HORIZON}.npy", w["split"])

    meta = {
        "info": info,
        "l_window": L_WINDOW,
        "horizon": HORIZON,
        "num_channels": NUM_CHANNELS,
        "cal_channels": CAL_CHANNELS,
        "ev_channels": EV_CHANNELS,
        "weather_classes": WEATHER_CLASSES,
        "splits": {"train_end": str(TRAIN_END), "val_end": str(VAL_END), "test_end": str(TEST_END)},
    }
    (PROC / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
