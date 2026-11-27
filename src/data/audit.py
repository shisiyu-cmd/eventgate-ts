"""Data-quality audit for the Metro Interstate Traffic Volume dataset.

Produces machine-readable JSON plus a human-readable markdown so every number
quoted in the report can be traced back to code.

Usage:  python -m src.data.audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/Metro_Interstate_Traffic_volume.csv.gz")
OUT = Path("results/audit")


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df.sort_values("date_time").reset_index(drop=True)


def audit(df: pd.DataFrame) -> dict:
    t = df["date_time"]
    full = pd.date_range(t.min(), t.max(), freq="h")
    missing = full.difference(pd.DatetimeIndex(t))

    rep = int(df.duplicated(subset="date_time").sum())

    vol = df["traffic_volume"]
    zero = vol == 0
    # contiguous runs of zero volume
    runs, cur = [], 0
    for z in zero.to_numpy():
        if z:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    temp = df["temp"]
    rain = df["rain_1h"]

    # hourly coverage by calendar year
    by_year = (
        df.assign(year=t.dt.year)
        .groupby("year")
        .agg(rows=("traffic_volume", "size"),
             mean_volume=("traffic_volume", "mean"),
             zero_rows=("traffic_volume", lambda s: int((s == 0).sum())))
    )

    return {
        "n_rows": int(len(df)),
        "time_start": str(t.min()),
        "time_end": str(t.max()),
        "expected_hours": int(len(full)),
        "missing_hours": int(len(missing)),
        "missing_hours_examples": [str(x) for x in missing[:10]],
        "duplicate_timestamps": rep,
        "holiday_rows": int(df["holiday"].notna().sum()),
        "holiday_values": sorted(df["holiday"].dropna().unique().tolist()),
        "weather_main_counts": df["weather_main"].value_counts().to_dict(),
        "weather_desc_n_classes": int(df["weather_description"].nunique()),
        "volume_zero_rows": int(zero.sum()),
        "volume_zero_ratio": float(zero.mean()),
        "volume_zero_run_lengths": dict(pd.Series(runs).value_counts().sort_index().astype(int).to_dict()),
        "volume_zero_longest_run": int(max(runs) if runs else 0),
        "volume_zero_by_year": df.assign(y=t.dt.year).groupby("y")["traffic_volume"]
                                 .apply(lambda s: int((s == 0).sum())).to_dict(),
        "temp_zero_rows": int((temp == 0).sum()),
        "temp_min_nonzero": float(temp[temp > 0].min()),
        "rain_max": float(rain.max()),
        "rain_p999": float(rain.quantile(0.999)),
        "rain_rows_over_50mm": int((rain > 50).sum()),
        "snow_max": float(df["snow_1h"].max()),
        "snow_rows_positive": int((df["snow_1h"] > 0).sum()),
        "clouds_out_of_range": int(((df["clouds_all"] < 0) | (df["clouds_all"] > 100)).sum()),
        "volume_stats": {
            "mean": float(vol.mean()), "std": float(vol.std()),
            "min": int(vol.min()), "max": int(vol.max()),
            "q01": float(vol.quantile(0.01)), "q99": float(vol.quantile(0.99)),
        },
        "by_year": by_year.round(2).reset_index().to_dict(orient="records"),
        "missing_by_year": pd.Series(1, index=missing).groupby(missing.year).size().astype(int).to_dict(),
    }


def render_md(a: dict) -> str:
    lines = [
        "# 数据质量审计报告 — Metro Interstate Traffic Volume",
        "",
        f"- 原始记录数：**{a['n_rows']:,}**",
        f"- 时间范围：**{a['time_start']} → {a['time_end']}**",
        f"- 应有小时数：**{a['expected_hours']:,}**；缺失小时：**{a['missing_hours']}**",
        f"- 重复时间戳：**{a['duplicate_timestamps']}**",
        f"- 节假日标注行：**{a['holiday_rows']}**（{len(a['holiday_values'])} 类）",
        "",
        "## 目标变量",
        "",
        f"- traffic_volume: mean={a['volume_stats']['mean']:.1f}, std={a['volume_stats']['std']:.1f}, "
        f"min={a['volume_stats']['min']}, max={a['volume_stats']['max']}",
        f"- **流量为 0 的记录：{a['volume_zero_rows']:,}（{a['volume_zero_ratio']*100:.2f}%）**，"
        f"最长连续 {a['volume_zero_longest_run']} 小时",
        f"- 零值按年：{a['volume_zero_by_year']}",
        "",
        "## 协变量",
        "",
        f"- temp==0 (K) 的异常记录：**{a['temp_zero_rows']}**（开尔文 0 不可能，视为缺失）",
        f"- rain_1h 最大值 **{a['rain_max']} mm/h**（>50mm/h 的行数 {a['rain_rows_over_50mm']}，疑似单位/录入错误）",
        f"- snow_1h 最大值 {a['snow_max']}，正样本 {a['snow_rows_positive']} 行",
        f"- clouds_all 越界：{a['clouds_out_of_range']}",
        "",
        "## 天气类别分布",
        "",
    ]
    for k, v in a["weather_main_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 逐年覆盖", "", "| year | rows | mean_volume | zero_rows |", "|---|---|---|---|"]
    for r in a["by_year"]:
        lines.append(f"| {r['year']} | {r['rows']} | {r['mean_volume']} | {r['zero_rows']} |")
    lines += ["", "## 缺失小时分布", "", str(a["missing_by_year"])]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    a = audit(df)
    (OUT / "audit.json").write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "audit.md").write_text(render_md(a), encoding="utf-8")
    print(render_md(a))


if __name__ == "__main__":
    main()
