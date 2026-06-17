from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FINAL_DATA_DIR = DATA_DIR / "final"
MODELING_DATA_DIR = DATA_DIR / "modeling"
REPORT_TABLE_DIR = ROOT / "reports" / "generated" / "tables"

INPUT_PATH = FINAL_DATA_DIR / "starbucks_model_features_final.csv"
OUTPUT_PATH = MODELING_DATA_DIR / "starbucks_engineered_features_final.csv"
SUMMARY_PATH = REPORT_TABLE_DIR / "starbucks_engineered_feature_summary.csv"

META_COLUMNS = ["상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"]
ENGINEERED_FEATURES = [
    "log_dist_subway",
    "subway_count_cat",
    "subway_ridership",
    "bus_stops_300m",
    "peak_avg",
    "restaurants_500m",
    "log_retail_500m",
    "convenience_500m",
    "indie_cafe_500m",
    "low_price_cafe_500m",
    "franchise_cafe_500m",
    "log_dist_starbucks",
    "avg_income",
    "offices",
    "living_pop",
    "land_price",
]
REQUIRED_COLUMNS = [
    *META_COLUMNS,
    "dist_nearest_subway",
    "num_subway_500m",
    "nearest_subway_ridership",
    "num_bus_stops_300m",
    "subway_morning_peak_500m",
    "subway_lunch_peak_500m",
    "subway_evening_peak_500m",
    "num_restaurants_500m",
    "num_retail_500m",
    "num_convenience_500m",
    "independent_cafe_count_500m",
    "low_price_cafe_count_500m",
    "other_franchise_cafe_count_500m",
    "dist_nearest_starbucks",
    "avg_income",
    "num_offices",
    "living_population",
    "land_price",
]


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    work = df[df["is_starbucks"] == 1].copy()
    features = pd.DataFrame(index=work.index)

    features["log_dist_subway"] = np.log1p(work["dist_nearest_subway"])
    features["subway_count_cat"] = np.select(
        [work["num_subway_500m"].eq(0), work["num_subway_500m"].eq(1)],
        [0, 1],
        default=2,
    ).astype(int)
    features["subway_ridership"] = work["nearest_subway_ridership"]
    features["bus_stops_300m"] = work["num_bus_stops_300m"]
    features["peak_avg"] = work[
        [
            "subway_morning_peak_500m",
            "subway_lunch_peak_500m",
            "subway_evening_peak_500m",
        ]
    ].mean(axis=1)
    features["restaurants_500m"] = work["num_restaurants_500m"]
    features["log_retail_500m"] = np.log1p(work["num_retail_500m"])
    features["convenience_500m"] = work["num_convenience_500m"]
    features["indie_cafe_500m"] = work["independent_cafe_count_500m"]
    features["low_price_cafe_500m"] = work["low_price_cafe_count_500m"]
    features["franchise_cafe_500m"] = work["other_franchise_cafe_count_500m"]
    features["log_dist_starbucks"] = np.log1p(work["dist_nearest_starbucks"])
    features["avg_income"] = work["avg_income"]
    features["offices"] = work["num_offices"]
    features["living_pop"] = work["living_population"]
    features["land_price"] = work["land_price"]

    return pd.concat([work[META_COLUMNS], features[ENGINEERED_FEATURES]], axis=1)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ENGINEERED_FEATURES:
        series = df[feature]
        rows.append(
            {
                "feature": feature,
                "missing": int(series.isna().sum()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "median": series.median(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows).round(6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Starbucks-only engineered feature set for modeling/clustering."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    engineered = build_engineered_features(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(args.output, index=False, encoding="utf-8-sig")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_table(engineered)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    print(f"Saved: {args.output.relative_to(ROOT)}")
    print(f"Shape: {engineered.shape[0]} rows x {engineered.shape[1]} columns")
    print(f"Feature missing values: {int(engineered[ENGINEERED_FEATURES].isna().sum().sum())}")
    print(f"Summary: {args.summary_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
