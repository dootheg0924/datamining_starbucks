from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FINAL_DATA_DIR = DATA_DIR / "final"
MODELING_DATA_DIR = DATA_DIR / "modeling"
REPORT_DIR = ROOT / "reports"
GENERATED_CLASSIFICATION_DIR = REPORT_DIR / "generated" / "classification"
CLASSIFICATION_DATA_DIR = GENERATED_CLASSIFICATION_DIR / "data"
CAFE_CLASSIFICATION_DIR = GENERATED_CLASSIFICATION_DIR / "01_cafe_level"
DISTRICT_CLASSIFICATION_DIR = GENERATED_CLASSIFICATION_DIR / "02_district_level"

CAFE_CLASSIFICATION_OUTPUT_DIR = CAFE_CLASSIFICATION_DIR / "outputs"
CAFE_CLASSIFICATION_FIGURE_DIR = CAFE_CLASSIFICATION_DIR / "figures"
CAFE_CLASSIFICATION_MODEL_DIR = CAFE_CLASSIFICATION_DIR / "models"
CAFE_CLASSIFICATION_LOG_DIR = CAFE_CLASSIFICATION_DIR / "logs"

DISTRICT_CLASSIFICATION_OUTPUT_DIR = DISTRICT_CLASSIFICATION_DIR / "outputs"
DISTRICT_CLASSIFICATION_FIGURE_DIR = DISTRICT_CLASSIFICATION_DIR / "figures"
DISTRICT_CLASSIFICATION_MODEL_DIR = DISTRICT_CLASSIFICATION_DIR / "models"
DISTRICT_CLASSIFICATION_LOG_DIR = DISTRICT_CLASSIFICATION_DIR / "logs"

# Backward-compatible aliases for stage 1 cafe-level scripts.
CLASSIFICATION_OUTPUT_DIR = CAFE_CLASSIFICATION_OUTPUT_DIR
CLASSIFICATION_FIGURE_DIR = CAFE_CLASSIFICATION_FIGURE_DIR
CLASSIFICATION_MODEL_DIR = CAFE_CLASSIFICATION_MODEL_DIR
CLASSIFICATION_LOG_DIR = CAFE_CLASSIFICATION_LOG_DIR

SEOUL_CAFE_FINAL_FEATURES_PATH = FINAL_DATA_DIR / "seoul_cafe_model_features_final.csv"
STARBUCKS_ENGINEERED_FEATURES_PATH = (
    MODELING_DATA_DIR / "starbucks_engineered_features_final.csv"
)
CLASSIFICATION_DATASET_PATH = CLASSIFICATION_DATA_DIR / "clf_dataset.parquet"
PREPARE_REPORT_PATH = CLASSIFICATION_DATA_DIR / "prepare_report.txt"

META_SOURCE_COLUMNS = ["상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"]
META_OUTPUT_COLUMNS = ["name", "brand", "is_starbucks", "lat", "lon", "sigungu", "address"]
META_RENAME = dict(zip(META_SOURCE_COLUMNS, META_OUTPUT_COLUMNS))

RAW_FEATURE_COLUMNS = [
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
SOURCE_REQUIRED_COLUMNS = [*META_SOURCE_COLUMNS, *RAW_FEATURE_COLUMNS]

CLF_FEATURES = [
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
    "avg_income",
    "offices",
    "living_pop",
    "land_price",
]

CLUSTER_FEATURES = [
    "log_dist_subway",
    "subway_count_cat",
    "subway_ridership",
    "bus_stops_300m",
    "peak_avg",
    "restaurants_500m",
    "convenience_500m",
    "indie_cafe_500m",
    "low_price_cafe_500m",
    "franchise_cafe_500m",
    "log_retail_500m",
    "log_dist_starbucks",
    "avg_income",
    "offices",
    "living_pop",
    "land_price",
]

PERSONA_NAMES = {
    0: "오피스고소득",
    1: "상업활성",
    2: "주거생활",
    3: "도심초밀집",
    4: "비역세권",
}


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_classification_dirs() -> None:
    ensure_dirs(
        CLASSIFICATION_DATA_DIR,
        CAFE_CLASSIFICATION_OUTPUT_DIR,
        CAFE_CLASSIFICATION_FIGURE_DIR,
        CAFE_CLASSIFICATION_MODEL_DIR,
        CAFE_CLASSIFICATION_LOG_DIR,
        DISTRICT_CLASSIFICATION_OUTPUT_DIR,
        DISTRICT_CLASSIFICATION_FIGURE_DIR,
        DISTRICT_CLASSIFICATION_MODEL_DIR,
        DISTRICT_CLASSIFICATION_LOG_DIR,
    )


def relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_seoul_cafe_final_features(path: Path = SEOUL_CAFE_FINAL_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_starbucks_engineered_features(
    path: Path = STARBUCKS_ENGINEERED_FEATURES_PATH,
) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_source_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in SOURCE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_classification_dataset(raw: pd.DataFrame, grid: float = 0.02) -> pd.DataFrame:
    validate_source_columns(raw)
    source = raw.rename(columns=META_RENAME).copy()

    df = source[META_OUTPUT_COLUMNS].copy()
    df["log_dist_subway"] = np.log1p(source["dist_nearest_subway"])
    df["subway_count_cat"] = source["num_subway_500m"].clip(upper=2).astype(int)
    df["subway_ridership"] = source["nearest_subway_ridership"]
    df["bus_stops_300m"] = source["num_bus_stops_300m"]
    df["peak_avg"] = source[
        [
            "subway_morning_peak_500m",
            "subway_lunch_peak_500m",
            "subway_evening_peak_500m",
        ]
    ].mean(axis=1)
    df["restaurants_500m"] = source["num_restaurants_500m"]
    df["log_retail_500m"] = np.log1p(source["num_retail_500m"])
    df["convenience_500m"] = source["num_convenience_500m"]
    df["indie_cafe_500m"] = source["independent_cafe_count_500m"]
    df["low_price_cafe_500m"] = source["low_price_cafe_count_500m"]
    df["franchise_cafe_500m"] = source["other_franchise_cafe_count_500m"]
    df["avg_income"] = source["avg_income"]
    df["offices"] = source["num_offices"]
    df["living_pop"] = source["living_population"]
    df["land_price"] = source["land_price"]
    df["log_dist_starbucks"] = np.log1p(source["dist_nearest_starbucks"])
    df["dist_nearest_starbucks"] = source["dist_nearest_starbucks"]

    df["grid_row"] = np.floor(df["lat"] / grid).astype(int)
    df["grid_col"] = np.floor(df["lon"] / grid).astype(int)
    df["spatial_block"] = df["grid_row"].astype(str) + "_" + df["grid_col"].astype(str)
    df["s"] = df["is_starbucks"].astype(int)

    for column in RAW_FEATURE_COLUMNS:
        df[column] = source[column]

    keep_columns = list(dict.fromkeys([
        "name",
        "brand",
        "is_starbucks",
        "s",
        "lat",
        "lon",
        "sigungu",
        "address",
        "spatial_block",
        "grid_row",
        "grid_col",
        *CLF_FEATURES,
        "log_dist_starbucks",
        "dist_nearest_starbucks",
        *RAW_FEATURE_COLUMNS,
    ]))
    return df[keep_columns]


def canonicalize_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Map arbitrary KMeans labels to the stable C0-C4 persona labels used in reports."""
    profiles = df.assign(_raw_cluster=labels).groupby("_raw_cluster")[CLUSTER_FEATURES].mean()
    remaining = set(int(label) for label in profiles.index)
    raw_to_canonical: dict[int, int] = {}

    raw_c0 = int(profiles.loc[list(remaining), "offices"].idxmax())
    raw_to_canonical[raw_c0] = 0
    remaining.remove(raw_c0)

    raw_c4 = int(profiles.loc[list(remaining), "log_dist_subway"].idxmax())
    raw_to_canonical[raw_c4] = 4
    remaining.remove(raw_c4)

    raw_c3 = int(profiles.loc[list(remaining), "restaurants_500m"].idxmax())
    raw_to_canonical[raw_c3] = 3
    remaining.remove(raw_c3)

    raw_c1 = int(profiles.loc[list(remaining), "restaurants_500m"].idxmax())
    raw_to_canonical[raw_c1] = 1
    remaining.remove(raw_c1)

    raw_c2 = remaining.pop()
    raw_to_canonical[raw_c2] = 2

    canonical = np.array([raw_to_canonical[int(label)] for label in labels], dtype=int)
    return canonical, raw_to_canonical


def canonicalize_centers(
    centers: np.ndarray,
    raw_to_canonical: dict[int, int],
) -> np.ndarray:
    canonical_to_raw = {
        canonical_label: raw_label for raw_label, canonical_label in raw_to_canonical.items()
    }
    return np.vstack([centers[canonical_to_raw[canonical_label]] for canonical_label in range(5)])


ensure_classification_dirs()
