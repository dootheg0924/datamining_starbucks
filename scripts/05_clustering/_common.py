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
GENERATED_CLUSTERING_DIR = REPORT_DIR / "generated" / "clustering"
GENERATED_CLUSTERING_TABLE_DIR = GENERATED_CLUSTERING_DIR / "tables"
GENERATED_CLUSTERING_FIGURE_DIR = GENERATED_CLUSTERING_DIR / "figures"

STARBUCKS_ENGINEERED_FEATURES_PATH = (
    MODELING_DATA_DIR / "starbucks_engineered_features_final.csv"
)
SEOUL_CAFE_FINAL_FEATURES_PATH = FINAL_DATA_DIR / "seoul_cafe_model_features_final.csv"

META_COLUMNS = ["상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"]
ENGINEERED_FEATURES = [
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
    0: "C0 오피스고소득",
    1: "C1 상업활성",
    2: "C2 주거생활",
    3: "C3 도심초밀집",
    4: "C4 비역세권",
}
FULL_FEATURE_INPUT_COLUMNS = [
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


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_starbucks_engineered_features(path: Path = STARBUCKS_ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_seoul_cafe_final_features(path: Path = SEOUL_CAFE_FINAL_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def build_engineered_features_for_all_cafes(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FULL_FEATURE_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = pd.DataFrame(index=df.index)
    features["log_dist_subway"] = np.log1p(df["dist_nearest_subway"])
    features["subway_count_cat"] = np.select(
        [df["num_subway_500m"].eq(0), df["num_subway_500m"].eq(1)],
        [0, 1],
        default=2,
    ).astype(int)
    features["subway_ridership"] = df["nearest_subway_ridership"]
    features["bus_stops_300m"] = df["num_bus_stops_300m"]
    features["peak_avg"] = df[
        [
            "subway_morning_peak_500m",
            "subway_lunch_peak_500m",
            "subway_evening_peak_500m",
        ]
    ].mean(axis=1)
    features["restaurants_500m"] = df["num_restaurants_500m"]
    features["convenience_500m"] = df["num_convenience_500m"]
    features["indie_cafe_500m"] = df["independent_cafe_count_500m"]
    features["low_price_cafe_500m"] = df["low_price_cafe_count_500m"]
    features["franchise_cafe_500m"] = df["other_franchise_cafe_count_500m"]
    features["log_retail_500m"] = np.log1p(df["num_retail_500m"])
    features["log_dist_starbucks"] = np.log1p(df["dist_nearest_starbucks"])
    features["avg_income"] = df["avg_income"]
    features["offices"] = df["num_offices"]
    features["living_pop"] = df["living_population"]
    features["land_price"] = df["land_price"]

    return pd.concat([df[META_COLUMNS], features[ENGINEERED_FEATURES]], axis=1)


def canonicalize_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Map arbitrary algorithm labels to stable C0-C4 persona labels."""
    profiles = df.assign(_raw_cluster=labels).groupby("_raw_cluster")[ENGINEERED_FEATURES].mean()
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
