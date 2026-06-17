from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FINAL_DATA_DIR = DATA_DIR / "final"
TABLE_DIR = ROOT / "reports" / "generated" / "tables"
FIGURE_DIR = ROOT / "reports" / "generated" / "figures" / "eda"

STARBUCKS_PATH = FINAL_DATA_DIR / "starbucks_model_features_final.csv"
FEATURES = [
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
LOG_VARS = [
    "dist_nearest_subway",
    "dist_nearest_starbucks",
    "num_retail_500m",
    "land_price",
    "avg_income",
    "num_offices",
    "nearest_subway_ridership",
    "living_population",
]
SQRT_VARS = [
    "subway_morning_peak_500m",
    "subway_lunch_peak_500m",
    "subway_evening_peak_500m",
]


def transformed_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df[FEATURES].copy()
    for column in LOG_VARS:
        result[column] = np.log1p(result[column])
    for column in SQRT_VARS:
        result[column] = np.sqrt(result[column])
    return result


def pairwise_correlations(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = df.corr(method="pearson")
    rows = []
    for i, feature_a in enumerate(corr.columns):
        for feature_b in corr.columns[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "pearson_r": value,
                    "abs_r": abs(value),
                    "strong_pair": abs(value) >= threshold,
                }
            )
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def skew_transform_table(raw: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        transform = "log1p" if feature in LOG_VARS else "sqrt" if feature in SQRT_VARS else "raw"
        rows.append(
            {
                "feature": feature,
                "transform": transform,
                "raw_skewness": raw[feature].skew(),
                "transformed_skewness": transformed[feature].skew(),
            }
        )
    return pd.DataFrame(rows).round(6)


def save_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(11, 9))
    matrix = axis.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(corr.columns)))
    axis.set_yticks(range(len(corr.index)))
    axis.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    axis.set_yticklabels(corr.index, fontsize=7)
    axis.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(matrix, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw/transformed correlation diagnostics.")
    parser.add_argument("--input", type=Path, default=STARBUCKS_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    starbucks = pd.read_csv(args.input, encoding="utf-8-sig")
    raw = starbucks[FEATURES].copy()
    transformed = transformed_features(starbucks)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw_corr = raw.corr(method="pearson")
    transformed_corr = transformed.corr(method="pearson")

    raw_corr.round(6).to_csv(TABLE_DIR / "eda_corr_raw_matrix.csv", encoding="utf-8-sig")
    transformed_corr.round(6).to_csv(TABLE_DIR / "eda_corr_transformed_matrix.csv", encoding="utf-8-sig")

    raw_pairs = pairwise_correlations(raw, args.threshold)
    transformed_pairs = pairwise_correlations(transformed, args.threshold)
    raw_pairs.to_csv(TABLE_DIR / "eda_corr_raw_pairs.csv", index=False, encoding="utf-8-sig")
    transformed_pairs.to_csv(
        TABLE_DIR / "eda_corr_transformed_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    skew = skew_transform_table(raw, transformed)
    skew.to_csv(TABLE_DIR / "eda_skew_transform_comparison.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_heatmap(raw_corr, FIGURE_DIR / "corr_raw_heatmap.png", "Raw Pearson Correlation")
        save_heatmap(
            transformed_corr,
            FIGURE_DIR / "corr_transformed_heatmap.png",
            "Transformed Pearson Correlation",
        )

    print(f"Raw strong pairs: {int(raw_pairs['strong_pair'].sum())}")
    print(f"Transformed strong pairs: {int(transformed_pairs['strong_pair'].sum())}")
    print(f"Tables written to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
