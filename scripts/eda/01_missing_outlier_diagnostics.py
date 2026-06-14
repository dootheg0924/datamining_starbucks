from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "reports" / "generated" / "tables"
FIGURE_DIR = ROOT / "reports" / "generated" / "figures" / "eda"

STARBUCKS_PATH = DATA_DIR / "starbucks_model_features_final.csv"
SEOUL_PATH = DATA_DIR / "seoul_cafe_model_features_final.csv"
ID_COLUMNS = {"상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"}


def model_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric if column not in ID_COLUMNS]


def missing_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "missing": missing,
                    "missing_rate": missing / len(df),
                }
            )
    return pd.DataFrame(rows)


def iqr_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        series = df[feature].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = (df[feature] < lower) | (df[feature] > upper)
        rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outlier": int(is_outlier.sum()),
                "outlier_rate": float(is_outlier.mean()),
                "skewness": series.skew(),
            }
        )
    return pd.DataFrame(rows).round(6)


def multi_outlier_stores(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for feature in features:
        series = df[feature].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags[feature] = ((df[feature] < lower) | (df[feature] > upper)).astype(int)

    flags["outlier_feature_count"] = flags.sum(axis=1)
    result = pd.concat(
        [df[["상호명", "시군구명", "도로명주소"]], flags[["outlier_feature_count"]]],
        axis=1,
    )
    return result[result["outlier_feature_count"] >= 2].sort_values(
        ["outlier_feature_count", "시군구명", "상호명"],
        ascending=[False, True, True],
    )


def save_grid_plots(df: pd.DataFrame, features: list[str]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = math.ceil(len(features) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.ravel(axes)
    for axis, feature in zip(axes, features):
        axis.boxplot(df[feature].dropna(), vert=False)
        axis.set_title(feature, fontsize=9)
    for axis in axes[len(features) :]:
        axis.axis("off")
    fig.suptitle("Starbucks Feature IQR Boxplots", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "outlier_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.ravel(axes)
    for axis, feature in zip(axes, features):
        series = df[feature].dropna()
        axis.hist(series, bins=30, color="#4c78a8", alpha=0.85)
        axis.axvline(series.median(), color="#f58518", linewidth=1)
        axis.set_title(f"{feature}\nskew={series.skew():.2f}", fontsize=9)
    for axis in axes[len(features) :]:
        axis.axis("off")
    fig.suptitle("Starbucks Feature Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Missing-value and outlier diagnostics.")
    parser.add_argument("--starbucks", type=Path, default=STARBUCKS_PATH)
    parser.add_argument("--seoul", type=Path, default=SEOUL_PATH)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    starbucks = pd.read_csv(args.starbucks, encoding="utf-8-sig")
    seoul = pd.read_csv(args.seoul, encoding="utf-8-sig")
    features = model_features(starbucks)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    missing = pd.concat(
        [missing_table(starbucks, "starbucks"), missing_table(seoul, "seoul_cafe")],
        ignore_index=True,
    )
    missing.to_csv(TABLE_DIR / "eda_missing_values.csv", index=False, encoding="utf-8-sig")

    iqr = iqr_summary(starbucks, features)
    iqr.to_csv(TABLE_DIR / "eda_starbucks_iqr_outliers.csv", index=False, encoding="utf-8-sig")

    multi = multi_outlier_stores(starbucks, features)
    multi.to_csv(TABLE_DIR / "eda_starbucks_multi_outlier_stores.csv", index=False, encoding="utf-8-sig")

    skew = (
        pd.DataFrame({"feature": features, "skewness": [starbucks[f].dropna().skew() for f in features]})
        .sort_values("skewness", key=lambda s: s.abs(), ascending=False)
        .round(6)
    )
    skew.to_csv(TABLE_DIR / "eda_starbucks_skewness.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        save_grid_plots(starbucks, features)

    print(f"Features checked: {len(features)}")
    print(f"Missing rows table: {TABLE_DIR / 'eda_missing_values.csv'}")
    print(f"IQR summary: {TABLE_DIR / 'eda_starbucks_iqr_outliers.csv'}")
    print(f"Multi-outlier stores: {len(multi)}")


if __name__ == "__main__":
    main()
