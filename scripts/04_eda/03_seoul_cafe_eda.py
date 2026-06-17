from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    ks_2samp = None


ROOT = Path(__file__).resolve().parents[2]
FINAL_DATA_DIR = ROOT / "data" / "final"
TABLE_DIR = ROOT / "reports" / "generated" / "tables"
ARCHIVE_TABLE_DIR = ROOT / "reports" / "archive" / "tables"
FIGURE_DIR = ROOT / "reports" / "generated" / "figures" / "eda" / "seoul_cafe"
REPORT_PATH = ROOT / "reports" / "seoul_cafe_eda_summary.md"

SEOUL_PATH = FINAL_DATA_DIR / "seoul_cafe_model_features_final.csv"

NAME_COL = "상호명"
BRAND_COL = "브랜드"
LABEL_COL = "is_starbucks"
LAT_COL = "위도"
LON_COL = "경도"
DISTRICT_COL = "시군구명"
ADDRESS_COL = "도로명주소"
NAN_REASON_COL = "nan_reason"

ID_COLUMNS = [
    NAME_COL,
    BRAND_COL,
    LABEL_COL,
    LAT_COL,
    LON_COL,
    DISTRICT_COL,
    ADDRESS_COL,
]

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

FEATURE_GROUPS = {
    "dist_nearest_subway": "transport",
    "num_subway_500m": "transport",
    "nearest_subway_ridership": "transport",
    "num_bus_stops_300m": "transport",
    "subway_morning_peak_500m": "subway_peak",
    "subway_lunch_peak_500m": "subway_peak",
    "subway_evening_peak_500m": "subway_peak",
    "num_restaurants_500m": "commerce",
    "num_retail_500m": "commerce",
    "num_convenience_500m": "commerce",
    "independent_cafe_count_500m": "cafe_competition",
    "low_price_cafe_count_500m": "cafe_competition",
    "other_franchise_cafe_count_500m": "cafe_competition",
    "dist_nearest_starbucks": "cafe_competition",
    "avg_income": "demographic_economic",
    "num_offices": "demographic_economic",
    "living_population": "demographic_economic",
    "land_price": "demographic_economic",
}

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

DISTRICT_MEDIAN_FEATURES = [
    "dist_nearest_subway",
    "num_restaurants_500m",
    "independent_cafe_count_500m",
    "dist_nearest_starbucks",
    "num_offices",
    "land_price",
]

ARCHIVE_TABLE_NAMES = {
    "seoul_cafe_basic_diagnostics.csv",
    "seoul_cafe_nan_reason_summary.csv",
    "seoul_cafe_feature_summary_by_group.csv",
    "seoul_cafe_starbucks_comparison.csv",
    "seoul_cafe_iqr_outliers.csv",
    "seoul_cafe_corr_pairs.csv",
    "seoul_cafe_district_summary.csv",
    "seoul_cafe_skew_transform_comparison.csv",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    if filename in ARCHIVE_TABLE_NAMES:
        df.to_csv(ARCHIVE_TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def relative_posix(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def validate_columns(df: pd.DataFrame) -> None:
    required = [*ID_COLUMNS, *FEATURES, NAN_REASON_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def dataset_splits(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    return [
        ("all", "전체", df),
        ("starbucks", "스타벅스", df[df[LABEL_COL] == 1].copy()),
        ("non_starbucks", "비스타벅스", df[df[LABEL_COL] == 0].copy()),
    ]


def to_numeric_frame(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    return df[features].apply(pd.to_numeric, errors="coerce")


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    result = to_numeric_frame(df).copy()
    for column in LOG_VARS:
        result[column] = np.log1p(result[column])
    for column in SQRT_VARS:
        result[column] = np.sqrt(result[column])
    return result


def transform_label(feature: str) -> str:
    if feature in LOG_VARS:
        return f"log1p({feature})"
    if feature in SQRT_VARS:
        return f"sqrt({feature})"
    return feature


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy() if max_rows else df.copy()
    work = work.fillna("")
    columns = [str(column) for column in work.columns]
    rows = [[str(value) for value in row] for row in work.to_numpy()]
    widths = [
        max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def basic_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, label, data in dataset_splits(df):
        feature_missing = int(data[FEATURES].isna().sum().sum())
        nan_marked = int(data[NAN_REASON_COL].notna().sum())
        duplicate_coord = int(data.duplicated([LAT_COL, LON_COL]).sum())
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "rows": len(data),
                "starbucks_count": int((data[LABEL_COL] == 1).sum()),
                "non_starbucks_count": int((data[LABEL_COL] == 0).sum()),
                "feature_missing_cells": feature_missing,
                "nan_reason_marked_rows": nan_marked,
                "duplicate_latlon_rows": duplicate_coord,
                "district_count": int(data[DISTRICT_COL].nunique()),
            }
        )
    return pd.DataFrame(rows)


def nan_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    reason = df[NAN_REASON_COL].fillna("none")
    rows = []
    for dataset, label, data in dataset_splits(df):
        counts = data[NAN_REASON_COL].fillna("none").value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "nan_reason": value,
                    "count": int(count),
                    "rate": float(count / len(data)) if len(data) else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if "none" not in set(reason):
        return result
    return result.sort_values(["dataset", "count"], ascending=[True, False]).reset_index(drop=True)


def feature_summary_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, label, data in dataset_splits(df):
        total = len(data)
        numeric = to_numeric_frame(data)
        for feature in FEATURES:
            series = numeric[feature].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = numeric[feature].notna() & (
                (numeric[feature] < lower) | (numeric[feature] > upper)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "feature": feature,
                    "group": FEATURE_GROUPS[feature],
                    "count": int(series.count()),
                    "missing_count": int(numeric[feature].isna().sum()),
                    "missing_rate": float(numeric[feature].isna().sum() / total) if total else np.nan,
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "p01": series.quantile(0.01),
                    "p05": series.quantile(0.05),
                    "q1": q1,
                    "q3": q3,
                    "p95": series.quantile(0.95),
                    "p99": series.quantile(0.99),
                    "max": series.max(),
                    "zero_count": int((series == 0).sum()),
                    "zero_rate": float((series == 0).sum() / total) if total else np.nan,
                    "skewness": series.skew(),
                    "iqr": iqr,
                    "iqr_lower_bound": lower,
                    "iqr_upper_bound": upper,
                    "iqr_outlier_count": int(outlier_mask.sum()),
                    "iqr_outlier_rate": float(outlier_mask.mean()) if total else np.nan,
                }
            )
    return pd.DataFrame(rows).round(6)


def iqr_outlier_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "label",
        "feature",
        "group",
        "q1",
        "q3",
        "iqr",
        "iqr_lower_bound",
        "iqr_upper_bound",
        "iqr_outlier_count",
        "iqr_outlier_rate",
        "skewness",
    ]
    return summary[columns].copy()


def multi_outlier_stores(df: pd.DataFrame) -> pd.DataFrame:
    numeric = to_numeric_frame(df)
    flags = pd.DataFrame(False, index=df.index, columns=FEATURES)
    for feature in FEATURES:
        series = numeric[feature].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags[feature] = numeric[feature].notna() & (
            (numeric[feature] < lower) | (numeric[feature] > upper)
        )

    outlier_count = flags.sum(axis=1)
    result = df[[NAME_COL, BRAND_COL, LABEL_COL, DISTRICT_COL, ADDRESS_COL]].copy()
    result["outlier_feature_count"] = outlier_count
    result["outlier_features"] = flags.apply(
        lambda row: "; ".join(feature for feature, is_outlier in row.items() if is_outlier),
        axis=1,
    )
    result = result[result["outlier_feature_count"] >= 2].sort_values(
        ["outlier_feature_count", DISTRICT_COL, NAME_COL],
        ascending=[False, True, True],
    )
    return result.reset_index(drop=True)


def standardized_mean_difference(positive: pd.Series, unlabeled: pd.Series) -> float:
    positive = positive.dropna()
    unlabeled = unlabeled.dropna()
    pooled = math.sqrt((positive.var(ddof=1) + unlabeled.var(ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((positive.mean() - unlabeled.mean()) / pooled)


def starbucks_comparison(df: pd.DataFrame) -> pd.DataFrame:
    positive = df[df[LABEL_COL] == 1]
    unlabeled = df[df[LABEL_COL] == 0]
    all_numeric = to_numeric_frame(df)
    pos_numeric = to_numeric_frame(positive)
    unlab_numeric = to_numeric_frame(unlabeled)
    rows = []
    for feature in FEATURES:
        pos_series = pos_numeric[feature].dropna()
        unlab_series = unlab_numeric[feature].dropna()
        all_series = all_numeric[feature].dropna()
        ks_statistic = np.nan
        ks_pvalue = np.nan
        if ks_2samp is not None and len(pos_series) and len(unlab_series):
            ks = ks_2samp(pos_series, unlab_series, alternative="two-sided", method="auto")
            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        pos_mean = pos_series.mean()
        unlab_mean = unlab_series.mean()
        pos_median = pos_series.median()
        unlab_median = unlab_series.median()
        rows.append(
            {
                "feature": feature,
                "group": FEATURE_GROUPS[feature],
                "all_mean": all_series.mean(),
                "starbucks_mean": pos_mean,
                "non_starbucks_mean": unlab_mean,
                "mean_diff_starbucks_minus_non": pos_mean - unlab_mean,
                "mean_ratio_starbucks_to_non": pos_mean / unlab_mean if unlab_mean else np.nan,
                "all_median": all_series.median(),
                "starbucks_median": pos_median,
                "non_starbucks_median": unlab_median,
                "median_diff_starbucks_minus_non": pos_median - unlab_median,
                "smd_starbucks_vs_non": standardized_mean_difference(pos_series, unlab_series),
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "direction": "starbucks_higher"
                if pos_median > unlab_median
                else "starbucks_lower"
                if pos_median < unlab_median
                else "similar_median",
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_smd=lambda data: data["smd_starbucks_vs_non"].abs())
        .sort_values("abs_smd", ascending=False)
        .round(6)
        .reset_index(drop=True)
    )


def corr_matrix_rows(df: pd.DataFrame, transformed: bool) -> pd.DataFrame:
    rows = []
    for dataset, label, data in dataset_splits(df):
        numeric = transform_features(data) if transformed else to_numeric_frame(data)
        corr = numeric.corr(method="pearson").round(6)
        for feature, values in corr.iterrows():
            row = {"dataset": dataset, "label": label, "feature": feature}
            row.update(values.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_correlations(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for transformed in [False, True]:
        transform_name = "transformed" if transformed else "raw"
        for dataset, label, data in dataset_splits(df):
            numeric = transform_features(data) if transformed else to_numeric_frame(data)
            corr = numeric.corr(method="pearson")
            for idx, feature_a in enumerate(FEATURES):
                for feature_b in FEATURES[idx + 1 :]:
                    value = corr.loc[feature_a, feature_b]
                    rows.append(
                        {
                            "dataset": dataset,
                            "label": label,
                            "transform": transform_name,
                            "feature_a": feature_a,
                            "feature_b": feature_b,
                            "pearson_r": value,
                            "abs_r": abs(value),
                            "strong_pair": abs(value) >= threshold,
                        }
                    )
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).round(6).reset_index(drop=True)


def skew_transform_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, label, data in dataset_splits(df):
        raw = to_numeric_frame(data)
        transformed = transform_features(data)
        for feature in FEATURES:
            transform = "log1p" if feature in LOG_VARS else "sqrt" if feature in SQRT_VARS else "raw"
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "feature": feature,
                    "transform": transform,
                    "raw_skewness": raw[feature].skew(),
                    "transformed_skewness": transformed[feature].skew(),
                }
            )
    return pd.DataFrame(rows).round(6)


def district_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(DISTRICT_COL, dropna=False)
    rows = []
    for district, data in grouped:
        row = {
            "district": district,
            "total_cafes": len(data),
            "starbucks_count": int((data[LABEL_COL] == 1).sum()),
            "non_starbucks_count": int((data[LABEL_COL] == 0).sum()),
            "starbucks_rate": float((data[LABEL_COL] == 1).mean()),
        }
        for feature in DISTRICT_MEDIAN_FEATURES:
            row[f"{feature}_median"] = pd.to_numeric(data[feature], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_cafes", ascending=False).round(6).reset_index(drop=True)


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


def save_grid_plots(df: pd.DataFrame) -> dict[str, str]:
    numeric = to_numeric_frame(df)
    ncols = 3
    nrows = math.ceil(len(FEATURES) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows))
    axes = np.ravel(axes)
    for axis, feature in zip(axes, FEATURES):
        series = numeric[feature].dropna()
        axis.hist(series, bins=35, color="#4c78a8", alpha=0.85)
        axis.axvline(series.median(), color="#f58518", linewidth=1)
        axis.set_title(f"{feature}\nskew={series.skew():.2f}", fontsize=8)
    for axis in axes[len(FEATURES) :]:
        axis.axis("off")
    fig.suptitle("Seoul cafe feature histograms", fontsize=14, fontweight="bold")
    fig.tight_layout()
    histogram_path = FIGURE_DIR / "seoul_cafe_feature_histograms.png"
    fig.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows))
    axes = np.ravel(axes)
    for axis, feature in zip(axes, FEATURES):
        axis.boxplot(numeric[feature].dropna(), vert=False, showfliers=True)
        axis.set_title(feature, fontsize=8)
    for axis in axes[len(FEATURES) :]:
        axis.axis("off")
    fig.suptitle("Seoul cafe feature boxplots", fontsize=14, fontweight="bold")
    fig.tight_layout()
    boxplot_path = FIGURE_DIR / "seoul_cafe_feature_boxplots.png"
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "histograms": relative_posix(histogram_path),
        "boxplots": relative_posix(boxplot_path),
    }


def save_overlay_plot(df: pd.DataFrame) -> str:
    starbucks = df[df[LABEL_COL] == 1]
    non_starbucks = df[df[LABEL_COL] == 0]
    starbucks_numeric = transform_features(starbucks)
    non_numeric = transform_features(non_starbucks)

    ncols = 3
    nrows = math.ceil(len(FEATURES) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows))
    axes = np.ravel(axes)
    for axis, feature in zip(axes, FEATURES):
        sb_series = starbucks_numeric[feature].dropna()
        non_series = non_numeric[feature].dropna()
        combined = pd.concat([sb_series, non_series], ignore_index=True)
        bins = np.histogram_bin_edges(combined, bins=35)
        axis.hist(non_series, bins=bins, density=True, alpha=0.42, color="#4c78a8", label="non-SB")
        axis.hist(sb_series, bins=bins, density=True, alpha=0.42, color="#f58518", label="SB")
        axis.set_title(transform_label(feature), fontsize=8)
    for axis in axes[len(FEATURES) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Starbucks vs non-Starbucks distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])
    path = FIGURE_DIR / "starbucks_vs_non_starbucks_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return relative_posix(path)


def save_smd_bar(comparison: pd.DataFrame) -> str:
    top = comparison.sort_values("abs_smd", ascending=True).tail(12)
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = np.where(top["smd_starbucks_vs_non"] >= 0, "#4c78a8", "#f58518")
    axis.barh(top["feature"], top["smd_starbucks_vs_non"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Standardized mean difference (Starbucks - non-Starbucks)")
    axis.set_title("Largest Starbucks vs non-Starbucks feature gaps")
    fig.tight_layout()
    path = FIGURE_DIR / "starbucks_smd_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return relative_posix(path)


def save_district_chart(districts: pd.DataFrame) -> str:
    top = districts.sort_values("total_cafes", ascending=False).head(15).copy()
    x = np.arange(len(top))
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x, top["total_cafes"], color="#4c78a8", alpha=0.85)
    axis.set_ylabel("Cafe count")
    axis.set_xticks(x)
    axis.set_xticklabels(top["district"], rotation=45, ha="right")
    rate_axis = axis.twinx()
    rate_axis.plot(x, top["starbucks_rate"], color="#f58518", marker="o")
    rate_axis.set_ylabel("Starbucks rate")
    axis.set_title("Top districts by cafe count and Starbucks rate")
    fig.tight_layout()
    path = FIGURE_DIR / "district_cafe_starbucks_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return relative_posix(path)


def save_figures(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    districts: pd.DataFrame,
) -> dict[str, str]:
    figure_paths = save_grid_plots(df)
    figure_paths["overlay"] = save_overlay_plot(df)
    raw_corr = to_numeric_frame(df).corr(method="pearson")
    transformed_corr = transform_features(df).corr(method="pearson")
    raw_heatmap_path = FIGURE_DIR / "seoul_cafe_corr_raw_heatmap.png"
    transformed_heatmap_path = FIGURE_DIR / "seoul_cafe_corr_transformed_heatmap.png"
    save_heatmap(raw_corr, raw_heatmap_path, "Seoul cafe raw Pearson correlation")
    save_heatmap(
        transformed_corr,
        transformed_heatmap_path,
        "Seoul cafe transformed Pearson correlation",
    )
    figure_paths["raw_corr_heatmap"] = relative_posix(raw_heatmap_path)
    figure_paths["transformed_corr_heatmap"] = relative_posix(transformed_heatmap_path)
    figure_paths["smd_bar"] = save_smd_bar(comparison)
    figure_paths["district_chart"] = save_district_chart(districts)
    return figure_paths


def round_for_report(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    work = df.copy()
    target_columns = columns or work.select_dtypes(include=[np.number]).columns.tolist()
    for column in target_columns:
        if column in work.columns and pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].round(3)
    return work


def write_report(
    df: pd.DataFrame,
    basic: pd.DataFrame,
    nan_reasons: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    skew: pd.DataFrame,
    districts: pd.DataFrame,
    figures: dict[str, str],
    skip_figures: bool,
) -> None:
    top_comparison = comparison[
        [
            "feature",
            "group",
            "starbucks_median",
            "non_starbucks_median",
            "median_diff_starbucks_minus_non",
            "smd_starbucks_vs_non",
            "ks_statistic",
            "direction",
        ]
    ].head(8)

    all_summary = summary[summary["dataset"] == "all"]
    top_skew = all_summary.sort_values("skewness", key=lambda data: data.abs(), ascending=False)[
        ["feature", "group", "median", "p95", "p99", "max", "skewness", "zero_rate"]
    ].head(8)

    top_outliers = all_summary.sort_values("iqr_outlier_rate", ascending=False)[
        ["feature", "group", "iqr_outlier_count", "iqr_outlier_rate", "skewness"]
    ].head(8)

    transformed_strong = corr_pairs[
        (corr_pairs["dataset"] == "all")
        & (corr_pairs["transform"] == "transformed")
        & (corr_pairs["strong_pair"])
    ][["feature_a", "feature_b", "pearson_r", "abs_r"]].head(10)

    district_top_count = districts[
        ["district", "total_cafes", "starbucks_count", "starbucks_rate"]
    ].head(10)
    district_top_rate = districts[districts["starbucks_count"] >= 5].sort_values(
        "starbucks_rate", ascending=False
    )[["district", "total_cafes", "starbucks_count", "starbucks_rate"]].head(10)

    nan_top = nan_reasons[nan_reasons["dataset"] == "all"][
        ["nan_reason", "count", "rate"]
    ].head(10)

    figure_lines = []
    if skip_figures:
        figure_lines.append("- Figures were skipped with `--skip-figures`.")
    else:
        for name, path in figures.items():
            figure_lines.append(f"- `{name}`: `{path}`")

    table_lines = [
        "- `reports/generated/tables/seoul_cafe_feature_summary_by_group.csv`",
        "- `reports/generated/tables/seoul_cafe_starbucks_comparison.csv`",
        "- `reports/generated/tables/seoul_cafe_iqr_outliers.csv`",
        "- `reports/generated/tables/seoul_cafe_multi_outlier_stores.csv`",
        "- `reports/generated/tables/seoul_cafe_corr_raw_matrix.csv`",
        "- `reports/generated/tables/seoul_cafe_corr_transformed_matrix.csv`",
        "- `reports/generated/tables/seoul_cafe_corr_pairs.csv`",
        "- `reports/generated/tables/seoul_cafe_district_summary.csv`",
        "- `reports/generated/tables/seoul_cafe_basic_diagnostics.csv`",
        "- `reports/generated/tables/seoul_cafe_nan_reason_summary.csv`",
        "- `reports/generated/tables/seoul_cafe_skew_transform_comparison.csv`",
    ]
    archive_table_lines = [
        f"- `reports/archive/tables/{filename}`" for filename in sorted(ARCHIVE_TABLE_NAMES)
    ]

    lines = [
        "# 서울 전체 카페 EDA 요약",
        "",
        f"- 입력 파일: `{relative_posix(SEOUL_PATH)}`",
        f"- 분석 대상: 서울 전체 카페 {len(df):,}개, 스타벅스 {int((df[LABEL_COL] == 1).sum()):,}개, 비스타벅스 {int((df[LABEL_COL] == 0).sum()):,}개",
        f"- 모델 feature: {len(FEATURES)}개",
        "",
        "## 1. 기본 진단",
        "",
        markdown_table(round_for_report(basic)),
        "",
        "`nan_reason`은 보정 전 결측 사유의 provenance로, feature 결측이 남았다는 뜻은 아닙니다.",
        "",
        markdown_table(round_for_report(nan_top)),
        "",
        "## 2. 스타벅스와 비스타벅스 차이",
        "",
        "아래 표는 standardized mean difference 절대값이 큰 feature 순서입니다. 양수는 스타벅스가 비스타벅스보다 큰 값, 음수는 작은 값을 뜻합니다.",
        "",
        markdown_table(round_for_report(top_comparison)),
        "",
        "## 3. 전체 카페 분포와 이상치",
        "",
        "왜도가 큰 변수는 변환 및 모델 해석에서 특히 주의해야 합니다.",
        "",
        markdown_table(round_for_report(top_skew)),
        "",
        "IQR 기준 이상치 비율이 큰 변수입니다. 이상치는 제거 대상이 아니라 고밀도 상권이나 특수 입지의 후보로 해석합니다.",
        "",
        markdown_table(round_for_report(top_outliers)),
        "",
        "## 4. 상관과 중복 가능성",
        "",
        "전체 카페 기준 변환 후에도 강하게 남는 Pearson 상관쌍입니다.",
        "",
        markdown_table(round_for_report(transformed_strong)),
        "",
        "## 5. 구별 요약",
        "",
        "카페 수 상위 구입니다.",
        "",
        markdown_table(round_for_report(district_top_count)),
        "",
        "스타벅스가 5개 이상 있는 구 중 스타벅스 비율이 높은 구입니다.",
        "",
        markdown_table(round_for_report(district_top_rate)),
        "",
        "## 6. 모델링/클러스터링 시사점",
        "",
        "- 비스타벅스는 확정 negative가 아니라 일반 카페 비교군으로만 해석해야 합니다.",
        "- 상권 밀도, 지하철/버스 접근성, 직장인구/지가 변수는 스타벅스와 일반 카페의 차이를 설명하는 핵심 후보입니다.",
        "- 카페 및 상권 count 변수는 서로 강한 상관을 보일 수 있으므로 선형 모델이나 해석 단계에서는 중복성을 확인해야 합니다.",
        "- 전체 카페 분포의 오른쪽 꼬리가 긴 변수는 기존 모델링 흐름과 같이 `log1p` 또는 `sqrt` 변환을 유지하는 것이 타당합니다.",
        "",
        "## 7. 산출물",
        "",
        "### Tables",
        "",
        *table_lines,
        "",
        "### Tracked table snapshots",
        "",
        "아래 CSV는 중요한 요약 산출물만 선별해 `reports/archive/tables/`에도 저장하므로 GitHub에 포함됩니다.",
        "",
        *archive_table_lines,
        "",
        "### Figures",
        "",
        *figure_lines,
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seoul-wide cafe EDA.")
    parser.add_argument("--input", type=Path, default=SEOUL_PATH)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    setup_plot_style()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    validate_columns(df)

    basic = basic_diagnostics(df)
    nan_reasons = nan_reason_summary(df)
    summary = feature_summary_by_group(df)
    outliers = iqr_outlier_summary(summary)
    multi = multi_outlier_stores(df)
    comparison = starbucks_comparison(df)
    raw_corr = corr_matrix_rows(df, transformed=False)
    transformed_corr = corr_matrix_rows(df, transformed=True)
    corr_pairs = pairwise_correlations(df, args.corr_threshold)
    skew = skew_transform_comparison(df)
    districts = district_summary(df)

    write_table(basic, "seoul_cafe_basic_diagnostics.csv")
    write_table(nan_reasons, "seoul_cafe_nan_reason_summary.csv")
    write_table(summary, "seoul_cafe_feature_summary_by_group.csv")
    write_table(comparison, "seoul_cafe_starbucks_comparison.csv")
    write_table(outliers, "seoul_cafe_iqr_outliers.csv")
    write_table(multi, "seoul_cafe_multi_outlier_stores.csv")
    write_table(raw_corr, "seoul_cafe_corr_raw_matrix.csv")
    write_table(transformed_corr, "seoul_cafe_corr_transformed_matrix.csv")
    write_table(corr_pairs, "seoul_cafe_corr_pairs.csv")
    write_table(districts, "seoul_cafe_district_summary.csv")
    write_table(skew, "seoul_cafe_skew_transform_comparison.csv")

    figures: dict[str, str] = {}
    if not args.skip_figures:
        figures = save_figures(df, comparison, districts)

    write_report(
        df=df,
        basic=basic,
        nan_reasons=nan_reasons,
        summary=summary,
        comparison=comparison,
        corr_pairs=corr_pairs,
        skew=skew,
        districts=districts,
        figures=figures,
        skip_figures=args.skip_figures,
    )

    print(f"Rows: {len(df):,}")
    print(f"Starbucks rows: {int((df[LABEL_COL] == 1).sum()):,}")
    print(f"Non-Starbucks rows: {int((df[LABEL_COL] == 0).sum()):,}")
    print(f"Tables written: {TABLE_DIR}")
    print(f"Archive table snapshots written: {ARCHIVE_TABLE_DIR}")
    print(f"Report written: {REPORT_PATH}")
    if args.skip_figures:
        print("Figures skipped")
    else:
        print(f"Figures written: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
