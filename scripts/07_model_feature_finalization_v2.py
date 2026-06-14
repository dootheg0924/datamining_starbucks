from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from _utils import (
    DATA_DIR,
    GENERATED_REPORT_DIR,
    GENERATED_TABLE_DIR,
    INTERMEDIATE_DATA_DIR,
    RAWDATA_DIR,
    ensure_dirs,
    markdown_table,
    read_csv_with_fallback,
    relative_posix,
)

TABLE_DIR = GENERATED_TABLE_DIR

BASE_PATH = INTERMEDIATE_DATA_DIR / "seoul_cafe_master_with_geo_features.csv"
PEAK_PATH = RAWDATA_DIR / "subway_time_group_analysis.csv"
SEOUL_OUTPUT_PATH = DATA_DIR / "seoul_cafe_model_features_v1.csv"
STARBUCKS_OUTPUT_PATH = DATA_DIR / "starbucks_model_features_v1.csv"
REPORT_PATH = GENERATED_REPORT_DIR / "07_model_feature_finalization_v2.md"

EARTH_RADIUS_KM = 6371.0088
RADIUS_500M_KM = 0.5

ID_COLUMNS = ["상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"]
EXISTING_FEATURES = [
    "dist_nearest_subway",
    "num_subway_500m",
    "nearest_subway_ridership",
    "num_restaurants_500m",
    "num_retail_500m",
    "num_convenience_500m",
    "avg_income",
    "num_offices",
    "living_population",
    "land_price",
]
PREVIOUSLY_CREATED_FEATURES = ["num_bus_stops_300m", "dist_nearest_starbucks"]
NEW_FEATURES = [
    "subway_morning_peak_500m",
    "subway_lunch_peak_500m",
    "subway_evening_peak_500m",
    "independent_cafe_count_500m",
    "low_price_cafe_count_500m",
    "other_franchise_cafe_count_500m",
]
FINAL_COLUMNS = [
    *ID_COLUMNS,
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
MODEL_FEATURES = [col for col in FINAL_COLUMNS if col not in ID_COLUMNS]
EXCLUDED_FEATURES = [
    "subway_ridership_500m",
    "num_competing_cafes_500m",
    "dist_nearest_bus_stop",
    "num_bus_stops_100m",
    "num_bus_stops_500m",
    "cafe_count_300m",
    "cafe_count_500m",
    "cafe_count_1000m",
    "premium_cafe_count_500m",
]
BRAND_TAXONOMY = {
    "independent_cafe_count_500m": ["프랜차이즈외"],
    "low_price_cafe_count_500m": ["메가MGC커피", "빽다방"],
    "other_franchise_cafe_count_500m": ["이디야커피", "투썸플레이스", "기타프랜차이즈"],
}


def find_station_master_path() -> Path:
    matches = [path for path in RAWDATA_DIR.glob("*.csv") if "역사마스터" in path.name]
    if not matches:
        raise FileNotFoundError("서울시_역사마스터_정보 파일을 찾지 못했습니다.")
    return matches[0]


def normalize_station_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\(.*?\)", "", text)
    if text.endswith("역"):
        text = text[:-1]
    return re.sub(r"\s+", "", text)


def normalize_line(value: object) -> str:
    text = str(value).strip().replace(" ", "")
    text = re.sub(r"\(.*?\)", "", text)
    numbered_line = re.match(r"^(\d+)호선.*$", text)
    if numbered_line:
        return numbered_line.group(1)
    text = text.replace("수도권광역급행철도", "GTX-A")
    if text in {"경의선", "중앙선"}:
        return "경의중앙선"
    if text == "공항철도1호선":
        return "공항철도"
    return text


def latlon_to_unit_xyz(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray) -> np.ndarray:
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def km_to_chord_radius(radius_km: float) -> float:
    return 2.0 * np.sin((radius_km / EARTH_RADIUS_KM) / 2.0)


def radius_sum(tree: cKDTree, values: np.ndarray, query_xyz: np.ndarray, radius_km: float) -> np.ndarray:
    radius = km_to_chord_radius(radius_km)
    neighbors = tree.query_ball_point(query_xyz, r=radius)
    return np.array([values[idx].sum() if len(idx) else 0.0 for idx in neighbors])


def radius_counts_for_brand(df: pd.DataFrame, brand_values: list[str], cafe_xyz: np.ndarray) -> np.ndarray:
    mask = df["브랜드"].isin(brand_values).to_numpy()
    category_xyz = cafe_xyz[mask]
    if len(category_xyz) == 0:
        return np.zeros(len(df), dtype=int)
    tree = cKDTree(category_xyz)
    radius = km_to_chord_radius(RADIUS_500M_KM)
    try:
        counts = tree.query_ball_point(cafe_xyz, r=radius, return_length=True).astype(int)
    except TypeError:
        counts = np.array([len(idx) for idx in tree.query_ball_point(cafe_xyz, r=radius)], dtype=int)
    counts = counts - mask.astype(int)
    return counts


def missing_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    total = len(df)
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "missing_count": int(df[feature].isna().sum()),
                "missing_rate": round(float(df[feature].isna().sum() / total), 6),
            }
            for feature in features
        ]
    )


def summary_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "Q1": series.quantile(0.25),
                "Q3": series.quantile(0.75),
                "max": series.max(),
                "zero_count": int((series == 0).sum()),
                "zero_rate": (series == 0).sum() / len(df),
                "skewness": series.skew(),
            }
        )
    return pd.DataFrame(rows).round(6)


def main() -> None:
    ensure_dirs(DATA_DIR, GENERATED_REPORT_DIR, TABLE_DIR)

    df, base_encoding = read_csv_with_fallback(BASE_PATH)
    peak_df, peak_encoding = read_csv_with_fallback(PEAK_PATH)
    station_path = find_station_master_path()
    station_df, station_encoding = read_csv_with_fallback(station_path)

    required_base = set(ID_COLUMNS + EXISTING_FEATURES + PREVIOUSLY_CREATED_FEATURES)
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Base file missing columns: {missing_base}")

    peak_required = {"호선명", "지하철역", "Morning Peak", "Lunch Peak", "Afternoon/Evening"}
    station_required = {"역사_ID", "역사명", "호선", "위도", "경도"}
    if not peak_required.issubset(peak_df.columns):
        raise ValueError(f"Peak file missing columns: {peak_required - set(peak_df.columns)}")
    if not station_required.issubset(station_df.columns):
        raise ValueError(f"Station file missing columns: {station_required - set(station_df.columns)}")

    cafe_xyz = latlon_to_unit_xyz(df["위도"], df["경도"])

    for output_col, brand_values in BRAND_TAXONOMY.items():
        df[output_col] = radius_counts_for_brand(df, brand_values, cafe_xyz)

    peak_work = peak_df.copy()
    peak_work["norm_line"] = peak_work["호선명"].map(normalize_line)
    peak_work["norm_station"] = peak_work["지하철역"].map(normalize_station_name)
    peak_agg = (
        peak_work.groupby(["norm_line", "norm_station"], as_index=False)
        .agg(
            original_rows=("지하철역", "count"),
            original_lines=("호선명", lambda x: "; ".join(sorted(set(map(str, x))))),
            original_stations=("지하철역", lambda x: "; ".join(sorted(set(map(str, x))))),
            **{
                "Morning Peak": ("Morning Peak", "sum"),
                "Lunch Peak": ("Lunch Peak", "sum"),
                "Afternoon/Evening": ("Afternoon/Evening", "sum"),
            },
        )
    )

    station_work = station_df.copy()
    station_work["norm_line"] = station_work["호선"].map(normalize_line)
    station_work["norm_station"] = station_work["역사명"].map(normalize_station_name)
    station_unique = station_work.drop_duplicates(subset=["norm_line", "norm_station", "위도", "경도"]).copy()
    merged_station_peak = station_unique.merge(
        peak_agg,
        on=["norm_line", "norm_station"],
        how="inner",
        validate="many_to_one",
    )

    peak_pairs = peak_agg[["norm_line", "norm_station", "original_lines", "original_stations"]]
    matched_pairs = merged_station_peak[["norm_line", "norm_station"]].drop_duplicates()
    merge_failures = peak_pairs.merge(
        matched_pairs,
        on=["norm_line", "norm_station"],
        how="left",
        indicator=True,
    )
    merge_failures = merge_failures[merge_failures["_merge"] == "left_only"].drop(columns=["_merge"])
    merge_failures = merge_failures.rename(
        columns={
            "norm_line": "normalized_line",
            "norm_station": "normalized_station",
            "original_lines": "source_line_names",
            "original_stations": "source_station_names",
        }
    )
    merge_failures.to_csv(
        TABLE_DIR / "subway_peak_station_merge_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_peak_xyz = latlon_to_unit_xyz(merged_station_peak["위도"], merged_station_peak["경도"])
    station_tree = cKDTree(station_peak_xyz)
    df["subway_morning_peak_500m"] = radius_sum(
        station_tree, merged_station_peak["Morning Peak"].to_numpy(dtype=float), cafe_xyz, RADIUS_500M_KM
    )
    df["subway_lunch_peak_500m"] = radius_sum(
        station_tree, merged_station_peak["Lunch Peak"].to_numpy(dtype=float), cafe_xyz, RADIUS_500M_KM
    )
    df["subway_evening_peak_500m"] = radius_sum(
        station_tree,
        merged_station_peak["Afternoon/Evening"].to_numpy(dtype=float),
        cafe_xyz,
        RADIUS_500M_KM,
    )

    df[NEW_FEATURES] = df[NEW_FEATURES].round(6)
    final_all = df[FINAL_COLUMNS].copy()
    final_starbucks = final_all[final_all["is_starbucks"] == 1].copy()

    final_all.to_csv(SEOUL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    final_starbucks.to_csv(STARBUCKS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    columns_table = pd.DataFrame(
        [
            {
                "order": idx + 1,
                "column": column,
                "role": "id_interpretation" if column in ID_COLUMNS else "model_feature",
                "group": (
                    "식별/해석"
                    if column in ID_COLUMNS
                    else "지리/교통"
                    if column
                    in {
                        "dist_nearest_subway",
                        "num_subway_500m",
                        "nearest_subway_ridership",
                        "num_bus_stops_300m",
                        "subway_morning_peak_500m",
                        "subway_lunch_peak_500m",
                        "subway_evening_peak_500m",
                    }
                    else "상권"
                    if column
                    in {
                        "num_restaurants_500m",
                        "num_retail_500m",
                        "num_convenience_500m",
                        "independent_cafe_count_500m",
                        "low_price_cafe_count_500m",
                        "other_franchise_cafe_count_500m",
                        "dist_nearest_starbucks",
                    }
                    else "인구/통계"
                ),
            }
            for idx, column in enumerate(FINAL_COLUMNS)
        ]
    )
    status_table = pd.DataFrame(
        [
            *[
                {"feature": feature, "status": "기존 master 변수 그대로 사용"}
                for feature in EXISTING_FEATURES
            ],
            *[
                {"feature": feature, "status": "이전 단계에서 생성되어 가져옴"}
                for feature in PREVIOUSLY_CREATED_FEATURES
            ],
            *[
                {"feature": feature, "status": "이번 v2에서 새로 생성"}
                for feature in NEW_FEATURES
            ],
            *[
                {"feature": feature, "status": "최종 output에서 제외"}
                for feature in EXCLUDED_FEATURES
            ],
        ]
    )
    missing_all = missing_table(final_all, MODEL_FEATURES)
    missing_starbucks = missing_table(final_starbucks, MODEL_FEATURES)
    summary_starbucks = summary_table(final_starbucks, MODEL_FEATURES)

    taxonomy_rows = []
    for output_col, brand_values in BRAND_TAXONOMY.items():
        taxonomy_rows.append(
            {
                "feature": output_col,
                "brand_group": ", ".join(brand_values),
                "brand_rows_in_master": int(df["브랜드"].isin(brand_values).sum()),
                "mean_all": df[output_col].mean(),
                "median_all": df[output_col].median(),
                "mean_starbucks": final_starbucks[output_col].mean(),
                "median_starbucks": final_starbucks[output_col].median(),
            }
        )
    taxonomy_rows.append(
        {
            "feature": "스타벅스 처리",
            "brand_group": "스타벅스",
            "brand_rows_in_master": int((df["브랜드"] == "스타벅스").sum()),
            "mean_all": np.nan,
            "median_all": np.nan,
            "mean_starbucks": np.nan,
            "median_starbucks": np.nan,
        }
    )
    taxonomy_summary = pd.DataFrame(taxonomy_rows).round(6)
    new_feature_summary = pd.concat(
        [
            summary_table(df, NEW_FEATURES).assign(dataset="all"),
            summary_table(final_starbucks, NEW_FEATURES).assign(dataset="starbucks"),
        ],
        ignore_index=True,
    )

    columns_table.to_csv(TABLE_DIR / "model_feature_v2_columns.csv", index=False, encoding="utf-8-sig")
    status_table.to_csv(TABLE_DIR / "model_feature_v2_status.csv", index=False, encoding="utf-8-sig")
    missing_all.to_csv(
        TABLE_DIR / "model_feature_v2_missing_values_all.csv", index=False, encoding="utf-8-sig"
    )
    missing_starbucks.to_csv(
        TABLE_DIR / "model_feature_v2_missing_values_starbucks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_starbucks.to_csv(
        TABLE_DIR / "model_feature_v2_summary_starbucks.csv", index=False, encoding="utf-8-sig"
    )
    taxonomy_summary.to_csv(
        TABLE_DIR / "cafe_brand_taxonomy_summary.csv", index=False, encoding="utf-8-sig"
    )
    new_feature_summary.to_csv(
        TABLE_DIR / "new_feature_v2_summary.csv", index=False, encoding="utf-8-sig"
    )

    merge_total = len(peak_agg)
    merge_success = merge_total - len(merge_failures)
    merge_rate = merge_success / merge_total if merge_total else np.nan
    peak_definition_note = (
        "`subway_time_group_analysis.csv`의 기존 peak 집계 컬럼을 사용했다. "
        "사용자가 확인한 시간대 정의는 Morning Peak=07-10, Lunch Peak=11-14, "
        "Afternoon/Evening=15-20이다."
    )
    shape_table = pd.DataFrame(
        [
            {
                "file": "data/seoul_cafe_model_features_v1.csv",
                "rows": final_all.shape[0],
                "columns": final_all.shape[1],
            },
            {
                "file": "data/starbucks_model_features_v1.csv",
                "rows": final_starbucks.shape[0],
                "columns": final_starbucks.shape[1],
            },
        ]
    )
    unit_table = pd.DataFrame(
        [
            {"item": "거리 변수", "unit_note": "km 단위"},
            {"item": "반경 count 변수", "unit_note": "변수명에 m 단위 유지"},
            {"item": "지하철 peak 변수", "unit_note": "500m 반경 내 peak 승하차량 합. Morning=07-10, Lunch=11-14, Evening=15-20"},
            {"item": "카페 분류 count 변수", "unit_note": "500m 반경 count"},
        ]
    )

    lines = [
        "# 07 Model Feature Finalization v2",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- base file: `{relative_posix(BASE_PATH)}` ({base_encoding})",
        f"- subway peak file: `{PEAK_PATH.name}` ({peak_encoding})",
        f"- station master file: `{station_path.name}` ({station_encoding})",
        "- 처리 원칙: 원본 덮어쓰기 없음, 결측치 대체 없음, 이상치 제거 없음, clustering/classification 실행 없음",
        "",
        "## 1. 기존 방향 수정",
        "",
        "이전 slim CSV는 clustering에 필요한 변수 일부만 남겨 원래 변수 체계를 충분히 반영하지 못했다. "
        "이번 v2에서는 사용자가 확정한 변수 체계를 기준으로 지리/교통, 상권, 인구/통계 변수를 균형 있게 포함했다.",
        "",
        "## 2. 출력 파일 shape",
        "",
        markdown_table(shape_table),
        "",
        "## 3. 최종 컬럼 목록",
        "",
        markdown_table(columns_table),
        "",
        "## 4. 변수 상태",
        "",
        markdown_table(status_table),
        "",
        "## 5. 지하철 peak 변수",
        "",
        peak_definition_note,
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "metric": "peak station normalized pairs",
                        "value": merge_total,
                    },
                    {
                        "metric": "merge success",
                        "value": merge_success,
                    },
                    {
                        "metric": "merge failures",
                        "value": len(merge_failures),
                    },
                    {
                        "metric": "merge success rate",
                        "value": round(merge_rate, 6),
                    },
                ]
            )
        ),
        "",
        "- 병합 실패 목록 저장: `reports/generated/tables/subway_peak_station_merge_failures.csv`",
        "- 병합 key: 호선명/역명에서 공백, 괄호, 일부 호선 suffix를 정규화한 key",
        "- 반경 500m 내 지하철역이 없는 카페는 peak 변수 값을 0으로 두었다.",
        "",
        "## 6. 카페 분류 수정",
        "",
        "기존 premium/low price 분류는 현재 브랜드 데이터 구조상 애매했다. "
        "따라서 이번 버전에서는 주변 카페를 independent, low price franchise, other franchise 세 가지로 나누었다. "
        "스타벅스는 이 세 count에 포함하지 않고, `dist_nearest_starbucks`로 별도 반영했다.",
        "",
        markdown_table(taxonomy_summary),
        "",
        "## 7. 최종 feature 결측치: 전체 데이터",
        "",
        markdown_table(missing_all),
        "",
        "## 8. 최종 feature 결측치: 스타벅스 데이터",
        "",
        markdown_table(missing_starbucks),
        "",
        "## 9. 스타벅스 681개 기준 기본 통계",
        "",
        markdown_table(summary_starbucks),
        "",
        "## 10. 이번 v2 신규 변수 기본 통계",
        "",
        markdown_table(new_feature_summary),
        "",
        "## 11. 단위 기록",
        "",
        markdown_table(unit_table),
        "",
        "## 12. 저장 산출물",
        "",
        "- `data/seoul_cafe_model_features_v1.csv`",
        "- `data/starbucks_model_features_v1.csv`",
        "- `reports/generated/07_model_feature_finalization_v2.md`",
        "- `reports/generated/tables/model_feature_v2_columns.csv`",
        "- `reports/generated/tables/model_feature_v2_status.csv`",
        "- `reports/generated/tables/model_feature_v2_missing_values_all.csv`",
        "- `reports/generated/tables/model_feature_v2_missing_values_starbucks.csv`",
        "- `reports/generated/tables/model_feature_v2_summary_starbucks.csv`",
        "- `reports/generated/tables/subway_peak_station_merge_failures.csv`",
        "- `reports/generated/tables/cafe_brand_taxonomy_summary.csv`",
        "- `reports/generated/tables/new_feature_v2_summary.csv`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"All output shape: {final_all.shape}")
    print(f"Starbucks output shape: {final_starbucks.shape}")
    print(f"Subway peak merge success rate: {merge_rate:.4f}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
