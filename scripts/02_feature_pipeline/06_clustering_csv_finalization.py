from __future__ import annotations

from datetime import datetime

import pandas as pd

from _utils import (
    GENERATED_REPORT_DIR,
    GENERATED_TABLE_DIR,
    INTERMEDIATE_DATA_DIR,
    ensure_dirs,
    markdown_table,
    relative_posix,
)

TABLE_DIR = GENERATED_TABLE_DIR

INPUT_PATH = INTERMEDIATE_DATA_DIR / "seoul_cafe_master_with_geo_features.csv"
SEOUL_OUTPUT_PATH = INTERMEDIATE_DATA_DIR / "seoul_cafe_clustering_features_v1.csv"
STARBUCKS_OUTPUT_PATH = INTERMEDIATE_DATA_DIR / "starbucks_clustering_features_v1.csv"
REPORT_PATH = GENERATED_REPORT_DIR / "06_clustering_csv_finalization.md"

ID_COLUMNS = [
    "상호명",
    "브랜드",
    "is_starbucks",
    "위도",
    "경도",
    "시군구명",
    "행정동코드",
    "행정동명",
    "도로명주소",
]
FEATURE_COLUMNS = [
    "dist_nearest_subway",
    "subway_ridership_500m",
    "num_bus_stops_300m",
    "cafe_count_300m",
    "dist_nearest_starbucks",
    "num_restaurants_500m",
]
FINAL_COLUMNS = [*ID_COLUMNS, *FEATURE_COLUMNS]
EXCLUDED_CANDIDATES = [
    "num_bus_stops_100m",
    "num_bus_stops_500m",
    "cafe_count_500m",
    "cafe_count_1000m",
    "num_competing_cafes_500m",
    "num_retail_500m",
    "num_convenience_500m",
]
EXCLUSION_REASONS = {
    "num_bus_stops_100m": "0값 비율이 높아 너무 희소함",
    "num_bus_stops_500m": "반경이 넓어 생활권 전체 성격이 강함",
    "cafe_count_500m": "기존 경쟁 카페 변수와 거의 중복",
    "cafe_count_1000m": "개별 매장 입지보다 지역 상권 규모를 반영",
    "num_competing_cafes_500m": "cafe_count_300m를 직접 경쟁권 대표 변수로 선택했으므로 제외",
    "num_retail_500m": "상권 밀도 변수와 중복 가능성이 높아 clustering 입력에서는 제외",
    "num_convenience_500m": "상권 밀도 변수와 중복 가능성이 높아 clustering 입력에서는 제외",
}


def missing_table(df: pd.DataFrame, features: list[str], dataset_name: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    for feature in features:
        missing_count = int(df[feature].isna().sum())
        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "missing_count": missing_count,
                "missing_rate": round(missing_count / total, 6),
            }
        )
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        s = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "count": int(s.count()),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "Q1": s.quantile(0.25),
                "Q3": s.quantile(0.75),
                "max": s.max(),
                "zero_count": int((s == 0).sum()),
                "zero_rate": (s == 0).sum() / len(df),
                "skewness": s.skew(),
            }
        )
    return pd.DataFrame(rows).round(6)


def main() -> None:
    ensure_dirs(INTERMEDIATE_DATA_DIR, GENERATED_REPORT_DIR, TABLE_DIR)

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    missing_columns = [col for col in FINAL_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    seoul_slim = df[FINAL_COLUMNS].copy()
    starbucks_slim = seoul_slim[seoul_slim["is_starbucks"] == 1].copy()

    seoul_slim.to_csv(SEOUL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    starbucks_slim.to_csv(STARBUCKS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    excluded_check = pd.DataFrame(
        [
            {
                "excluded_candidate": col,
                "present_in_seoul_output": col in seoul_slim.columns,
                "present_in_starbucks_output": col in starbucks_slim.columns,
                "reason": EXCLUSION_REASONS[col],
            }
            for col in EXCLUDED_CANDIDATES
        ]
    )

    missing_values = pd.concat(
        [
            missing_table(seoul_slim, FEATURE_COLUMNS, "seoul_cafe_clustering_features_v1"),
            missing_table(starbucks_slim, FEATURE_COLUMNS, "starbucks_clustering_features_v1"),
        ],
        ignore_index=True,
    )
    summary_starbucks = summary_table(starbucks_slim, FEATURE_COLUMNS)

    missing_values.to_csv(
        TABLE_DIR / "clustering_feature_missing_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_starbucks.to_csv(
        TABLE_DIR / "clustering_feature_summary_starbucks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shape_table = pd.DataFrame(
        [
            {
                "file": "data/archive/intermediate/seoul_cafe_clustering_features_v1.csv",
                "rows": seoul_slim.shape[0],
                "columns": seoul_slim.shape[1],
            },
            {
                "file": "data/archive/intermediate/starbucks_clustering_features_v1.csv",
                "rows": starbucks_slim.shape[0],
                "columns": starbucks_slim.shape[1],
            },
        ]
    )
    column_table = pd.DataFrame(
        {
            "order": range(1, len(FINAL_COLUMNS) + 1),
            "column": FINAL_COLUMNS,
            "role": ["식별/해석용"] * len(ID_COLUMNS) + ["clustering feature"] * len(FEATURE_COLUMNS),
        }
    )
    unit_table = pd.DataFrame(
        [
            {"feature": "dist_nearest_subway", "unit_note": "km 단위 거리 변수"},
            {"feature": "dist_nearest_starbucks", "unit_note": "km 단위 거리 변수"},
            {"feature": "num_bus_stops_300m", "unit_note": "300m 반경 count. 변수명에 m 유지"},
            {"feature": "cafe_count_300m", "unit_note": "300m 반경 count. 변수명에 m 유지"},
            {"feature": "subway_ridership_500m", "unit_note": "500m 반경 지하철 승하차량 집계"},
            {"feature": "num_restaurants_500m", "unit_note": "500m 반경 음식점 수"},
        ]
    )

    lines = [
        "# 06 Clustering CSV Finalization",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 입력 파일: `{relative_posix(INPUT_PATH)}`",
        "- 목적: 후보 반경 변수를 모두 담은 master에서 clustering 담당자가 바로 쓸 slim CSV 생성",
        "- 처리 원칙: 원본 파일 덮어쓰기 없음, 결측치 대체 없음, 이상치 제거 없음, clustering 실행 없음",
        "",
        "## 1. 출력 파일 shape",
        "",
        markdown_table(shape_table),
        "",
        "## 2. 포함 컬럼",
        "",
        markdown_table(column_table),
        "",
        "## 3. 제외 후보 변수 확인",
        "",
        markdown_table(excluded_check),
        "",
        "## 4. Clustering feature 결측치",
        "",
        markdown_table(missing_values),
        "",
        "## 5. 스타벅스 파일 기준 기본 통계",
        "",
        markdown_table(summary_starbucks),
        "",
        "## 6. 단위 기록",
        "",
        markdown_table(unit_table),
        "",
        "거리 변수 `dist_nearest_subway`, `dist_nearest_starbucks`는 km 단위로 기록한다. "
        "반경 count 변수 `num_bus_stops_300m`, `cafe_count_300m`는 변수명에 m를 유지해 반경 기준을 명확히 했다.",
        "",
        "## 7. 저장 산출물",
        "",
        "- `data/archive/intermediate/seoul_cafe_clustering_features_v1.csv`",
        "- `data/archive/intermediate/starbucks_clustering_features_v1.csv`",
        "- `reports/06_clustering_csv_finalization.md`",
        "- `reports/generated/tables/clustering_feature_missing_values.csv`",
        "- `reports/generated/tables/clustering_feature_summary_starbucks.csv`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Seoul output shape: {seoul_slim.shape}")
    print(f"Starbucks output shape: {starbucks_slim.shape}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
