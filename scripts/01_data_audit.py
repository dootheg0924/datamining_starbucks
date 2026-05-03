from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "seoul_cafe_master.csv"
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_PATH = REPORT_DIR / "01_data_audit.md"


def read_master_csv(path: Path) -> pd.DataFrame:
    """Load the master CSV without mutating or imputing any values."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    table = df.copy()
    table = table.astype(object).where(pd.notna(table), "")
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    header_line = "| " + " | ".join(
        headers[i].ljust(widths[i]) for i in range(len(headers))
    ) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_rate = missing_count / len(df) * 100 if len(df) else missing_count
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": [int(missing_count[col]) for col in df.columns],
            "missing_rate_pct": [round(float(missing_rate[col]), 4) for col in df.columns],
        }
    )


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame(
            columns=["column", "mean", "median", "std", "min", "max", "Q1", "Q3"]
        )

    stats = pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "std": numeric_df.std(),
            "min": numeric_df.min(),
            "max": numeric_df.max(),
            "Q1": numeric_df.quantile(0.25),
            "Q3": numeric_df.quantile(0.75),
        }
    )
    stats = stats.reset_index(names="column")
    return stats.round(4)


def value_counts_table(series: pd.Series, column_name: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = [column_name, "count"]
    counts[column_name] = counts[column_name].astype("string").fillna("<NA>")
    return counts


def infer_distance_unit(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    col = "dist_nearest_subway"
    if col not in df.columns:
        return pd.DataFrame(), "`dist_nearest_subway` 컬럼이 없어 단위 추정을 하지 못했습니다."

    distance = pd.to_numeric(df[col], errors="coerce").dropna()
    if distance.empty:
        return pd.DataFrame(), "`dist_nearest_subway` 유효 숫자값이 없어 단위 추정을 하지 못했습니다."

    summary = pd.DataFrame(
        {
            "metric": ["min", "Q1", "median", "Q3", "max"],
            "value": [
                distance.min(),
                distance.quantile(0.25),
                distance.median(),
                distance.quantile(0.75),
                distance.max(),
            ],
        }
    ).round(4)

    max_value = float(distance.max())
    median_value = float(distance.median())
    if max_value <= 20 and median_value <= 5:
        note = (
            "값 범위가 대체로 소수점 단위이며 최댓값도 20 이하이므로 "
            "`dist_nearest_subway`는 km 단위일 가능성이 높다고 추정됩니다. "
            "다만 원천 데이터 정의를 확인하기 전까지 확정하지 않습니다."
        )
    elif max_value >= 100:
        note = (
            "값 범위가 수백 이상의 크기를 포함하므로 `dist_nearest_subway`는 "
            "m 단위일 가능성이 있다고 추정됩니다. "
            "다만 원천 데이터 정의를 확인하기 전까지 확정하지 않습니다."
        )
    else:
        note = (
            "값 범위만으로는 `dist_nearest_subway`의 단위를 명확히 보기 어렵습니다. "
            "현재 단계에서는 단위 미확정으로 기록합니다."
        )

    return summary, note


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = read_master_csv(DATA_PATH)
    df_starbucks = df[df["is_starbucks"] == 1].copy()

    columns_dtypes = pd.DataFrame(
        {"column": df.columns, "dtype": [str(dtype) for dtype in df.dtypes]}
    )
    starbucks_counts = value_counts_table(df["is_starbucks"], "is_starbucks")
    brand_counts = value_counts_table(df["브랜드"], "브랜드")
    missing_all = missing_table(df)
    missing_starbucks = missing_table(df_starbucks)
    stats_all = numeric_stats(df)
    stats_starbucks = numeric_stats(df_starbucks)
    distance_range, distance_note = infer_distance_unit(df)

    missing_all.to_csv(TABLE_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")
    missing_starbucks.to_csv(
        TABLE_DIR / "missing_values_starbucks.csv", index=False, encoding="utf-8-sig"
    )
    stats_all.to_csv(TABLE_DIR / "basic_stats_all.csv", index=False, encoding="utf-8-sig")
    stats_starbucks.to_csv(
        TABLE_DIR / "basic_stats_starbucks.csv", index=False, encoding="utf-8-sig"
    )

    lines = [
        "# 01 Data Audit: seoul_cafe_master.csv",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 원본 파일: `{DATA_PATH.name}`",
        "- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, 모델링 없음",
        "",
        "## 1. 데이터 크기",
        "",
        f"- 전체 행 수: {len(df):,}",
        f"- 전체 열 수: {df.shape[1]:,}",
        f"- 스타벅스 필터 행 수 (`df_starbucks`): {len(df_starbucks):,}",
        "",
        "## 2. 컬럼명과 dtype",
        "",
        markdown_table(columns_dtypes),
        "",
        "## 3. is_starbucks 기준 개수",
        "",
        markdown_table(starbucks_counts),
        "",
        "## 4. 브랜드 value counts",
        "",
        markdown_table(brand_counts),
        "",
        "## 5. 컬럼별 결측치: 전체 데이터",
        "",
        markdown_table(missing_all),
        "",
        "## 6. 숫자형 변수 기본통계: 전체 데이터",
        "",
        markdown_table(stats_all),
        "",
        "## 7. 컬럼별 결측치: 스타벅스 데이터",
        "",
        markdown_table(missing_starbucks),
        "",
        "## 8. 숫자형 변수 기본통계: 스타벅스 데이터",
        "",
        markdown_table(stats_starbucks),
        "",
        "## 9. dist_nearest_subway 값 범위와 단위 추정",
        "",
        markdown_table(distance_range),
        "",
        f"- 단위 기록: {distance_note}",
        "",
        "## 10. 저장된 표",
        "",
        "- `reports/tables/missing_values.csv`",
        "- `reports/tables/missing_values_starbucks.csv`",
        "- `reports/tables/basic_stats_all.csv`",
        "- `reports/tables/basic_stats_starbucks.csv`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Loaded: {DATA_PATH.name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Report written: {REPORT_PATH}")
    print(f"Tables written: {TABLE_DIR}")


if __name__ == "__main__":
    main()
