from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "seoul_cafe_master_with_geo_features.csv"
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures" / "radius_selection"
REPORT_PATH = REPORT_DIR / "05_radius_selection_eda.md"

NEW_FEATURES = [
    "dist_nearest_bus_stop",
    "num_bus_stops_100m",
    "num_bus_stops_300m",
    "num_bus_stops_500m",
    "cafe_count_300m",
    "cafe_count_500m",
    "cafe_count_1000m",
    "dist_nearest_starbucks",
]
OLD_FEATURES = [
    "dist_nearest_subway",
    "num_subway_500m",
    "subway_ridership_500m",
    "num_competing_cafes_500m",
    "num_restaurants_500m",
    "num_retail_500m",
    "num_convenience_500m",
]
ANALYSIS_FEATURES = [*NEW_FEATURES, *OLD_FEATURES]
BUS_RADIUS_FEATURES = ["num_bus_stops_100m", "num_bus_stops_300m", "num_bus_stops_500m"]
CAFE_RADIUS_FEATURES = ["cafe_count_300m", "cafe_count_500m", "cafe_count_1000m"]
ID_COLUMNS = ["상호명", "시군구명", "도로명주소"]


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
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
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for feature in features:
        s = pd.to_numeric(df[feature], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        missing_count = int(s.isna().sum())
        zero_count = int((s == 0).sum())
        rows.append(
            {
                "feature": feature,
                "count": int(s.count()),
                "missing_count": missing_count,
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "Q1": q1,
                "Q3": q3,
                "max": s.max(),
                "zero_count": zero_count,
                "zero_rate": zero_count / total,
                "skewness": s.skew(),
                "unique_count": int(s.nunique(dropna=True)),
                "cv": s.std() / s.mean() if s.mean() != 0 else pd.NA,
            }
        )
    return pd.DataFrame(rows).round(6)


def pairwise_corr_rows(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for left, right in pairs:
        rows.append(
            {
                "feature_1": left,
                "feature_2": right,
                "pearson": df[left].corr(df[right], method="pearson"),
                "spearman": df[left].corr(df[right], method="spearman"),
            }
        )
    return pd.DataFrame(rows).round(6)


def build_top_bottom(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        cols = [*ID_COLUMNS, feature]
        nonmissing = df[cols].dropna(subset=[feature]).copy()
        for rank_type, selected in [
            ("top", nonmissing.sort_values(feature, ascending=False).head(10)),
            ("bottom", nonmissing.sort_values(feature, ascending=True).head(10)),
        ]:
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                rows.append(
                    {
                        "feature": feature,
                        "rank_type": rank_type,
                        "rank": rank,
                        "상호명": row["상호명"],
                        "시군구명": row["시군구명"],
                        "도로명주소": row["도로명주소"],
                        "value": row[feature],
                    }
                )
    return pd.DataFrame(rows).round({"value": 6})


def recommendation_table() -> pd.DataFrame:
    rows = [
        {
            "feature": "dist_nearest_bus_stop",
            "recommendation": "candidate",
            "reason": "가장 가까운 버스정류장까지 거리라 해석은 명확하지만 num_bus_stops_300m와 함께 쓰면 버스 접근성 축이 과대표현될 수 있음",
        },
        {
            "feature": "num_bus_stops_100m",
            "recommendation": "drop_too_sparse",
            "reason": "스타벅스 기준 0값 비율이 높아 근접 정류장 유무에 치우침",
        },
        {
            "feature": "num_bus_stops_300m",
            "recommendation": "select",
            "reason": "0값이 거의 없고 매장 주변 도보권 접근성을 잘 나타내며 500m보다 국지성이 좋음",
        },
        {
            "feature": "num_bus_stops_500m",
            "recommendation": "drop_too_broad",
            "reason": "0값이 없고 300m와 상관이 높아 더 넓은 생활권 성격이 강함",
        },
        {
            "feature": "cafe_count_300m",
            "recommendation": "select",
            "reason": "직접 경쟁권으로 해석하기 쉽고 500m/1000m보다 매장 인접 상권 차이를 더 잘 보존",
        },
        {
            "feature": "cafe_count_500m",
            "recommendation": "drop_duplicate",
            "reason": "기존 num_competing_cafes_500m와 거의 동일하며 cafe_count_300m와도 높은 상관",
        },
        {
            "feature": "cafe_count_1000m",
            "recommendation": "drop_too_broad",
            "reason": "개별 매장 입지보다 지역 상권 규모를 강하게 반영하고 반경별 카페 수와 중복성이 큼",
        },
        {
            "feature": "dist_nearest_starbucks",
            "recommendation": "select",
            "reason": "스타벅스 내부에서 도심 밀집형과 독립 입지형을 구분하는 해석력이 있음",
        },
        {
            "feature": "dist_nearest_subway",
            "recommendation": "candidate",
            "reason": "지하철 접근성 축으로 기존 후보 유지. 버스 접근성 변수와는 다른 교통 차원을 제공",
        },
        {
            "feature": "num_subway_500m",
            "recommendation": "use_for_interpretation_only",
            "reason": "이산형이고 IQR이 작아 모델 입력보다는 역세권 여부 해석에 적합",
        },
        {
            "feature": "subway_ridership_500m",
            "recommendation": "candidate",
            "reason": "역세권 규모를 나타내지만 0값과 왜도가 있어 변환 후 후보로 유지",
        },
        {
            "feature": "num_competing_cafes_500m",
            "recommendation": "drop_duplicate",
            "reason": "cafe_count_500m와 거의 중복. 반경 대표로 cafe_count_300m를 쓰면 동시에 넣지 않는 편이 좋음",
        },
        {
            "feature": "num_restaurants_500m",
            "recommendation": "candidate",
            "reason": "카페 수와 다르면서 상권 활성을 나타내는 기존 상권 후보. 카페 밀도와 상관 확인 후 선택",
        },
        {
            "feature": "num_retail_500m",
            "recommendation": "use_for_interpretation_only",
            "reason": "상권 규모 변수들과 중복 가능성이 커서 모델 입력보다 유형 설명에 유용",
        },
        {
            "feature": "num_convenience_500m",
            "recommendation": "use_for_interpretation_only",
            "reason": "상권 밀도 보조 지표로 해석용 가치가 있으나 음식점/카페 변수와 중복 가능성이 큼",
        },
    ]
    return pd.DataFrame(rows)


def save_histograms(df: pd.DataFrame) -> list[dict[str, str]]:
    figure_rows = []
    for feature in [*BUS_RADIUS_FEATURES, *CAFE_RADIUS_FEATURES, "dist_nearest_starbucks"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], bins=30, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(f"{feature} histogram")
        ax.set_xlabel(f"{feature} (km)" if feature.startswith("dist_") else feature)
        ax.set_ylabel("Starbucks count")
        fig.tight_layout()
        path = FIGURE_DIR / f"{feature}_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_rows.append({"feature": feature, "figure": str(path.relative_to(ROOT)).replace("\\", "/")})
    return figure_rows


def save_heatmap(corr: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Radius selection feature correlation (Starbucks only)")
    fig.tight_layout()
    path = FIGURE_DIR / "radius_feature_correlation_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    missing_cols = [col for col in ["is_starbucks", *ANALYSIS_FEATURES, *ID_COLUMNS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_starbucks = df[df["is_starbucks"] == 1].copy()

    summary = feature_summary(df_starbucks, NEW_FEATURES)
    corr = df_starbucks[ANALYSIS_FEATURES].corr(method="pearson").round(6)
    corr.to_csv(TABLE_DIR / "radius_feature_correlations.csv", encoding="utf-8-sig")
    summary.to_csv(
        TABLE_DIR / "radius_feature_summary_starbucks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recommendations = recommendation_table()
    recommendations.to_csv(
        TABLE_DIR / "radius_feature_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_bottom = build_top_bottom(df_starbucks, [*NEW_FEATURES, "num_competing_cafes_500m"])
    top_bottom.to_csv(
        TABLE_DIR / "radius_feature_top_bottom_starbucks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_rows = save_histograms(df_starbucks)
    heatmap_path = save_heatmap(corr)

    bus_pair_corr = pairwise_corr_rows(
        df_starbucks,
        [
            ("num_bus_stops_100m", "num_bus_stops_300m"),
            ("num_bus_stops_100m", "num_bus_stops_500m"),
            ("num_bus_stops_300m", "num_bus_stops_500m"),
        ],
    )
    cafe_pair_corr = pairwise_corr_rows(
        df_starbucks,
        [
            ("cafe_count_300m", "cafe_count_500m"),
            ("cafe_count_300m", "cafe_count_1000m"),
            ("cafe_count_500m", "cafe_count_1000m"),
        ],
    )
    competing_compare = pd.DataFrame(
        [
            {
                "comparison": "num_competing_cafes_500m vs cafe_count_500m",
                "pearson": df_starbucks["num_competing_cafes_500m"].corr(
                    df_starbucks["cafe_count_500m"], method="pearson"
                ),
                "spearman": df_starbucks["num_competing_cafes_500m"].corr(
                    df_starbucks["cafe_count_500m"], method="spearman"
                ),
                "mean_existing": df_starbucks["num_competing_cafes_500m"].mean(),
                "mean_new": df_starbucks["cafe_count_500m"].mean(),
                "mean_new_minus_existing": (
                    df_starbucks["cafe_count_500m"] - df_starbucks["num_competing_cafes_500m"]
                ).mean(),
                "median_new_minus_existing": (
                    df_starbucks["cafe_count_500m"] - df_starbucks["num_competing_cafes_500m"]
                ).median(),
            }
        ]
    ).round(6)

    selected_corr_checks = pd.DataFrame(
        [
            {
                "relationship": "dist_nearest_bus_stop vs dist_nearest_subway",
                "pearson": corr.loc["dist_nearest_bus_stop", "dist_nearest_subway"],
                "interpretation": "버스 접근성과 지하철 접근성이 같은 축인지 확인",
            },
            {
                "relationship": "dist_nearest_starbucks vs cafe_count_300m",
                "pearson": corr.loc["dist_nearest_starbucks", "cafe_count_300m"],
                "interpretation": "스타벅스 독립성과 직접 카페 밀도 관계",
            },
            {
                "relationship": "dist_nearest_starbucks vs num_restaurants_500m",
                "pearson": corr.loc["dist_nearest_starbucks", "num_restaurants_500m"],
                "interpretation": "스타벅스 독립성과 상권 활성도 관계",
            },
            {
                "relationship": "cafe_count_500m vs num_restaurants_500m",
                "pearson": corr.loc["cafe_count_500m", "num_restaurants_500m"],
                "interpretation": "카페 밀도와 음식점 밀도의 중복성",
            },
            {
                "relationship": "cafe_count_500m vs num_retail_500m",
                "pearson": corr.loc["cafe_count_500m", "num_retail_500m"],
                "interpretation": "카페 밀도와 소매업 밀도의 중복성",
            },
        ]
    ).round(6)

    feature_set = pd.DataFrame(
        [
            {
                "feature_set": "Geo Radius Set 추천안",
                "features": "dist_nearest_bus_stop, num_bus_stops_300m, cafe_count_300m, dist_nearest_starbucks, num_restaurants_500m",
                "note": "버스 접근성, 직접 카페 경쟁권, 스타벅스 밀집/독립성, 상권 활성도를 균형 있게 포함",
            },
            {
                "feature_set": "간결형",
                "features": "num_bus_stops_300m, cafe_count_300m, dist_nearest_starbucks",
                "note": "반경 변수만 최소로 넣어 중복을 줄인 구성",
            },
        ]
    )

    bus_summary = summary[summary["feature"].isin(BUS_RADIUS_FEATURES)][
        ["feature", "mean", "median", "std", "zero_count", "zero_rate", "skewness", "cv"]
    ]
    cafe_summary = summary[summary["feature"].isin(CAFE_RADIUS_FEATURES)][
        ["feature", "mean", "median", "std", "zero_count", "zero_rate", "skewness", "cv"]
    ]
    nearest_sb_summary = summary[summary["feature"] == "dist_nearest_starbucks"][
        ["feature", "mean", "median", "std", "min", "Q1", "Q3", "max", "skewness", "cv"]
    ]
    nearest_sb_top_bottom = top_bottom[top_bottom["feature"] == "dist_nearest_starbucks"]

    presentation_logic = (
        "버스 접근성은 100m, 300m, 500m 반경을 비교한 결과, 100m는 0값 비율이 높아 희소했고 "
        "500m는 모든 스타벅스 매장에 값이 존재해 생활권 전체를 반영하는 경향이 컸다. "
        "300m는 대부분의 매장에서 값이 존재하면서도 매장 간 차이를 유지해, clustering용 버스 접근성 대표 변수로 선택했다. "
        "카페 밀도는 300m, 500m, 1000m가 서로 높은 상관을 보였고, 500m는 기존 경쟁 카페 변수와 거의 중복되었다. "
        "따라서 개별 매장의 직접 경쟁권으로 해석 가능한 300m 카페 수를 대표 변수로 선택하고, 1000m는 지역 상권 규모 해석용으로 남긴다."
    )

    lines = [
        "# 05 Radius Selection EDA",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 입력 파일: `{str(INPUT_PATH.relative_to(ROOT)).replace(chr(92), '/')}`",
        "- 분석 대상: `df[df[\"is_starbucks\"] == 1]`",
        f"- 스타벅스 매장 수: {len(df_starbucks):,}",
        "- 목적: clustering 실행 전 좌표 기반 반경 변수와 중복 변수를 선택",
        "- 처리 원칙: clustering 없음, 결측치 대체 없음, 이상치 제거 없음, 비스타벅스 비교 없음",
        "",
        "## 1. 새 변수 기본 통계",
        "",
        markdown_table(summary),
        "",
        "## 2. 버스정류장 반경 선택",
        "",
        markdown_table(bus_summary),
        "",
        "### 버스 반경 간 상관",
        "",
        markdown_table(bus_pair_corr),
        "",
        "- `num_bus_stops_100m`: 스타벅스 기준 0값 비율이 높아 가장 가까운 정류장 유무에 치우친다.",
        "- `num_bus_stops_300m`: 0값이 거의 없고 도보 접근권으로 해석 가능하며, 분산도 유지된다.",
        "- `num_bus_stops_500m`: 0값이 없어 안정적이지만 300m와 상관이 높고 생활권 전체를 반영하는 성격이 강하다.",
        "- 판단: 버스 접근성 대표 반경은 `num_bus_stops_300m`를 선택한다. `dist_nearest_bus_stop`은 보조 후보로 유지한다.",
        "",
        "## 3. 카페 수 반경 선택",
        "",
        markdown_table(cafe_summary),
        "",
        "### 카페 반경 간 상관",
        "",
        markdown_table(cafe_pair_corr),
        "",
        "- `cafe_count_300m`: 매장 주변 직접 경쟁권으로 해석하기 가장 쉽다.",
        "- `cafe_count_500m`: 기존 `num_competing_cafes_500m`와 거의 중복되어 둘을 동시에 쓰지 않는 것이 좋다.",
        "- `cafe_count_1000m`: 개별 매장 입지보다 지역 상권 전체 규모를 반영한다.",
        "- 판단: clustering 입력 대표 반경은 `cafe_count_300m`를 선택하고, `cafe_count_1000m`는 해석용으로 남긴다.",
        "",
        "## 4. 기존 경쟁 카페 변수와 비교",
        "",
        "`cafe_count_500m`는 현재 master의 전체 카페 좌표를 기준으로 자기 자신만 제외하고 다시 계산한 변수다. "
        "`num_competing_cafes_500m`는 기존 master에 있던 변수로 원천과 세부 정의가 다를 수 있다.",
        "",
        markdown_table(competing_compare),
        "",
        "상관이 매우 높으므로 `cafe_count_500m`와 `num_competing_cafes_500m`를 동시에 clustering feature로 쓰는 것은 피한다. "
        "반경 선택 관점에서는 직접 경쟁권인 `cafe_count_300m`를 새 대표 변수로 쓰고, 기존 `num_competing_cafes_500m`는 비교/해석용으로 남기는 편이 낫다.",
        "",
        "## 5. dist_nearest_starbucks 평가",
        "",
        markdown_table(nearest_sb_summary),
        "",
        "### 상위/하위 10개",
        "",
        markdown_table(nearest_sb_top_bottom),
        "",
        "상위 매장은 종로평창동, 청계산입구역, 홍제역 등 주변 스타벅스와 거리가 먼 독립 입지로 해석 가능하다. "
        "하위 매장은 타임스퀘어, IFC, 잠실, 광화문 등 한 건물 또는 초근접 상권 내 복수 매장이 있는 도심 밀집형으로 해석된다. "
        "따라서 `dist_nearest_starbucks`는 도심 밀집형과 독립 입지형 cluster를 나누는 데 도움이 되는 변수로 판단한다.",
        "",
        "## 6. 새 변수와 기존 변수 상관관계",
        "",
        f"- correlation table: `reports/tables/radius_feature_correlations.csv`",
        f"- heatmap: `{str(heatmap_path.relative_to(ROOT)).replace(chr(92), '/')}`",
        "",
        markdown_table(selected_corr_checks),
        "",
        "## 7. 변수별 최종 recommendation",
        "",
        markdown_table(recommendations),
        "",
        "## 8. 제안 geo feature set",
        "",
        markdown_table(feature_set),
        "",
        "## 9. 발표용 반경 선택 논리",
        "",
        presentation_logic,
        "",
        "## 10. 저장된 시각화",
        "",
        markdown_table(pd.DataFrame(figure_rows)),
        "",
        "## 11. 저장 산출물",
        "",
        "- `reports/05_radius_selection_eda.md`",
        "- `reports/tables/radius_feature_summary_starbucks.csv`",
        "- `reports/tables/radius_feature_correlations.csv`",
        "- `reports/tables/radius_feature_recommendation.csv`",
        "- `reports/tables/radius_feature_top_bottom_starbucks.csv`",
        "- `reports/figures/radius_selection/`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Starbucks rows: {len(df_starbucks)}")
    print(f"Report written: {REPORT_PATH}")
    print(f"Tables written: {TABLE_DIR}")
    print(f"Figures written: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
