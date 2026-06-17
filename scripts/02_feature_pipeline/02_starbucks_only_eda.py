from __future__ import annotations

import math
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _utils import (
    GENERATED_FIGURE_DIR,
    GENERATED_REPORT_DIR,
    GENERATED_TABLE_DIR,
    RAWDATA_DIR,
    ensure_dirs,
    markdown_table,
    read_csv,
    relative_posix,
)

DATA_PATH = RAWDATA_DIR / "seoul_cafe_master.csv"
TABLE_DIR = GENERATED_TABLE_DIR
FIGURE_DIR = GENERATED_FIGURE_DIR / "starbucks_only"
REPORT_PATH = GENERATED_REPORT_DIR / "02_starbucks_only_eda.md"

FEATURES = [
    "dist_nearest_subway",
    "num_subway_500m",
    "nearest_subway_ridership",
    "subway_ridership_500m",
    "num_competing_cafes_500m",
    "num_restaurants_500m",
    "num_retail_500m",
    "num_convenience_500m",
    "avg_income",
    "num_offices",
    "living_population",
    "land_price",
]

ID_COLUMNS = ["상호명", "시군구명", "도로명주소"]

FEATURE_GROUPS = {
    "dist_nearest_subway": "지하철 접근성",
    "num_subway_500m": "지하철 접근성",
    "nearest_subway_ridership": "지하철 접근성",
    "subway_ridership_500m": "지하철 접근성",
    "num_competing_cafes_500m": "상권 밀도",
    "num_restaurants_500m": "상권 밀도",
    "num_retail_500m": "상권 밀도",
    "num_convenience_500m": "상권 밀도",
    "avg_income": "인구/경제",
    "num_offices": "인구/경제",
    "living_population": "인구/경제",
    "land_price": "인구/경제",
}

FEATURE_LABELS_KO = {
    "dist_nearest_subway": "가장 가까운 지하철역 거리",
    "num_subway_500m": "500m 내 지하철역 수",
    "nearest_subway_ridership": "가장 가까운 지하철역 승하차 인원",
    "subway_ridership_500m": "500m 내 지하철 승하차 인원",
    "num_competing_cafes_500m": "500m 내 경쟁 카페 수",
    "num_restaurants_500m": "500m 내 음식점 수",
    "num_retail_500m": "500m 내 소매업 수",
    "num_convenience_500m": "500m 내 편의점 수",
    "avg_income": "평균 소득",
    "num_offices": "오피스 수",
    "living_population": "생활 인구",
    "land_price": "공시지가/지가",
}


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for feature in FEATURES:
        s = pd.to_numeric(df[feature], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        missing_count = int(s.isna().sum())
        zero_count = int((s == 0).sum())
        rows.append(
            {
                "feature": feature,
                "group": FEATURE_GROUPS[feature],
                "count": int(s.count()),
                "missing_count": missing_count,
                "missing_rate": round(missing_count / total, 4),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "Q1": q1,
                "Q3": q3,
                "max": s.max(),
                "IQR": q3 - q1,
                "zero_count": zero_count,
                "zero_rate": round(zero_count / total, 4),
                "skewness": s.skew(),
            }
        )
    return pd.DataFrame(rows).round(4)


def top_bottom_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
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
    return pd.DataFrame(rows).round({"value": 4})


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for feature in FEATURES:
        s = pd.to_numeric(df[feature], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = s.notna() & ((s < lower) | (s > upper))
        outliers = df.loc[mask, [*ID_COLUMNS, feature]].copy()
        outlier_count = int(mask.sum())
        outlier_rate = outlier_count / total
        if outliers.empty:
            rows.append(
                {
                    "feature": feature,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "outlier_count": outlier_count,
                    "outlier_rate": outlier_rate,
                    "outlier_direction": "",
                    "상호명": "",
                    "시군구명": "",
                    "도로명주소": "",
                    "value": math.nan,
                }
            )
            continue

        for _, row in outliers.sort_values(feature, ascending=False).iterrows():
            value = row[feature]
            rows.append(
                {
                    "feature": feature,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "outlier_count": outlier_count,
                    "outlier_rate": outlier_rate,
                    "outlier_direction": "low" if value < lower else "high",
                    "상호명": row["상호명"],
                    "시군구명": row["시군구명"],
                    "도로명주소": row["도로명주소"],
                    "value": value,
                }
            )
    return pd.DataFrame(rows).round(4)


def outlier_summary(outlier_rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        outlier_rows.groupby("feature", as_index=False)
        .agg(
            lower_bound=("lower_bound", "first"),
            upper_bound=("upper_bound", "first"),
            outlier_count=("outlier_count", "first"),
            outlier_rate=("outlier_rate", "first"),
        )
        .round(4)
    )
    return summary


def choose_recommendation(
    row: pd.Series, corr_pairs: dict[str, list[str]]
) -> tuple[str, str, str]:
    feature = row["feature"]
    reasons = []
    transform_note = ""
    status = "use"

    if row["missing_rate"] > 0:
        reasons.append(f"결측률 {row['missing_rate']:.2%}")
        status = "caution"
    if abs(row["skewness"]) >= 1:
        reasons.append(f"왜도 {row['skewness']:.2f}")
        transform_note = "log1p 변환 검토"
        status = "use_with_transform" if status == "use" else status
    if row["zero_rate"] >= 0.1:
        reasons.append(f"0값 비율 {row['zero_rate']:.2%}")
        status = "caution"
    if row["IQR"] == 0:
        reasons.append("IQR이 0이라 IQR 이상치 기준 해석 주의")
        status = "caution"
    if feature in corr_pairs:
        related = ", ".join(corr_pairs[feature][:3])
        reasons.append(f"중복 가능 변수: {related}")
        if status == "use":
            status = "caution"

    if feature == "dist_nearest_subway":
        reasons.append("방향성이 명확한 접근성 변수")
        transform_note = "원본은 km 추정, 시각화만 m 변환; clustering 전 scaling/log 변환 검토"
    elif feature == "num_subway_500m":
        reasons.append("해석은 쉽지만 이산형 변수")
        if status == "use":
            status = "use_with_transform"
            transform_note = "scaling 후 사용 권장"
    elif feature == "nearest_subway_ridership":
        reasons.append("역세권 규모를 직접 반영")
        if not transform_note:
            transform_note = "scaling 후 사용 권장"
    elif feature == "subway_ridership_500m":
        reasons.append("0값은 500m 내 역 없음의 구조적 값일 수 있음")
    elif feature in {"num_retail_500m", "num_offices", "land_price"}:
        if abs(row["skewness"]) >= 1:
            transform_note = "log1p 변환 후 사용 권장"
    elif feature == "avg_income":
        if row["missing_rate"] > 0:
            status = "caution"
        transform_note = "scaling 후 사용 권장"

    if not reasons:
        reasons.append("분포와 결측 상태가 비교적 안정적")
        transform_note = "scaling 후 사용 권장"

    return status, "; ".join(reasons), transform_note


def correlation_redundancy(corr: pd.DataFrame, threshold: float = 0.8) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = {}
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.setdefault(left, []).append(f"{right} ({value:.2f})")
                pairs.setdefault(right, []).append(f"{left} ({value:.2f})")
    return pairs


def save_distribution_plots(df: pd.DataFrame, summary: pd.DataFrame) -> list[dict[str, str]]:
    plot_rows = []
    plot_df = df.copy()
    plot_df["dist_nearest_subway_m"] = plot_df["dist_nearest_subway"] * 1000

    for feature in FEATURES:
        source_feature = "dist_nearest_subway_m" if feature == "dist_nearest_subway" else feature
        label = (
            f"{FEATURE_LABELS_KO[feature]} (m, 시각화용)"
            if feature == "dist_nearest_subway"
            else FEATURE_LABELS_KO[feature]
        )
        s = pd.to_numeric(plot_df[source_feature], errors="coerce").dropna()
        slug = slugify(feature)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(s, bins=30, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(f"{label} histogram")
        ax.set_xlabel(label)
        ax.set_ylabel("매장 수")
        fig.tight_layout()
        hist_path = FIGURE_DIR / f"{slug}_hist.png"
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.boxplot(x=s, ax=ax, color="#72B7B2")
        ax.set_title(f"{label} boxplot")
        ax.set_xlabel(label)
        fig.tight_layout()
        box_path = FIGURE_DIR / f"{slug}_boxplot.png"
        fig.savefig(box_path, dpi=160)
        plt.close(fig)

        log_path = ""
        skewness = float(summary.loc[summary["feature"] == feature, "skewness"].iloc[0])
        if abs(skewness) >= 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(s.map(lambda x: math.log1p(x)), bins=30, kde=True, ax=ax, color="#F58518")
            ax.set_title(f"{label} log1p histogram")
            ax.set_xlabel(f"log1p({label})")
            ax.set_ylabel("매장 수")
            fig.tight_layout()
            log_path_obj = FIGURE_DIR / f"{slug}_log1p_hist.png"
            fig.savefig(log_path_obj, dpi=160)
            plt.close(fig)
            log_path = relative_posix(log_path_obj)

        plot_rows.append(
            {
                "feature": feature,
                "histogram": relative_posix(hist_path),
                "boxplot": relative_posix(box_path),
                "log1p_histogram": log_path,
            }
        )

    return plot_rows


def save_heatmap(corr: pd.DataFrame, filename: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 9))
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
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_recommendations(summary: pd.DataFrame, spearman_corr: pd.DataFrame) -> pd.DataFrame:
    corr_pairs = correlation_redundancy(spearman_corr)
    rows = []
    for _, row in summary.iterrows():
        status, rationale, transform_note = choose_recommendation(row, corr_pairs)
        rows.append(
            {
                "feature": row["feature"],
                "group": row["group"],
                "recommendation": status,
                "rationale": rationale,
                "transform_note": transform_note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs(GENERATED_REPORT_DIR, TABLE_DIR, FIGURE_DIR)
    setup_plot_style()

    df = read_csv(DATA_PATH)
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    missing_id_cols = [col for col in ID_COLUMNS if col not in df.columns]
    if missing_features or missing_id_cols:
        raise ValueError(f"Missing columns: {missing_features + missing_id_cols}")

    df_starbucks = df[df["is_starbucks"] == 1].copy()
    feature_df = df_starbucks[FEATURES].apply(pd.to_numeric, errors="coerce")

    summary = feature_summary(df_starbucks)
    top_bottom = top_bottom_by_feature(df_starbucks)
    outliers = iqr_outliers(df_starbucks)
    outlier_stats = outlier_summary(outliers)
    pearson_corr = feature_df.corr(method="pearson")
    spearman_corr = feature_df.corr(method="spearman")
    recommendations = make_recommendations(summary, spearman_corr)

    summary.to_csv(
        TABLE_DIR / "starbucks_only_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outliers.to_csv(
        TABLE_DIR / "starbucks_only_outliers_iqr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_bottom.to_csv(
        TABLE_DIR / "starbucks_only_top_bottom_by_feature.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pearson_corr.round(4).to_csv(
        TABLE_DIR / "starbucks_only_corr_pearson.csv",
        encoding="utf-8-sig",
    )
    spearman_corr.round(4).to_csv(
        TABLE_DIR / "starbucks_only_corr_spearman.csv",
        encoding="utf-8-sig",
    )
    recommendations.to_csv(
        TABLE_DIR / "starbucks_only_feature_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_rows = save_distribution_plots(df_starbucks, summary)
    pearson_heatmap = save_heatmap(
        pearson_corr, "correlation_heatmap_pearson.png", "Pearson correlation heatmap"
    )
    spearman_heatmap = save_heatmap(
        spearman_corr, "correlation_heatmap_spearman.png", "Spearman correlation heatmap"
    )

    high_corr_pairs = []
    for i, left in enumerate(spearman_corr.columns):
        for right in spearman_corr.columns[i + 1 :]:
            value = spearman_corr.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.7:
                high_corr_pairs.append(
                    {"feature_1": left, "feature_2": right, "spearman_corr": round(value, 4)}
                )
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        "spearman_corr", key=lambda s: s.abs(), ascending=False
    )

    feature_sets = pd.DataFrame(
        [
            {
                "feature_set": "A. 교통 접근성 중심",
                "features": "dist_nearest_subway, num_subway_500m, nearest_subway_ridership, subway_ridership_500m",
                "note": "역 접근성과 역세권 유동량 차이를 중심으로 스타벅스 입지를 나누는 최소 세트",
            },
            {
                "feature_set": "B. 교통 + 상권 밀도",
                "features": "dist_nearest_subway, nearest_subway_ridership, subway_ridership_500m, num_competing_cafes_500m, num_restaurants_500m, num_retail_500m, num_convenience_500m",
                "note": "유동량과 주변 업종 밀도를 함께 반영하되, 상관 높은 변수는 다음 단계에서 축소 검토",
            },
            {
                "feature_set": "C. 균형형",
                "features": "dist_nearest_subway, nearest_subway_ridership, num_competing_cafes_500m, num_restaurants_500m, avg_income, num_offices, living_population, land_price",
                "note": "교통, 상권, 경제/인구를 균형 있게 포함하는 중간발표용 후보 세트",
            },
        ]
    )

    lines = [
        "# 02 Starbucks-only EDA",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 입력 파일: `{DATA_PATH.name}`",
        "- 분석 대상: `df_starbucks = df[df[\"is_starbucks\"] == 1]`",
        f"- 스타벅스 매장 수: {len(df_starbucks):,}",
        "- 분석 범위: 스타벅스 내부 입지 차이 파악 및 clustering용 기존 변수 후보 정리",
        "- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, clustering 실행 없음, 비스타벅스 비교 없음",
        "",
        "## 1. 분석 대상 변수",
        "",
        markdown_table(
            pd.DataFrame(
                {
                    "feature": FEATURES,
                    "group": [FEATURE_GROUPS[f] for f in FEATURES],
                    "label": [FEATURE_LABELS_KO[f] for f in FEATURES],
                }
            )
        ),
        "",
        "## 2. 변수별 기본 통계",
        "",
        "아래 통계는 스타벅스 681개 매장만 대상으로 계산했다.",
        "",
        markdown_table(summary),
        "",
        "## 3. 거리 변수 단위 기록",
        "",
        "`dist_nearest_subway`는 1단계 audit에서 값 범위가 소수점 km 형태로 보여 km 단위일 가능성이 높다고 추정했다. "
        "이번 EDA에서는 원본 컬럼을 덮어쓰지 않고, 그래프 표시용으로만 `dist_nearest_subway_m = dist_nearest_subway * 1000`을 만들어 m 단위로 표시했다. "
        "단위는 여전히 추정이며 원천 데이터 정의 확인 전까지 확정하지 않는다.",
        "",
        "## 4. 분포 그래프",
        "",
        "각 변수별 histogram과 boxplot을 저장했다. 왜도 절댓값이 1 이상인 변수는 log1p 변환 histogram도 추가했다.",
        "",
        markdown_table(pd.DataFrame(plot_rows)),
        "",
        "## 5. IQR 기준 이상치 요약",
        "",
        "이상치는 제거하지 않고, 해석 가능한 매장인지 확인하기 위한 점검 대상으로만 기록했다. 전체 매장 리스트는 `reports/generated/tables/starbucks_only_outliers_iqr.csv`에 저장했다.",
        "`num_subway_500m`는 Q1과 Q3가 모두 1이라 IQR이 0이다. 따라서 IQR 기준에서는 값이 1이 아닌 매장이 모두 이상치로 잡히며, 이 변수는 연속형 변수의 이상치 판단처럼 해석하면 안 된다.",
        "",
        markdown_table(outlier_stats),
        "",
        "## 6. 상위/하위 10개 매장",
        "",
        "각 변수별 상위 10개와 하위 10개 매장은 `reports/generated/tables/starbucks_only_top_bottom_by_feature.csv`에 저장했다. "
        "보고서에는 일부 예시만 표시한다.",
        "",
        markdown_table(top_bottom.head(24)),
        "",
        "## 7. 변수 간 상관관계",
        "",
        f"- Pearson correlation table: `reports/generated/tables/starbucks_only_corr_pearson.csv`",
        f"- Spearman correlation table: `reports/generated/tables/starbucks_only_corr_spearman.csv`",
        f"- Pearson heatmap: `{relative_posix(pearson_heatmap)}`",
        f"- Spearman heatmap: `{relative_posix(spearman_heatmap)}`",
        "",
        "Spearman 기준 절댓값 0.7 이상인 변수쌍은 다음과 같다.",
        "",
        markdown_table(high_corr_df) if not high_corr_df.empty else "_절댓값 0.7 이상인 Spearman 상관 변수쌍 없음._",
        "",
        "### 변수군별 중복성 메모",
        "",
        "- 지하철 접근성: `dist_nearest_subway`, `num_subway_500m`, `nearest_subway_ridership`, `subway_ridership_500m`는 접근성, 역 수, 승하차 규모를 서로 다른 관점에서 담는다. `subway_ridership_500m`는 500m 내 역이 없는 경우 0이 될 수 있어 구조적 0값을 주의해야 한다.",
        "- 상권 밀도: `num_competing_cafes_500m`, `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m`는 같은 반경 내 업종 밀도를 재는 변수라 상호 중복 가능성이 있다. 다음 단계에서는 상관이 높은 변수 조합을 줄이거나 표준화 후 사용해야 한다.",
        "- 인구/경제: `avg_income`, `num_offices`, `living_population`, `land_price`는 결측과 왜도가 함께 존재한다. 특히 `num_offices`, `land_price`는 큰 값 쪽 꼬리가 길어 log1p 변환 후보로 보는 것이 좋다.",
        "",
        "## 8. Clustering 관점 변수 평가",
        "",
        markdown_table(recommendations),
        "",
        "## 9. Clustering용 기존 변수 후보 feature set",
        "",
        markdown_table(feature_sets),
        "",
        "이번 단계에서는 위 feature set을 제안만 하며 clustering은 실행하지 않았다.",
        "",
        "## 10. 다음 단계에서 추가해야 할 변수",
        "",
        "- 버스 접근성 변수",
        "- 후보 반경별 카페 수 변수",
        "- 시간대별 지하철 유동량 변수",
        "- 대학 접근성 변수",
        "- 20-30대 인구 비율",
        "- 직장인/상주인구 대비 생활인구 비율",
        "- 관광/상업 핵심지 여부를 나타내는 공간 태그",
        "",
        "## 11. 저장 산출물",
        "",
        "- `reports/02_starbucks_only_eda.md`",
        "- `reports/generated/tables/starbucks_only_feature_summary.csv`",
        "- `reports/generated/tables/starbucks_only_outliers_iqr.csv`",
        "- `reports/generated/tables/starbucks_only_top_bottom_by_feature.csv`",
        "- `reports/generated/tables/starbucks_only_corr_pearson.csv`",
        "- `reports/generated/tables/starbucks_only_corr_spearman.csv`",
        "- `reports/generated/tables/starbucks_only_feature_recommendation.csv`",
        "- `reports/generated/figures/starbucks_only/`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Loaded Starbucks rows: {len(df_starbucks)}")
    print(f"Report written: {REPORT_PATH}")
    print(f"Tables written: {TABLE_DIR}")
    print(f"Figures written: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
