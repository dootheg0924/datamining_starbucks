from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures" / "geo_features"

MASTER_PATH = ROOT / "seoul_cafe_master.csv"
BUS_PATH = ROOT / "서울시_버스정류소_위치정보.csv"
OUTPUT_MASTER_PATH = DATA_DIR / "seoul_cafe_master_with_geo_features.csv"
REPORT_PATH = REPORT_DIR / "04_geo_feature_engineering.md"

EARTH_RADIUS_KM = 6371.0088
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
BUS_COUNT_FEATURES = [
    "num_bus_stops_100m",
    "num_bus_stops_300m",
    "num_bus_stops_500m",
]
CAFE_COUNT_FEATURES = [
    "cafe_count_300m",
    "cafe_count_500m",
    "cafe_count_1000m",
]
ID_COLUMNS = ["상호명", "브랜드", "is_starbucks", "시군구명", "도로명주소"]


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    for encoding in ["utf-8-sig", "euc-kr", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path), "default"


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def latlon_to_unit_xyz(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray) -> np.ndarray:
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def km_to_chord_radius(radius_km: float) -> float:
    return 2.0 * np.sin((radius_km / EARTH_RADIUS_KM) / 2.0)


def chord_to_km(chord_distance: np.ndarray) -> np.ndarray:
    clipped = np.clip(chord_distance / 2.0, 0.0, 1.0)
    return 2.0 * np.arcsin(clipped) * EARTH_RADIUS_KM


def radius_counts(tree: cKDTree, query_xyz: np.ndarray, radius_km: float) -> np.ndarray:
    radius = km_to_chord_radius(radius_km)
    try:
        return tree.query_ball_point(query_xyz, r=radius, return_length=True)
    except TypeError:
        return np.array([len(indices) for indices in tree.query_ball_point(query_xyz, r=radius)])


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
                "missing_rate": round(missing_count / total, 6),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "Q1": q1,
                "Q3": q3,
                "max": s.max(),
                "zero_count": zero_count,
                "zero_rate": round(zero_count / total, 6),
                "skewness": s.skew(),
            }
        )
    return pd.DataFrame(rows).round(6)


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


def save_histograms(df: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for feature in [*BUS_COUNT_FEATURES, *CAFE_COUNT_FEATURES, "dist_nearest_starbucks"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], bins=30, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(f"{feature} histogram")
        ax.set_xlabel(f"{feature} (km)" if feature.startswith("dist_") else feature)
        ax.set_ylabel("count")
        fig.tight_layout()
        path = FIGURE_DIR / f"{feature}_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rows.append({"feature": feature, "figure": str(path.relative_to(ROOT)).replace("\\", "/")})
    return rows


def save_correlation_heatmap(corr: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
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
    ax.set_title("New geo feature correlation")
    fig.tight_layout()
    path = FIGURE_DIR / "new_geo_feature_correlation_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    df, master_encoding = read_csv_with_fallback(MASTER_PATH)
    bus_df, bus_encoding = read_csv_with_fallback(BUS_PATH)

    required_master_cols = {"상호명", "브랜드", "is_starbucks", "위도", "경도", "시군구명", "도로명주소"}
    required_bus_cols = {"정류소명", "X좌표", "Y좌표"}
    missing_master = required_master_cols - set(df.columns)
    missing_bus = required_bus_cols - set(bus_df.columns)
    if missing_master or missing_bus:
        raise ValueError(f"Missing columns: master={missing_master}, bus={missing_bus}")

    cafe_xyz = latlon_to_unit_xyz(df["위도"], df["경도"])
    bus_clean = bus_df.dropna(subset=["X좌표", "Y좌표"]).copy()
    bus_xyz = latlon_to_unit_xyz(bus_clean["Y좌표"], bus_clean["X좌표"])
    bus_tree = cKDTree(bus_xyz)

    bus_chord_dist, _ = bus_tree.query(cafe_xyz, k=1)
    df["dist_nearest_bus_stop"] = chord_to_km(bus_chord_dist)
    df["num_bus_stops_100m"] = radius_counts(bus_tree, cafe_xyz, 0.1)
    df["num_bus_stops_300m"] = radius_counts(bus_tree, cafe_xyz, 0.3)
    df["num_bus_stops_500m"] = radius_counts(bus_tree, cafe_xyz, 0.5)

    cafe_tree = cKDTree(cafe_xyz)
    df["cafe_count_300m"] = radius_counts(cafe_tree, cafe_xyz, 0.3) - 1
    df["cafe_count_500m"] = radius_counts(cafe_tree, cafe_xyz, 0.5) - 1
    df["cafe_count_1000m"] = radius_counts(cafe_tree, cafe_xyz, 1.0) - 1

    starbucks_mask = df["is_starbucks"] == 1
    starbucks_df = df.loc[starbucks_mask].copy()
    starbucks_xyz = latlon_to_unit_xyz(starbucks_df["위도"], starbucks_df["경도"])
    starbucks_tree = cKDTree(starbucks_xyz)
    k = min(2, len(starbucks_df))
    sb_chord_dist, sb_neighbor_pos = starbucks_tree.query(cafe_xyz, k=k)
    if k == 1:
        df["dist_nearest_starbucks"] = chord_to_km(np.asarray(sb_chord_dist))
    else:
        starbucks_source_indices = starbucks_df.index.to_numpy()
        nearest_dist_km = np.empty(len(df), dtype=float)
        for row_position, row_index in enumerate(df.index):
            neighbor_positions = np.atleast_1d(sb_neighbor_pos[row_position])
            distances = np.atleast_1d(sb_chord_dist[row_position])
            chosen_distance = distances[0]
            if df.at[row_index, "is_starbucks"] == 1:
                for distance, neighbor_position in zip(distances, neighbor_positions):
                    if starbucks_source_indices[neighbor_position] != row_index:
                        chosen_distance = distance
                        break
            nearest_dist_km[row_position] = chord_to_km(np.array([chosen_distance]))[0]
        df["dist_nearest_starbucks"] = nearest_dist_km

    df[NEW_FEATURES] = df[NEW_FEATURES].round(6)
    df.to_csv(OUTPUT_MASTER_PATH, index=False, encoding="utf-8-sig")

    df_starbucks = df[df["is_starbucks"] == 1].copy()
    summary_all = feature_summary(df, NEW_FEATURES)
    summary_starbucks = feature_summary(df_starbucks, NEW_FEATURES)

    summary_all.to_csv(
        TABLE_DIR / "new_geo_feature_summary_all.csv", index=False, encoding="utf-8-sig"
    )
    summary_starbucks.to_csv(
        TABLE_DIR / "new_geo_feature_summary_starbucks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr_features = [
        *NEW_FEATURES,
        "num_competing_cafes_500m",
    ]
    corr = df[corr_features].corr(method="pearson").round(6)
    corr.to_csv(TABLE_DIR / "new_geo_feature_correlations.csv", encoding="utf-8-sig")

    top_bottom_rows = []
    for rank_type, selected in [
        ("top", df_starbucks.sort_values("dist_nearest_starbucks", ascending=False).head(10)),
        ("bottom", df_starbucks.sort_values("dist_nearest_starbucks", ascending=True).head(10)),
    ]:
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            top_bottom_rows.append(
                {
                    "feature": "dist_nearest_starbucks",
                    "rank_type": rank_type,
                    "rank": rank,
                    "상호명": row["상호명"],
                    "시군구명": row["시군구명"],
                    "도로명주소": row["도로명주소"],
                    "value_km": row["dist_nearest_starbucks"],
                }
            )
    top_bottom = pd.DataFrame(top_bottom_rows)
    top_bottom.to_csv(
        TABLE_DIR / "starbucks_new_geo_top_bottom.csv", index=False, encoding="utf-8-sig"
    )

    figures = save_histograms(df_starbucks)
    heatmap_path = save_correlation_heatmap(df_starbucks[NEW_FEATURES].corr(method="pearson"))

    missing_new = (
        df[NEW_FEATURES]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_count"})
    )
    missing_new["missing_rate"] = missing_new["missing_count"] / len(df)
    missing_new = missing_new.round(6)

    bus_zero_rates = summary_starbucks[summary_starbucks["feature"].isin(BUS_COUNT_FEATURES)][
        ["feature", "zero_count", "zero_rate"]
    ]
    cafe_distribution = summary_starbucks[summary_starbucks["feature"].isin(CAFE_COUNT_FEATURES)][
        ["feature", "mean", "median", "skewness"]
    ]
    competing_corr = pd.DataFrame(
        [
            {
                "comparison": "num_competing_cafes_500m vs cafe_count_500m",
                "pearson_corr_all": df["num_competing_cafes_500m"].corr(df["cafe_count_500m"], method="pearson"),
                "spearman_corr_all": df["num_competing_cafes_500m"].corr(df["cafe_count_500m"], method="spearman"),
                "pearson_corr_starbucks": df_starbucks["num_competing_cafes_500m"].corr(
                    df_starbucks["cafe_count_500m"], method="pearson"
                ),
                "spearman_corr_starbucks": df_starbucks["num_competing_cafes_500m"].corr(
                    df_starbucks["cafe_count_500m"], method="spearman"
                ),
                "mean_difference_all_new_minus_existing": (
                    df["cafe_count_500m"] - df["num_competing_cafes_500m"]
                ).mean(),
                "mean_difference_starbucks_new_minus_existing": (
                    df_starbucks["cafe_count_500m"] - df_starbucks["num_competing_cafes_500m"]
                ).mean(),
            }
        ]
    ).round(6)

    zero_distance_starbucks = df_starbucks[df_starbucks["dist_nearest_starbucks"] == 0][
        [*ID_COLUMNS, "위도", "경도", "dist_nearest_starbucks"]
    ]

    selected_corr = corr.loc[
        [*BUS_COUNT_FEATURES, *CAFE_COUNT_FEATURES],
        [*BUS_COUNT_FEATURES, *CAFE_COUNT_FEATURES],
    ]

    lines = [
        "# 04 Geo Feature Engineering",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 입력 master: `{MASTER_PATH.name}` ({master_encoding})",
        f"- 입력 버스정류장: `{BUS_PATH.name}` ({bus_encoding})",
        f"- 출력 master: `data/seoul_cafe_master_with_geo_features.csv`",
        "- 처리 범위: `seoul_cafe_master.csv` 전체 행에 좌표 기반 변수 추가",
        "- 거리 계산: Haversine great-circle distance. 거리 단위는 km",
        "- 반경 기준: 100m=0.1km, 300m=0.3km, 500m=0.5km, 1000m=1.0km",
        "- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, clustering 없음, 지하철 peak/premium/low price 변수 생성 없음",
        "",
        "## 1. 생성 변수",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {"feature": "dist_nearest_bus_stop", "unit": "km", "definition": "가장 가까운 버스정류장까지 Haversine 거리"},
                    {"feature": "num_bus_stops_100m", "unit": "count", "definition": "0.1km 반경 내 버스정류장 수"},
                    {"feature": "num_bus_stops_300m", "unit": "count", "definition": "0.3km 반경 내 버스정류장 수"},
                    {"feature": "num_bus_stops_500m", "unit": "count", "definition": "0.5km 반경 내 버스정류장 수"},
                    {"feature": "cafe_count_300m", "unit": "count", "definition": "0.3km 반경 내 전체 카페 수. 자기 자신 제외"},
                    {"feature": "cafe_count_500m", "unit": "count", "definition": "0.5km 반경 내 전체 카페 수. 자기 자신 제외"},
                    {"feature": "cafe_count_1000m", "unit": "count", "definition": "1.0km 반경 내 전체 카페 수. 자기 자신 제외"},
                    {"feature": "dist_nearest_starbucks", "unit": "km", "definition": "가장 가까운 스타벅스까지 거리. 스타벅스 행은 자기 자신 제외"},
                ]
            )
        ),
        "",
        "## 2. 새 변수 결측치 확인",
        "",
        markdown_table(missing_new),
        "",
        "## 3. 기본 통계: 전체 데이터",
        "",
        markdown_table(summary_all),
        "",
        "## 4. 기본 통계: 스타벅스 681개",
        "",
        markdown_table(summary_starbucks),
        "",
        "## 5. 스타벅스 기준 검증",
        "",
        "### 버스정류장 수 zero rate",
        "",
        markdown_table(bus_zero_rates),
        "",
        "### 반경별 카페 수 분포",
        "",
        markdown_table(cafe_distribution),
        "",
        "### dist_nearest_starbucks 상위/하위 10개",
        "",
        markdown_table(top_bottom),
        "",
        "### 스타벅스 행 중 dist_nearest_starbucks = 0",
        "",
        markdown_table(zero_distance_starbucks),
        "",
        "## 6. 기존 num_competing_cafes_500m와 새 cafe_count_500m 비교",
        "",
        "`cafe_count_500m`는 현재 master 파일의 전체 카페 좌표를 사용해 0.5km 반경 내 카페 수를 다시 계산하고 자기 자신만 제외한 변수다. "
        "`num_competing_cafes_500m`는 기존 master에 이미 있던 변수로, 생성 원천과 필터 정의가 다를 수 있어 값이 완전히 같다고 가정하지 않았다.",
        "",
        markdown_table(competing_corr),
        "",
        "## 7. 새 변수 상관관계",
        "",
        f"- 전체 correlation table: `reports/tables/new_geo_feature_correlations.csv`",
        f"- heatmap: `{str(heatmap_path.relative_to(ROOT)).replace(chr(92), '/')}`",
        "",
        "아래는 버스정류장 count 변수와 카페 count 변수만 발췌한 Pearson correlation이다.",
        "",
        markdown_table(selected_corr.reset_index().rename(columns={"index": "feature"})),
        "",
        "## 8. 저장된 시각화",
        "",
        markdown_table(pd.DataFrame(figures)),
        "",
        "## 9. 저장 산출물",
        "",
        "- `data/seoul_cafe_master_with_geo_features.csv`",
        "- `reports/04_geo_feature_engineering.md`",
        "- `reports/tables/new_geo_feature_summary_all.csv`",
        "- `reports/tables/new_geo_feature_summary_starbucks.csv`",
        "- `reports/tables/new_geo_feature_correlations.csv`",
        "- `reports/tables/starbucks_new_geo_top_bottom.csv`",
        "- `reports/figures/geo_features/`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Rows processed: {len(df)}")
    print(f"Starbucks rows: {len(df_starbucks)}")
    print(f"New master written: {OUTPUT_MASTER_PATH}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
