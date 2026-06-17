# -*- coding: utf-8 -*-
"""
Project newly opened Starbucks stores onto the final district classification.

Input:
  - ROOT/신규_스타벅스.CSV

Method:
  - Rebuild the final DBSCAN auto-eps district assignment used by
    19_district_final.py and 20_district_persona_bridge.py.
  - Assign each new store to the district of its nearest existing cafe.
  - Attach district classification score and C0-C4 persona composition.

Outputs:
  - outputs/new_starbucks_district_projection.csv
  - figures/F22_new_starbucks_district_persona_fingerprint.png
  - archive table/figure/log copies
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    DISTRICT_CLASSIFICATION_FIGURE_DIR,
    DISTRICT_CLASSIFICATION_LOG_DIR,
    DISTRICT_CLASSIFICATION_OUTPUT_DIR,
    ROOT,
)


EARTH_RADIUS_M = 6_371_000.0
SHARE_COLUMNS = [f"share_C{k}" for k in range(5)]

ARCHIVE_TABLE_DIR = ROOT / "reports" / "archive" / "tables" / "classification" / "02_district_level"
ARCHIVE_FIGURE_DIR = ROOT / "reports" / "archive" / "figures" / "classification" / "02_district_level"
ARCHIVE_LOG_DIR = ROOT / "reports" / "archive" / "classification" / "logs" / "02_district_level"

NEW_STARBUCKS_CANDIDATES = [
    ROOT / "신규_스타벅스.CSV",
    ROOT / "신규_스타벅스.csv",
]

PERSONA_COLORS = {
    0: "#7dd3fc",
    1: "#34d399",
    2: "#fbbf24",
    3: "#f472b6",
    4: "#a78bfa",
}
PERSONA_LABELS = {
    0: "C0 오피스고소득",
    1: "C1 상업활성",
    2: "C2 주거생활",
    3: "C3 도심초밀집",
    4: "C4 비역세권",
}


def ensure_dirs() -> None:
    for path in [
        DISTRICT_CLASSIFICATION_OUTPUT_DIR,
        DISTRICT_CLASSIFICATION_FIGURE_DIR,
        DISTRICT_CLASSIFICATION_LOG_DIR,
        ARCHIVE_TABLE_DIR,
        ARCHIVE_FIGURE_DIR,
        ARCHIVE_LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    for font in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#0b1020"
    plt.rcParams["axes.facecolor"] = "#0b1020"


def find_new_starbucks_path() -> Path:
    for path in NEW_STARBUCKS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing 신규_스타벅스 CSV. Tried: "
        + ", ".join(str(path) for path in NEW_STARBUCKS_CANDIDATES)
    )


def read_new_starbucks(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            raw = pd.read_csv(path, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not read {path}: {last_error}") from last_error

    raw = raw.dropna(axis=1, how="all")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    required = {"상호명", "위도", "경도", "시군구명"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    stores = raw.copy()
    stores["위도"] = pd.to_numeric(stores["위도"], errors="coerce")
    stores["경도"] = pd.to_numeric(stores["경도"], errors="coerce")
    stores = stores.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    if stores.empty:
        raise ValueError(f"{path.name} has no valid lat/lon rows")
    return stores


def auto_eps(xy_rad: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(xy_rad)
    distances, _ = nn.kneighbors(xy_rad)
    kth = np.sort(distances[:, k - 1])[::-1]
    second_diff = np.diff(np.diff(kth))
    lo, hi = int(len(kth) * 0.05), int(len(kth) * 0.6)
    return float(kth[lo + int(np.argmax(second_diff[lo:hi])) + 1])


def assign_existing_cafe_districts(cafes: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    xy_rad = np.radians(cafes[["lat", "lon"]].to_numpy(float))
    eps = auto_eps(xy_rad)
    labels = DBSCAN(eps=eps, min_samples=5, metric="haversine").fit_predict(xy_rad)

    assigned = labels.copy()
    non_noise = labels != -1
    if non_noise.any() and (~non_noise).any():
        nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(xy_rad[non_noise])
        _, nearest = nn.kneighbors(xy_rad[~non_noise])
        assigned[~non_noise] = labels[non_noise][nearest[:, 0]]

    result = cafes.copy()
    result["district"] = assigned
    return result, eps * EARTH_RADIUS_M


def assign_new_to_districts(new_stores: pd.DataFrame, cafes: pd.DataFrame) -> pd.DataFrame:
    cafe_xy = np.radians(cafes[["lat", "lon"]].to_numpy(float))
    new_xy = np.radians(new_stores[["위도", "경도"]].to_numpy(float))
    nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(cafe_xy)
    distance_rad, nearest_idx = nn.kneighbors(new_xy)
    nearest = cafes.iloc[nearest_idx[:, 0]].reset_index(drop=True)

    assigned = new_stores.copy()
    assigned["assigned_district"] = nearest["district"].astype(int).to_numpy()
    assigned["nearest_cafe_name"] = nearest["name"].to_numpy()
    assigned["nearest_cafe_sigungu"] = nearest["sigungu"].to_numpy()
    assigned["nearest_cafe_distance_m"] = distance_rad[:, 0] * EARTH_RADIUS_M
    return assigned


def attach_district_outputs(assigned: pd.DataFrame) -> pd.DataFrame:
    bridge_path = DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_persona_bridge.csv"
    if not bridge_path.exists():
        raise FileNotFoundError(
            f"Missing {bridge_path}. "
            "Run scripts/06_classification/02_district_level/20_district_persona_bridge.py first."
        )
    bridge = pd.read_csv(bridge_path, encoding="utf-8-sig")
    needed = {
        "district",
        "dominant_sigungu",
        "n_cafe",
        "n_starbucks",
        "score",
        "score_pct",
        "dominant_persona_name",
        "sb_core_share",
        "weak_access_share",
        *SHARE_COLUMNS,
    }
    missing = needed.difference(bridge.columns)
    if missing:
        raise ValueError(f"{bridge_path.name} missing columns: {sorted(missing)}")

    output = assigned.merge(
        bridge[list(needed)],
        left_on="assigned_district",
        right_on="district",
        how="left",
    )
    if output["score"].isna().any():
        missing_districts = output.loc[output["score"].isna(), "assigned_district"].tolist()
        raise ValueError(f"Failed to attach district outputs for districts: {missing_districts}")

    output = output.rename(
        columns={
            "상호명": "new_store",
            "브랜드": "brand",
            "위도": "lat",
            "경도": "lon",
            "시군구명": "source_sigungu",
            "도로명주소": "address",
            "dominant_sigungu": "district_sigungu",
            "n_cafe": "district_n_cafe",
            "n_starbucks": "district_existing_starbucks",
            "score": "district_score",
            "score_pct": "district_score_pct",
        }
    )
    output["rank_by_district_score"] = output["district_score_pct"].rank(
        method="first", ascending=False
    ).astype(int)
    ordered_columns = [
        "rank_by_district_score",
        "new_store",
        "brand",
        "lat",
        "lon",
        "source_sigungu",
        "address",
        "assigned_district",
        "district_sigungu",
        "nearest_cafe_name",
        "nearest_cafe_sigungu",
        "nearest_cafe_distance_m",
        "district_n_cafe",
        "district_existing_starbucks",
        "district_score",
        "district_score_pct",
        "dominant_persona_name",
        "sb_core_share",
        "weak_access_share",
        *SHARE_COLUMNS,
    ]
    return output[ordered_columns].sort_values(
        ["district_score_pct", "district_score"], ascending=False
    ).reset_index(drop=True)


def write_table(projected: pd.DataFrame) -> None:
    generated = DISTRICT_CLASSIFICATION_OUTPUT_DIR / "new_starbucks_district_projection.csv"
    archive = ARCHIVE_TABLE_DIR / "new_starbucks_district_projection.csv"
    projected.to_csv(generated, index=False, encoding="utf-8-sig")
    projected.to_csv(archive, index=False, encoding="utf-8-sig")
    print(f"[table] {generated}")


def make_figure(projected: pd.DataFrame) -> None:
    setup_plot_style()
    data = projected.sort_values("district_score_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    fig, (ax_bar, ax_dot) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.8),
        width_ratios=[4.4, 1.1],
        facecolor="#0b1020",
        sharey=True,
    )

    left = np.zeros(len(data))
    for k in range(5):
        values = data[f"share_C{k}"].to_numpy(float)
        ax_bar.barh(y, values, left=left, color=PERSONA_COLORS[k], label=PERSONA_LABELS[k], height=0.66)
        left += values

    labels = [
        f"{row.new_store}({row.source_sigungu}) | D{int(row.assigned_district)}/{row.district_sigungu} | {row.nearest_cafe_distance_m:.0f}m"
        for row in data.itertuples(index=False)
    ]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, color="#e5e7eb", fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Assigned district persona share", color="#cbd5e1")
    ax_bar.set_title("New Starbucks Assigned District Persona", color="#f8fafc", fontsize=16, weight="bold")
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.grid(axis="x", color="#1e293b", alpha=0.7)
    ax_bar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, labelcolor="#e5e7eb")

    ax_dot.scatter(
        data["district_score_pct"],
        y,
        c="#38bdf8",
        s=120,
        edgecolors="#ffffff",
        linewidth=0.8,
    )
    ax_dot.set_xlim(0, 101)
    ax_dot.set_xlabel("District score pct", color="#cbd5e1")
    ax_dot.set_title("Classification", color="#f8fafc", fontsize=12, weight="bold")
    ax_dot.tick_params(colors="#94a3b8")
    ax_dot.grid(axis="x", color="#1e293b", alpha=0.7)
    for idx, row in data.iterrows():
        ax_dot.text(
            row["district_score_pct"] + 1.2,
            idx,
            f"{row['district_score_pct']:.0f}",
            color="#e5e7eb",
            fontsize=8,
            va="center",
        )
    for spine in [*ax_bar.spines.values(), *ax_dot.spines.values()]:
        spine.set_color("#334155")

    generated = DISTRICT_CLASSIFICATION_FIGURE_DIR / "F22_new_starbucks_district_persona_fingerprint.png"
    archive = ARCHIVE_FIGURE_DIR / "F22_new_starbucks_district_persona_fingerprint.png"
    fig.savefig(generated, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    shutil.copy2(generated, archive)
    print(f"[figure] {generated}")


def markdown_table(projected: pd.DataFrame) -> str:
    columns = [
        "new_store",
        "source_sigungu",
        "assigned_district",
        "district_sigungu",
        "nearest_cafe_distance_m",
        "district_score",
        "district_score_pct",
        "dominant_persona_name",
        "sb_core_share",
        "weak_access_share",
    ]
    work = projected[columns].copy()
    work["nearest_cafe_distance_m"] = work["nearest_cafe_distance_m"].map(lambda value: f"{value:.0f}")
    for column in ["district_score", "district_score_pct", "sb_core_share", "weak_access_share"]:
        work[column] = work[column].map(lambda value: f"{value:.3f}")
    headers = list(work.columns)
    rows = work.astype(str).to_numpy().tolist()
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_log(projected: pd.DataFrame, source_path: Path, eps_m: float) -> None:
    lines = [
        "# 신규 스타벅스 상권 배정",
        "",
        f"> 생성 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 입력: `{source_path.name}`",
        f"- 기존 상권 정의: DBSCAN auto-eps={eps_m:.0f}m + 노이즈 최근접 흡수",
        "- 신규 매장은 직접 featurize하지 않고, 가장 가까운 기존 카페의 상권에 배정했다.",
        "- 적합도와 C0-C4 분포는 배정된 기존 상권의 값을 사용한다.",
        "",
        "## 신규 매장별 배정 상권",
        "",
        markdown_table(projected),
        "",
    ]
    generated = DISTRICT_CLASSIFICATION_LOG_DIR / "NEW_STARBUCKS_DISTRICT_REPORT.md"
    archive = ARCHIVE_LOG_DIR / "NEW_STARBUCKS_DISTRICT_REPORT.md"
    text = "\n".join(lines)
    generated.write_text(text, encoding="utf-8")
    archive.write_text(text, encoding="utf-8")
    print(f"[log] {generated}")


def validate_outputs() -> None:
    required = [
        DISTRICT_CLASSIFICATION_OUTPUT_DIR / "new_starbucks_district_projection.csv",
        DISTRICT_CLASSIFICATION_FIGURE_DIR / "F22_new_starbucks_district_persona_fingerprint.png",
        DISTRICT_CLASSIFICATION_LOG_DIR / "NEW_STARBUCKS_DISTRICT_REPORT.md",
        ARCHIVE_TABLE_DIR / "new_starbucks_district_projection.csv",
        ARCHIVE_FIGURE_DIR / "F22_new_starbucks_district_persona_fingerprint.png",
        ARCHIVE_LOG_DIR / "NEW_STARBUCKS_DISTRICT_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Missing expected outputs: {missing}")


def main() -> None:
    ensure_dirs()
    source_path = find_new_starbucks_path()
    new_stores = read_new_starbucks(source_path)
    cafes = pd.read_parquet(CLASSIFICATION_DATASET_PATH).reset_index(drop=True)
    cafes, eps_m = assign_existing_cafe_districts(cafes)
    assigned = assign_new_to_districts(new_stores, cafes)
    projected = attach_district_outputs(assigned)

    write_table(projected)
    make_figure(projected)
    write_log(projected, source_path, eps_m)
    validate_outputs()

    print("\n[신규 스타벅스 상권 배정]")
    print(
        projected[
            [
                "new_store",
                "source_sigungu",
                "assigned_district",
                "district_sigungu",
                "nearest_cafe_distance_m",
                "district_score",
                "district_score_pct",
                "dominant_persona_name",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()


