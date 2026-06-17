# -*- coding: utf-8 -*-
"""
District-persona bridge for the final district-level classification.

This script does not retrain the final district PU model. It reuses the
existing C0-C4 cafe persona assignment and attaches persona composition to the
district scores produced by 19_district_final.py.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    ROOT,
)


HERE = Path(__file__).resolve().parent
DISTRICT_FINAL_SCRIPT = HERE / "19_district_final.py"

ARCHIVE_TABLE_DIR = ROOT / "reports" / "archive" / "tables" / "classification"
ARCHIVE_FIGURE_DIR = ROOT / "reports" / "archive" / "figures" / "classification"
ARCHIVE_LOG_DIR = ROOT / "reports" / "archive" / "classification" / "logs"

PERSONA_COLORS = {
    0: "#7dd3fc",  # office/high-income
    1: "#34d399",  # commercial
    2: "#fbbf24",  # residential
    3: "#f472b6",  # dense central
    4: "#a78bfa",  # weak subway access
}
PERSONA_LABELS = {
    0: "C0 오피스고소득",
    1: "C1 상업활성",
    2: "C2 주거생활",
    3: "C3 도심초밀집",
    4: "C4 비역세권",
}
PERSONA_SHORT = {k: f"C{k}" for k in range(5)}
SHARE_COLUMNS = [f"share_C{k}" for k in range(5)]

SNU = (37.4591, 126.9520)
UNIVERSITIES = {
    "서울대": (37.4591, 126.9520),
    "연세대": (37.5665, 126.9388),
    "고려대": (37.5894, 127.0327),
    "한양대": (37.5559, 127.0438),
    "홍익대": (37.5511, 126.9250),
    "중앙대": (37.5051, 126.9571),
    "경희대": (37.5970, 127.0517),
    "이화여대": (37.5620, 126.9469),
    "건국대": (37.5403, 127.0793),
    "서강대": (37.5511, 126.9410),
}


def ensure_dirs() -> None:
    for path in [
        CLASSIFICATION_OUTPUT_DIR,
        CLASSIFICATION_FIGURE_DIR,
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


def haversine_km(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius_km = 6371.0
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat) * p / 2.0) ** 2
        + np.cos(lat * p) * np.cos(lat2 * p) * np.sin((lon2 - lon) * p / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def auto_eps(xy_rad: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(xy_rad)
    distances, _ = nn.kneighbors(xy_rad)
    kth = np.sort(distances[:, k - 1])[::-1]
    second_diff = np.diff(np.diff(kth))
    lo, hi = int(len(kth) * 0.05), int(len(kth) * 0.6)
    return float(kth[lo + int(np.argmax(second_diff[lo:hi])) + 1])


def assign_districts(df: pd.DataFrame) -> tuple[pd.Series, float]:
    xy_rad = np.radians(df[["lat", "lon"]].to_numpy(float))
    eps = auto_eps(xy_rad)
    labels = DBSCAN(eps=eps, min_samples=5, metric="haversine").fit_predict(xy_rad)

    assigned = labels.copy()
    non_noise = labels != -1
    if non_noise.any() and (~non_noise).any():
        nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(xy_rad[non_noise])
        _, nearest = nn.kneighbors(xy_rad[~non_noise])
        assigned[~non_noise] = labels[non_noise][nearest[:, 0]]

    return pd.Series(assigned, index=df.index, name="district"), eps * 6371000.0


def require_or_build_district_scores() -> None:
    scores_path = CLASSIFICATION_OUTPUT_DIR / "district_scores.csv"
    university_path = CLASSIFICATION_OUTPUT_DIR / "district_university.csv"
    if scores_path.exists() and university_path.exists():
        return

    print("[bridge] district score outputs missing; running 19_district_final.py first")
    subprocess.run([sys.executable, str(DISTRICT_FINAL_SCRIPT)], check=True, cwd=str(ROOT))


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_or_build_district_scores()
    cafes = pd.read_parquet(CLASSIFICATION_DATASET_PATH).reset_index(drop=True)
    personas = pd.read_parquet(CLASSIFICATION_OUTPUT_DIR / "cafe_personas.parquet").reset_index(drop=True)
    scores = pd.read_csv(CLASSIFICATION_OUTPUT_DIR / "district_scores.csv", encoding="utf-8-sig")

    if len(cafes) != len(personas):
        raise ValueError(f"Row count mismatch: clf_dataset={len(cafes)}, cafe_personas={len(personas)}")
    if not (cafes["name"].to_numpy() == personas["name"].to_numpy()).all():
        raise ValueError("Row order mismatch: name columns differ between clf_dataset and cafe_personas")
    if not np.allclose(cafes["lat"].to_numpy(float), personas["lat"].to_numpy(float)):
        raise ValueError("Row order mismatch: lat columns differ between clf_dataset and cafe_personas")
    if personas["persona"].isna().any():
        raise ValueError("Missing persona assignment exists")

    return cafes, personas, scores


def persona_entropy(shares: pd.DataFrame) -> pd.Series:
    values = shares.to_numpy(float)
    safe = np.where(values > 0, values, 1.0)
    entropy = -(values * np.log(safe)).sum(axis=1) / math.log(values.shape[1])
    return pd.Series(entropy, index=shares.index)


def build_bridge_table(cafes: pd.DataFrame, personas: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    working = cafes[["name", "s", "lat", "lon", "sigungu"]].copy()
    working["persona"] = personas["persona"].astype(int)
    working["persona_name"] = personas["persona_name"].astype(str)
    working["district"], eps_m = assign_districts(working)

    dominant_sigungu = working.groupby("district")["sigungu"].agg(lambda values: values.value_counts().idxmax())
    base = working.groupby("district").agg(
        n_cafe=("name", "size"),
        n_starbucks=("s", "sum"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )
    counts = pd.crosstab(working["district"], working["persona"]).reindex(columns=range(5), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = SHARE_COLUMNS

    bridge = base.join(shares)
    bridge["dominant_sigungu"] = dominant_sigungu
    bridge["dominant_persona"] = shares.idxmax(axis=1).str.replace("share_", "", regex=False)
    bridge["dominant_persona_name"] = bridge["dominant_persona"].map(
        {f"C{k}": PERSONA_LABELS[k] for k in range(5)}
    )
    bridge["persona_entropy"] = persona_entropy(shares)
    bridge["sb_core_share"] = bridge["share_C0"] + bridge["share_C1"] + bridge["share_C3"]
    bridge["weak_access_share"] = bridge["share_C4"]

    score_cols = ["district", "score", "score_pct"]
    if "y" in scores.columns:
        score_cols.append("y")
    score_frame = scores[score_cols].drop_duplicates("district").set_index("district")
    bridge = bridge.join(score_frame[["score", "score_pct"]], how="left")
    if bridge["score"].isna().any():
        missing = bridge.index[bridge["score"].isna()].tolist()[:10]
        raise ValueError(f"District score join failed; missing districts include {missing}")

    bridge["has_starbucks"] = bridge["n_starbucks"].gt(0).astype(int)
    bridge = bridge.reset_index()
    bridge["score_decile"] = pd.qcut(
        bridge["score_pct"], q=10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop"
    ).astype(str)
    bridge = bridge.sort_values("score_pct", ascending=False).reset_index(drop=True)

    cafes_with_district = working.copy()
    return bridge, cafes_with_district, eps_m


def build_decile_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    decile = (
        bridge.groupby("score_decile", observed=False)
        .agg(
            n_district=("district", "size"),
            mean_score_pct=("score_pct", "mean"),
            mean_sb_core_share=("sb_core_share", "mean"),
            mean_weak_access_share=("weak_access_share", "mean"),
            **{column: (column, "mean") for column in SHARE_COLUMNS},
        )
        .reset_index()
    )
    decile["decile_num"] = decile["score_decile"].str.extract(r"(\d+)").astype(int)
    decile = decile.sort_values("decile_num").drop(columns="decile_num")
    return decile


def build_top_district_persona(bridge: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    columns = [
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
    ]
    top = bridge.sort_values(["score_pct", "score"], ascending=False).head(n)[columns].copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top


def load_university_scores() -> pd.DataFrame:
    scores = pd.read_csv(CLASSIFICATION_OUTPUT_DIR / "district_university.csv", encoding="utf-8-sig")
    required = {"대학", "n_cafe", "상권적합도", "백분위", "스벅수"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"district_university.csv missing columns: {sorted(missing)}")
    scores = scores.rename(
        columns={
            "대학": "university_label",
            "n_cafe": "synthetic_n_cafe",
            "상권적합도": "synthetic_score",
            "백분위": "synthetic_score_pct",
            "스벅수": "synthetic_n_starbucks",
        }
    )
    scores["university"] = scores["university_label"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    return scores


def build_university_persona(cafes_with_district: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    bridge_lookup = bridge.set_index("district")
    synthetic_scores = load_university_scores().set_index("university")
    rows: list[dict[str, object]] = []
    for name, (lat, lon) in UNIVERSITIES.items():
        distance = haversine_km(cafes_with_district["lat"], cafes_with_district["lon"], lat, lon)
        campus = cafes_with_district.loc[distance <= 1.0].copy()
        if len(campus) < 5:
            continue
        if name not in synthetic_scores.index:
            raise ValueError(f"Missing synthetic university score for {name}")
        synthetic = synthetic_scores.loc[name]
        counts = campus["persona"].value_counts().reindex(range(5), fill_value=0)
        shares = counts / counts.sum()
        campus_district = int(campus["district"].value_counts().idxmax())
        district_row = bridge_lookup.loc[campus_district]
        row: dict[str, object] = {
            "university": name,
            "n_cafe": int(synthetic["synthetic_n_cafe"]),
            "n_starbucks": int(synthetic["synthetic_n_starbucks"]),
            "persona_n_cafe": int(len(campus)),
            "main_district": campus_district,
            "score": float(synthetic["synthetic_score"]),
            "score_pct": float(synthetic["synthetic_score_pct"]),
            "actual_district_score": float(district_row["score"]),
            "actual_district_score_pct": float(district_row["score_pct"]),
            "dominant_persona": PERSONA_LABELS[int(shares.idxmax())],
            "sb_core_share": float(shares[0] + shares[1] + shares[3]),
            "weak_access_share": float(shares[4]),
        }
        for k in range(5):
            row[f"share_C{k}"] = float(shares[k])
        rows.append(row)

    universities = pd.DataFrame(rows)
    if universities.empty:
        raise ValueError("University persona table is empty")
    universities = universities.sort_values("score_pct", ascending=False).reset_index(drop=True)
    return universities


def write_tables(bridge: pd.DataFrame, universities: pd.DataFrame, decile: pd.DataFrame, top_districts: pd.DataFrame) -> None:
    targets = {
        "district_persona_bridge.csv": bridge,
        "district_university_persona.csv": universities,
        "district_persona_decile_summary.csv": decile,
        "top20_district_persona.csv": top_districts,
    }
    for filename, frame in targets.items():
        frame.to_csv(CLASSIFICATION_OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
        frame.to_csv(ARCHIVE_TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    bridge.to_parquet(CLASSIFICATION_OUTPUT_DIR / "district_persona_bridge.parquet", index=False)


def savefig(fig: plt.Figure, filename: str) -> None:
    generated = CLASSIFICATION_FIGURE_DIR / filename
    archive = ARCHIVE_FIGURE_DIR / filename
    fig.savefig(generated, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    shutil.copy2(generated, archive)
    print(f"[figure] {generated}")


def plot_bridge_map(bridge: pd.DataFrame, universities: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9), facecolor="#0b1020")
    cmap = LinearSegmentedColormap.from_list("scoreglow", ["#1f2937", "#155e75", "#22c55e", "#fde047"])
    sizes = 28 + np.sqrt(bridge["n_cafe"].to_numpy(float)) * 18
    edge_colors = np.where(bridge["has_starbucks"].eq(1), "#f8fafc", "#334155")
    ax.scatter(
        bridge["lon"],
        bridge["lat"],
        c=bridge["score_pct"],
        s=sizes,
        cmap=cmap,
        alpha=0.82,
        linewidth=1.1,
        edgecolors=edge_colors,
    )

    for _, row in universities.iterrows():
        lat, lon = UNIVERSITIES[str(row["university"])]
        is_snu = row["university"] == "서울대"
        ax.scatter(
            lon,
            lat,
            marker="*",
            s=280 if is_snu else 125,
            c="#ef4444" if is_snu else "#e2e8f0",
            edgecolors="#ffffff",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            lon + 0.004,
            lat + 0.003,
            str(row["university"]),
            color="#fecaca" if is_snu else "#e2e8f0",
            fontsize=11 if is_snu else 9,
            weight="bold" if is_snu else "normal",
            zorder=6,
        )

    cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("District score percentile", color="#e5e7eb")
    cbar.ax.yaxis.set_tick_params(color="#e5e7eb")
    plt.setp(cbar.ax.get_yticklabels(), color="#e5e7eb")

    ax.set_title("District Persona Bridge Map", color="#f8fafc", fontsize=18, weight="bold", pad=15)
    ax.set_xlabel("Longitude", color="#cbd5e1")
    ax.set_ylabel("Latitude", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#1e293b", lw=0.7, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    savefig(fig, "F16_district_persona_bridge_map.png")


def plot_university_fingerprint(universities: pd.DataFrame) -> None:
    data = universities.sort_values("score_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    fig, (ax_bar, ax_dot) = plt.subplots(
        1,
        2,
        figsize=(12, 6.5),
        width_ratios=[3.8, 1.1],
        facecolor="#0b1020",
        sharey=True,
    )
    left = np.zeros(len(data))
    for k in range(5):
        values = data[f"share_C{k}"].to_numpy(float)
        ax_bar.barh(y, values, left=left, color=PERSONA_COLORS[k], label=PERSONA_LABELS[k], height=0.68)
        left += values

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(data["university"], color="#e5e7eb", fontsize=10)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Persona share", color="#cbd5e1")
    ax_bar.set_title("University Persona Fingerprint", color="#f8fafc", fontsize=16, weight="bold")
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.grid(axis="x", color="#1e293b", alpha=0.7)
    ax_bar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, labelcolor="#e5e7eb")

    colors = ["#ef4444" if name == "서울대" else "#38bdf8" for name in data["university"]]
    ax_dot.scatter(data["score_pct"], y, c=colors, s=120, edgecolors="#ffffff", linewidth=0.8)
    ax_dot.set_xlim(0, 100)
    ax_dot.set_xlabel("Score pct", color="#cbd5e1")
    ax_dot.set_title("Classification", color="#f8fafc", fontsize=12, weight="bold")
    ax_dot.tick_params(colors="#94a3b8")
    ax_dot.grid(axis="x", color="#1e293b", alpha=0.7)
    for spine in [*ax_bar.spines.values(), *ax_dot.spines.values()]:
        spine.set_color("#334155")

    savefig(fig, "F17_university_persona_fingerprint.png")


def plot_top_district_fingerprint(top_districts: pd.DataFrame) -> None:
    data = top_districts.sort_values("rank", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    fig, (ax_bar, ax_dot) = plt.subplots(
        1,
        2,
        figsize=(13.5, 10),
        width_ratios=[4.4, 1.05],
        facecolor="#0b1020",
        sharey=True,
    )

    left = np.zeros(len(data))
    for k in range(5):
        values = data[f"share_C{k}"].to_numpy(float)
        ax_bar.barh(y, values, left=left, color=PERSONA_COLORS[k], label=PERSONA_LABELS[k], height=0.68)
        left += values

    labels = [
        f"#{int(row.rank):02d} D{int(row.district)} | {row.dominant_sigungu} | n={int(row.n_cafe)}, SB={int(row.n_starbucks)}"
        for row in data.itertuples(index=False)
    ]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, color="#e5e7eb", fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Persona share", color="#cbd5e1")
    ax_bar.set_title("Top 20 District Persona Fingerprint", color="#f8fafc", fontsize=17, weight="bold")
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.grid(axis="x", color="#1e293b", alpha=0.7)
    ax_bar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, labelcolor="#e5e7eb")

    dot_colors = np.where(data["n_starbucks"].gt(0), "#38bdf8", "#f97316")
    ax_dot.scatter(data["score_pct"], y, c=dot_colors, s=115, edgecolors="#ffffff", linewidth=0.8)
    ax_dot.set_xlim(84, 101)
    ax_dot.set_xlabel("Score pct", color="#cbd5e1")
    ax_dot.set_title("Classification", color="#f8fafc", fontsize=12, weight="bold")
    ax_dot.tick_params(colors="#94a3b8")
    ax_dot.grid(axis="x", color="#1e293b", alpha=0.7)
    for _, row in data.iterrows():
        ax_dot.text(
            row["score_pct"] + 0.15,
            y[int(row.name)],
            f"{row['score_pct']:.0f}",
            color="#e5e7eb",
            fontsize=8,
            va="center",
        )
    for spine in [*ax_bar.spines.values(), *ax_dot.spines.values()]:
        spine.set_color("#334155")

    savefig(fig, "F21_top20_district_persona_fingerprint.png")


def plot_decile_heatmap(decile: pd.DataFrame) -> None:
    matrix = decile.set_index("score_decile")[SHARE_COLUMNS].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor="#0b1020")
    cmap = LinearSegmentedColormap.from_list("personaheat", ["#111827", "#1d4ed8", "#22c55e", "#facc15"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.55, float(matrix.max())))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([PERSONA_SHORT[k] for k in range(5)], color="#e5e7eb")
    ax.set_yticks(np.arange(len(decile)))
    ax.set_yticklabels(decile["score_decile"], color="#e5e7eb")
    ax.set_xlabel("Persona", color="#cbd5e1")
    ax.set_ylabel("District score decile", color="#cbd5e1")
    ax.set_title("Persona Share by Score Decile", color="#f8fafc", fontsize=16, weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", color="#f8fafc", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Mean share", color="#e5e7eb")
    cbar.ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    savefig(fig, "F18_persona_score_decile_heatmap.png")


def plot_snu_card(universities: pd.DataFrame) -> None:
    snu = universities.loc[universities["university"].eq("서울대")].iloc[0]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="#0b1020")
    ax.axis("off")
    ax.text(0.04, 0.88, "SNU District Interpretation", color="#f8fafc", fontsize=24, weight="bold")
    ax.text(
        0.04,
        0.79,
        "Classification score is low because the district persona mix is not close to the core Starbucks personas.",
        color="#cbd5e1",
        fontsize=12,
    )

    cards = [
        ("Score percentile", f"{snu['score_pct']:.0f}", "#38bdf8"),
        ("Dominant persona", str(snu["dominant_persona"]), "#fbbf24"),
        ("C0+C1+C3 core share", f"{snu['sb_core_share']:.0%}", "#22c55e"),
        ("C4 weak-access share", f"{snu['weak_access_share']:.0%}", "#a78bfa"),
    ]
    x0s = [0.04, 0.28, 0.52, 0.76]
    for x0, (label, value, color) in zip(x0s, cards):
        rect = plt.Rectangle((x0, 0.42), 0.20, 0.26, transform=ax.transAxes, color="#111827", ec="#334155", lw=1.2)
        ax.add_patch(rect)
        ax.text(x0 + 0.02, 0.61, label, transform=ax.transAxes, color="#94a3b8", fontsize=10)
        ax.text(x0 + 0.02, 0.49, value, transform=ax.transAxes, color=color, fontsize=19, weight="bold")

    left = 0.04
    for k in range(5):
        width = float(snu[f"share_C{k}"]) * 0.82
        ax.add_patch(
            plt.Rectangle((left, 0.24), width, 0.085, transform=ax.transAxes, color=PERSONA_COLORS[k], ec="none")
        )
        if width > 0.055:
            ax.text(
                left + width / 2,
                0.282,
                f"C{k}",
                transform=ax.transAxes,
                color="#0b1020",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )
        left += width

    ax.text(0.04, 0.15, "Persona mix: " + "  ".join(f"C{k} {snu[f'share_C{k}']:.0%}" for k in range(5)), color="#e5e7eb", fontsize=12)
    ax.text(
        0.04,
        0.08,
        "Use this as a post-hoc bridge: existing C0-C4 clustering explains the district score, not a new predictive feature claim.",
        color="#94a3b8",
        fontsize=10,
    )
    savefig(fig, "F19_snu_persona_interpretation_card.png")


def make_figures(bridge: pd.DataFrame, universities: pd.DataFrame, decile: pd.DataFrame, top_districts: pd.DataFrame) -> None:
    setup_plot_style()
    plot_bridge_map(bridge, universities)
    plot_university_fingerprint(universities)
    plot_top_district_fingerprint(top_districts)
    plot_decile_heatmap(decile)
    plot_snu_card(universities)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    work = df[columns].head(max_rows).copy() if max_rows else df[columns].copy()
    for column in work.columns:
        if pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda value: f"{value:.3f}")
    work = work.fillna("").astype(str)
    headers = [str(column) for column in work.columns]
    rows = work.to_numpy().tolist()
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_log(bridge: pd.DataFrame, universities: pd.DataFrame, decile: pd.DataFrame, top_districts: pd.DataFrame, eps_m: float) -> None:
    snu = universities.loc[universities["university"].eq("서울대")].iloc[0]
    lines = [
        "# District Persona Bridge",
        "",
        f"- DBSCAN auto-eps with noise absorption: {eps_m:.0f}m",
        f"- Districts: {len(bridge)}",
        f"- SNU score percentile: {snu['score_pct']:.0f}",
        f"- SNU dominant persona: {snu['dominant_persona']}",
        f"- SNU core share(C0+C1+C3): {snu['sb_core_share']:.3f}",
        f"- SNU weak-access share(C4): {snu['weak_access_share']:.3f}",
        "",
        "## University Persona",
        "",
        markdown_table(
            universities,
            ["university", "n_cafe", "n_starbucks", "score_pct", "dominant_persona", "sb_core_share", "weak_access_share"],
        ),
        "",
        "## Score Decile Persona Summary",
        "",
        markdown_table(decile, ["score_decile", "mean_score_pct", "mean_sb_core_share", "mean_weak_access_share", *SHARE_COLUMNS]),
        "",
        "## Top 20 District Persona",
        "",
        markdown_table(
            top_districts,
            ["rank", "district", "dominant_sigungu", "n_cafe", "n_starbucks", "score_pct", "dominant_persona_name", "sb_core_share", "weak_access_share"],
        ),
        "",
    ]
    (ARCHIVE_LOG_DIR / "PERSONA_BRIDGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(bridge: pd.DataFrame, universities: pd.DataFrame) -> None:
    share_sum = bridge[SHARE_COLUMNS].sum(axis=1)
    if not np.allclose(share_sum, 1.0, atol=1e-9):
        raise ValueError("Persona share columns do not sum to 1 for every district")
    if universities.empty:
        raise ValueError("University persona output is empty")
    required = [
        CLASSIFICATION_OUTPUT_DIR / "district_persona_bridge.csv",
        CLASSIFICATION_OUTPUT_DIR / "district_university_persona.csv",
        CLASSIFICATION_OUTPUT_DIR / "district_persona_decile_summary.csv",
        CLASSIFICATION_OUTPUT_DIR / "top20_district_persona.csv",
        CLASSIFICATION_OUTPUT_DIR / "district_persona_bridge.parquet",
        ARCHIVE_FIGURE_DIR / "F16_district_persona_bridge_map.png",
        ARCHIVE_FIGURE_DIR / "F17_university_persona_fingerprint.png",
        ARCHIVE_FIGURE_DIR / "F21_top20_district_persona_fingerprint.png",
        ARCHIVE_FIGURE_DIR / "F18_persona_score_decile_heatmap.png",
        ARCHIVE_FIGURE_DIR / "F19_snu_persona_interpretation_card.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Missing expected outputs: {missing}")


def main() -> None:
    ensure_dirs()
    cafes, personas, scores = read_inputs()
    bridge, cafes_with_district, eps_m = build_bridge_table(cafes, personas, scores)
    decile = build_decile_summary(bridge)
    top_districts = build_top_district_persona(bridge, n=20)
    universities = build_university_persona(cafes_with_district, bridge)

    write_tables(bridge, universities, decile, top_districts)
    make_figures(bridge, universities, decile, top_districts)
    write_log(bridge, universities, decile, top_districts, eps_m)
    validate_outputs(bridge, universities)

    snu = universities.loc[universities["university"].eq("서울대")].iloc[0]
    print(f"[bridge] districts={len(bridge)} eps_m={eps_m:.0f}")
    print(
        "[bridge] SNU "
        f"score_pct={snu['score_pct']:.0f}, "
        f"dominant={snu['dominant_persona']}, "
        f"core_share={snu['sb_core_share']:.3f}, "
        f"C4={snu['weak_access_share']:.3f}"
    )
    print(f"[bridge] tables: {CLASSIFICATION_OUTPUT_DIR}")
    print(f"[bridge] archive tables: {ARCHIVE_TABLE_DIR}")
    print(f"[bridge] archive figures: {ARCHIVE_FIGURE_DIR}")


if __name__ == "__main__":
    main()
