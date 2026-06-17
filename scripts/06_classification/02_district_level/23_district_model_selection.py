# -*- coding: utf-8 -*-
"""
District-level PU model selection and hyperparameter comparison.

This script keeps the final district definition from 19_district_final.py:
  - DBSCAN auto-eps
  - nearest-district absorption for DBSCAN noise
  - district features = mean cafe-level classification features
  - label = any Starbucks in the district

Outputs:
  - outputs/district_model_selection.csv
  - outputs/district_model_selection_best.json
  - figures/F23_district_model_selection.png
  - logs/DISTRICT_MODEL_SELECTION_REPORT.md
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CLASSIFICATION_DATASET_PATH,
    CLF_FEATURES,
    DISTRICT_CLASSIFICATION_FIGURE_DIR,
    DISTRICT_CLASSIFICATION_LOG_DIR,
    DISTRICT_CLASSIFICATION_OUTPUT_DIR,
    ROOT,
)


RNG = 42
N_BAGS = 15
EARTH_RADIUS_M = 6_371_000.0

ARCHIVE_TABLE_DIR = ROOT / "reports" / "archive" / "tables" / "classification" / "02_district_level"
ARCHIVE_FIGURE_DIR = ROOT / "reports" / "archive" / "figures" / "classification" / "02_district_level"
ARCHIVE_LOG_DIR = ROOT / "reports" / "archive" / "classification" / "logs" / "02_district_level"


@dataclass(frozen=True)
class Candidate:
    model_id: str
    family: str
    params: dict[str, object]


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


def auto_eps(xy_rad: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(xy_rad)
    distances, _ = nn.kneighbors(xy_rad)
    kth = np.sort(distances[:, k - 1])[::-1]
    second_diff = np.diff(np.diff(kth))
    lo, hi = int(len(kth) * 0.05), int(len(kth) * 0.6)
    return float(kth[lo + int(np.argmax(second_diff[lo:hi])) + 1])


def assign_districts(cafes: pd.DataFrame) -> tuple[pd.Series, float]:
    xy_rad = np.radians(cafes[["lat", "lon"]].to_numpy(float))
    eps = auto_eps(xy_rad)
    labels = DBSCAN(eps=eps, min_samples=5, metric="haversine").fit_predict(xy_rad)

    assigned = labels.copy()
    non_noise = labels != -1
    if non_noise.any() and (~non_noise).any():
        nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(xy_rad[non_noise])
        _, nearest = nn.kneighbors(xy_rad[~non_noise])
        assigned[~non_noise] = labels[non_noise][nearest[:, 0]]

    return pd.Series(assigned, index=cafes.index, name="district"), eps * EARTH_RADIUS_M


def build_district_frame() -> tuple[pd.DataFrame, float]:
    cafes = pd.read_parquet(CLASSIFICATION_DATASET_PATH).copy()
    cafes["district"], eps_m = assign_districts(cafes)
    districts = (
        cafes.groupby("district")
        .agg(
            **{feature: (feature, "mean") for feature in CLF_FEATURES},
            y=("s", "max"),
            n_cafe=("name", "size"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
    )
    districts["spatial_group"] = (
        np.floor(districts["lat"] / 0.02).astype(int).astype(str)
        + "_"
        + np.floor(districts["lon"] / 0.02).astype(int).astype(str)
    )
    return districts, eps_m


def candidate_grid() -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate(
            "RF_current_leaf2",
            "RandomForest",
            {"n_estimators": 100, "min_samples_leaf": 2, "max_features": "sqrt"},
        )
    ]

    for leaf in [1, 2, 5]:
        for max_features in ["sqrt", 0.7]:
            candidates.append(
                Candidate(
                    f"RF_leaf{leaf}_mf{max_features}",
                    "RandomForest",
                    {"n_estimators": 160, "min_samples_leaf": leaf, "max_features": max_features},
                )
            )
            candidates.append(
                Candidate(
                    f"ET_leaf{leaf}_mf{max_features}",
                    "ExtraTrees",
                    {"n_estimators": 160, "min_samples_leaf": leaf, "max_features": max_features},
                )
            )

    for n_estimators in [80, 140]:
        for learning_rate in [0.05, 0.10]:
            for depth in [2, 3]:
                candidates.append(
                    Candidate(
                        f"GBM_n{n_estimators}_lr{learning_rate:g}_d{depth}",
                        "GradientBoosting",
                        {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": depth,
                        },
                    )
                )

    for c_value in [0.3, 1.0, 3.0]:
        candidates.append(
            Candidate(
                f"LOGIT_C{c_value:g}",
                "LogisticRegression",
                {"C": c_value},
            )
        )

    return candidates


def build_estimator(candidate: Candidate, seed: int):
    if candidate.family == "RandomForest":
        return RandomForestClassifier(
            **candidate.params,
            n_jobs=1,
            random_state=seed,
        )
    if candidate.family == "ExtraTrees":
        return ExtraTreesClassifier(
            **candidate.params,
            n_jobs=1,
            random_state=seed,
        )
    if candidate.family == "GradientBoosting":
        return GradientBoostingClassifier(
            **candidate.params,
            random_state=seed,
        )
    if candidate.family == "LogisticRegression":
        return LogisticRegression(
            **candidate.params,
            max_iter=2000,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown candidate family: {candidate.family}")


def pu_bag_predict(
    candidate: Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    positives = np.where(y_train == 1)[0]
    unlabeled = np.where(y_train == 0)[0]
    if len(positives) == 0 or len(unlabeled) == 0:
        raise ValueError("PU bagging requires both positive and unlabeled rows in the train fold")

    rng = np.random.RandomState(seed)
    scores = np.zeros(len(x_test), dtype=float)
    for bag_idx in range(N_BAGS):
        sample_u = rng.choice(
            unlabeled,
            size=len(positives),
            replace=len(unlabeled) < len(positives),
        )
        sample_idx = np.concatenate([positives, sample_u])
        sample_y = np.r_[np.ones(len(positives), dtype=int), np.zeros(len(sample_u), dtype=int)]
        model = build_estimator(candidate, seed + bag_idx)
        model.fit(x_train[sample_idx], sample_y)
        scores += model.predict_proba(x_test)[:, 1]
    return scores / N_BAGS


def top_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores)[::-1]
    base_rate = float(y.mean())
    total_pos = int(y.sum())
    metrics: dict[str, float] = {}
    for pct in [0.05, 0.10, 0.20]:
        n_top = max(1, int(np.ceil(len(y) * pct)))
        top = order[:n_top]
        hit_rate = float(y[top].mean())
        key = int(pct * 100)
        metrics[f"precision_at_{key}"] = hit_rate
        metrics[f"lift_at_{key}"] = hit_rate / base_rate if base_rate > 0 else np.nan
        metrics[f"gains_at_{key}"] = float(y[top].sum() / total_pos) if total_pos > 0 else np.nan
    return metrics


def evaluate_candidate(candidate: Candidate, districts: pd.DataFrame) -> dict[str, object]:
    x = districts[CLF_FEATURES].to_numpy(float)
    y = districts["y"].to_numpy(int)
    groups = districts["spatial_group"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(GroupKFold(5).split(x, y, groups)):
        scaler = StandardScaler().fit(x[train_idx])
        x_train = scaler.transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        fold_scores = pu_bag_predict(
            candidate,
            x_train,
            y[train_idx],
            x_test,
            seed=RNG + fold_idx * 10_000,
        )
        oof[test_idx] = fold_scores
        if 0 < y[test_idx].sum() < len(test_idx):
            fold_aucs.append(float(roc_auc_score(y[test_idx], fold_scores)))

    if np.isnan(oof).any():
        raise ValueError(f"OOF prediction contains NaN for {candidate.model_id}")

    row: dict[str, object] = {
        "model_id": candidate.model_id,
        "family": candidate.family,
        "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        "pu_auc": float(roc_auc_score(y, oof)),
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "n_bags": N_BAGS,
    }
    row.update(top_metrics(y, oof))
    return row


def evaluate_all(districts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = candidate_grid()
    for idx, candidate in enumerate(candidates, start=1):
        row = evaluate_candidate(candidate, districts)
        rows.append(row)
        print(
            "[model-selection] "
            f"{idx:02d}/{len(candidates):02d} {candidate.model_id}: "
            f"PU-AUC={row['pu_auc']:.4f}, lift@5={row['lift_at_5']:.2f}, gains@20={row['gains_at_20']:.3f}"
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["pu_auc", "lift_at_5", "gains_at_20"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def save_tables(result: pd.DataFrame) -> dict[str, object]:
    table_path = DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_model_selection.csv"
    archive_path = ARCHIVE_TABLE_DIR / "district_model_selection.csv"
    result.to_csv(table_path, index=False, encoding="utf-8-sig")
    result.to_csv(archive_path, index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict()
    best_path = DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_model_selection_best.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[table] {table_path}")
    return best


def setup_plot_style() -> None:
    for font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def make_figure(result: pd.DataFrame) -> None:
    setup_plot_style()
    top = result.head(12).sort_values("pu_auc", ascending=True)
    family_best = result.sort_values("pu_auc", ascending=False).groupby("family", as_index=False).head(1)
    colors = {
        "RandomForest": "#2563eb",
        "ExtraTrees": "#16a34a",
        "GradientBoosting": "#dc2626",
        "LogisticRegression": "#9333ea",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].barh(top["model_id"], top["pu_auc"], color=[colors[fam] for fam in top["family"]])
    axes[0].set_title("Top model/hyperparameter candidates", fontsize=13, weight="bold")
    axes[0].set_xlabel("OOF PU-AUC")
    axes[0].set_xlim(max(0.5, top["pu_auc"].min() - 0.025), min(1.0, top["pu_auc"].max() + 0.025))
    axes[0].grid(axis="x", color="#dddddd", lw=0.7, alpha=0.7)

    for family, frame in result.groupby("family"):
        axes[1].scatter(
            frame["gains_at_20"],
            frame["pu_auc"],
            s=75,
            alpha=0.78,
            label=family,
            color=colors[family],
            edgecolors="white",
            linewidth=0.7,
        )
    for _, row in family_best.iterrows():
        axes[1].annotate(
            row["model_id"],
            (row["gains_at_20"], row["pu_auc"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_title("PU-AUC vs gains@20%", fontsize=13, weight="bold")
    axes[1].set_xlabel("gains@20%")
    axes[1].set_ylabel("OOF PU-AUC")
    axes[1].grid(color="#dddddd", lw=0.7, alpha=0.7)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("F23. District PU Model Selection", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figure_path = DISTRICT_CLASSIFICATION_FIGURE_DIR / "F23_district_model_selection.png"
    archive_path = ARCHIVE_FIGURE_DIR / "F23_district_model_selection.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(figure_path, archive_path)
    print(f"[figure] {figure_path}")


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    work = frame[columns].head(max_rows).copy() if max_rows else frame[columns].copy()
    for column in work.columns:
        if pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda value: f"{value:.4f}")
    work = work.fillna("").astype(str)
    headers = list(work.columns)
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


def write_log(result: pd.DataFrame, districts: pd.DataFrame, eps_m: float, best: dict[str, object]) -> None:
    family_best = result.sort_values("pu_auc", ascending=False).groupby("family", as_index=False).head(1)
    current = result.loc[result["model_id"].eq("RF_current_leaf2")].iloc[0]
    auc_delta = float(best["pu_auc"] - current["pu_auc"])
    lines = [
        "# 상권 단위 모델 선택 및 하이퍼파라미터 비교",
        "",
        f"> 생성 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 상권 정의: DBSCAN auto-eps={eps_m:.0f}m + 노이즈 최근접 흡수",
        f"- 상권 수: {len(districts)}개, 스벅상권: {int(districts['y'].sum())}개",
        f"- 검증: spatial GroupKFold(5), 각 fold train 내부 PU bagging {N_BAGS}회",
        "- 선택 기준: OOF PU-AUC 1순위, lift@5%와 gains@20%를 보조 지표로 확인",
        "",
        "## 선택 결과",
        "",
        f"- Best: **{best['model_id']}** ({best['family']})",
        f"- Best OOF PU-AUC: **{best['pu_auc']:.4f}**",
        f"- Current RF 기준선 OOF PU-AUC: {current['pu_auc']:.4f}",
        f"- Best - current delta: {auc_delta:+.4f}",
        f"- Best lift@5%: {best['lift_at_5']:.2f}",
        f"- Best gains@20%: {best['gains_at_20']:.3f}",
        "",
        "## Top 10 Candidates",
        "",
        markdown_table(
            result,
            ["rank", "model_id", "family", "pu_auc", "fold_auc_mean", "fold_auc_std", "lift_at_5", "gains_at_20"],
            max_rows=10,
        ),
        "",
        "## Family Best",
        "",
        markdown_table(
            family_best.sort_values("pu_auc", ascending=False),
            ["model_id", "family", "pu_auc", "lift_at_5", "gains_at_20", "params"],
        ),
        "",
        "## 해석",
        "",
        "- 기존 19번 상권 최종 모델은 고정 RandomForest 설정을 사용했지만, 이 비교는 상권 단위에서도 모델 family와 hyperparameter 선택 과정을 명시적으로 추가한다.",
        "- PU 구조상 이 값들은 확정 음성 기준의 절대 정확도가 아니라, 스타벅스 상권을 U 상권보다 위에 두는 순위 성능으로 해석해야 한다.",
        "- 최종 채점 모델을 교체할 때는 위 best 후보를 기준으로 19번 scoring 모델을 재학습하고 F16-F22 downstream 산출물을 함께 재생성하면 된다.",
        "",
    ]
    text = "\n".join(lines)
    log_path = DISTRICT_CLASSIFICATION_LOG_DIR / "DISTRICT_MODEL_SELECTION_REPORT.md"
    archive_path = ARCHIVE_LOG_DIR / "DISTRICT_MODEL_SELECTION_REPORT.md"
    log_path.write_text(text, encoding="utf-8")
    archive_path.write_text(text, encoding="utf-8")
    print(f"[log] {log_path}")


def validate_outputs() -> None:
    required = [
        DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_model_selection.csv",
        DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_model_selection_best.json",
        DISTRICT_CLASSIFICATION_FIGURE_DIR / "F23_district_model_selection.png",
        DISTRICT_CLASSIFICATION_LOG_DIR / "DISTRICT_MODEL_SELECTION_REPORT.md",
        ARCHIVE_TABLE_DIR / "district_model_selection.csv",
        ARCHIVE_FIGURE_DIR / "F23_district_model_selection.png",
        ARCHIVE_LOG_DIR / "DISTRICT_MODEL_SELECTION_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Missing expected outputs: {missing}")


def main() -> None:
    ensure_dirs()
    districts, eps_m = build_district_frame()
    result = evaluate_all(districts)
    best = save_tables(result)
    make_figure(result)
    write_log(result, districts, eps_m, best)
    validate_outputs()

    print("\n[상권 모델 선택]")
    print(f"districts={len(districts)}, positives={int(districts['y'].sum())}, eps_m={eps_m:.0f}")
    print(
        result[
            ["rank", "model_id", "family", "pu_auc", "fold_auc_mean", "fold_auc_std", "lift_at_5", "gains_at_20"]
        ]
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
