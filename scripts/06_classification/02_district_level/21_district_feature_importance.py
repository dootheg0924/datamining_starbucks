# -*- coding: utf-8 -*-
"""
District-level feature importance for the final PU classification model.

This mirrors 19_district_final.py:
  - DBSCAN auto-eps with nearest-district absorption
  - district features = mean of cafe-level classification features
  - label = any Starbucks in the district
  - spatial GroupKFold(5)

Outputs:
  - outputs/district_feature_importance.csv
  - figures/F20_district_feature_importance.png
  - logs/DISTRICT_FEATURE_IMPORTANCE_REPORT.md
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    DISTRICT_CLASSIFICATION_FIGURE_DIR,
    DISTRICT_CLASSIFICATION_LOG_DIR,
    DISTRICT_CLASSIFICATION_OUTPUT_DIR,
    CLF_FEATURES,
    ROOT,
)


RNG = 42
T_BAGS = 30
N_PERM = 8
EARTH_RADIUS_M = 6_371_000.0

ARCHIVE_TABLE_DIR = ROOT / "reports" / "archive" / "tables" / "classification" / "02_district_level"
ARCHIVE_FIGURE_DIR = ROOT / "reports" / "archive" / "figures" / "classification" / "02_district_level"
ARCHIVE_LOG_DIR = ROOT / "reports" / "archive" / "classification" / "logs" / "02_district_level"

FEATURE_LABELS_KO = {
    "log_dist_subway": "지하철거리(log)",
    "subway_count_cat": "지하철수(범주)",
    "subway_ridership": "역승하차",
    "bus_stops_300m": "버스정류장",
    "peak_avg": "피크평균",
    "restaurants_500m": "음식점",
    "log_retail_500m": "소매(log)",
    "convenience_500m": "편의점",
    "indie_cafe_500m": "독립카페",
    "low_price_cafe_500m": "저가카페",
    "franchise_cafe_500m": "프랜차이즈카페",
    "avg_income": "평균소득",
    "offices": "직장인구",
    "living_pop": "생활인구",
    "land_price": "공시지가",
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
    for font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


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

    return pd.Series(assigned, index=df.index, name="district"), eps * EARTH_RADIUS_M


def build_district_frame() -> tuple[pd.DataFrame, float]:
    cafes = pd.read_parquet(CLASSIFICATION_DATASET_PATH)
    cafes = cafes.copy()
    cafes["district"], eps_m = assign_districts(cafes)
    agg = (
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
    agg["spatial_group"] = (
        np.floor(agg["lat"] / 0.02).astype(int).astype(str)
        + "_"
        + np.floor(agg["lon"] / 0.02).astype(int).astype(str)
    )
    return agg, eps_m


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=RNG,
    )


def fit_pu_bag(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.RandomState,
    n_bags: int = T_BAGS,
) -> list[RandomForestClassifier]:
    pos = np.where(y_train == 1)[0]
    unlabeled = np.where(y_train == 0)[0]
    models: list[RandomForestClassifier] = []
    for _ in range(n_bags):
        sample_u = rng.choice(
            unlabeled,
            size=len(pos),
            replace=len(unlabeled) < len(pos),
        )
        sample_idx = np.concatenate([pos, sample_u])
        sample_y = np.r_[np.ones(len(pos)), np.zeros(len(sample_u))]
        models.append(rf().fit(X_train[sample_idx], sample_y))
    return models


def fit_logreg_bag(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.RandomState,
    n_bags: int = T_BAGS,
) -> list[LogisticRegression]:
    pos = np.where(y_train == 1)[0]
    unlabeled = np.where(y_train == 0)[0]
    models: list[LogisticRegression] = []
    for _ in range(n_bags):
        sample_u = rng.choice(
            unlabeled,
            size=len(pos),
            replace=len(unlabeled) < len(pos),
        )
        sample_idx = np.concatenate([pos, sample_u])
        sample_y = np.r_[np.ones(len(pos)), np.zeros(len(sample_u))]
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        models.append(model.fit(X_train[sample_idx], sample_y))
    return models


def bag_predict(models: list[RandomForestClassifier], X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0)


def compute_importance(districts: pd.DataFrame) -> tuple[pd.DataFrame, float, list[float]]:
    X = districts[CLF_FEATURES].to_numpy(float)
    y = districts["y"].to_numpy(int)
    groups = districts["spatial_group"].to_numpy()
    splits = list(GroupKFold(5).split(X, y, groups))

    n_features = len(CLF_FEATURES)
    perm = np.zeros((len(splits), n_features))
    rf_imp = np.zeros((len(splits), n_features))
    coefs = np.zeros((len(splits), n_features))
    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        rng = np.random.RandomState(RNG + fold_idx)
        models = fit_pu_bag(X_train, y_train, rng)
        base_score = bag_predict(models, X_test)
        base_auc = roc_auc_score(y_test, base_score)
        oof[test_idx] = base_score
        fold_aucs.append(base_auc)
        rf_imp[fold_idx] = np.mean([model.feature_importances_ for model in models], axis=0)

        perm_rng = np.random.RandomState(RNG + 1000 + fold_idx)
        for feature_idx in range(n_features):
            drops = []
            for _ in range(N_PERM):
                X_perm = X_test.copy()
                perm_rng.shuffle(X_perm[:, feature_idx])
                drops.append(base_auc - roc_auc_score(y_test, bag_predict(models, X_perm)))
            perm[fold_idx, feature_idx] = float(np.mean(drops))

        logreg_models = fit_logreg_bag(
            X_train,
            y_train,
            np.random.RandomState(RNG + 2000 + fold_idx),
        )
        coefs[fold_idx] = np.mean([model.coef_[0] for model in logreg_models], axis=0)
        print(f"[fold {fold_idx}] base_auc={base_auc:.4f}")

    oof_auc = roc_auc_score(y, oof)
    result = pd.DataFrame(
        {
            "feature": CLF_FEATURES,
            "label_ko": [FEATURE_LABELS_KO[feature] for feature in CLF_FEATURES],
            "perm_imp": perm.mean(axis=0),
            "perm_imp_std": perm.std(axis=0),
            "rf_imp": rf_imp.mean(axis=0),
            "logreg_coef": coefs.mean(axis=0),
            "logreg_coef_std": coefs.std(axis=0),
        }
    )
    result["direction"] = np.where(result["logreg_coef"] >= 0, "양(+, 스벅상권↑)", "음(-, 스벅상권↓)")
    result["coef_sign_stable"] = [
        float((np.sign(coefs[:, idx]) == np.sign(coefs[:, idx].mean())).mean())
        for idx in range(n_features)
    ]
    result = result.sort_values("perm_imp", ascending=False).reset_index(drop=True)
    return result, oof_auc, fold_aucs


def save_outputs(result: pd.DataFrame) -> None:
    output_path = DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_feature_importance.csv"
    archive_path = ARCHIVE_TABLE_DIR / "district_feature_importance.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    result.to_csv(archive_path, index=False, encoding="utf-8-sig")
    print(f"[table] {output_path}")


def make_figure(result: pd.DataFrame) -> Path:
    setup_plot_style()
    ranked = result.sort_values("perm_imp", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [1.12, 1]})

    axes[0].barh(ranked["label_ko"], ranked["perm_imp"], color="#1f7a3f")
    axes[0].set_title("상권 단위 순열중요도", fontsize=13, weight="bold")
    axes[0].set_xlabel("PU-AUC 하락폭")
    axes[0].grid(axis="x", color="#dddddd", lw=0.7, alpha=0.7)

    coef_colors = np.where(ranked["logreg_coef"].to_numpy() >= 0, "#c0392b", "#2c3e50")
    axes[1].barh(ranked["label_ko"], ranked["logreg_coef"], color=coef_colors)
    axes[1].axvline(0, color="#111111", lw=0.8)
    axes[1].set_title("방향성: LogReg 표준화계수", fontsize=13, weight="bold")
    axes[1].set_xlabel("계수 (빨강=스벅상권↑, 남색=스벅상권↓)")
    axes[1].grid(axis="x", color="#dddddd", lw=0.7, alpha=0.7)

    fig.suptitle("F20. 상권 단위 Classification Feature Importance", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    figure_path = DISTRICT_CLASSIFICATION_FIGURE_DIR / "F20_district_feature_importance.png"
    archive_path = ARCHIVE_FIGURE_DIR / "F20_district_feature_importance.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(figure_path, archive_path)
    print(f"[figure] {figure_path}")
    return figure_path


def write_log(
    result: pd.DataFrame,
    districts: pd.DataFrame,
    eps_m: float,
    oof_auc: float,
    fold_aucs: list[float],
) -> None:
    top = result.head(6)
    lines = [
        "# 상권 단위 Feature Importance",
        "",
        f"> 생성 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 상권 정의: DBSCAN auto-eps={eps_m:.0f}m + 노이즈 최근접 흡수",
        f"- 상권 수: {len(districts)}개, 스벅상권: {int(districts['y'].sum())}개",
        f"- OOF PU-AUC: {oof_auc:.4f}",
        f"- fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}",
        "",
        "## 변수별 중요도",
        "",
        "| 변수 | 순열중요도 | RF중요도 | LogReg계수 | 방향 | 부호안정성 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(
            "| %s | %.4f | %.4f | %+.4f | %s | %.0f%% |"
            % (
                row["label_ko"],
                row["perm_imp"],
                row["rf_imp"],
                row["logreg_coef"],
                row["direction"],
                row["coef_sign_stable"] * 100,
            )
        )

    lines += [
        "",
        "## 핵심 요약",
        "",
        "- 상권 단위 중요도 상위 변수: " + ", ".join(top["label_ko"].tolist()),
        "- 양(+) 방향 변수: " + ", ".join(result.loc[result["logreg_coef"] > 0, "label_ko"].tolist()),
        "- 음(-) 방향 변수: " + ", ".join(result.loc[result["logreg_coef"] < 0, "label_ko"].tolist()),
        "",
    ]
    log_text = "\n".join(lines)
    log_path = DISTRICT_CLASSIFICATION_LOG_DIR / "DISTRICT_FEATURE_IMPORTANCE_REPORT.md"
    archive_path = ARCHIVE_LOG_DIR / "DISTRICT_FEATURE_IMPORTANCE_REPORT.md"
    log_path.write_text(log_text, encoding="utf-8")
    archive_path.write_text(log_text, encoding="utf-8")
    print(f"[log] {log_path}")


def validate_outputs() -> None:
    required = [
        DISTRICT_CLASSIFICATION_OUTPUT_DIR / "district_feature_importance.csv",
        DISTRICT_CLASSIFICATION_FIGURE_DIR / "F20_district_feature_importance.png",
        DISTRICT_CLASSIFICATION_LOG_DIR / "DISTRICT_FEATURE_IMPORTANCE_REPORT.md",
        ARCHIVE_TABLE_DIR / "district_feature_importance.csv",
        ARCHIVE_FIGURE_DIR / "F20_district_feature_importance.png",
        ARCHIVE_LOG_DIR / "DISTRICT_FEATURE_IMPORTANCE_REPORT.md",
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise ValueError(f"Missing expected outputs: {missing}")


def main() -> None:
    ensure_dirs()
    districts, eps_m = build_district_frame()
    result, oof_auc, fold_aucs = compute_importance(districts)
    save_outputs(result)
    make_figure(result)
    write_log(result, districts, eps_m, oof_auc, fold_aucs)
    validate_outputs()

    print("\n[상권 feature importance]")
    print(f"districts={len(districts)}, positives={int(districts['y'].sum())}, eps_m={eps_m:.0f}")
    print(f"OOF PU-AUC={oof_auc:.4f}")
    print(
        result[["label_ko", "perm_imp", "rf_imp", "logreg_coef", "direction"]]
        .head(8)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()


