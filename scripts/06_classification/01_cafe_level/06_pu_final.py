# -*- coding: utf-8 -*-
"""
[PU 최종 확정] PU bagging (base=GBM) + 적합도 점수 보정

모델주/PU실험 결론: PU bagging 이 최선, base 는 GBM≈RF≈XGB(동률). GBM 채택.
산출:
  1) spatial-CV OOF 적합도 점수(누수 없는 정직한 점수) — 카페별
       - score_raw  : bag 평균 확률 [0,1]
       - score_pct  : 전체 카페 중 백분위(0~100) — 해석 친화
       - score_cal  : Elkan c-보정 확률 (c=양성 평균 OOF), 클립 [0,1]
  2) 전체데이터 앙상블(서울대 후보지 등 신규 입력 채점용) → joblib 저장
검증: spatial GroupKFold(5). 기록: logs/PU_LOG.md (최종 모델 카드)
"""
import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = 42
T_BAGS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
DATA = CLASSIFICATION_DATASET_PATH
OUTDIR = CLASSIFICATION_OUTPUT_DIR
MODELDIR = CLASSIFICATION_MODEL_DIR
LOGDIR = CLASSIFICATION_LOG_DIR
PULOG = os.path.join(LOGDIR, "PU_LOG.md")

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def gbm():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RNG)

def fit_bag(Xtr, pos_idx, neg_idx, rs, T=T_BAGS):
    """PU bagging: 학습된 base 추정기 리스트 반환."""
    models = []
    for _ in range(T):
        u = rs.choice(neg_idx, size=len(pos_idx), replace=len(neg_idx) < len(pos_idx))
        idx = np.concatenate([pos_idx, u]); yy = np.r_[np.ones(len(pos_idx)), np.zeros(len(u))]
        models.append(gbm().fit(Xtr[idx], yy))
    return models

def bag_score(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    groups = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, groups))

    # 1) spatial-CV OOF 점수 (정직)
    oof = np.full(len(s), np.nan); perfold = []
    for fold, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        str_ = s[tr]; rs = np.random.RandomState(RNG + fold)
        pos = np.where(str_ == 1)[0]; neg = np.where(str_ == 0)[0]
        models = fit_bag(Xtr, pos, neg, rs)
        oof[te] = bag_score(models, Xte)
        perfold.append(roc_auc_score(s[te], oof[te]))
        print(f"[fold {fold}] AUC={perfold[-1]:.4f}")

    auc = roc_auc_score(s, oof); ap = average_precision_score(s, oof)
    pf = np.array(perfold)
    # 보정 점수
    c = oof[s == 1].mean()                       # Elkan c = 양성 평균 OOF
    score_cal = np.clip(oof / max(c, 1e-6), 0, 1)
    score_pct = pd.Series(oof).rank(pct=True).to_numpy() * 100

    print("\n[PU 최종] base=GBM, PU bagging  | OOF PU-AUC=%.4f  AP=%.4f  fold=%.4f±%.3f"
          % (auc, ap, pf.mean(), pf.std()))
    print("Elkan c (양성 평균 OOF) = %.4f  -> score_cal = score_raw / c" % c)

    # 점수 저장
    out = df[["name", "brand", "is_starbucks", "sigungu", "lat", "lon", "spatial_block"]].copy()
    out["score_raw"] = oof
    out["score_pct"] = score_pct
    out["score_cal"] = score_cal
    out.to_parquet(os.path.join(OUTDIR, "final_pu_scores.parquet"), index=False)

    # 2) 전체데이터 앙상블(신규 입력 채점용) 저장
    sc_full = StandardScaler().fit(X)
    pos = np.where(s == 1)[0]; neg = np.where(s == 0)[0]
    models_full = fit_bag(sc_full.transform(X), pos, neg, np.random.RandomState(RNG))
    joblib.dump({"scaler": sc_full, "models": models_full, "features": CLF_FEATURES,
                 "elkan_c": float(c), "ref_scores": oof},
                os.path.join(MODELDIR, "pu_bagging_gbm.joblib"))

    # 점수 분포 확인(상위 비스벅 = 숨은 후보)
    top = out[(out.is_starbucks == 0)].sort_values("score_raw", ascending=False).head(15)
    print("\n[비스벅 카페 적합도 top15 (숨은 스벅형)]")
    print(top[["name", "sigungu", "score_raw", "score_pct"]].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    write_log(auc, ap, pf, c, out)
    print("\n[저장] outputs/final_pu_scores.parquet, models/pu_bagging_gbm.joblib")
    print("[기록] logs/PU_LOG.md (최종 모델 카드)")

def write_log(auc, ap, pf, c, out):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sb = out[out.is_starbucks == 1]["score_raw"]
    nb = out[out.is_starbucks == 0]["score_raw"]
    L = ["", "---", "", f"## 최종 PU 모델 카드 @ {ts}", "",
         "- **방법**: PU bagging (전체 U, 1:1 부트스트랩 ×%d)" % T_BAGS,
         "- **base estimator**: GradientBoosting(n=150, depth=3, lr=0.1)",
         "- **검증**: spatial GroupKFold(5)",
         "- **피처**: raw 17개 (dist_nearest_starbucks 제외, 누수 차단)", "",
         "### 성능", "",
         f"- OOF PU-AUC: **{auc:.4f}** | AP: {ap:.4f} | fold AUC: {pf.mean():.4f}±{pf.std():.3f}",
         f"- Elkan c (양성 평균 OOF 점수): {c:.4f}", "",
         "### 적합도 점수 정의", "",
         "- `score_raw`: bag 평균 확률 [0,1]",
         "- `score_pct`: 전체 카페 대비 백분위 (서울대 후보 해석용)",
         "- `score_cal`: Elkan 보정 확률 = score_raw / c",
         f"- 스벅 평균 score_raw={sb.mean():.3f} vs 비스벅 평균={nb.mean():.3f}", "",
         "### 산출물", "",
         "- outputs/final_pu_scores.parquet (카페별 3종 점수)",
         "- models/pu_bagging_gbm.joblib (전체데이터 앙상블 — 서울대 후보 채점용)", "",
         "메모: 향후 base를 LogReg로 바꾼 '해석 트랙' 점수도 별도 생성 예정.", ""]
    with open(PULOG, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()


