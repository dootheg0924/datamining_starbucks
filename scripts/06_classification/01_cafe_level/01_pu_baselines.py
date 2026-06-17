# -*- coding: utf-8 -*-
"""
[1단계] 라벨/샘플링 4전략 비교 (공간 블록 CV)

목적: "미입점을 음성으로 강제하는 전략(①~③) vs PU(④)" 를 동일 조건에서 비교.
전략
  ① weighted     : U=negative, RandomForest(class_weight='balanced')
  ② undersample  : U에서 |P| 만큼 랜덤 추출(1:1), RF
  ③ smote        : SMOTE 로 P 합성 오버샘플(균형), RF
  ④ pu_bagging   : Mordelet&Vert PU bagging (P 전체 + U 부트스트랩 앙상블)
  ⑤ pu_elkan     : Elkan-Noto (s 분류 후 c=P(s=1|y=1) 보정)

평가: U에 진짜 음성이 없으므로 PU 친화 지표 사용.
  - PU-AUC (P vs U): U를 음성처럼 둔 ROC-AUC. 절대값은 과소추정이나 전략 간 상대비교 유효.
  - AP (average precision)
  - precision@681 / recall@681 : 상위 681개(=실제 P 수) 안에 held-out P 가 얼마나 들어오나
  - lift@5% : 상위 5% 안의 P 밀도 / 전체 P 밀도
검증: spatial_block 기준 GroupKFold(5). OOF 점수로 집계.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RNG = 42
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
os.makedirs(OUTDIR, exist_ok=True)

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def rf(**kw):
    p = dict(n_estimators=300, n_jobs=-1, random_state=RNG, min_samples_leaf=2)
    p.update(kw)
    return RandomForestClassifier(**p)

# --- 전략별: (Xtr, str, Xte) -> 테스트 점수 ---------------------------------
def strat_weighted(Xtr, s, Xte, rs):
    m = rf(class_weight="balanced").fit(Xtr, s)
    return m.predict_proba(Xte)[:, 1]

def strat_undersample(Xtr, s, Xte, rs):
    pos = np.where(s == 1)[0]; neg = np.where(s == 0)[0]
    neg_s = rs.choice(neg, size=len(pos), replace=False)
    idx = np.concatenate([pos, neg_s])
    m = rf().fit(Xtr[idx], s[idx])
    return m.predict_proba(Xte)[:, 1]

def strat_smote(Xtr, s, Xte, rs):
    Xr, yr = SMOTE(random_state=RNG, k_neighbors=5).fit_resample(Xtr, s)
    m = rf().fit(Xr, yr)
    return m.predict_proba(Xte)[:, 1]

def strat_pu_bagging(Xtr, s, Xte, rs, T=30):
    pos = np.where(s == 1)[0]; neg = np.where(s == 0)[0]
    acc = np.zeros(Xte.shape[0])
    for t in range(T):
        u_s = rs.choice(neg, size=len(pos), replace=True)  # U 부트스트랩
        idx = np.concatenate([pos, u_s])
        yy = np.concatenate([np.ones(len(pos)), np.zeros(len(u_s))])
        m = rf(n_estimators=100).fit(Xtr[idx], yy)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / T

def strat_pu_elkan(Xtr, s, Xte, rs):
    # s(labeled vs unlabeled) 분류 후 c = 라벨양성에서의 평균 g(x) 로 보정
    m = rf().fit(Xtr, s)
    pos = np.where(s == 1)[0]
    c = m.predict_proba(Xtr[pos])[:, 1].mean()
    c = max(c, 1e-6)
    return np.clip(m.predict_proba(Xte)[:, 1] / c, 0, 1)

STRATS = {
    "weighted": strat_weighted,
    "undersample": strat_undersample,
    "smote": strat_smote,
    "pu_bagging": strat_pu_bagging,
    "pu_elkan": strat_pu_elkan,
}

# --- 평가 -------------------------------------------------------------------
def lift_at(s_true, score, frac=0.05):
    n = max(1, int(len(score) * frac))
    order = np.argsort(-score)[:n]
    return (s_true[order].mean()) / s_true.mean()

def prec_rec_at(s_true, score, k):
    order = np.argsort(-score)[:k]
    tp = s_true[order].sum()
    return tp / k, tp / s_true.sum()

def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(dtype=float)
    s = df["s"].to_numpy(dtype=int)
    groups = df["spatial_block"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    oof = {k: np.full(len(df), np.nan) for k in STRATS}

    for fold, (tr, te) in enumerate(gkf.split(X, s, groups)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        str_tr = s[tr]
        rs = np.random.RandomState(RNG + fold)
        ptr, pte = str_tr.sum(), s[te].sum()
        print(f"[fold {fold}] train P={ptr} U={len(tr)-ptr} | test P={pte} U={len(te)-pte}")
        for name, fn in STRATS.items():
            oof[name][te] = fn(Xtr, str_tr, Xte, rs)

    # 집계
    rows = []
    for name in STRATS:
        sc = oof[name]
        rows.append({
            "strategy": name,
            "PU_AUC": roc_auc_score(s, sc),
            "AP": average_precision_score(s, sc),
            "prec@681": prec_rec_at(s, sc, int(s.sum()))[0],
            "recall@681": prec_rec_at(s, sc, int(s.sum()))[1],
            "lift@5%": lift_at(s, sc, 0.05),
        })
    res = pd.DataFrame(rows).sort_values("PU_AUC", ascending=False)
    pd.set_option("display.width", 160)
    print("\n" + "=" * 78)
    print("[1단계] 4(+1)전략 비교 — 공간 블록 GroupKFold(5), OOF 집계")
    print("=" * 78)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 저장: 점수 + 요약
    oof_df = df[["name", "brand", "is_starbucks", "sigungu", "spatial_block"]].copy()
    for k in STRATS:
        oof_df[f"score_{k}"] = oof[k]
    oof_df.to_parquet(os.path.join(OUTDIR, "oof_scores.parquet"), index=False)
    res.to_csv(os.path.join(OUTDIR, "strategy_comparison.csv"), index=False, encoding="utf-8-sig")
    print("\n[저장] outputs/oof_scores.parquet, outputs/strategy_comparison.csv")

if __name__ == "__main__":
    main()


