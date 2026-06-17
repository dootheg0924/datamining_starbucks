# -*- coding: utf-8 -*-
"""
[PU 고도화 2] 비-RN Two-step:  A) Reliable Positive 확장   B) EM soft-weighted

1단계 점수는 baseline PU bagging 의 OOB 점수(각 U를, 그 U를 학습에 쓰지 않은
bag 들만 평균)로 산출 → 낙관 편향 차단.

  A. twostep_RP  : OOB 상위 RP_FRAC 의 train-U 를 의사양성으로 P 에 편입 →
                   (P∪RP) vs 나머지 U 로 PU bagging 재학습.
                   편입된 U = '아직 스벅 없지만 스벅형' 숨은 후보지(산출물).
  B. twostep_EM  : OOB 점수 w 를 U 의 양성확률로 보고 Elkan 복제 가중 학습
                   (U 를 +가중 w, -가중 1-w 로 동시 투입). U 를 버리지 않음.
비교 기준 baseline: pu_bagging(전체 U).  검증: spatial GroupKFold(5).
기록: classification/logs/PU_LOG.md (append)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = 42
T_BAGS = 30
RP_FRAC = 0.03      # train-U 중 상위 3%를 reliable positive 로
HERE = os.path.dirname(os.path.abspath(__file__))
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
LOGDIR = CLASSIFICATION_LOG_DIR
PULOG = os.path.join(LOGDIR, "PU_LOG.md")

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def rf(n=100):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=2, n_jobs=-1, random_state=RNG)

# --- baseline bagging + train-U OOB 점수 동시 산출 --------------------------
def bagging_with_oob(Xtr, pos_idx, U_idx, Xte, rs, T=T_BAGS, neg_pool=None):
    """반환: 테스트 점수, train-U OOB 점수(U_idx 정렬)."""
    if neg_pool is None:
        neg_pool = U_idx
    test = np.zeros(Xte.shape[0])
    Uall = Xtr[U_idx]
    oob_sum = np.zeros(len(U_idx)); oob_cnt = np.zeros(len(U_idx))
    pos_in_U = {u: i for i, u in enumerate(U_idx)}
    for _ in range(T):
        u = rs.choice(neg_pool, size=len(pos_idx), replace=len(neg_pool) < len(pos_idx))
        idx = np.concatenate([pos_idx, u]); yy = np.r_[np.ones(len(pos_idx)), np.zeros(len(u))]
        m = rf(100).fit(Xtr[idx], yy)
        test += m.predict_proba(Xte)[:, 1]
        pr = m.predict_proba(Uall)[:, 1]
        sampled = np.zeros(len(U_idx), dtype=bool)
        for uu in u:
            j = pos_in_U.get(uu)
            if j is not None: sampled[j] = True
        keep = ~sampled
        oob_sum[keep] += pr[keep]; oob_cnt[keep] += 1
    return test / T, oob_sum / np.maximum(oob_cnt, 1)

# --- A. Reliable Positive 확장 ---------------------------------------------
def twostep_RP(Xtr, pos_idx, U_idx, Xte, rs, U_oob):
    n_rp = max(1, int(len(U_idx) * RP_FRAC))
    rp_local = np.argsort(-U_oob)[:n_rp]      # OOB 상위
    rp_idx = U_idx[rp_local]
    rest = np.setdiff1d(U_idx, rp_idx)
    pos_aug = np.concatenate([pos_idx, rp_idx])
    test, _ = bagging_with_oob(Xtr, pos_aug, rest, Xte, rs, neg_pool=rest)
    return test, rp_idx

# --- B. EM soft-weighted (Elkan 복제) --------------------------------------
def twostep_EM(Xtr, pos_idx, U_idx, Xte, w):
    w = np.clip(w, 1e-3, 1 - 1e-3)
    Xp, Xu = Xtr[pos_idx], Xtr[U_idx]
    X = np.vstack([Xp, Xu, Xu])
    y = np.r_[np.ones(len(pos_idx)), np.ones(len(U_idx)), np.zeros(len(U_idx))]
    sw = np.r_[np.ones(len(pos_idx)), w, 1 - w]
    m = rf(300).fit(X, y, sample_weight=sw)
    return m.predict_proba(Xte)[:, 1]

def lift_at(y, sc, frac=0.05):
    n = max(1, int(len(sc) * frac)); o = np.argsort(-sc)[:n]
    return y[o].mean() / y.mean()
def prec_at(y, sc, k):
    o = np.argsort(-sc)[:k]; return y[o].sum() / k

METHODS = ["pu_bagging", "twostep_RP", "twostep_EM"]

def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    groups = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, groups))

    oof = {m: np.full(len(s), np.nan) for m in METHODS}
    perfold = {m: [] for m in METHODS}
    rp_global = []   # 숨은 후보지(의사양성)로 뽑힌 원본 행 인덱스

    for fold, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        str_ = s[tr]; rs = np.random.RandomState(RNG + fold)
        pos_tr = np.where(str_ == 1)[0]; U_tr = np.where(str_ == 0)[0]

        base_test, U_oob = bagging_with_oob(Xtr, pos_tr, U_tr, Xte, rs)
        oof["pu_bagging"][te] = base_test

        rp_test, rp_local = twostep_RP(Xtr, pos_tr, U_tr, Xte, np.random.RandomState(RNG + 100 + fold), U_oob)
        oof["twostep_RP"][te] = rp_test
        rp_global.extend(tr[rp_local].tolist())   # 원본 인덱스로 환원

        oof["twostep_EM"][te] = twostep_EM(Xtr, pos_tr, U_tr, Xte, U_oob)

        for m in METHODS:
            if 0 < s[te].sum() < len(te):
                perfold[m].append(roc_auc_score(s[te], oof[m][te]))
        print(f"[fold {fold}] RP={len(rp_local)}  done")

    rows = []
    for m in METHODS:
        pf = np.array(perfold[m])
        rows.append({"method": m, "PU_AUC": roc_auc_score(s, oof[m]),
                     "AP": average_precision_score(s, oof[m]),
                     "fold_AUC_mean": pf.mean(), "fold_AUC_std": pf.std(),
                     "lift@5%": lift_at(s, oof[m]), "prec@681": prec_at(s, oof[m], int(s.sum()))})
    res = pd.DataFrame(rows).sort_values("PU_AUC", ascending=False)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("[PU 고도화 2] 비-RN Two-step (A:RP, B:EM) vs baseline")
    print("=" * 78)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # A 산출물: 숨은 후보지(의사양성 U) — 시군구 분포 + 예시
    rp = df.iloc[sorted(set(rp_global))]
    rp_only = rp[rp["is_starbucks"] == 0]
    print("\n[A 산출물] Reliable Positive 로 뽑힌 '숨은 스벅형' 카페:", len(rp_only), "곳")
    print("시군구 top10:")
    print(rp_only["sigungu"].value_counts().head(10).to_string())

    res.to_csv(os.path.join(OUTDIR, "pu_twostep_pos_comparison.csv"), index=False, encoding="utf-8-sig")
    rp_only[["name", "brand", "sigungu", "lat", "lon"]].to_csv(
        os.path.join(OUTDIR, "hidden_candidates_RP.csv"), index=False, encoding="utf-8-sig")
    write_log(res, rp_only)
    print("\n[저장] outputs/pu_twostep_pos_comparison.csv, hidden_candidates_RP.csv")
    print("[기록] logs/PU_LOG.md")

def write_log(res, rp_only):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["", "---", "", f"## 실행 @ {ts}  — 비-RN Two-step (A:RP, B:EM)", "",
         f"설정: T_BAGS={T_BAGS}, RP_FRAC={RP_FRAC}, 1단계 점수=baseline bagging OOB", "",
         "### 방법 비교 (PU-AUC 내림차순)", "",
         "| 방법 | PU-AUC | fold AUC(평균±std) | AP | lift@5% | prec@681 |",
         "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append("| %s | %.4f | %.4f±%.3f | %.4f | %.3f | %.4f |" % (
            r["method"], r["PU_AUC"], r["fold_AUC_mean"], r["fold_AUC_std"],
            r["AP"], r["lift@5%"], r["prec@681"]))
    L += ["", "### A 산출물 — Reliable Positive(숨은 스벅형 카페) 시군구 분포 top10", ""]
    vc = rp_only["sigungu"].value_counts().head(10)
    L += ["| 시군구 | 수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vc.items()]
    L += ["", f"(총 {len(rp_only)}곳, outputs/hidden_candidates_RP.csv)", "",
          "메모: (A/B가 baseline 대비 어떤지, RP 후보지 해석은 여기에)", ""]
    with open(PULOG, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

