# -*- coding: utf-8 -*-
"""
[PU 고도화 3] 반복적 Two-step (S-EM, Liu 2014 계열) — RN 재정제로 경계 좁히기

각 spatial fold 내부에서 RN을 반복 정제:
  iter1: neg_pool = 전체 U → PU-bagging(baseline)
  iterk: U 전체를 현재 모델로 채점 → 하위 q분위를 RN으로 재식별 → 다음 학습은 RN으로
공정 평가: 반복은 '학습 절차'일 뿐, PU-AUC는 매 반복 **전체 test U**로 동일 산출.
모드:
  A 고정분위(q=0.6 유지)  — RN 수렴 관찰
  B 점감분위(0.6→0.35)    — 경계 능동적 축소
기록: 반복별 PU-AUC/lift/|RN|/Jaccard → outputs/pu_iterative.csv, figures/F10_iterative.png
검증: GroupKFold(5). base=RF (PU 실험과 동일).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = 42
T_BAGS = 20
K_ITER = 6
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
OUT = CLASSIFICATION_OUTPUT_DIR; FIG = CLASSIFICATION_FIGURE_DIR

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RNG)

def bag(Xtr, pos, neg_pool, Xte, Xu, rs, T=T_BAGS):
    """PU-bagging. 반환: test 점수, U_train 점수(앙상블)."""
    te = np.zeros(Xte.shape[0]); us = np.zeros(Xu.shape[0])
    for _ in range(T):
        u = rs.choice(neg_pool, size=len(pos), replace=len(neg_pool) < len(pos))
        idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
        m = rf().fit(Xtr[idx], yy)
        te += m.predict_proba(Xte)[:, 1]; us += m.predict_proba(Xu)[:, 1]
    return te / T, us / T

def q_schedule(mode):
    if mode == "A_fixed":
        return [0.6] * K_ITER
    return list(np.linspace(0.6, 0.35, K_ITER))   # B_shrink

def run_mode(df, mode):
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    g = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, g))
    qs = q_schedule(mode)
    auc = np.zeros((5, K_ITER)); lift = np.zeros((5, K_ITER))
    rn_size = np.zeros((5, K_ITER)); jac = np.full((5, K_ITER), np.nan)

    for fi, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        s_tr, s_te = s[tr], s[te]
        pos = np.where(s_tr == 1)[0]; U = np.where(s_tr == 0)[0]
        Xu = Xtr[U]
        rs = np.random.RandomState(RNG + fi)
        neg_pool = U.copy(); prev_rn = None
        for k in range(K_ITER):
            te_score, u_score = bag(Xtr, pos, neg_pool, Xte, Xu, rs)
            auc[fi, k] = roc_auc_score(s_te, te_score)
            n = max(1, int(len(te) * 0.05)); o = np.argsort(-te_score)[:n]
            lift[fi, k] = s_te[o].mean() / s_te.mean()
            # RN 재식별: U 전체 점수 하위 q분위
            thr = np.quantile(u_score, qs[k])
            rn_local = np.where(u_score < thr)[0]
            rn = U[rn_local]
            rn_size[fi, k] = len(rn)
            if prev_rn is not None:
                a, b = set(rn.tolist()), set(prev_rn.tolist())
                jac[fi, k] = len(a & b) / len(a | b) if (a | b) else 0
            prev_rn = rn
            neg_pool = rn if len(rn) >= len(pos) else U  # RN이 너무 작으면 전체로 보호
        print(f"  [{mode} fold{fi}] AUC by iter: " + " ".join(f"{auc[fi,k]:.3f}" for k in range(K_ITER)))
    rows = []
    for k in range(K_ITER):
        rows.append({"mode": mode, "iter": k + 1, "q": round(qs[k], 3),
                     "PU_AUC": auc[:, k].mean(), "AUC_std": auc[:, k].std(),
                     "lift@5%": lift[:, k].mean(), "RN_size": rn_size[:, k].mean(),
                     "jaccard_prev": np.nanmean(jac[:, k])})
    return pd.DataFrame(rows)

def main():
    df = pd.read_parquet(DATA)
    print("=" * 72); print("[반복 Two-step] 모드 A(고정) / B(점감) — 전체 test 공정 평가"); print("=" * 72)
    res = pd.concat([run_mode(df, "A_fixed"), run_mode(df, "B_shrink")], ignore_index=True)
    pd.set_option("display.width", 200)
    print("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res.to_csv(os.path.join(OUT, "pu_iterative.csv"), index=False, encoding="utf-8-sig")
    fig(res)
    print("\n[저장] outputs/pu_iterative.csv, figures/F10_iterative.png")

def fig(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
        try: plt.rcParams["font.family"] = f; break
        except Exception: pass
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, c in [("A_fixed", "#1f7a3f"), ("B_shrink", "#c0392b")]:
        d = res[res["mode"] == mode]
        ax[0].plot(d["iter"], d["PU_AUC"], "o-", color=c, label=mode)
        ax[1].plot(d["iter"], d["lift@5%"], "o-", color=c, label=mode)
    base = res[(res["mode"] == "A_fixed") & (res["iter"] == 1)]["PU_AUC"].iloc[0]
    ax[0].axhline(base, ls="--", c="gray", label=f"baseline(iter1)={base:.3f}")
    ax[0].set_title("F10a. 반복별 PU-AUC (전체 test 공정평가)"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("PU-AUC"); ax[0].legend(fontsize=8)
    ax[1].set_title("F10b. 반복별 lift@5%"); ax[1].set_xlabel("iteration"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F10_iterative.png"), dpi=130)

if __name__ == "__main__":
    main()


