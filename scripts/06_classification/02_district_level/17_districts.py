# -*- coding: utf-8 -*-
"""
[상권 단위 재구성] 좌표 영역 → 상권(DBSCAN) 단위 + '스벅상권 카페는 U 아님' 검증

배경(팀 논의): 스벅이 이미 있는 '상권' 안의 카페는 음성(U)으로 볼 수 없다(강현).
  단 그 카페를 평가에서까지 빼면 문제만 쉬워져 성능이 무조건 오른다(승민).
→ 단위를 상권으로 바꾸고, 카페 단위는 '학습만 제외/평가도 제외'를 분리해 정직하게 비교.

상권 정의: 카페 좌표 DBSCAN(haversine, eps=150m, min_samples=5). 스벅 포함 상권 vs 미포함.
A) 상권 단위 PU: 상권을 표본으로(피처=구성 카페 평균), 스벅포함=양성/미포함=U. PU-AUC.
B) 카페 단위 PU 3버전:
   baseline  : U=전체 비스벅, 평가=전체
   B1_cleanU : U=비스벅상권 카페만(학습), 평가=전체        ← 공정(강현 효과 격리)
   B2_cleanE : U=비스벅상권 카페만, 평가도 스벅상권 비스벅 제외 ← 더 쉬운 문제(부풀림)
검증: spatial GroupKFold(5). base=RF(PU 실험 일관). 기록: logs/PU_LOG.md
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RNG = 42; T = 30
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    DISTRICT_CLASSIFICATION_FIGURE_DIR,
    DISTRICT_CLASSIFICATION_LOG_DIR,
    DISTRICT_CLASSIFICATION_MODEL_DIR,
    DISTRICT_CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
DATA = CLASSIFICATION_DATASET_PATH
OUT = DISTRICT_CLASSIFICATION_OUTPUT_DIR; LOG = DISTRICT_CLASSIFICATION_LOG_DIR
CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RNG)

def bag(Xtr, pos, neg, Xte, rs, T=T):
    acc = np.zeros(Xte.shape[0])
    for _ in range(T):
        u = rs.choice(neg, len(pos), replace=len(neg) < len(pos))
        idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
        acc += rf().fit(Xtr[idx], yy).predict_proba(Xte)[:, 1]
    return acc / T

def lift5(y, sc):
    n = max(1, int(len(sc)*0.05)); o = np.argsort(-sc)[:n]; return y[o].mean()/y.mean()

def assign_districts(df):
    eps = 150/6371000.0
    lab = DBSCAN(eps=eps, min_samples=5, metric="haversine").fit_predict(np.radians(df[["lat","lon"]].values))
    df = df.copy(); df["district"] = lab
    sb_dist = set(df.loc[(df.district != -1) & (df.s == 1), "district"].unique())
    df["in_sb_district"] = df["district"].isin(sb_dist) & (df["district"] != -1)
    return df, sb_dist

def part_A(df, sb_dist):
    d = df[df.district != -1]
    agg = d.groupby("district").agg({**{f: "mean" for f in CLF_FEATURES},
                                     "s": "max", "lat": "mean", "lon": "mean"})
    agg["y"] = agg["s"].astype(int)
    agg["blk"] = (np.floor(agg.lat/0.02).astype(int).astype(str) + "_" +
                  np.floor(agg.lon/0.02).astype(int).astype(str))
    X = agg[CLF_FEATURES].to_numpy(float); y = agg["y"].to_numpy(int); g = agg["blk"].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pos = np.where(y[tr] == 1)[0]; neg = np.where(y[tr] == 0)[0]
        oof[te] = bag(Xtr, pos, neg, Xte, np.random.RandomState(RNG))
    auc = roc_auc_score(y, oof)
    return {"setting": "A_상권단위", "n": len(y), "pos": int(y.sum()),
            "PU_AUC": auc, "lift@5%": lift5(y, oof)}

def part_B(df):
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    insb = df["in_sb_district"].to_numpy(bool); g = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, g))
    res = []
    for mode in ["baseline", "B1_cleanU", "B2_cleanE"]:
        oof = np.full(len(s), np.nan); evalmask = np.zeros(len(s), bool)
        for tr, te in splits:
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            pos = np.where(s[tr] == 1)[0]
            if mode == "baseline":
                neg = np.where(s[tr] == 0)[0]
            else:  # 비스벅상권 카페만 음성으로
                neg = np.where((s[tr] == 0) & (~insb[tr]))[0]
            oof[te] = bag(Xtr, pos, neg, Xte, np.random.RandomState(RNG))
            if mode == "B2_cleanE":   # 평가도 스벅상권 비스벅 제외
                keep = (s[te] == 1) | (~insb[te])
                evalmask[te[keep]] = True
            else:
                evalmask[te] = True
        m = evalmask
        res.append({"setting": mode, "n": int(m.sum()), "pos": int(s[m].sum()),
                    "PU_AUC": roc_auc_score(s[m], oof[m]), "lift@5%": lift5(s[m], oof[m])})
    return res

def main():
    df = pd.read_parquet(DATA)
    df, sb_dist = assign_districts(df)
    n_dist = df.loc[df.district != -1, "district"].nunique()
    n_sbd = len(sb_dist); n_nonsbd = n_dist - n_sbd
    excl = int(((df.s == 0) & df.in_sb_district).sum())
    print("=" * 74)
    print(f"[상권 DBSCAN] 상권 {n_dist} (스벅상권 {n_sbd}/비스벅상권 {n_nonsbd}), "
          f"노이즈카페 {(df.district==-1).sum()}")
    print(f"스벅상권 내 비스벅 카페(=U에서 제외 대상) {excl} / 비스벅 {int((df.s==0).sum())}")
    print("=" * 74)
    rows = [part_A(df, sb_dist)] + part_B(df)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res.to_csv(os.path.join(OUT, "district_comparison.csv"), index=False, encoding="utf-8-sig")
    write_log(res, n_dist, n_sbd, n_nonsbd, excl, int((df.s==0).sum()))
    print("\n[저장] outputs/district_comparison.csv | [기록] logs/PU_LOG.md")

def write_log(res, n_dist, n_sbd, n_nonsbd, excl, n_u):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["", "---", "", f"## 실행 @ {ts} — 상권(DBSCAN) 단위 재구성", "",
         f"상권 DBSCAN(eps150m,ms5): {n_dist}개(스벅상권 {n_sbd}/비스벅상권 {n_nonsbd}). "
         f"스벅상권 내 비스벅 {excl}/{n_u} 가 U제외 대상.", "",
         "| 설정 | n | pos | PU-AUC | lift@5% |", "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append("| %s | %d | %d | %.4f | %.3f |" % (r["setting"], r["n"], r["pos"], r["PU_AUC"], r["lift@5%"]))
    L += ["",
          "해석: B1_cleanU(학습만 정제, 전체평가)=강현 아이디어의 *공정* 효과. "
          "B2_cleanE(평가도 제외)=더 쉬운 문제(승민 우려, 부풀림). A_상권단위=단위 자체를 상권으로(집계).",
          "baseline 대비 B1이 오르면 '상권기반 U정제'가 실효, B2와 격차가 '평가 아티팩트' 크기.", ""]
    with open(os.path.join(LOG, "PU_LOG.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()



