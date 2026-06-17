# -*- coding: utf-8 -*-
"""
[상권 정의 고도화] 여러 상권 정의 → 상권 단위 PU-AUC 비교 + 민감도/안정성

상권 정의 후보:
  - DBSCAN 고정 eps (100/150/250m, ms=5)
  - DBSCAN 자동 eps (k거리 2차미분 elbow — 클러스터링 단계와 동일 방법론)
  - HDBSCAN (가변 밀도, min_cluster_size 10/20)
  - + 노이즈 흡수(고립 카페를 최근접 상권에 편입) 변형
각 정의로 상권 단위 PU(피처=구성카페 평균, 라벨=스벅 1개+, spatial CV) → PU-AUC.
목적: 더 원리적인 상권 정의 채택 + "상권-PU 성능이 정의 방식에 robust" 입증.
기록: outputs/district_def_comparison.csv, logs/PU_LOG.md
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RNG = 42; T = 30
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
OUT = CLASSIFICATION_OUTPUT_DIR; LOG = CLASSIFICATION_LOG_DIR
CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def auto_eps(XY_rad, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(XY_rad)
    d, _ = nn.kneighbors(XY_rad)
    kd = np.sort(d[:, k-1])[::-1]
    d2 = np.diff(np.diff(kd)); lo, hi = int(len(kd)*0.05), int(len(kd)*0.6)
    idx = lo + int(np.argmax(d2[lo:hi])) + 1
    return float(kd[idx])

def absorb_noise(df, XY_rad):
    m = df["d"].values != -1
    if m.all() or not m.any(): return df
    nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(XY_rad[m])
    _, idx = nn.kneighbors(XY_rad[~m])
    lab = df["d"].values.copy(); lab[~m] = df["d"].values[m][idx[:, 0]]
    df = df.copy(); df["d"] = lab; return df

def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RNG)

def district_pu(df, labels, XY_rad, absorb=False, boot=False):
    df = df.copy(); df["d"] = labels
    if absorb: df = absorb_noise(df, XY_rad)
    d = df[df["d"] != -1]
    cov = len(d) / len(df)
    agg = d.groupby("d").agg({**{f: "mean" for f in CLF_FEATURES},
                              "s": "max", "lat": "mean", "lon": "mean"})
    y = agg["s"].astype(int).to_numpy(); X = agg[CLF_FEATURES].to_numpy(float)
    g = (np.floor(agg.lat/0.02).astype(int).astype(str) + "_" +
         np.floor(agg.lon/0.02).astype(int).astype(str)).to_numpy()
    if y.sum() < 5 or (y == 0).sum() < 5:
        return dict(n=len(y), pos=int(y.sum()), cov=cov, PU_AUC=np.nan, ci="-")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pos = np.where(y[tr] == 1)[0]; neg = np.where(y[tr] == 0)[0]
        rs = np.random.RandomState(RNG); acc = np.zeros(len(te))
        for _ in range(T):
            u = rs.choice(neg, len(pos), replace=len(neg) < len(pos))
            ii = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
            acc += rf().fit(Xtr[ii], yy).predict_proba(Xte)[:, 1]
        oof[te] = acc / T
    auc = roc_auc_score(y, oof)
    ci = "-"
    if boot:
        rs = np.random.RandomState(0); bs = []
        for _ in range(1000):
            ix = rs.randint(0, len(y), len(y))
            if 0 < y[ix].sum() < len(ix): bs.append(roc_auc_score(y[ix], oof[ix]))
        ci = f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]"
    return dict(n=len(y), pos=int(y.sum()), cov=cov, PU_AUC=auc, ci=ci)

def main():
    df = pd.read_parquet(DATA)
    XY = np.radians(df[["lat", "lon"]].values)
    R = 6371000.0
    defs = []
    for em in [100, 150, 250]:
        defs.append((f"DBSCAN {em}m/ms5", DBSCAN(eps=em/R, min_samples=5, metric="haversine").fit_predict(XY), False))
    ae = auto_eps(XY, k=5)
    defs.append((f"DBSCAN auto-eps({ae*R:.0f}m)/ms5",
                 DBSCAN(eps=ae, min_samples=5, metric="haversine").fit_predict(XY), False))
    for mcs in [10, 20]:
        defs.append((f"HDBSCAN mcs{mcs}",
                     HDBSCAN(min_cluster_size=mcs, metric="haversine").fit_predict(XY), False))

    rows = []
    for name, lab, _ in defs:
        r = district_pu(df, lab, XY, absorb=False); r["정의"] = name; r["노이즈흡수"] = "X"; rows.append(r)
    # 노이즈 흡수 변형: DBSCAN 150m, HDBSCAN mcs20
    for name, lab in [("DBSCAN 150m/ms5", defs[1][1]), ("HDBSCAN mcs20", defs[-1][1])]:
        r = district_pu(df, lab, XY, absorb=True); r["정의"] = name; r["노이즈흡수"] = "O"; rows.append(r)

    res = pd.DataFrame(rows)[["정의", "노이즈흡수", "n", "pos", "cov", "PU_AUC", "ci"]]
    res = res.sort_values("PU_AUC", ascending=False)
    pd.set_option("display.width", 200)
    print("=" * 80); print("[상권 정의 고도화] 정의별 상권-단위 PU-AUC (cov=상권소속 카페비율)"); print("=" * 80)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 최고 정의 bootstrap CI
    best = res.iloc[0]
    print(f"\n최고: {best['정의']}(흡수 {best['노이즈흡수']}) PU-AUC={best['PU_AUC']:.4f}")
    # 재계산 with boot for the best
    res.to_csv(os.path.join(OUT, "district_def_comparison.csv"), index=False, encoding="utf-8-sig")
    write_log(res, ae*R)
    print("\n[저장] outputs/district_def_comparison.csv | [기록] logs/PU_LOG.md")

def write_log(res, ae_m):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["", "---", "", f"## 실행 @ {ts} — 상권 정의 고도화 비교", "",
         f"auto-eps(k거리 elbow) = {ae_m:.0f}m. 정의별 상권-단위 PU-AUC:", "",
         "| 정의 | 노이즈흡수 | 상권수 | 스벅상권 | 카페커버 | PU-AUC |", "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append("| %s | %s | %d | %d | %.2f | %.4f |" % (
            r["정의"], r["노이즈흡수"], r["n"], r["pos"], r["cov"], r["PU_AUC"]))
    L += ["", "해석: 상권 정의(DBSCAN 고정/자동eps/HDBSCAN/노이즈흡수)를 바꿔도 상권-PU-AUC가 "
          "0.7대로 안정하면 → 상권단위 우월성이 특정 정의 산물 아님(robust). 최적 정의 채택.", ""]
    with open(os.path.join(LOG, "PU_LOG.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

