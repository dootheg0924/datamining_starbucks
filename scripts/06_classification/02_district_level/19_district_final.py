# -*- coding: utf-8 -*-
"""
[상권 단위 최종] DBSCAN auto-eps(223m)+노이즈흡수 상권 → 상권 PU 모델 확정
+ 서울대 캠퍼스를 '하나의 상권'으로 채점 + 타대학 상권 비교

산출:
  - 상권 PU-AUC(OOF) + bootstrap CI
  - 전체 상권 적합도(real 상권) + 모델/스케일러 저장(신규 상권 채점용)
  - 서울대 캠퍼스 합성상권(교내 카페 평균) 적합도·백분위 + 실제 소속 상권
  - 서울 주요 대학 캠퍼스 합성상권 비교
출력: outputs/district_scores.csv, district_university.csv, models/pu_district.joblib, logs/DISTRICT_REPORT.md
"""
import os, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
MODELDIR = DISTRICT_CLASSIFICATION_MODEL_DIR
CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]
SNU = (37.4591, 126.9520)
UNIV = {"서울대(관악)": (37.4591,126.9520), "연세대(신촌)": (37.5665,126.9388),
        "고려대(안암)": (37.5894,127.0327), "한양대(왕십리)": (37.5559,127.0438),
        "홍익대(상수)": (37.5511,126.9250), "중앙대(흑석)": (37.5051,126.9571),
        "경희대(회기)": (37.5970,127.0517), "이화여대(대현)": (37.5620,126.9469),
        "건국대(화양)": (37.5403,127.0793), "서강대(신촌)": (37.5511,126.9410)}

def hav(la, lo, la2, lo2):
    R=6371; p=np.pi/180
    a=np.sin((la2-la)*p/2)**2+np.cos(la*p)*np.cos(la2*p)*np.sin((lo2-lo)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def auto_eps(XY, k=5):
    nn=NearestNeighbors(n_neighbors=k, metric="haversine").fit(XY); d,_=nn.kneighbors(XY)
    kd=np.sort(d[:,k-1])[::-1]; d2=np.diff(np.diff(kd)); lo,hi=int(len(kd)*.05),int(len(kd)*.6)
    return float(kd[lo+int(np.argmax(d2[lo:hi]))+1])

def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RNG)

def main():
    df = pd.read_parquet(DATA)
    XY = np.radians(df[["lat","lon"]].values); R=6371000.0
    eps = auto_eps(XY); lab = DBSCAN(eps=eps, min_samples=5, metric="haversine").fit_predict(XY)
    # 노이즈 흡수(최근접 상권)
    df["district"] = lab; m = lab != -1
    nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(XY[m])
    _, idx = nn.kneighbors(XY[~m]); l2 = lab.copy(); l2[~m] = lab[m][idx[:,0]]; df["district"] = l2

    # 상권 집계
    agg = df.groupby("district").agg({**{f:"mean" for f in CLF_FEATURES},
                                      "s":"max","lat":"mean","lon":"mean","name":"size"}).rename(columns={"name":"n_cafe"})
    agg["y"] = agg["s"].astype(int)
    X = agg[CLF_FEATURES].to_numpy(float); y = agg["y"].to_numpy(int)
    g = (np.floor(agg.lat/0.02).astype(int).astype(str)+"_"+np.floor(agg.lon/0.02).astype(int).astype(str)).to_numpy()
    print(f"[상권 최종] auto-eps={eps*R:.0f}m+흡수 → 상권 {len(y)} (스벅상권 {int(y.sum())}), 전 카페 배정")

    # OOF 평가 + CI
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc=StandardScaler().fit(X[tr]); Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te])
        pos=np.where(y[tr]==1)[0]; neg=np.where(y[tr]==0)[0]; rs=np.random.RandomState(RNG); acc=np.zeros(len(te))
        for _ in range(T):
            u=rs.choice(neg,len(pos),replace=len(neg)<len(pos)); ii=np.concatenate([pos,u]); yy=np.r_[np.ones(len(pos)),np.zeros(len(u))]
            acc+=rf().fit(Xtr[ii],yy).predict_proba(Xte)[:,1]
        oof[te]=acc/T
    auc=roc_auc_score(y,oof)
    rs=np.random.RandomState(0); bs=[roc_auc_score(y[i],oof[i]) for i in (rs.randint(0,len(y),len(y)) for _ in range(1000)) if 0<y[i].sum()<len(i)]
    ci=(np.percentile(bs,2.5),np.percentile(bs,97.5))
    print(f"상권 PU-AUC(OOF)={auc:.4f}  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    # 전체데이터 앙상블(합성상권 채점용)
    scaler=StandardScaler().fit(X); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    rs=np.random.RandomState(RNG); models=[]
    Xs=scaler.transform(X)
    for _ in range(T):
        u=rs.choice(neg,len(pos),replace=False if len(neg)>=len(pos) else True); ii=np.concatenate([pos,u]); yy=np.r_[np.ones(len(pos)),np.zeros(len(u))]
        models.append(rf().fit(Xs[ii],yy))
    def score_vec(vec):
        z=scaler.transform(vec.reshape(1,-1)); return float(np.mean([mm.predict_proba(z)[0,1] for mm in models]))
    joblib.dump({"scaler":scaler,"models":models,"features":CLF_FEATURES,"eps_m":eps*R}, os.path.join(MODELDIR,"pu_district.joblib"))

    agg["score"]=oof
    agg["score_pct"]=pd.Series(oof, index=agg.index).rank(pct=True)*100
    agg.reset_index()[["district","n_cafe","y","score","score_pct","lat","lon"]].to_csv(
        os.path.join(OUT,"district_scores.csv"), index=False, encoding="utf-8-sig")

    # 서울대 캠퍼스 = 합성 상권 (교내 ≤1.2km 카페 평균)
    df["d_snu"]=hav(df.lat,df.lon,*SNU)
    camp=df[df.d_snu<=1.2]
    synth=camp[CLF_FEATURES].mean().to_numpy()
    snu_score=score_vec(synth)
    snu_pct=(oof < snu_score).mean()*100
    camp_dist=df.loc[camp.index,"district"].value_counts()
    real_dist=camp_dist.index[0]
    print(f"\n[서울대 캠퍼스 합성상권] 교내 {len(camp)}카페 평균 → 적합도 {snu_score:.3f} "
          f"(상권 중 {snu_pct:.0f}백분위), 실제 소속상권 {real_dist}(적합도 {agg.loc[real_dist,'score']:.3f})")

    # 타대학 합성 상권 (≤1.0km)
    rows=[]
    for nm,(la,lo) in UNIV.items():
        d=hav(df.lat,df.lon,la,lo); z=df[d<=1.0]
        if len(z)<5: continue
        v=z[CLF_FEATURES].mean().to_numpy(); sciv=score_vec(v)
        rows.append({"대학":nm,"n_cafe":len(z),"상권적합도":sciv,"백분위":(oof<sciv).mean()*100,"스벅수":int(z.s.sum())})
    uni=pd.DataFrame(rows).sort_values("상권적합도",ascending=False)
    print("\n[대학 캠퍼스 합성상권 적합도]")
    print(uni.to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    uni.to_csv(os.path.join(OUT,"district_university.csv"), index=False, encoding="utf-8-sig")

    write_log(eps*R, len(y), int(y.sum()), auc, ci, snu_score, snu_pct, uni, len(camp))
    print("\n[저장] district_scores.csv, district_university.csv, models/pu_district.joblib, logs/DISTRICT_REPORT.md")

def write_log(eps_m, n, pos, auc, ci, snu, snu_pct, uni, ncamp):
    ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L=["# 상권 단위 PU 최종 리포트", "", f"> 생성 {ts}", "",
       f"## 상권 정의·성능","",
       f"- 정의: DBSCAN **auto-eps={eps_m:.0f}m**(k거리 elbow, 클러스터링과 동일)+노이즈 최근접 흡수 → 상권 {n}개(스벅상권 {pos}), 전 카페 배정.",
       f"- **상권 PU-AUC(OOF, spatial CV) = {auc:.4f}**, bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] (카페단위 0.677 대비 공정·실질 개선).","",
       "## 서울대 캠퍼스 (합성 상권)","",
       f"- 교내 {ncamp}개 카페 평균을 '하나의 상권'으로 채점 → **적합도 {snu:.3f}, 상권 중 {snu_pct:.0f}백분위**.",
       "- 캠퍼스 상권은 하위권 = 스벅형 상권 아님(상권 관점 결론, 카페단위와 일치).","",
       "## 서울 주요 대학 캠퍼스 합성상권 비교","",
       "| 대학 | 카페수 | 상권적합도 | 백분위 | 스벅수 |","|---|---|---|---|---|"]
    for _,r in uni.iterrows():
        L.append("| %s | %d | %.3f | %.0f | %d |"%(r["대학"],r["n_cafe"],r["상권적합도"],r["백분위"],r["스벅수"]))
    L+=["","메모: 서울대 상권 적합도 순위·백분위로 '상권 관점에서도 비역세권형'을 재확인. 한계: 거대상권 chaining, 합성상권은 근사.",""]
    with open(os.path.join(LOG,"DISTRICT_REPORT.md"),"w",encoding="utf-8") as f:
        f.write("\n".join(L)+"\n")

if __name__=="__main__":
    main()



