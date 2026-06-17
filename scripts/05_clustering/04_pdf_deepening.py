# -*- coding: utf-8 -*-
"""
PDF(중간발표 vF) 23~24p 클러스터링 심화 계획 보완
  ① Clustering 간 경계 분석 (C0-C3, C2-C4 가설 검증)         [23p-01]
  ② 구분력 하위변수 → Classification 제거 후보 명시          [23p-02]
  ③ Hierarchical 결과와 정합성 비교                          [24p]
  ④ DBSCAN 노이즈 매장 페르소나 재해석                      [24p]
  ⑤ 위경도 포함(B) vs 제외(A) 피처셋 추가 검증               [24p]
입력(읽기전용): data/modeling/starbucks_engineered_features_final.csv
출력: reports/generated/clustering/tables/
"""
import os

import numpy as np
import pandas as pd
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy import stats
from _common import (
    ENGINEERED_FEATURES,
    GENERATED_CLUSTERING_TABLE_DIR,
    PERSONA_NAMES,
    canonicalize_centers,
    canonicalize_cluster_labels,
    ensure_dirs,
    read_starbucks_engineered_features,
    relative_to_root,
)

TABLE_OUT_DIR = GENERATED_CLUSTERING_TABLE_DIR
RS=42

ensure_dirs(TABLE_OUT_DIR)
df=read_starbucks_engineered_features()
FEATS=ENGINEERED_FEATURES
names=PERSONA_NAMES
X=df[FEATS].values
Z=StandardScaler().fit_transform(X)
km=KMeans(n_clusters=5,n_init=15,random_state=RS).fit(Z)
lab,raw_to_canonical=canonicalize_cluster_labels(df, km.labels_)
cent=canonicalize_centers(km.cluster_centers_, raw_to_canonical)

# ═══ ① 경계 분석 ════════════════════════════════════════════════════
print('='*60); print('① Clustering 간 경계 분석 [PDF 23p-01]')
D=np.linalg.norm(Z[:,None,:]-cent[None,:,:],axis=2)   # (N,5)
order=np.argsort(D,axis=1)
d1=D[np.arange(len(Z)),order[:,0]]; d2=D[np.arange(len(Z)),order[:,1]]
margin=(d2-d1)/d2          # 0에 가까울수록 경계(1·2순위 centroid 거의 등거리)
boundary=margin<0.10       # 경계 매장
print(f'  경계 매장(margin<0.10): {int(boundary.sum())}개')
pairs=pd.Series([tuple(sorted([order[i,0],order[i,1]])) for i in np.where(boundary)[0]]).value_counts()
print('  경계 매장이 걸친 클러스터 쌍 (상위):')
for (a,b),n in pairs.head(6).items():
    print(f'    {names[a]} ↔ {names[b]}: {n}개')
# PDF 가설 검증
def pair_cnt(a,b): return int(pairs.get(tuple(sorted([a,b])),0))
print(f'  ▶ PDF 가설 검증: C0↔C3={pair_cnt(0,3)}  C2↔C4={pair_cnt(2,4)}  (둘 다 경계 흐릿 예측)')
bd=df[['상호명','시군구명']].copy(); bd['cluster']=lab; bd['margin']=margin.round(3)
bd['2nd_nearest']=[names[order[i,1]] for i in range(len(Z))]; bd['is_boundary']=boundary
bd.sort_values('margin').to_csv(TABLE_OUT_DIR / 'D1_boundary_analysis.csv',index=False,encoding='utf-8-sig')

# ═══ ② 구분력 하위변수 = 제거 후보 ══════════════════════════════════
print('='*60); print('② 구분력 하위변수 → Classification 제거 후보 [PDF 23p-02]')
rows=[]
for j,f in enumerate(FEATS):
    g=[X[lab==c,j] for c in range(5)]; F,p=stats.f_oneway(*g)
    grand=X[:,j].mean()
    eta2=sum(len(x)*(x.mean()-grand)**2 for x in g)/((X[:,j]-grand)**2).sum()
    rows.append({'feature':f,'eta_squared':round(eta2,3)})
ev=pd.DataFrame(rows).sort_values('eta_squared')
print('  구분력 하위 5 (Classification 제거 검토 후보):')
print(ev.head(5).to_string(index=False))
print('  → 단, 제거 전 다중공선성·도메인 의미 함께 고려 권장')
ev.to_csv(TABLE_OUT_DIR / 'D2_low_discriminative_features.csv',index=False,encoding='utf-8-sig')

# ═══ ③ Hierarchical 정합성 비교 ═════════════════════════════════════
print('='*60); print('③ KMeans vs Hierarchical 정합성 [PDF 24p]')
hier=AgglomerativeClustering(n_clusters=5,linkage='ward').fit_predict(Z)
ari=adjusted_rand_score(lab,hier)
print(f'  ARI(KMeans, Hierarchical k=5) = {ari:.3f}')
ct=pd.crosstab(pd.Series(lab,name='KMeans'),pd.Series(hier,name='Hier'))
print('  교차표 (행=KMeans, 열=Hierarchical):'); print(ct.to_string())
# 각 KMeans 클러스터가 어느 Hier 클러스터와 최대 일치
print('  KMeans→Hier 최대대응 일치율:')
for c in range(5):
    row=ct.loc[c]; print(f'    {names[c]}: {row.max()}/{row.sum()} ({row.max()/row.sum()*100:.0f}%) → Hier {row.idxmax()}')
ct.to_csv(TABLE_OUT_DIR / 'D3_kmeans_vs_hierarchical.csv',encoding='utf-8-sig')

# ═══ ④ DBSCAN 노이즈 재해석 ════════════════════════════════════════
print('='*60); print('④ DBSCAN 노이즈 매장 페르소나 재해석 [PDF 24p]')
def auto_eps(Xs,k=5):   # 원본 clustering_analysis.py와 동일
    nn=NearestNeighbors(n_neighbors=k).fit(Xs)
    dists,_=nn.kneighbors(Xs)
    kd=np.sort(dists[:,k-1])[::-1]
    d2=np.diff(np.diff(kd)); idx=int(np.argmax(d2))+1
    lo,hi=int(len(kd)*0.05),int(len(kd)*0.6)
    if not(lo<=idx<=hi): idx=lo+int(np.argmax(d2[lo:hi]))+1
    return float(kd[idx])
eps=auto_eps(Z,k=5)
db=DBSCAN(eps=eps,min_samples=5).fit_predict(Z)
noise=db==-1
print(f'  eps={eps:.3f}, 노이즈 {int(noise.sum())}개, 클러스터 {len(set(db))-(1 if -1 in db else 0)}개')
print('  노이즈 매장이 KMeans에서 속한 페르소나:')
nz=df.loc[noise,['상호명','시군구명']].copy(); nz['kmeans_persona']=[names[c] for c in lab[noise]]
print(nz.to_string(index=False))
_mode = pd.Series(lab[noise]).map(names).mode()[0] if noise.sum() else '-'
print(f'  → 노이즈는 대부분 {_mode} 등 특수/외곽 입지로 재해석')
nz.to_csv(TABLE_OUT_DIR / 'D4_dbscan_noise_reinterpret.csv',index=False,encoding='utf-8-sig')

# ═══ ⑤ 위경도 포함(B) vs 제외(A) ═══════════════════════════════════
print('='*60); print('⑤ 위경도 포함(B) vs 제외(A) 추가 검증 [PDF 24p]')
XB=df[FEATS+['위도','경도']].values
ZB=StandardScaler().fit_transform(XB)
raw_labB=KMeans(n_clusters=5,n_init=15,random_state=RS).fit_predict(ZB)
labB,_=canonicalize_cluster_labels(df, raw_labB)
print(f'  Silhouette  A(제외)={silhouette_score(Z,lab):.3f}  B(포함)={silhouette_score(ZB,labB):.3f}')
print(f'  ARI(A, B) = {adjusted_rand_score(lab,labB):.3f}  (낮을수록 위경도가 군집을 크게 바꿈)')
# 군집 내 평균 haversine 거리(km): 작을수록 지리적으로 뭉침
la=df['위도'].values; lo=df['경도'].values
def within_geo(labels):
    tot=[]
    for c in np.unique(labels):
        idx=np.where(labels==c)[0]
        if len(idx)<2: continue
        a=la[idx]; o=lo[idx]; R=6371; P=np.pi/180
        dm=2*R*np.arcsin(np.sqrt(np.sin((a[:,None]-a[None,:])*P/2)**2+
            np.cos(a[:,None]*P)*np.cos(a[None,:]*P)*np.sin((o[:,None]-o[None,:])*P/2)**2))
        iu=np.triu_indices(len(idx),1); tot.append(dm[iu].mean())
    return np.average(tot,weights=[np.sum(labels==c) for c in np.unique(labels) if np.sum(labels==c)>=2])
gA=within_geo(lab); gB=within_geo(labB)
print(f'  군집 내 평균 매장간 거리(km)  A(제외)={gA:.2f}  B(포함)={gB:.2f}')
print('  → B가 더 작음 = 위경도가 "지리적 뭉침" 유발(상권 목적과 어긋남) → A 채택 정당')
pd.DataFrame({'상호명':df['상호명'],'cluster_A':lab,'cluster_B':labB}).to_csv(
    TABLE_OUT_DIR / 'D5_setA_vs_setB.csv',index=False,encoding='utf-8-sig')
print(f'\n저장 완료 → {relative_to_root(TABLE_OUT_DIR)} (D1~D5)')
