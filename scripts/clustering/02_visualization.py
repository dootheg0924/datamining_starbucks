# -*- coding: utf-8 -*-
"""
클러스터링 고도화 3 - 비선형 차원축소 시각화 (학생[28] 반영)
PCA(기존 56.9%) vs t-SNE vs UMAP 비교 + 표준화 프로파일 히트맵
입력(읽기전용): data/modeling/starbucks_engineered_features_final.csv
출력: reports/generated/clustering/figures/
"""
import os

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from _common import (
    ENGINEERED_FEATURES,
    GENERATED_CLUSTERING_FIGURE_DIR,
    PERSONA_NAMES,
    canonicalize_cluster_labels,
    ensure_dirs,
    read_starbucks_engineered_features,
    relative_to_root,
)

for f in ['Malgun Gothic','NanumGothic','AppleGothic']:
    if any(f.lower()==x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams['font.family']=f; break
plt.rcParams['axes.unicode_minus']=False
FIGURE_OUT_DIR = GENERATED_CLUSTERING_FIGURE_DIR
RS=42

ensure_dirs(FIGURE_OUT_DIR)
df=read_starbucks_engineered_features()
FEATS=ENGINEERED_FEATURES
Z=StandardScaler().fit_transform(df[FEATS].values)
raw_lab=KMeans(n_clusters=5,n_init=15,random_state=RS).fit_predict(Z)
lab,_=canonicalize_cluster_labels(df, raw_lab)
names=PERSONA_NAMES
colors=['#d62728','#ff7f0e','#2ca02c','#9467bd','#1f77b4']

# ── 3개 임베딩 ────────────────────────────────────────────────────
pca=PCA(n_components=2,random_state=RS).fit(Z); P=pca.transform(Z)
evr=pca.explained_variance_ratio_.sum()*100
T=TSNE(n_components=2,perplexity=30,random_state=RS,init='pca').fit_transform(Z)
U=umap.UMAP(n_neighbors=15,min_dist=0.1,random_state=RS,n_jobs=1).fit_transform(Z)

fig,axes=plt.subplots(1,3,figsize=(21,7))
for ax,emb,title in [(axes[0],P,f'PCA (설명분산 {evr:.1f}%)'),
                     (axes[1],T,'t-SNE (perplexity=30)'),
                     (axes[2],U,'UMAP (n_neighbors=15)')]:
    for c in range(5):
        m=lab==c
        ax.scatter(emb[m,0],emb[m,1],s=18,c=colors[c],label=names[c],alpha=0.7,edgecolor='none')
    ax.set_title(title,fontsize=13); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc='best',fontsize=9)
plt.suptitle('차원축소 비교: 선형(PCA) vs 비선형(t-SNE/UMAP) — KMeans k=5',fontsize=15)
plt.tight_layout(); plt.savefig(FIGURE_OUT_DIR / 'V1_dimreduction_comparison.png',dpi=120); plt.close()
print(f'PCA 설명분산: {evr:.1f}%  | V1 저장')

# ── 표준화 프로파일 히트맵 (페르소나 해석 근거) ──────────────────
prof=pd.DataFrame(Z,columns=FEATS); prof['c']=lab
cmean=prof.groupby('c')[FEATS].mean()
fig,ax=plt.subplots(figsize=(13,6))
im=ax.imshow(cmean.values,cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto')
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS,rotation=45,ha='right',fontsize=9)
ax.set_yticks(range(5)); ax.set_yticklabels([names[c] for c in range(5)],fontsize=10)
for i in range(5):
    for j in range(len(FEATS)):
        ax.text(j,i,f'{cmean.values[i,j]:.1f}',ha='center',va='center',fontsize=7,
                color='white' if abs(cmean.values[i,j])>0.9 else 'black')
plt.colorbar(im,label='표준화 평균 (z-score)')
ax.set_title('클러스터별 표준화 피처 프로파일 (페르소나 해석 근거)',fontsize=13)
plt.tight_layout(); plt.savefig(FIGURE_OUT_DIR / 'V2_profile_heatmap.png',dpi=120); plt.close()
print('V2 히트맵 저장')
print(f'\n저장 완료 → {relative_to_root(FIGURE_OUT_DIR)} (V1, V2)')
