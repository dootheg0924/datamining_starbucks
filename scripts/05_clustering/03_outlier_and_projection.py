# -*- coding: utf-8 -*-
"""
클러스터링 고도화 5·6
  5. 이상치/특수입지 페르소나 별도 해석   (교수[34], 학생[02] 한옥스타벅스)
  6. 비스타벅스 카페 클러스터 투영        (교수[35] '일괄 부적합' 재고)
입력(읽기전용): data/modeling/starbucks_engineered_features_final.csv,
              data/final/seoul_cafe_model_features_final.csv
출력: reports/generated/clustering/tables/, reports/generated/clustering/figures/
"""
import os

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from _common import (
    ENGINEERED_FEATURES,
    GENERATED_CLUSTERING_FIGURE_DIR,
    GENERATED_CLUSTERING_TABLE_DIR,
    PERSONA_NAMES,
    build_engineered_features_for_all_cafes,
    canonicalize_centers,
    canonicalize_cluster_labels,
    ensure_dirs,
    read_seoul_cafe_final_features,
    read_starbucks_engineered_features,
    relative_to_root,
)

for f in ['Malgun Gothic','NanumGothic','AppleGothic']:
    if any(f.lower()==x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams['font.family']=f; break
plt.rcParams['axes.unicode_minus']=False
TABLE_OUT_DIR = GENERATED_CLUSTERING_TABLE_DIR
FIGURE_OUT_DIR = GENERATED_CLUSTERING_FIGURE_DIR
RS=42

ensure_dirs(TABLE_OUT_DIR, FIGURE_OUT_DIR)
FEATS=ENGINEERED_FEATURES
names=PERSONA_NAMES

sb=read_starbucks_engineered_features()
scaler=StandardScaler().fit(sb[FEATS].values)
Zsb=scaler.transform(sb[FEATS].values)
km=KMeans(n_clusters=5,n_init=15,random_state=RS).fit(Zsb)
lab,raw_to_canonical=canonicalize_cluster_labels(sb, km.labels_)
cent=canonicalize_centers(km.cluster_centers_, raw_to_canonical)

# ═══ 고도화 5: 이상치/특수입지 별도 해석 ════════════════════════════
# centroid까지 거리로 '클러스터 내 이상치' 식별 (cluster_dist 상위 = 전형성 낮음)
d2cent=np.linalg.norm(Zsb-cent[lab],axis=1)
sb_o=sb[['상호명','시군구명']].copy(); sb_o['cluster']=lab; sb_o['centroid_dist']=d2cent.round(2)
# 클러스터별 z-score (자기 클러스터 내 이상 정도)
thr=pd.Series(d2cent).groupby(lab).transform(lambda x:x.mean()+2*x.std())
sb_o['is_outlier']=(d2cent>thr.values)
n_out=int(sb_o['is_outlier'].sum())
print(f'[고도화5] 클러스터 내 이상치(centroid 2σ 초과): {n_out}개')
print('  이상치 TOP12 (전형성 낮은 = 데이터로 설명 어려운 특수입지 후보):')
top=sb_o.sort_values('centroid_dist',ascending=False).head(12)
print(top.to_string(index=False))
sb_o.sort_values('centroid_dist',ascending=False).to_csv(
    TABLE_OUT_DIR / 'O1_within_cluster_outliers.csv',index=False,encoding='utf-8-sig')

# 발표 12p 특수입지 키워드 매칭 (목적지형/DT/특수시설)
kw=['북한산','한강','타워','DT','드라이브','이케아','하나로','휴게소','공원','산','아울렛']
mask_special=sb['상호명'].astype(str).str.contains('|'.join(kw),na=False)
print(f"\n  특수입지 키워드 매칭 매장: {int(mask_special.sum())}개")
sp=sb.loc[mask_special,['상호명','시군구명']].copy(); sp['cluster']=lab[mask_special.values]
sp['cluster_name']=sp['cluster'].map(names)
print(sp.head(15).to_string(index=False))
sp.to_csv(TABLE_OUT_DIR / 'O2_special_location_stores.csv',index=False,encoding='utf-8-sig')

# ═══ 고도화 6: 비스타벅스 카페 클러스터 투영 (교수[35]) ═════════════
full=read_seoul_cafe_final_features()
f=build_engineered_features_for_all_cafes(full)[FEATS]
is_sb=full['is_starbucks'].values==1
neg=f[~is_sb]
Zneg=scaler.transform(neg.values)
# 최근접 centroid 배정
dist_neg=np.linalg.norm(Zneg[:,None,:]-cent[None,:,:],axis=2)
neg_lab=dist_neg.argmin(axis=1)

# 분포 비교: 스타벅스 vs 비스타벅스
sb_dist=np.bincount(lab,minlength=5)/len(lab)*100
neg_dist=np.bincount(neg_lab,minlength=5)/len(neg_lab)*100
print('\n[고도화6] 클러스터별 분포: 스타벅스 vs 비스타벅스 카페')
comp=pd.DataFrame({'cluster':[names[c] for c in range(5)],
                   '스타벅스_%':sb_dist.round(1),'비스타벅스_%':neg_dist.round(1),
                   '비스타벅스_수':np.bincount(neg_lab,minlength=5)})
print(comp.to_string(index=False))

# 핵심 메시지: 비스타벅스도 '스타벅스형 클러스터'(C1/C3 상업·도심)에 다수 포함
sb_like=int(((neg_lab==1)|(neg_lab==3)).sum())
print(f"\n  ★ 비스타벅스 {len(neg)}개 중 스타벅스 핵심상권형(C1상업+C3도심)에 속한 카페: "
      f"{sb_like}개 ({sb_like/len(neg)*100:.1f}%)")
print("  → 비스타벅스 카페를 일괄 '입지 부적합'으로 볼 수 없음 (교수[35] 반영)")

full_out=full[['상호명','시군구명','위도','경도']].copy()
full_out['is_starbucks']=full['is_starbucks']
proj=np.full(len(full),-1); proj[~is_sb]=neg_lab; proj[is_sb]=lab
full_out['projected_cluster']=proj
full_out.to_csv(TABLE_OUT_DIR / 'O3_all_cafe_cluster_projection.csv',index=False,encoding='utf-8-sig')

# 시각화: 분포 막대
fig,ax=plt.subplots(figsize=(10,6))
x=np.arange(5); w=0.38
ax.bar(x-w/2,sb_dist,w,label='스타벅스(681)',color='#2ca02c')
ax.bar(x+w/2,neg_dist,w,label='비스타벅스(21,624)',color='#cccccc',edgecolor='#888')
ax.set_xticks(x); ax.set_xticklabels([names[c] for c in range(5)],rotation=20,fontsize=10)
ax.set_ylabel('비율 (%)'); ax.legend()
ax.set_title('클러스터 분포: 스타벅스 vs 비스타벅스 카페\n비스타벅스도 상업·도심형에 다수 분포 → "일괄 부적합" 가정 재고',fontsize=12)
plt.tight_layout(); plt.savefig(FIGURE_OUT_DIR / 'O4_sb_vs_nonsb_distribution.png',dpi=120); plt.close()
print(f'\n저장 완료 → {relative_to_root(TABLE_OUT_DIR)}, {relative_to_root(FIGURE_OUT_DIR)} (O1~O4)')
