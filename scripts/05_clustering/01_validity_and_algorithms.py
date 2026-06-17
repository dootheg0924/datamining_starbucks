# -*- coding: utf-8 -*-
"""
클러스터링 고도화 1 - 정량적 구별성 / 대안 알고리즘 / 안정성
입력(읽기전용): data/modeling/starbucks_engineered_features_final.csv
출력: reports/generated/clustering/tables/

반영 피드백:
  교수[36] 페르소나→근거 논리 보강  → ANOVA F + η²(효과크기)
  학생[05] 정량적 구별성(효과크기·지리응집도)
  학생[28] GMM 등 대안 알고리즘
  학생[14] k 다각도 재검토
  학생[25] train/test(부트스트랩) 안정성 검증 + 발표23p 계획
"""
import json
import os

import numpy as np
import pandas as pd
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score, f1_score)
from scipy import stats

from _common import (
    ENGINEERED_FEATURES,
    GENERATED_CLUSTERING_TABLE_DIR,
    canonicalize_cluster_labels,
    ensure_dirs,
    read_starbucks_engineered_features,
    relative_to_root,
)

TABLE_OUT_DIR = GENERATED_CLUSTERING_TABLE_DIR
RS = 42

# ── 데이터 로드 (읽기전용) ────────────────────────────────────────────
ensure_dirs(TABLE_OUT_DIR)
df = read_starbucks_engineered_features()
FEATS = ENGINEERED_FEATURES
X = df[FEATS].values
gu = df['시군구명'].values
lat, lon = df['위도'].values, df['경도'].values
Z = StandardScaler().fit_transform(X)
print(f'스타벅스 {len(df)}개 · 피처 {len(FEATS)}개')

# ── 기준 모델: KMeans k=5 (발표 최종) ────────────────────────────────
km5 = KMeans(n_clusters=5, n_init=15, random_state=RS).fit(Z)
lab, raw_to_canonical = canonicalize_cluster_labels(df, km5.labels_)
sizes = np.bincount(lab)
print('KMeans k=5 persona 크기:', {f'C{c}': int(sizes[c]) for c in range(5)})
print('원시 label → persona label:', raw_to_canonical)

results = {}

# ═══ 고도화 1A: ANOVA F-stat + η²(eta-squared, 효과크기) ═══════════════
# η² = 클러스터가 설명하는 분산 비율 (0~1). 0.14↑ large effect (Cohen)
rows = []
for j, f in enumerate(FEATS):
    groups = [X[lab == c, j] for c in range(5)]
    F, p = stats.f_oneway(*groups)
    grand = X[:, j].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((X[:, j] - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    rows.append({'feature': f, 'F_stat': round(F, 1), 'p_value': p,
                 'eta_squared': round(eta2, 3),
                 'effect': 'large' if eta2 >= 0.14 else ('medium' if eta2 >= 0.06 else 'small')})
anova = pd.DataFrame(rows).sort_values('eta_squared', ascending=False)
anova.to_csv(TABLE_OUT_DIR / 'A1_feature_discriminative_power.csv', index=False, encoding='utf-8-sig')
print('\n[고도화1A] 클러스터 구분력 상위 5 (η² 효과크기):')
print(anova.head(5).to_string(index=False))
print(f"  large-effect 피처 수: {(anova['effect']=='large').sum()}/{len(FEATS)}")

# ═══ 고도화 1B: 지리적 응집도 ═══════════════════════════════════════
# 각 클러스터 매장들이 지리적으로 얼마나 모여있나 (km 평균 최근접거리)
def geo_cohesion(mask):
    la, lo = lat[mask], lon[mask]
    if mask.sum() < 2: return np.nan
    R = 6371; P = np.pi/180
    n = mask.sum(); d = np.zeros((n, n))
    for i in range(n):
        dd = 2*R*np.arcsin(np.sqrt(np.sin((la-la[i])*P/2)**2 +
             np.cos(la[i]*P)*np.cos(la*P)*np.sin((lo-lo[i])*P/2)**2))
        d[i] = dd
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1).mean()  # 평균 최근접 매장 거리(km)
geo_rows = []
for c in range(5):
    m = lab == c
    geo_rows.append({'cluster': c, 'n': int(m.sum()),
                     '대표구': pd.Series(gu[m]).mode()[0],
                     '구_집중도(%)': round(pd.Series(gu[m]).value_counts(normalize=True).iloc[0]*100, 1),
                     '평균최근접거리_km': round(geo_cohesion(m), 3)})
geo = pd.DataFrame(geo_rows)
geo.to_csv(TABLE_OUT_DIR / 'A2_geographic_cohesion.csv', index=False, encoding='utf-8-sig')
print('\n[고도화1B] 클러스터별 지리적 응집도:')
print(geo.to_string(index=False))

# ═══ 고도화 2: 대안 알고리즘 & k 재검토 (다중지표) ═══════════════════
print('\n[고도화2] k=2~8 다중지표 (KMeans / GMM / Hierarchical)')
krows = []
for k in range(2, 9):
    for name, model in [
        ('KMeans', KMeans(n_clusters=k, n_init=15, random_state=RS)),
        ('GMM', GaussianMixture(n_components=k, covariance_type='full', random_state=RS, n_init=3)),
        ('Hierarchical', AgglomerativeClustering(n_clusters=k, linkage='ward'))]:
        l = model.fit_predict(Z)
        bic = model.bic(Z) if name == 'GMM' else np.nan
        krows.append({'k': k, 'algorithm': name,
                      'silhouette': round(silhouette_score(Z, l), 3),
                      'davies_bouldin': round(davies_bouldin_score(Z, l), 3),
                      'calinski_harabasz': round(calinski_harabasz_score(Z, l), 1),
                      'BIC': round(bic, 0) if not np.isnan(bic) else None})
kcomp = pd.DataFrame(krows)
kcomp.to_csv(TABLE_OUT_DIR / 'A3_k_and_algorithm_comparison.csv', index=False, encoding='utf-8-sig')
print(kcomp.to_string(index=False))

# GMM k=5 vs KMeans k=5 일치도
gmm5 = GaussianMixture(n_components=5, covariance_type='full', random_state=RS, n_init=3).fit(Z)
print(f"\n  GMM k=5 vs KMeans k=5 일치도(ARI): {adjusted_rand_score(lab, gmm5.predict(Z)):.3f}")

# ═══ 고도화 4: 부트스트랩 안정성 (학생[25], 발표23p) ════════════════
print('\n[고도화4] 부트스트랩 안정성 (80% 표본 × 100회)')
N = len(Z); B = 100; frac = 0.8
co_same = np.zeros((N, N)); co_both = np.zeros((N, N))
rng = np.random.RandomState(RS)
for b in range(B):
    idx = rng.choice(N, int(N*frac), replace=False)
    lb = KMeans(n_clusters=5, n_init=5, random_state=b).fit_predict(Z[idx])
    same = (lb[:, None] == lb[None, :]).astype(float)
    co_same[np.ix_(idx, idx)] += same
    co_both[np.ix_(idx, idx)] += 1
with np.errstate(invalid='ignore'):
    cocluster = np.where(co_both > 0, co_same/co_both, np.nan)
# 각 매장: 자기 클러스터 동료들과 같이 묶이는 평균 확률
stab = []
for i in range(N):
    peers = (lab == lab[i]); peers[i] = False
    stab.append(np.nanmean(cocluster[i, peers]))
df_stab = pd.DataFrame({'cluster': lab, 'stability': stab})
clus_stab = df_stab.groupby('cluster')['stability'].mean().round(3)
print('  클러스터별 평균 안정성 (1=항상 같이 묶임):')
for c in range(5):
    print(f'    C{c} (n={int(sizes[c])}): {clus_stab[c]:.3f}')
print(f'  전체 평균 안정성: {np.nanmean(stab):.3f}')

# 결과 저장
out_df = df[['상호명','시군구명','위도','경도']].copy()
out_df['cluster_kmeans5'] = lab
out_df['stability'] = np.round(stab, 3)
out_df.to_csv(TABLE_OUT_DIR / 'A4_store_cluster_stability.csv', index=False, encoding='utf-8-sig')

summary = {
    'large_effect_features': int((anova['effect']=='large').sum()),
    'top_discriminator': anova.iloc[0]['feature'],
    'mean_stability': round(float(np.nanmean(stab)), 3),
    'gmm_kmeans_ari': round(adjusted_rand_score(lab, gmm5.predict(Z)), 3),
}
with (TABLE_OUT_DIR / 'A_summary.json').open('w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f'\n저장 완료 → {relative_to_root(TABLE_OUT_DIR)} (A1~A4)')
