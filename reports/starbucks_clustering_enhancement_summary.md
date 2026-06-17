# 스타벅스 입점 예측 - 클러스터링 고도화 보고서

> **과목**: 2026-1 데이터마이닝 텀프로젝트  
> **작성일**: 2026-06-14  
> **범위**: 중간발표 이후 클러스터링 단계 고도화  
> **입력**: `data/modeling/starbucks_engineered_features_final.csv`, `data/final/seoul_cafe_model_features_final.csv`  
> **재현 코드**: `scripts/05_clustering/`  
> **보존 산출물**: `reports/archive/tables/clustering/`, `reports/archive/figures/clustering/`

이 보고서는 현재 repo의 공식 feature 파일을 기준으로 재생성한 클러스터링 고도화 결과를 정리한다. `incoming_clustering/`의 예전 보고서는 별도 입력 파일을 기준으로 작성되어 일부 수치가 달랐으므로, Git 편입본에서는 현재 재현 파이프라인과 일치하는 수치만 사용한다.

---

## 0. Executive Summary

| 영역 | 결과 |
|---|---|
| 기준 모델 | KMeans k=5, persona label 고정 후 C0~C4 크기 47, 224, 238, 73, 99 |
| K=5 정당성 | 16개 피처 중 15개가 large-effect(eta squared >= 0.14), Silhouette 0.172 |
| 안정성 | 80% bootstrap 100회 기준 평균 안정성 0.789 |
| 비선형 시각화 | PCA 설명분산 56.9%, t-SNE/UMAP 비교 그림 보존 |
| 이상치/특수입지 | centroid 2 sigma 초과 이상치 31개, 특수입지 키워드 매장 100개 |
| 비스타벅스 재해석 | 비스타벅스 21,624개 중 7,546개(34.9%)가 스타벅스 핵심상권형 C1+C3에 투영 |
| PDF 심화 | 경계분석, 구분력 하위변수, Hierarchical 정합성, DBSCAN 노이즈, 위경도 포함/제외 검증 완료 |

핵심 메시지는 세 가지다.

1. KMeans k=5는 Silhouette 단일 지표만으로는 약하지만, 효과크기, 안정성, 비선형 시각화, 지리 응집도를 함께 보면 해석 가능한 persona 구조를 갖는다.
2. 비스타벅스 카페는 일괄 negative sample로 두기 어렵다. 상당수가 스타벅스형 상권에 있으므로 PU Learning 또는 unlabeled 관점이 더 적절하다.
3. 특수입지와 landmark 매장은 군집 평균으로 설명하기보다 별도 해석 트랙으로 관리해야 한다.

---

## 1. 기준 클러스터

KMeans k=5의 원시 label은 실행 환경과 입력에 따라 의미 순서가 바뀔 수 있다. `scripts/05_clustering/_common.py`는 cluster profile을 기준으로 C0~C4 persona label을 고정한다.

| Persona | 크기 | 해석 |
|---|---:|---|
| C0 오피스고소득 | 47 | 업무지구 및 고소득 수요가 강한 입지 |
| C1 상업활성 | 224 | 상권 밀도와 유동 수요가 큰 입지 |
| C2 주거생활 | 238 | 생활권 기반의 대형 cluster |
| C3 도심초밀집 | 73 | 도심, 고지가, 초밀집 상권 |
| C4 비역세권 | 99 | 지하철 접근성이 상대적으로 약한 입지 |

---

## 2. K=5 정당성 보강

### 2.1 피처 구분력

ANOVA F와 eta squared로 cluster가 각 피처 분산을 얼마나 설명하는지 확인했다. 16개 피처 중 15개가 large-effect다.

| 순위 | 피처 | eta squared | F |
|---:|---|---:|---:|
| 1 | `offices` | 0.741 | 483.8 |
| 2 | `indie_cafe_500m` | 0.710 | 413.1 |
| 3 | `restaurants_500m` | 0.681 | 360.0 |
| 4 | `convenience_500m` | 0.596 | 249.8 |
| 5 | `peak_avg` | 0.538 | 196.5 |

산출물: `reports/archive/tables/clustering/A1_feature_discriminative_power.csv`

### 2.2 지리 응집도

| Persona | n | 대표구 | 구 집중도 | 평균 최근접거리 |
|---|---:|---|---:|---:|
| C0 오피스고소득 | 47 | 강남구 | 51.1% | 0.224 km |
| C1 상업활성 | 224 | 강남구 | 15.2% | 0.264 km |
| C2 주거생활 | 238 | 강서구 | 9.2% | 0.564 km |
| C3 도심초밀집 | 73 | 중구 | 43.8% | 0.148 km |
| C4 비역세권 | 99 | 강남구 | 14.1% | 1.156 km |

산출물: `reports/archive/tables/clustering/A2_geographic_cohesion.csv`

### 2.3 안정성

80% bootstrap 100회 기준 평균 안정성은 0.789다.

| Persona | 안정성 |
|---|---:|
| C0 오피스고소득 | 0.854 |
| C1 상업활성 | 0.746 |
| C2 주거생활 | 0.790 |
| C3 도심초밀집 | 0.836 |
| C4 비역세권 | 0.820 |

산출물: `reports/archive/tables/clustering/A4_store_cluster_stability.csv`

---

## 3. 피드백 반영 결과

| 피드백 | 반영 방식 | 결과 |
|---|---|---|
| K=5 정당화 부족 | eta squared, 지리 응집도, bootstrap 안정성, 대안 알고리즘 비교 | 15/16 large-effect, 평균 안정성 0.789 |
| GMM, t-SNE, UMAP 등 대안 검토 | k=2~8 x KMeans/GMM/Hierarchical 비교, PCA/t-SNE/UMAP 시각화 | KMeans k=5 Silhouette 0.172, PCA 설명분산 56.9% |
| 이상치 persona 별도 해석 | cluster centroid 거리 기반 이상치와 특수입지 키워드 식별 | 이상치 31개, 특수입지 100개 |
| 비스타벅스 일괄 부적합 가정 재고 | 전체 카페를 스타벅스 centroid에 투영 | 비스타벅스 34.9%가 C1 상업활성 + C3 도심초밀집에 분포 |

비스타벅스 분포:

| Persona | 스타벅스 % | 비스타벅스 % | 비스타벅스 수 |
|---|---:|---:|---:|
| C0 오피스고소득 | 6.9 | 3.5 | 765 |
| C1 상업활성 | 32.9 | 29.1 | 6,295 |
| C2 주거생활 | 34.9 | 34.4 | 7,443 |
| C3 도심초밀집 | 10.7 | 5.8 | 1,251 |
| C4 비역세권 | 14.5 | 27.1 | 5,870 |

산출물:

- `reports/archive/tables/clustering/A3_k_and_algorithm_comparison.csv`
- `reports/archive/figures/clustering/V1_dimreduction_comparison.png`
- `reports/archive/figures/clustering/V2_profile_heatmap.png`
- `reports/archive/tables/clustering/O1_within_cluster_outliers.csv`
- `reports/archive/tables/clustering/O2_special_location_stores.csv`
- `reports/archive/tables/clustering/O3_all_cafe_cluster_projection.csv`
- `reports/archive/figures/clustering/O4_sb_vs_nonsb_distribution.png`

---

## 4. PDF 심화 계획 반영

### 4.1 Cluster 경계 분석

1순위와 2순위 centroid가 거의 등거리인 경계 매장(margin < 0.10)은 104개다.

| 경계 쌍 | 매장 수 |
|---|---:|
| C1 상업활성 - C2 주거생활 | 49 |
| C1 상업활성 - C3 도심초밀집 | 22 |
| C2 주거생활 - C4 비역세권 | 16 |
| C1 상업활성 - C4 비역세권 | 7 |
| C0 오피스고소득 - C1 상업활성 | 5 |
| C0 오피스고소득 - C2 주거생활 | 3 |

PDF 가설 중 C0-C3 경계가 흐릿할 것이라는 예상은 현재 데이터 기준 1개로 강하게 나타나지 않았다. C2-C4 경계는 16개로 일부 확인된다.

### 4.2 구분력 하위변수

| 피처 | eta squared |
|---|---:|
| `avg_income` | 0.071 |
| `bus_stops_300m` | 0.172 |
| `log_dist_starbucks` | 0.372 |
| `subway_ridership` | 0.398 |
| `land_price` | 0.427 |

`avg_income`은 cluster 구분력 최하위이므로 classification 단계에서 제거 후보로 검토할 수 있다. 다만 제거 여부는 분류 목적, 다중공선성, 도메인 의미를 함께 확인해야 한다.

### 4.3 Hierarchical 정합성

ARI(KMeans, Hierarchical k=5)는 0.514다.

| Persona | 최대대응 일치율 |
|---|---:|
| C0 오피스고소득 | 62% |
| C1 상업활성 | 50% |
| C2 주거생활 | 100% |
| C3 도심초밀집 | 63% |
| C4 비역세권 | 98% |

C2와 C4는 알고리즘이 바뀌어도 안정적이고, C1은 경계분석과 정합성 양쪽에서 가장 흔들리는 cluster다.

### 4.4 DBSCAN 노이즈 재해석

원본 `auto_eps` 방식으로 eps=3.103을 얻었고, DBSCAN 결과는 노이즈 18개와 cluster 3개다. 노이즈에는 강남R, 강남역신분당역사, 여의도한강공원, 잠실역, 정부서울청사R, 신세계본점6F, 서울역사 등 landmark, 환승, 특수시설 성격의 매장이 포함된다.

### 4.5 위경도 포함/제외 검증

| 지표 | A: 위경도 제외 | B: 위경도 포함 |
|---|---:|---:|
| Silhouette | 0.172 | 0.133 |
| 군집 내 평균 매장간 거리 | 9.97 km | 9.43 km |
| ARI(A, B) | 0.571 | - |

B는 지리적으로 더 뭉치지만 Silhouette이 낮다. 위경도가 상권 특성보다 지리적 근접성을 과하게 반영할 수 있으므로, 현재 feature set에서는 A(위경도 제외)를 유지한다.

산출물:

- `reports/archive/tables/clustering/D1_boundary_analysis.csv`
- `reports/archive/tables/clustering/D2_low_discriminative_features.csv`
- `reports/archive/tables/clustering/D3_kmeans_vs_hierarchical.csv`
- `reports/archive/tables/clustering/D4_dbscan_noise_reinterpret.csv`
- `reports/archive/tables/clustering/D5_setA_vs_setB.csv`

---

## 5. 재현 방법

```bash
python scripts/05_clustering/01_validity_and_algorithms.py
python scripts/05_clustering/02_visualization.py
python scripts/05_clustering/03_outlier_and_projection.py
python scripts/05_clustering/04_pdf_deepening.py
```

기본 출력은 `reports/generated/clustering/`에 저장되며 Git에는 포함하지 않는다. 보고서에 사용한 현재 스냅샷은 `reports/archive/tables/clustering/`, `reports/archive/figures/clustering/`에 보존한다.
