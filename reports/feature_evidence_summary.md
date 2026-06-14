# 최종 Feature 근거 요약

이 문서는 최종 모델 feature가 왜 포함되었는지, 어떤 통계와 시각화 근거를 확인했는지 빠르게 파악하기 위한 인수인계용 요약입니다.

- 최종 데이터
  - `data/final/starbucks_model_features_final.csv`: 스타벅스 681개 매장 기준 최종 모델 feature
  - `data/final/seoul_cafe_model_features_final.csv`: 서울 카페 22,305개 기준 최종 모델 feature. `nan_reason`은 보정 전 결측 원인을 기록한 provenance 컬럼입니다.
  - `data/modeling/starbucks_engineered_features_final.csv`: 스타벅스 전용 모델링 및 클러스터링용 파생 feature set
- 정리된 분석 archive
  - `reports/archive/analysis_archive_summary.md`
- 스타벅스 feature engineering 근거
  - `reports/starbucks_feature_engineering_summary.md`
- 클러스터링 고도화 근거
  - `reports/starbucks_clustering_enhancement_summary.md`
- 주요 근거 산출물
  - `reports/archive/tables/model_feature_v2_columns.csv`
  - `reports/archive/tables/model_feature_v2_status.csv`
  - `reports/archive/tables/model_feature_v2_summary_starbucks.csv`
  - `reports/archive/tables/new_feature_v2_summary.csv`
  - `reports/archive/tables/radius_feature_recommendation.csv`
  - `reports/archive/tables/clustering/`
  - `reports/archive/figures/clustering/`

## 1. 최종 Feature 구성

최종 모델 feature CSV는 식별 및 해석용 컬럼 7개와 모델 feature 18개로 구성됩니다. 서울 전체 final CSV에는 `nan_reason` 1개 컬럼이 추가되어 총 26개 컬럼이고, 스타벅스 전용 final CSV는 `nan_reason`을 제외한 25개 컬럼입니다.

| 구분 | 컬럼 |
| --- | --- |
| 식별/해석 | `상호명`, `브랜드`, `is_starbucks`, `위도`, `경도`, `시군구명`, `도로명주소` |
| 지리/교통 | `dist_nearest_subway`, `num_subway_500m`, `nearest_subway_ridership`, `num_bus_stops_300m`, `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` |
| 상권 | `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m`, `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m`, `dist_nearest_starbucks` |
| 인구/통계 | `avg_income`, `num_offices`, `living_population`, `land_price` |

## 2. 반경 선택 근거

### 2.1 버스정류장 반경

최종 선택: `num_bus_stops_300m`

| 후보 | 스타벅스 기준 평균 | 중앙값 | 0개 비율 | 판단 |
| --- | ---: | ---: | ---: | --- |
| `num_bus_stops_100m` | 1.72 | 2.0 | 23.79% | 반경이 너무 좁아 0개 매장이 많습니다. |
| `num_bus_stops_300m` | 9.86 | 9.0 | 0.73% | 도보 접근권으로 해석 가능하고 매장 간 차이도 남습니다. |
| `num_bus_stops_500m` | 22.43 | 21.0 | 0.00% | 반경이 넓어 생활권 성격이 강하고 300m와 상관이 높습니다. |

추가 근거:

- `num_bus_stops_300m`와 `num_bus_stops_500m`의 상관은 0.793입니다.
- `dist_nearest_bus_stop`은 해석은 명확하지만, `num_bus_stops_300m`가 주변 버스 접근성을 더 안정적으로 표현하므로 최종 output에서는 제외했습니다.
- 관련 시각화는 `reports/archive/figures/radius_selection/`의 버스정류장 histogram을 참고합니다.

### 2.2 카페 반경

반경 비교 단계에서는 `cafe_count_300m`가 직접 경쟁권 지표로 가장 해석 가능하다고 판단했습니다. 다만 최종 모델 feature에서는 단순 전체 카페 수 대신 500m 반경의 카페 유형별 count를 사용했습니다.

| 후보 | 스타벅스 기준 평균 | 중앙값 | 판단 |
| --- | ---: | ---: | --- |
| `cafe_count_300m` | 40.22 | 35.0 | 직접 경쟁권으로 해석하기 쉬워 clustering 후보로 적합합니다. |
| `cafe_count_500m` | 90.01 | 79.0 | 기존 `num_competing_cafes_500m`와 거의 중복됩니다. |
| `cafe_count_1000m` | 276.28 | 235.0 | 개별 매장보다 지역 상권 규모 성격이 강합니다. |

추가 근거:

- `cafe_count_300m`와 `cafe_count_500m`의 상관은 0.905입니다.
- `cafe_count_500m`와 기존 `num_competing_cafes_500m`의 상관은 0.999로 사실상 중복입니다.
- 최종 모델에서는 상권 구조를 더 구체적으로 표현하기 위해 `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m`를 사용했습니다.

### 2.3 500m 반경을 유지한 변수

| 변수 | 500m 유지 이유 |
| --- | --- |
| `num_subway_500m` | 역세권 여부와 주변 지하철 접근성을 해석하기 위한 기준 변수입니다. |
| `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` | 500m 내 지하철역의 시간대별 승하차 규모를 합산해 역세권 수요를 반영합니다. |
| `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m` | 주변 상권 규모를 같은 반경에서 비교하기 위한 master 기반 변수입니다. |
| `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 단순 전체 카페 수보다 카페 유형별 경쟁 및 상권 구성을 표현합니다. |

## 3. 변수별 모델링 주의사항

| 변수군 | 변수 | 최종 판단 및 주의사항 |
| --- | --- | --- |
| 지리/교통 | `dist_nearest_subway` | 지하철 접근성의 방향성이 명확해 포함했습니다. 척도가 크므로 모델링 시 scaling 또는 log 변환을 검토합니다. |
| 지리/교통 | `num_subway_500m` | 역세권 여부를 해석하기 쉬워 포함했습니다. 이산형 변수이고 IQR이 작아 이상치 해석에 주의합니다. |
| 지리/교통 | `nearest_subway_ridership` | 가장 가까운 역의 규모를 직접 반영합니다. 원천 결측은 final 생성 과정에서 보정했고 `nan_reason`에 원인을 남겼습니다. |
| 지리/교통 | `num_bus_stops_300m` | 100m보다 안정적이고 500m보다 구역성이 좋아 포함했습니다. |
| 지리/교통 | `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` | 출근, 점심, 저녁 시간대 역세권 수요를 반영합니다. 500m 내 역이 없으면 구조적 0입니다. |
| 상권 | `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m` | 주변 상권 활성도와 생활 편의성을 반영합니다. 서로 상관이 높을 수 있어 선형 모델에서는 공선성 확인이 필요합니다. |
| 상권 | `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 전체 카페 수보다 주변 카페 구성 차이를 더 구체적으로 표현합니다. |
| 상권 | `dist_nearest_starbucks` | 스타벅스 밀집형 입지와 독립 입지를 구분할 해석력이 있어 포함했습니다. 척도가 크므로 변환을 검토합니다. |
| 인구/통계 | `avg_income` | 지역 구매력 지표로 포함했습니다. 원천 결측은 final 생성 과정에서 보정했습니다. |
| 인구/통계 | `num_offices` | 업무지구 성격을 반영하기 위해 포함했습니다. 분포 치우침과 결측 보정 여부를 확인해야 합니다. |
| 인구/통계 | `living_population` | 지역 체류 및 생활 인구 규모를 반영합니다. |
| 인구/통계 | `land_price` | 입지 가치와 상권 수준을 반영합니다. 척도가 커서 log 변환 검토가 필요합니다. |

## 4. 확인 상태

| 구분 | 참고 산출물 |
| --- | --- |
| 최종 컬럼/상태 | `reports/archive/tables/model_feature_v2_columns.csv`, `reports/archive/tables/model_feature_v2_status.csv` |
| 결측/통계 | `reports/archive/tables/model_feature_v2_missing_values_all.csv`, `reports/archive/tables/model_feature_v2_missing_values_starbucks.csv`, `reports/archive/tables/model_feature_v2_summary_starbucks.csv` |
| 신규 feature | `reports/archive/tables/new_feature_v2_summary.csv`, `reports/archive/tables/cafe_brand_taxonomy_summary.csv`, `reports/archive/tables/subway_peak_station_merge_failures.csv` |
| 반경 선택 | `reports/archive/tables/radius_feature_recommendation.csv`, `reports/archive/tables/radius_feature_correlations.csv`, `reports/archive/figures/radius_selection/radius_feature_correlation_heatmap.png` |
| 주요 상관 시각화 | `reports/archive/figures/geo_features/new_geo_feature_correlation_heatmap.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_pearson.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_spearman.png` |
| 클러스터링 고도화 | `reports/starbucks_clustering_enhancement_summary.md`, `reports/archive/tables/clustering/`, `reports/archive/figures/clustering/` |

## 5. 모델링 전 주의사항

- 최종 CSV는 feature 결측 보정을 적용했지만 이상치 제거, scaling, log 변환은 하지 않은 상태입니다.
- 거리 변수는 km 단위입니다.
- 반경 count 변수는 변수명에 반경 단위가 포함되어 있습니다.
- `nan_reason`은 현재 결측 여부가 아니라 보정 전 결측 원인을 설명하는 컬럼입니다. 모델 feature로 직접 쓰기보다 데이터 provenance로 해석하는 편이 좋습니다.
- `dist_nearest_subway`, `dist_nearest_starbucks`, `num_retail_500m`, `num_offices`, `land_price`는 척도와 치우침이 커서 log 변환 또는 robust scaling을 검토합니다.
- 상권 변수들은 서로 상관이 높을 수 있으므로 선형 모델에 사용할 때는 공선성을 확인합니다.
- peak 지하철 변수에서 0은 결측이 아니라 500m 내 역이 없는 구조적 0으로 해석합니다.
