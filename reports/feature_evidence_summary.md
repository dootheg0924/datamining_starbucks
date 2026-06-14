# Feature Evidence Summary

최종 모델 feature가 왜 포함되었는지, 어떤 통계/시각화 근거를 확인했는지 빠르게 파악하기 위한 핵심 요약 문서다.

- 최종 데이터:
  - `data/starbucks_model_features_final.csv`: 스타벅스 681개 매장 기준 최종 모델 feature
  - `data/seoul_cafe_model_features_final.csv`: 서울 카페 22,305개 기준 최종 모델 feature. `nan_reason`은 보정 전 결측 원인을 기록한 provenance 컬럼이다.
- 압축 분석 기록:
  - `reports/archive/analysis_archive_summary.md`
- 핵심 근거 표:
  - `reports/archive/tables/model_feature_v2_columns.csv`
  - `reports/archive/tables/model_feature_v2_status.csv`
  - `reports/archive/tables/model_feature_v2_summary_starbucks.csv`
  - `reports/archive/tables/new_feature_v2_summary.csv`
  - `reports/archive/tables/radius_feature_recommendation.csv`

## 1. 최종 feature 구성

최종 모델 feature CSV는 식별/해석용 컬럼 7개와 모델 feature 18개로 구성된다. 서울 전체 final CSV에는 추가로 `nan_reason` 1개 컬럼이 있어 총 26개 컬럼이고, 스타벅스 전용 final CSV는 `nan_reason`을 제외한 25개 컬럼이다.

| 구분 | 컬럼 |
| --- | --- |
| 식별/해석 | `상호명`, `브랜드`, `is_starbucks`, `위도`, `경도`, `시군구명`, `도로명주소` |
| 지리/교통 | `dist_nearest_subway`, `num_subway_500m`, `nearest_subway_ridership`, `num_bus_stops_300m`, `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` |
| 상권 | `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m`, `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m`, `dist_nearest_starbucks` |
| 인구/통계 | `avg_income`, `num_offices`, `living_population`, `land_price` |

## 2. 반경 선택 핵심 근거

### 2.1 버스정류장 반경

최종 선택: `num_bus_stops_300m`

| 후보 | 스타벅스 기준 평균 | 중앙값 | 0값 비율 | 판단 |
| --- | ---: | ---: | ---: | --- |
| `num_bus_stops_100m` | 1.72 | 2.0 | 23.79% | 너무 근접한 반경이라 0값이 많고 희소함 |
| `num_bus_stops_300m` | 9.86 | 9.0 | 0.73% | 도보 접근권으로 해석 가능하고 매장 간 차이도 유지됨 |
| `num_bus_stops_500m` | 22.43 | 21.0 | 0.00% | 너무 넓은 생활권 성격이 강하고 300m와 상관이 높음 |

추가 근거:

- `num_bus_stops_300m`와 `num_bus_stops_500m`의 상관은 0.793이다.
- `dist_nearest_bus_stop`은 해석이 명확하지만, `num_bus_stops_300m`와 함께 쓰면 버스 접근성 축이 과대표현될 수 있어 최종 output에서는 제외했다.
- 관련 시각화는 `reports/archive/figures/radius_selection/num_bus_stops_100m_hist.png`, `num_bus_stops_300m_hist.png`, `num_bus_stops_500m_hist.png`를 유지한다.

### 2.2 카페 수 반경

반경 비교 단계에서는 `cafe_count_300m`가 직접 경쟁권 대표로 가장 적합하다고 판단했다. 다만 최종 모델 feature에서는 단순 카페 수 대신 500m 반경의 카페 유형별 count를 사용했다.

| 후보 | 스타벅스 기준 평균 | 중앙값 | 판단 |
| --- | ---: | ---: | --- |
| `cafe_count_300m` | 40.22 | 35.0 | 직접 경쟁권으로 해석하기 쉬워 clustering 후보로 적합 |
| `cafe_count_500m` | 90.01 | 79.0 | 기존 `num_competing_cafes_500m`와 거의 중복 |
| `cafe_count_1000m` | 276.28 | 235.0 | 개별 매장보다 지역 상권 규모 성격이 강함 |

추가 근거:

- `cafe_count_300m`와 `cafe_count_500m`의 상관은 0.905이다.
- `cafe_count_500m`와 기존 `num_competing_cafes_500m`의 상관은 0.999로 사실상 중복이다.
- 최종 모델에서는 상권을 더 구체적으로 표현하기 위해 `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m`를 사용했다.
- 관련 시각화는 `reports/archive/figures/radius_selection/cafe_count_300m_hist.png`, `cafe_count_500m_hist.png`, `cafe_count_1000m_hist.png`를 유지한다.

### 2.3 500m 반경을 유지한 변수

| 변수 | 500m 유지 이유 |
| --- | --- |
| `num_subway_500m` | 역세권 여부와 주변 지하철 접근성을 해석하기 위한 기존 변수 |
| `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` | 500m 내 지하철역의 시간대별 승하차 규모를 합산해 역세권 수요를 반영 |
| `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m` | 기존 master의 상권 밀도 변수로, 주변 상권 규모를 같은 반경에서 비교 가능 |
| `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 단순 카페 수 대신 카페 유형별 경쟁/상권 구성을 표현 |

## 3. 최종 포함 변수별 근거

| 변수군 | 변수 | 최종 판단 및 모델링 주의점 |
| --- | --- | --- |
| 지리/교통 | `dist_nearest_subway` | 지하철 접근성의 방향성이 명확해 포함. 왜도가 커서 모델링 전 scaling 또는 log 변환 검토 |
| 지리/교통 | `num_subway_500m` | 역세권 여부를 해석하기 쉬워 포함. 이산형 변수이고 IQR이 작아 이상치 해석 주의 |
| 지리/교통 | `nearest_subway_ridership` | 가까운 역의 규모를 직접 반영하므로 포함. 원천 결측은 final 생성 과정에서 보정하고 `nan_reason`에 원인을 남김 |
| 지리/교통 | `num_bus_stops_300m` | 100m보다 안정적이고 500m보다 국지성이 좋아 포함 |
| 지리/교통 | `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` | 출근/점심/저녁 시간대 역세권 수요를 반영하기 위해 포함. 500m 내 역이 없으면 구조적 0 |
| 상권 | `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m` | 주변 상권 활성도와 생활 상권 밀도를 반영. 서로 상관이 높을 수 있어 선형 모델에서는 공선성 확인 필요 |
| 상권 | `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 단순 전체 카페 수보다 주변 카페 구성을 더 잘 표현하므로 포함 |
| 상권 | `dist_nearest_starbucks` | 스타벅스 밀집형/독립 입지를 구분하는 해석력이 있어 포함. 왜도가 커서 변환 검토 |
| 인구/통계 | `avg_income` | 구매력/지역 소득 수준을 반영하기 위해 포함. 원천 결측은 final 생성 과정에서 보정 |
| 인구/통계 | `num_offices` | 업무지구 성격을 반영하기 위해 포함. 원천 결측 보정 후에도 왜도 처리 검토 필요 |
| 인구/통계 | `living_population` | 지역 체류/생활 규모를 반영하기 위해 포함. 원천 결측은 final 생성 과정에서 보정 |
| 인구/통계 | `land_price` | 입지 가치와 상권 수준을 반영하기 위해 포함. 왜도가 커서 log 변환 검토 |

## 4. 확인 상태

| 구분 | 유지 근거 |
| --- | --- |
| 최종 컬럼/상태 | `reports/archive/tables/model_feature_v2_columns.csv`, `reports/archive/tables/model_feature_v2_status.csv` |
| 결측/통계 | `reports/archive/tables/model_feature_v2_missing_values_all.csv`, `reports/archive/tables/model_feature_v2_missing_values_starbucks.csv`, `reports/archive/tables/model_feature_v2_summary_starbucks.csv` |
| 신규 feature | `reports/archive/tables/new_feature_v2_summary.csv`, `reports/archive/tables/cafe_brand_taxonomy_summary.csv`, `reports/archive/tables/subway_peak_station_merge_failures.csv` |
| 반경 선택 | `reports/archive/tables/radius_feature_recommendation.csv`, `reports/archive/tables/radius_feature_correlations.csv`, `reports/archive/figures/radius_selection/radius_feature_correlation_heatmap.png` |
| 주요 상관 시각화 | `reports/archive/figures/geo_features/new_geo_feature_correlation_heatmap.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_pearson.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_spearman.png` |

## 5. 다음 담당자에게 전달할 모델링 주의점

- 최종 CSV는 feature 결측 보정을 적용했지만, 이상치 제거, scaling, log 변환은 하지 않은 상태다.
- 거리 변수는 km 단위다.
- 반경 count 변수는 변수명에 반경 단위가 포함되어 있다.
- `nan_reason`은 현재 결측 여부가 아니라 보정 전 결측 원인을 설명하는 컬럼이므로, 모델 feature로 직접 쓰기보다 데이터 provenance로 해석하는 것이 좋다.
- `dist_nearest_subway`, `dist_nearest_starbucks`, `num_retail_500m`, `num_offices`, `land_price`는 왜도가 커서 log 변환 또는 robust scaling 검토가 필요하다.
- 상권 밀도 변수들은 서로 상관이 높을 수 있으므로, 선형 모델을 사용할 경우 다중공선성을 확인하는 것이 좋다.
- peak 지하철 변수는 500m 내 역이 없는 경우 0이다. 이 0은 결측이 아니라 구조적 0으로 해석해야 한다.
