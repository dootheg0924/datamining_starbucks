# Feature Evidence Summary

- 작성 목적: 최종 모델 feature가 왜 포함되었는지, 어떤 통계/시각화 근거를 확인했는지 다음 담당자가 빠르게 파악하도록 정리한다.
- 최종 데이터:
  - `data/starbucks_model_features_v1.csv`: 스타벅스 681개 매장 기준 최종 모델 feature
  - `data/seoul_cafe_model_features_v1.csv`: 서울 카페 22,305개 기준 최종 모델 feature
- 기준 보고서:
  - `reports/07_model_feature_finalization_v2.md`: 최종 feature 확정 문서
  - `reports/04_geo_feature_engineering.md`: 좌표 기반 geo feature 생성 문서
  - `reports/archive/05_radius_selection_eda.md`: 반경 후보 비교 및 선택 근거
  - `reports/archive/02_starbucks_only_eda.md`: 기존 master 변수의 스타벅스 기준 EDA

## 1. 최종 feature 구성

최종 CSV는 총 25개 컬럼이다. 이 중 7개는 식별/해석용 컬럼이고, 모델 feature는 18개다.

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
- 관련 시각화:
  - `reports/archive/figures/radius_selection/num_bus_stops_100m_hist.png`
  - `reports/archive/figures/radius_selection/num_bus_stops_300m_hist.png`
  - `reports/archive/figures/radius_selection/num_bus_stops_500m_hist.png`

### 2.2 카페 수 반경

반경 비교 단계에서는 `cafe_count_300m`가 직접 경쟁권 대표로 가장 적합하다고 판단했다. 다만 최종 v2 모델 feature에서는 단순 카페 수 대신 500m 반경의 카페 유형별 count를 사용했다.

| 후보 | 스타벅스 기준 평균 | 중앙값 | 판단 |
| --- | ---: | ---: | --- |
| `cafe_count_300m` | 40.22 | 35.0 | 직접 경쟁권으로 해석하기 쉬워 clustering 후보로 적합 |
| `cafe_count_500m` | 90.01 | 79.0 | 기존 `num_competing_cafes_500m`와 거의 중복 |
| `cafe_count_1000m` | 276.28 | 235.0 | 개별 매장보다 지역 상권 규모 성격이 강함 |

추가 근거:

- `cafe_count_300m`와 `cafe_count_500m`의 상관은 0.905이다.
- `cafe_count_500m`와 기존 `num_competing_cafes_500m`의 상관은 0.999로 사실상 중복이다.
- 최종 v2에서는 상권을 더 구체적으로 표현하기 위해 `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m`를 사용했다.
- 관련 시각화:
  - `reports/archive/figures/radius_selection/cafe_count_300m_hist.png`
  - `reports/archive/figures/radius_selection/cafe_count_500m_hist.png`
  - `reports/archive/figures/radius_selection/cafe_count_1000m_hist.png`

### 2.3 500m 반경을 유지한 변수

아래 변수들은 원천 master 또는 최종 v2 정의에서 500m 반경을 사용했다.

| 변수 | 500m 유지 이유 |
| --- | --- |
| `num_subway_500m` | 역세권 여부와 주변 지하철 접근성을 해석하기 위한 기존 변수 |
| `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m` | 500m 내 지하철역의 시간대별 승하차 규모를 합산해 역세권 수요를 반영 |
| `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m` | 기존 master의 상권 밀도 변수로, 주변 상권 규모를 같은 반경에서 비교 가능 |
| `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 단순 카페 수 대신 카페 유형별 경쟁/상권 구성을 표현 |

## 3. 최종 포함 변수별 근거

### 3.1 지리/교통

| 변수 | 정의 | 확인한 근거 | 시각화 | 최종 판단 및 주의점 |
| --- | --- | --- | --- | --- |
| `dist_nearest_subway` | 가장 가까운 지하철역까지 거리. km 단위 | 스타벅스 기준 평균 0.341km, 중앙값 0.273km, 왜도 3.29, 결측 없음 | `reports/archive/figures/starbucks_only/dist_nearest_subway_hist.png`, `reports/archive/figures/starbucks_only/dist_nearest_subway_boxplot.png`, `reports/archive/figures/starbucks_only/dist_nearest_subway_log1p_hist.png` | 지하철 접근성의 방향성이 명확해 포함. 왜도가 커서 모델링 전 scaling 또는 log 변환 검토 |
| `num_subway_500m` | 500m 내 지하철역 수 | 평균 1.0, 중앙값 1.0, 0값 비율 18.06%, 결측 없음 | `reports/archive/figures/starbucks_only/num_subway_500m_hist.png`, `reports/archive/figures/starbucks_only/num_subway_500m_boxplot.png` | 역세권 여부를 해석하기 쉬워 포함. 이산형 변수이고 IQR이 작아 이상치 해석 주의 |
| `nearest_subway_ridership` | 가장 가까운 지하철역의 승하차 규모 | 스타벅스 기준 결측 8개, 평균 59,446, 중앙값 46,029, 왜도 1.37 | `reports/archive/figures/starbucks_only/nearest_subway_ridership_hist.png`, `reports/archive/figures/starbucks_only/nearest_subway_ridership_boxplot.png`, `reports/archive/figures/starbucks_only/nearest_subway_ridership_log1p_hist.png` | 가까운 역의 규모를 직접 반영하므로 포함. 결측과 왜도 처리 필요 |
| `num_bus_stops_300m` | 300m 반경 내 버스정류장 수 | 반경 비교 결과 선택. 스타벅스 기준 평균 9.86, 중앙값 9, 0값 비율 0.73%, 결측 없음 | `reports/archive/figures/radius_selection/num_bus_stops_300m_hist.png` | 100m보다 안정적이고 500m보다 국지성이 좋아 포함 |
| `subway_morning_peak_500m` | 500m 내 지하철역의 Morning Peak 승하차량 합. 07-10시 | 스타벅스 기준 평균 441,009, 중앙값 316,405, 0값 비율 17.47%, 결측 없음 | 별도 단일 변수 그래프 없음. 통계는 `reports/07_model_feature_finalization_v2.md`와 `reports/archive/tables/new_feature_v2_summary.csv` 확인 | 출근 시간대 역세권 수요를 반영하기 위해 포함. 500m 내 역이 없으면 0 |
| `subway_lunch_peak_500m` | 500m 내 지하철역의 Lunch Peak 승하차량 합. 11-14시 | 스타벅스 기준 평균 281,017, 중앙값 185,671, 0값 비율 17.47%, 결측 없음 | 별도 단일 변수 그래프 없음 | 점심 시간대 유동 수요를 반영하기 위해 포함 |
| `subway_evening_peak_500m` | 500m 내 지하철역의 Afternoon/Evening 승하차량 합. 15-20시 | 스타벅스 기준 평균 739,374, 중앙값 484,485, 0값 비율 17.47%, 결측 없음 | 별도 단일 변수 그래프 없음 | 퇴근/저녁 시간대 역세권 수요를 반영하기 위해 포함 |

### 3.2 상권

| 변수 | 정의 | 확인한 근거 | 시각화 | 최종 판단 및 주의점 |
| --- | --- | --- | --- | --- |
| `num_restaurants_500m` | 500m 반경 내 음식점 수 | 스타벅스 기준 평균 527.83, 중앙값 479, 왜도 0.70, 결측 없음 | `reports/archive/figures/starbucks_only/num_restaurants_500m_hist.png`, `reports/archive/figures/starbucks_only/num_restaurants_500m_boxplot.png` | 주변 상권 활성도를 나타내는 대표 변수로 포함. 카페/편의점 밀도와 상관이 높을 수 있음 |
| `num_retail_500m` | 500m 반경 내 소매업 수 | 평균 431.90, 중앙값 324, 왜도 2.71, 결측 없음 | `reports/archive/figures/starbucks_only/num_retail_500m_hist.png`, `reports/archive/figures/starbucks_only/num_retail_500m_boxplot.png`, `reports/archive/figures/starbucks_only/num_retail_500m_log1p_hist.png` | 상권 규모를 보조적으로 반영하기 위해 포함. 왜도가 커서 log 변환 검토 |
| `num_convenience_500m` | 500m 반경 내 편의점 수 | 평균 30.08, 중앙값 27, 왜도 0.62, 결측 없음 | `reports/archive/figures/starbucks_only/num_convenience_500m_hist.png`, `reports/archive/figures/starbucks_only/num_convenience_500m_boxplot.png` | 생활 상권 밀도를 보조적으로 반영하기 위해 포함. 음식점 수와 상관이 높을 수 있음 |
| `independent_cafe_count_500m` | 500m 반경 내 독립 카페 수. 스타벅스 제외 | 전체 평균 67.52, 스타벅스 평균 75.14, 스타벅스 기준 결측 없음 | 별도 단일 변수 그래프 없음. 통계는 `reports/archive/tables/new_feature_v2_summary.csv` 확인 | 단순 전체 카페 수보다 주변 카페 구성을 더 잘 표현하므로 포함 |
| `low_price_cafe_count_500m` | 500m 반경 내 저가 프랜차이즈 카페 수. 메가MGC커피, 빽다방 기준 | 전체 평균 3.63, 스타벅스 평균 4.19, 스타벅스 기준 0값 비율 4.41% | 별도 단일 변수 그래프 없음 | 가격대가 다른 프랜차이즈 경쟁 구도를 반영하기 위해 포함 |
| `other_franchise_cafe_count_500m` | 500m 반경 내 기타 프랜차이즈 카페 수. 이디야커피, 투썸플레이스, 기타프랜차이즈 기준 | 전체 평균 5.82, 스타벅스 평균 7.21, 스타벅스 기준 0값 비율 4.11% | 별도 단일 변수 그래프 없음 | 스타벅스 외 프랜차이즈 경쟁 구도를 반영하기 위해 포함 |
| `dist_nearest_starbucks` | 가장 가까운 다른 스타벅스까지 거리. km 단위 | 스타벅스 기준 평균 0.340km, 중앙값 0.243km, 왜도 2.46, 결측 없음 | `reports/archive/figures/geo_features/dist_nearest_starbucks_hist.png`, `reports/archive/figures/radius_selection/dist_nearest_starbucks_hist.png` | 스타벅스 밀집형/독립 입지를 구분하는 해석력이 있어 포함 |

### 3.3 인구/통계

| 변수 | 정의 | 확인한 근거 | 시각화 | 최종 판단 및 주의점 |
| --- | --- | --- | --- | --- |
| `avg_income` | 행정동 또는 지역 단위 평균 소득 | 스타벅스 기준 결측 26개, 평균 3,940,507, 중앙값 3,607,265, 왜도 0.65 | `reports/archive/figures/starbucks_only/avg_income_hist.png`, `reports/archive/figures/starbucks_only/avg_income_boxplot.png` | 구매력/지역 소득 수준을 반영하기 위해 포함. 결측 처리 필요 |
| `num_offices` | 주변 또는 지역 단위 오피스 수 | 스타벅스 기준 결측 30개, 평균 45,775, 중앙값 15,114, 왜도 1.85 | `reports/archive/figures/starbucks_only/num_offices_hist.png`, `reports/archive/figures/starbucks_only/num_offices_boxplot.png`, `reports/archive/figures/starbucks_only/num_offices_log1p_hist.png` | 업무지구 성격을 반영하기 위해 포함. 왜도와 결측 처리 필요 |
| `living_population` | 생활인구 | 스타벅스 기준 결측 29개, 평균 39,124, 중앙값 35,699, 왜도 1.32 | `reports/archive/figures/starbucks_only/living_population_hist.png`, `reports/archive/figures/starbucks_only/living_population_boxplot.png`, `reports/archive/figures/starbucks_only/living_population_log1p_hist.png` | 지역 체류/생활 규모를 반영하기 위해 포함. 결측 처리 필요 |
| `land_price` | 공시지가 또는 토지가격 계열 변수 | 스타벅스 기준 평균 8,639,840, 중앙값 6,399,509, 왜도 1.93, 결측 없음 | `reports/archive/figures/starbucks_only/land_price_hist.png`, `reports/archive/figures/starbucks_only/land_price_boxplot.png`, `reports/archive/figures/starbucks_only/land_price_log1p_hist.png` | 입지 가치와 상권 수준을 반영하기 위해 포함. 왜도가 커서 log 변환 검토 |


## 4. 시각화/통계 확인 상태

| 구분 | 변수 | 상태 |
| --- | --- | --- |
| 시각화와 통계 모두 확인 | `dist_nearest_subway`, `num_subway_500m`, `nearest_subway_ridership`, `num_bus_stops_300m`, `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m`, `dist_nearest_starbucks`, `avg_income`, `num_offices`, `living_population`, `land_price` | 기존 EDA 또는 반경 선택 EDA에서 histogram/boxplot/log histogram 일부 확인 |
| 통계 확인 중심 | `subway_morning_peak_500m`, `subway_lunch_peak_500m`, `subway_evening_peak_500m`, `independent_cafe_count_500m`, `low_price_cafe_count_500m`, `other_franchise_cafe_count_500m` | 최종 v2에서 생성된 변수. 기본 통계, 결측치, 0값 비율은 확인했으나 별도 단일 변수 시각화는 생성하지 않음 |
| 반경 비교 시각화 확인 | `num_bus_stops_100m`, `num_bus_stops_300m`, `num_bus_stops_500m`, `cafe_count_300m`, `cafe_count_500m`, `cafe_count_1000m` | 반경 선택 과정에서 분포와 상관을 확인 |
| 상관관계 확인 | 주요 지리/상권 후보 변수 | `reports/archive/figures/radius_selection/radius_feature_correlation_heatmap.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_pearson.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_spearman.png` |

## 6. 다음 담당자에게 전달할 모델링 주의점

- 최종 CSV는 결측치 대체, 이상치 제거, scaling, log 변환을 하지 않은 상태다.
- 거리 변수는 km 단위다.
- 반경 count 변수는 변수명에 반경 단위가 포함되어 있다.
- `avg_income`, `num_offices`, `living_population`, `nearest_subway_ridership`는 스타벅스 기준 결측이 있으므로 모델링 전에 처리 방법을 정해야 한다.
- `dist_nearest_subway`, `dist_nearest_starbucks`, `num_retail_500m`, `num_offices`, `land_price`는 왜도가 커서 log 변환 또는 robust scaling 검토가 필요하다.
- 상권 밀도 변수들은 서로 상관이 높을 수 있으므로, 선형 모델을 사용할 경우 다중공선성을 확인하는 것이 좋다.
- peak 지하철 변수는 500m 내 역이 없는 경우 0이다. 이 0은 결측이 아니라 구조적 0으로 해석해야 한다.
