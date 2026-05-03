# 07 Model Feature Finalization v2

- 생성 시각: 2026-05-04 00:50:11
- base file: `data/seoul_cafe_master_with_geo_features.csv` (utf-8-sig)
- subway peak file: `subway_time_group_analysis.csv` (utf-8-sig)
- station master file: `서울시_역사마스터_정보 (1).csv` (euc-kr)
- 처리 원칙: 원본 덮어쓰기 없음, 결측치 대체 없음, 이상치 제거 없음, clustering/classification 실행 없음

## 1. 기존 방향 수정

이전 slim CSV는 clustering에 필요한 변수 일부만 남겨 원래 변수 체계를 충분히 반영하지 못했다. 이번 v2에서는 사용자가 확정한 변수 체계를 기준으로 지리/교통, 상권, 인구/통계 변수를 균형 있게 포함했다.

## 2. 출력 파일 shape

| file                                  | rows  | columns |
| ------------------------------------- | ----- | ------- |
| data/seoul_cafe_model_features_v1.csv | 22305 | 25      |
| data/starbucks_model_features_v1.csv  | 681   | 25      |

## 3. 최종 컬럼 목록

| order | column                          | role              | group |
| ----- | ------------------------------- | ----------------- | ----- |
| 1     | 상호명                             | id_interpretation | 식별/해석 |
| 2     | 브랜드                             | id_interpretation | 식별/해석 |
| 3     | is_starbucks                    | id_interpretation | 식별/해석 |
| 4     | 위도                              | id_interpretation | 식별/해석 |
| 5     | 경도                              | id_interpretation | 식별/해석 |
| 6     | 시군구명                            | id_interpretation | 식별/해석 |
| 7     | 도로명주소                           | id_interpretation | 식별/해석 |
| 8     | dist_nearest_subway             | model_feature     | 지리/교통 |
| 9     | num_subway_500m                 | model_feature     | 지리/교통 |
| 10    | nearest_subway_ridership        | model_feature     | 지리/교통 |
| 11    | num_bus_stops_300m              | model_feature     | 지리/교통 |
| 12    | subway_morning_peak_500m        | model_feature     | 지리/교통 |
| 13    | subway_lunch_peak_500m          | model_feature     | 지리/교통 |
| 14    | subway_evening_peak_500m        | model_feature     | 지리/교통 |
| 15    | num_restaurants_500m            | model_feature     | 상권    |
| 16    | num_retail_500m                 | model_feature     | 상권    |
| 17    | num_convenience_500m            | model_feature     | 상권    |
| 18    | independent_cafe_count_500m     | model_feature     | 상권    |
| 19    | low_price_cafe_count_500m       | model_feature     | 상권    |
| 20    | other_franchise_cafe_count_500m | model_feature     | 상권    |
| 21    | dist_nearest_starbucks          | model_feature     | 상권    |
| 22    | avg_income                      | model_feature     | 인구/통계 |
| 23    | num_offices                     | model_feature     | 인구/통계 |
| 24    | living_population               | model_feature     | 인구/통계 |
| 25    | land_price                      | model_feature     | 인구/통계 |

## 4. 변수 상태

| feature                         | status              |
| ------------------------------- | ------------------- |
| dist_nearest_subway             | 기존 master 변수 그대로 사용 |
| num_subway_500m                 | 기존 master 변수 그대로 사용 |
| nearest_subway_ridership        | 기존 master 변수 그대로 사용 |
| num_restaurants_500m            | 기존 master 변수 그대로 사용 |
| num_retail_500m                 | 기존 master 변수 그대로 사용 |
| num_convenience_500m            | 기존 master 변수 그대로 사용 |
| avg_income                      | 기존 master 변수 그대로 사용 |
| num_offices                     | 기존 master 변수 그대로 사용 |
| living_population               | 기존 master 변수 그대로 사용 |
| land_price                      | 기존 master 변수 그대로 사용 |
| num_bus_stops_300m              | 이전 단계에서 생성되어 가져옴    |
| dist_nearest_starbucks          | 이전 단계에서 생성되어 가져옴    |
| subway_morning_peak_500m        | 이번 v2에서 새로 생성       |
| subway_lunch_peak_500m          | 이번 v2에서 새로 생성       |
| subway_evening_peak_500m        | 이번 v2에서 새로 생성       |
| independent_cafe_count_500m     | 이번 v2에서 새로 생성       |
| low_price_cafe_count_500m       | 이번 v2에서 새로 생성       |
| other_franchise_cafe_count_500m | 이번 v2에서 새로 생성       |
| subway_ridership_500m           | 최종 output에서 제외      |
| num_competing_cafes_500m        | 최종 output에서 제외      |
| dist_nearest_bus_stop           | 최종 output에서 제외      |
| num_bus_stops_100m              | 최종 output에서 제외      |
| num_bus_stops_500m              | 최종 output에서 제외      |
| cafe_count_300m                 | 최종 output에서 제외      |
| cafe_count_500m                 | 최종 output에서 제외      |
| cafe_count_1000m                | 최종 output에서 제외      |
| premium_cafe_count_500m         | 최종 output에서 제외      |

## 5. 지하철 peak 변수

`subway_time_group_analysis.csv`의 기존 peak 집계 컬럼을 사용했다. 사용자가 확인한 시간대 정의는 Morning Peak=07-10, Lunch Peak=11-14, Afternoon/Evening=15-20이다.

| metric                        | value    |
| ----------------------------- | -------- |
| peak station normalized pairs | 638.0    |
| merge success                 | 625.0    |
| merge failures                | 13.0     |
| merge success rate            | 0.979624 |

- 병합 실패 목록 저장: `reports/tables/subway_peak_station_merge_failures.csv`
- 병합 key: 호선명/역명에서 공백, 괄호, 일부 호선 suffix를 정규화한 key
- 반경 500m 내 지하철역이 없는 카페는 peak 변수 값을 0으로 두었다.

## 6. 카페 분류 수정

기존 premium/low price 분류는 현재 브랜드 데이터 구조상 애매했다. 따라서 이번 버전에서는 주변 카페를 independent, low price franchise, other franchise 세 가지로 나누었다. 스타벅스는 이 세 count에 포함하지 않고, `dist_nearest_starbucks`로 별도 반영했다.

| feature                         | brand_group            | brand_rows_in_master | mean_all  | median_all | mean_starbucks | median_starbucks |
| ------------------------------- | ---------------------- | -------------------- | --------- | ---------- | -------------- | ---------------- |
| independent_cafe_count_500m     | 프랜차이즈외                 | 18689                | 67.520108 | 53.0       | 75.142438      | 64.0             |
| low_price_cafe_count_500m       | 메가MGC커피, 빽다방           | 1213                 | 3.625017  | 3.0        | 4.193833       | 4.0              |
| other_franchise_cafe_count_500m | 이디야커피, 투썸플레이스, 기타프랜차이즈 | 1722                 | 5.815288  | 5.0        | 7.207048       | 6.0              |
| 스타벅스 처리                         | 스타벅스                   | 681                  |           |            |                |                  |

## 7. 최종 feature 결측치: 전체 데이터

| feature                         | missing_count | missing_rate |
| ------------------------------- | ------------- | ------------ |
| dist_nearest_subway             | 0             | 0.0          |
| num_subway_500m                 | 0             | 0.0          |
| nearest_subway_ridership        | 291           | 0.013046     |
| num_bus_stops_300m              | 0             | 0.0          |
| subway_morning_peak_500m        | 0             | 0.0          |
| subway_lunch_peak_500m          | 0             | 0.0          |
| subway_evening_peak_500m        | 0             | 0.0          |
| num_restaurants_500m            | 0             | 0.0          |
| num_retail_500m                 | 0             | 0.0          |
| num_convenience_500m            | 0             | 0.0          |
| independent_cafe_count_500m     | 0             | 0.0          |
| low_price_cafe_count_500m       | 0             | 0.0          |
| other_franchise_cafe_count_500m | 0             | 0.0          |
| dist_nearest_starbucks          | 0             | 0.0          |
| avg_income                      | 181           | 0.008115     |
| num_offices                     | 328           | 0.014705     |
| living_population               | 392           | 0.017575     |
| land_price                      | 0             | 0.0          |

## 8. 최종 feature 결측치: 스타벅스 데이터

| feature                         | missing_count | missing_rate |
| ------------------------------- | ------------- | ------------ |
| dist_nearest_subway             | 0             | 0.0          |
| num_subway_500m                 | 0             | 0.0          |
| nearest_subway_ridership        | 8             | 0.011747     |
| num_bus_stops_300m              | 0             | 0.0          |
| subway_morning_peak_500m        | 0             | 0.0          |
| subway_lunch_peak_500m          | 0             | 0.0          |
| subway_evening_peak_500m        | 0             | 0.0          |
| num_restaurants_500m            | 0             | 0.0          |
| num_retail_500m                 | 0             | 0.0          |
| num_convenience_500m            | 0             | 0.0          |
| independent_cafe_count_500m     | 0             | 0.0          |
| low_price_cafe_count_500m       | 0             | 0.0          |
| other_franchise_cafe_count_500m | 0             | 0.0          |
| dist_nearest_starbucks          | 0             | 0.0          |
| avg_income                      | 26            | 0.038179     |
| num_offices                     | 30            | 0.044053     |
| living_population               | 29            | 0.042584     |
| land_price                      | 0             | 0.0          |

## 9. 스타벅스 681개 기준 기본 통계

| feature                         | count | mean           | median    | std            | min       | Q1        | Q3         | max        | zero_count | zero_rate | skewness |
| ------------------------------- | ----- | -------------- | --------- | -------------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | -------- |
| dist_nearest_subway             | 681   | 0.341431       | 0.2725    | 0.307308       | 0.0127    | 0.1402    | 0.4425     | 3.0724     | 0          | 0.0       | 3.289291 |
| num_subway_500m                 | 681   | 1.0            | 1.0       | 0.634776       | 0.0       | 1.0       | 1.0        | 4.0        | 123        | 0.180617  | 0.518977 |
| nearest_subway_ridership        | 673   | 59446.173848   | 46029.0   | 43107.702338   | 4235.0    | 28961.0   | 83493.0    | 238671.0   | 0          | 0.0       | 1.365557 |
| num_bus_stops_300m              | 681   | 9.861968       | 9.0       | 4.596838       | 0.0       | 6.0       | 13.0       | 29.0       | 5          | 0.007342  | 0.693703 |
| subway_morning_peak_500m        | 681   | 441008.577093  | 316405.0  | 446046.010227  | 0.0       | 120198.0  | 635608.0   | 2606832.0  | 119        | 0.174743  | 1.499606 |
| subway_lunch_peak_500m          | 681   | 281017.295154  | 185671.0  | 298521.92695   | 0.0       | 73728.0   | 392649.0   | 1586750.0  | 119        | 0.174743  | 1.586472 |
| subway_evening_peak_500m        | 681   | 739373.550661  | 484485.0  | 766006.346572  | 0.0       | 182991.0  | 1084419.0  | 4171883.0  | 119        | 0.174743  | 1.473186 |
| num_restaurants_500m            | 681   | 527.834068     | 479.0     | 311.909886     | 6.0       | 280.0     | 735.0      | 1656.0     | 0          | 0.0       | 0.699732 |
| num_retail_500m                 | 681   | 431.89721      | 324.0     | 391.667372     | 5.0       | 192.0     | 518.0      | 2886.0     | 0          | 0.0       | 2.712894 |
| num_convenience_500m            | 681   | 30.0837        | 27.0      | 15.564299      | 1.0       | 19.0      | 40.0       | 85.0       | 0          | 0.0       | 0.617344 |
| independent_cafe_count_500m     | 681   | 75.142438      | 64.0      | 49.547974      | 1.0       | 39.0      | 103.0      | 263.0      | 0          | 0.0       | 1.008811 |
| low_price_cafe_count_500m       | 681   | 4.193833       | 4.0       | 2.508929       | 0.0       | 3.0       | 5.0        | 13.0       | 30         | 0.044053  | 0.763934 |
| other_franchise_cafe_count_500m | 681   | 7.207048       | 6.0       | 5.098703       | 0.0       | 3.0       | 10.0       | 27.0       | 28         | 0.041116  | 0.925439 |
| dist_nearest_starbucks          | 681   | 0.340141       | 0.243029  | 0.294381       | 0.009603  | 0.15324   | 0.438721   | 2.638043   | 0          | 0.0       | 2.456194 |
| avg_income                      | 655   | 3940506.683969 | 3607265.0 | 1091893.110713 | 2152244.0 | 3130657.0 | 4846324.5  | 7421305.0  | 0          | 0.0       | 0.650863 |
| num_offices                     | 651   | 45774.884793   | 15114.0   | 63728.541549   | 29.0      | 3005.0    | 62687.0    | 250897.0   | 0          | 0.0       | 1.845503 |
| living_population               | 652   | 39123.653374   | 35698.5   | 21304.248611   | 5870.0    | 24288.0   | 46455.25   | 98668.0    | 0          | 0.0       | 1.315541 |
| land_price                      | 681   | 8639839.735683 | 6399509.0 | 6336621.55539  | 1098262.0 | 3864306.0 | 13166568.0 | 59244720.0 | 0          | 0.0       | 1.931309 |

## 10. 이번 v2 신규 변수 기본 통계

| feature                         | count | mean          | median   | std           | min | Q1       | Q3        | max       | zero_count | zero_rate | skewness | dataset   |
| ------------------------------- | ----- | ------------- | -------- | ------------- | --- | -------- | --------- | --------- | ---------- | --------- | -------- | --------- |
| subway_morning_peak_500m        | 22305 | 318799.037122 | 213387.0 | 379023.164371 | 0.0 | 0.0      | 464960.0  | 2606832.0 | 6546       | 0.293477  | 1.825534 | all       |
| subway_lunch_peak_500m          | 22305 | 206465.912889 | 120795.0 | 256599.96372  | 0.0 | 0.0      | 298401.0  | 1722940.0 | 6546       | 0.293477  | 1.846151 | all       |
| subway_evening_peak_500m        | 22305 | 538003.76633  | 322074.0 | 662081.703503 | 0.0 | 0.0      | 783543.0  | 4466485.0 | 6546       | 0.293477  | 1.817016 | all       |
| independent_cafe_count_500m     | 22305 | 67.520108     | 53.0     | 49.853008     | 0.0 | 31.0     | 92.0      | 281.0     | 28         | 0.001255  | 1.323711 | all       |
| low_price_cafe_count_500m       | 22305 | 3.625017      | 3.0      | 2.33624       | 0.0 | 2.0      | 5.0       | 15.0      | 1255       | 0.056265  | 0.874253 | all       |
| other_franchise_cafe_count_500m | 22305 | 5.815288      | 5.0      | 4.698815      | 0.0 | 2.0      | 8.0       | 28.0      | 1309       | 0.058686  | 1.180167 | all       |
| subway_morning_peak_500m        | 681   | 441008.577093 | 316405.0 | 446046.010227 | 0.0 | 120198.0 | 635608.0  | 2606832.0 | 119        | 0.174743  | 1.499606 | starbucks |
| subway_lunch_peak_500m          | 681   | 281017.295154 | 185671.0 | 298521.92695  | 0.0 | 73728.0  | 392649.0  | 1586750.0 | 119        | 0.174743  | 1.586472 | starbucks |
| subway_evening_peak_500m        | 681   | 739373.550661 | 484485.0 | 766006.346572 | 0.0 | 182991.0 | 1084419.0 | 4171883.0 | 119        | 0.174743  | 1.473186 | starbucks |
| independent_cafe_count_500m     | 681   | 75.142438     | 64.0     | 49.547974     | 1.0 | 39.0     | 103.0     | 263.0     | 0          | 0.0       | 1.008811 | starbucks |
| low_price_cafe_count_500m       | 681   | 4.193833      | 4.0      | 2.508929      | 0.0 | 3.0      | 5.0       | 13.0      | 30         | 0.044053  | 0.763934 | starbucks |
| other_franchise_cafe_count_500m | 681   | 7.207048      | 6.0      | 5.098703      | 0.0 | 3.0      | 10.0      | 27.0      | 28         | 0.041116  | 0.925439 | starbucks |

## 11. 단위 기록

| item           | unit_note                                                        |
| -------------- | ---------------------------------------------------------------- |
| 거리 변수          | km 단위                                                            |
| 반경 count 변수    | 변수명에 m 단위 유지                                                     |
| 지하철 peak 변수    | 500m 반경 내 peak 승하차량 합. Morning=07-10, Lunch=11-14, Evening=15-20 |
| 카페 분류 count 변수 | 500m 반경 count                                                    |

## 12. 저장 산출물

- `data/seoul_cafe_model_features_v1.csv`
- `data/starbucks_model_features_v1.csv`
- `reports/07_model_feature_finalization_v2.md`
- `reports/tables/model_feature_v2_columns.csv`
- `reports/tables/model_feature_v2_status.csv`
- `reports/tables/model_feature_v2_missing_values_all.csv`
- `reports/tables/model_feature_v2_missing_values_starbucks.csv`
- `reports/tables/model_feature_v2_summary_starbucks.csv`
- `reports/tables/subway_peak_station_merge_failures.csv`
- `reports/tables/cafe_brand_taxonomy_summary.csv`
- `reports/tables/new_feature_v2_summary.csv`
