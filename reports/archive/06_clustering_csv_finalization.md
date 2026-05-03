# 06 Clustering CSV Finalization

- 생성 시각: 2026-05-04 00:11:43
- 입력 파일: `data/seoul_cafe_master_with_geo_features.csv`
- 목적: 후보 반경 변수를 모두 담은 master에서 clustering 담당자가 바로 쓸 slim CSV 생성
- 처리 원칙: 원본 파일 덮어쓰기 없음, 결측치 대체 없음, 이상치 제거 없음, clustering 실행 없음

## 1. 출력 파일 shape

| file                                       | rows  | columns |
| ------------------------------------------ | ----- | ------- |
| data/seoul_cafe_clustering_features_v1.csv | 22305 | 15      |
| data/starbucks_clustering_features_v1.csv  | 681   | 15      |

## 2. 포함 컬럼

| order | column                 | role               |
| ----- | ---------------------- | ------------------ |
| 1     | 상호명                    | 식별/해석용             |
| 2     | 브랜드                    | 식별/해석용             |
| 3     | is_starbucks           | 식별/해석용             |
| 4     | 위도                     | 식별/해석용             |
| 5     | 경도                     | 식별/해석용             |
| 6     | 시군구명                   | 식별/해석용             |
| 7     | 행정동코드                  | 식별/해석용             |
| 8     | 행정동명                   | 식별/해석용             |
| 9     | 도로명주소                  | 식별/해석용             |
| 10    | dist_nearest_subway    | clustering feature |
| 11    | subway_ridership_500m  | clustering feature |
| 12    | num_bus_stops_300m     | clustering feature |
| 13    | cafe_count_300m        | clustering feature |
| 14    | dist_nearest_starbucks | clustering feature |
| 15    | num_restaurants_500m   | clustering feature |

## 3. 제외 후보 변수 확인

| excluded_candidate       | present_in_seoul_output | present_in_starbucks_output | reason                                   |
| ------------------------ | ----------------------- | --------------------------- | ---------------------------------------- |
| num_bus_stops_100m       | False                   | False                       | 0값 비율이 높아 너무 희소함                         |
| num_bus_stops_500m       | False                   | False                       | 반경이 넓어 생활권 전체 성격이 강함                     |
| cafe_count_500m          | False                   | False                       | 기존 경쟁 카페 변수와 거의 중복                       |
| cafe_count_1000m         | False                   | False                       | 개별 매장 입지보다 지역 상권 규모를 반영                  |
| num_competing_cafes_500m | False                   | False                       | cafe_count_300m를 직접 경쟁권 대표 변수로 선택했으므로 제외 |
| num_retail_500m          | False                   | False                       | 상권 밀도 변수와 중복 가능성이 높아 clustering 입력에서는 제외 |
| num_convenience_500m     | False                   | False                       | 상권 밀도 변수와 중복 가능성이 높아 clustering 입력에서는 제외 |

## 4. Clustering feature 결측치

| dataset                           | feature                | missing_count | missing_rate |
| --------------------------------- | ---------------------- | ------------- | ------------ |
| seoul_cafe_clustering_features_v1 | dist_nearest_subway    | 0             | 0.0          |
| seoul_cafe_clustering_features_v1 | subway_ridership_500m  | 0             | 0.0          |
| seoul_cafe_clustering_features_v1 | num_bus_stops_300m     | 0             | 0.0          |
| seoul_cafe_clustering_features_v1 | cafe_count_300m        | 0             | 0.0          |
| seoul_cafe_clustering_features_v1 | dist_nearest_starbucks | 0             | 0.0          |
| seoul_cafe_clustering_features_v1 | num_restaurants_500m   | 0             | 0.0          |
| starbucks_clustering_features_v1  | dist_nearest_subway    | 0             | 0.0          |
| starbucks_clustering_features_v1  | subway_ridership_500m  | 0             | 0.0          |
| starbucks_clustering_features_v1  | num_bus_stops_300m     | 0             | 0.0          |
| starbucks_clustering_features_v1  | cafe_count_300m        | 0             | 0.0          |
| starbucks_clustering_features_v1  | dist_nearest_starbucks | 0             | 0.0          |
| starbucks_clustering_features_v1  | num_restaurants_500m   | 0             | 0.0          |

## 5. 스타벅스 파일 기준 기본 통계

| feature                | count | mean         | median   | std          | min      | Q1      | Q3       | max      | zero_count | zero_rate | skewness |
| ---------------------- | ----- | ------------ | -------- | ------------ | -------- | ------- | -------- | -------- | ---------- | --------- | -------- |
| dist_nearest_subway    | 681   | 0.341431     | 0.2725   | 0.307308     | 0.0127   | 0.1402  | 0.4425   | 3.0724   | 0          | 0.0       | 3.289291 |
| subway_ridership_500m  | 681   | 60711.201175 | 43283.0  | 60650.235136 | 0.0      | 18762.0 | 87230.0  | 352395.0 | 125        | 0.183554  | 1.478396 |
| num_bus_stops_300m     | 681   | 9.861968     | 9.0      | 4.596838     | 0.0      | 6.0     | 13.0     | 29.0     | 5          | 0.007342  | 0.693703 |
| cafe_count_300m        | 681   | 40.217327    | 35.0     | 25.14329     | 0.0      | 21.0    | 57.0     | 115.0    | 2          | 0.002937  | 0.665423 |
| dist_nearest_starbucks | 681   | 0.340141     | 0.243029 | 0.294381     | 0.009603 | 0.15324 | 0.438721 | 2.638043 | 0          | 0.0       | 2.456194 |
| num_restaurants_500m   | 681   | 527.834068   | 479.0    | 311.909886   | 6.0      | 280.0   | 735.0    | 1656.0   | 0          | 0.0       | 0.699732 |

## 6. 단위 기록

| feature                | unit_note                |
| ---------------------- | ------------------------ |
| dist_nearest_subway    | km 단위 거리 변수              |
| dist_nearest_starbucks | km 단위 거리 변수              |
| num_bus_stops_300m     | 300m 반경 count. 변수명에 m 유지 |
| cafe_count_300m        | 300m 반경 count. 변수명에 m 유지 |
| subway_ridership_500m  | 500m 반경 지하철 승하차량 집계      |
| num_restaurants_500m   | 500m 반경 음식점 수            |

거리 변수 `dist_nearest_subway`, `dist_nearest_starbucks`는 km 단위로 기록한다. 반경 count 변수 `num_bus_stops_300m`, `cafe_count_300m`는 변수명에 m를 유지해 반경 기준을 명확히 했다.

## 7. 저장 산출물

- `data/seoul_cafe_clustering_features_v1.csv`
- `data/starbucks_clustering_features_v1.csv`
- `reports/06_clustering_csv_finalization.md`
- `reports/tables/clustering_feature_missing_values.csv`
- `reports/tables/clustering_feature_summary_starbucks.csv`
