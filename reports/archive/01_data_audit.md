# 01 Data Audit: seoul_cafe_master.csv

- 생성 시각: 2026-05-03 23:25:25
- 원본 파일: `seoul_cafe_master.csv`
- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, 모델링 없음

## 1. 데이터 크기

- 전체 행 수: 22,305
- 전체 열 수: 21
- 스타벅스 필터 행 수 (`df_starbucks`): 681

## 2. 컬럼명과 dtype

| column                   | dtype   |
| ------------------------ | ------- |
| 상호명                      | str     |
| 브랜드                      | str     |
| is_starbucks             | int64   |
| 위도                       | float64 |
| 경도                       | float64 |
| 시군구명                     | str     |
| 행정동코드                    | float64 |
| 행정동명                     | str     |
| 도로명주소                    | str     |
| dist_nearest_subway      | float64 |
| num_subway_500m          | int64   |
| nearest_subway_ridership | float64 |
| subway_ridership_500m    | float64 |
| num_competing_cafes_500m | int64   |
| num_restaurants_500m     | int64   |
| num_retail_500m          | int64   |
| num_convenience_500m     | int64   |
| avg_income               | float64 |
| num_offices              | float64 |
| living_population        | float64 |
| land_price               | float64 |

## 3. is_starbucks 기준 개수

| is_starbucks | count |
| ------------ | ----- |
| 0            | 21624 |
| 1            | 681   |

## 4. 브랜드 value counts

| 브랜드     | count |
| ------- | ----- |
| 프랜차이즈외  | 18689 |
| 기타프랜차이즈 | 947   |
| 메가MGC커피 | 857   |
| 스타벅스    | 681   |
| 이디야커피   | 433   |
| 빽다방     | 356   |
| 투썸플레이스  | 342   |

## 5. 컬럼별 결측치: 전체 데이터

| column                   | dtype   | missing_count | missing_rate_pct |
| ------------------------ | ------- | ------------- | ---------------- |
| 상호명                      | str     | 0             | 0.0              |
| 브랜드                      | str     | 0             | 0.0              |
| is_starbucks             | int64   | 0             | 0.0              |
| 위도                       | float64 | 0             | 0.0              |
| 경도                       | float64 | 0             | 0.0              |
| 시군구명                     | str     | 0             | 0.0              |
| 행정동코드                    | float64 | 25            | 0.1121           |
| 행정동명                     | str     | 681           | 3.0531           |
| 도로명주소                    | str     | 0             | 0.0              |
| dist_nearest_subway      | float64 | 0             | 0.0              |
| num_subway_500m          | int64   | 0             | 0.0              |
| nearest_subway_ridership | float64 | 291           | 1.3046           |
| subway_ridership_500m    | float64 | 0             | 0.0              |
| num_competing_cafes_500m | int64   | 0             | 0.0              |
| num_restaurants_500m     | int64   | 0             | 0.0              |
| num_retail_500m          | int64   | 0             | 0.0              |
| num_convenience_500m     | int64   | 0             | 0.0              |
| avg_income               | float64 | 181           | 0.8115           |
| num_offices              | float64 | 328           | 1.4705           |
| living_population        | float64 | 392           | 1.7575           |
| land_price               | float64 | 0             | 0.0              |

## 6. 숫자형 변수 기본통계: 전체 데이터

| column                   | mean          | median     | std          | min        | max        | Q1         | Q3          |
| ------------------------ | ------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ----------- |
| is_starbucks             | 0.0305        | 0.0        | 0.172        | 0.0        | 1.0        | 0.0        | 0.0         |
| 위도                       | 37.5442       | 37.5451    | 0.0461       | 37.4341    | 37.6904    | 37.5064    | 37.5709     |
| 경도                       | 126.9906      | 126.9973   | 0.0808       | 126.7942   | 127.1807   | 126.9239   | 127.0501    |
| 행정동코드                    | 11441656.9461 | 11440720.0 | 199702.5434  | 11110515.0 | 11740700.0 | 11260570.0 | 11628208.75 |
| dist_nearest_subway      | 0.4311        | 0.3637     | 0.3117       | 0.0017     | 3.2369     | 0.2233     | 0.5493      |
| num_subway_500m          | 0.8203        | 1.0        | 0.6465       | 0.0        | 4.0        | 0.0        | 1.0         |
| nearest_subway_ridership | 52310.0326    | 38747.0    | 41767.9279   | 2665.0     | 238671.0   | 22612.0    | 69204.0     |
| subway_ridership_500m    | 44370.447     | 28961.0    | 53044.5437   | 0.0        | 352395.0   | 0.0        | 62771.0     |
| num_competing_cafes_500m | 76.9608       | 61.0       | 54.3806      | 0.0        | 293.0      | 37.0       | 105.0       |
| num_restaurants_500m     | 472.0677      | 411.0      | 300.5986     | 1.0        | 1670.0     | 247.0      | 651.0       |
| num_retail_500m          | 355.9204      | 282.0      | 302.2058     | 0.0        | 2890.0     | 179.0      | 427.0       |
| num_convenience_500m     | 27.2273       | 25.0       | 13.9256      | 0.0        | 88.0       | 17.0       | 35.0        |
| avg_income               | 3551032.4513  | 3261019.0  | 951870.3014  | 2099146.0  | 7421305.0  | 2860315.0  | 4069239.0   |
| num_offices              | 28737.6401    | 5852.0     | 49992.3846   | 3.0        | 250897.0   | 1885.0     | 31123.0     |
| living_population        | 33592.8157    | 28550.0    | 19240.7462   | 3527.0     | 98668.0    | 19613.0    | 41121.0     |
| land_price               | 7107929.5608  | 5072899.0  | 5831229.9692 | 195882.0   | 68290599.0 | 3433797.0  | 8486933.0   |

## 7. 컬럼별 결측치: 스타벅스 데이터

| column                   | dtype   | missing_count | missing_rate_pct |
| ------------------------ | ------- | ------------- | ---------------- |
| 상호명                      | str     | 0             | 0.0              |
| 브랜드                      | str     | 0             | 0.0              |
| is_starbucks             | int64   | 0             | 0.0              |
| 위도                       | float64 | 0             | 0.0              |
| 경도                       | float64 | 0             | 0.0              |
| 시군구명                     | str     | 0             | 0.0              |
| 행정동코드                    | float64 | 25            | 3.6711           |
| 행정동명                     | str     | 681           | 100.0            |
| 도로명주소                    | str     | 0             | 0.0              |
| dist_nearest_subway      | float64 | 0             | 0.0              |
| num_subway_500m          | int64   | 0             | 0.0              |
| nearest_subway_ridership | float64 | 8             | 1.1747           |
| subway_ridership_500m    | float64 | 0             | 0.0              |
| num_competing_cafes_500m | int64   | 0             | 0.0              |
| num_restaurants_500m     | int64   | 0             | 0.0              |
| num_retail_500m          | int64   | 0             | 0.0              |
| num_convenience_500m     | int64   | 0             | 0.0              |
| avg_income               | float64 | 26            | 3.8179           |
| num_offices              | float64 | 30            | 4.4053           |
| living_population        | float64 | 29            | 4.2584           |
| land_price               | float64 | 0             | 0.0              |

## 8. 숫자형 변수 기본통계: 스타벅스 데이터

| column                   | mean          | median     | std          | min        | max        | Q1         | Q3         |
| ------------------------ | ------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ---------- |
| is_starbucks             | 1.0           | 1.0        | 0.0          | 1.0        | 1.0        | 1.0        | 1.0        |
| 위도                       | 37.5391       | 37.5364    | 0.0424       | 37.4473    | 37.6695    | 37.5051    | 37.5657    |
| 경도                       | 126.9931      | 126.9995   | 0.0773       | 126.8005   | 127.1749   | 126.9347   | 127.046    |
| 행정동코드                    | 11456836.8354 | 11500603.0 | 210456.1433  | 11110530.0 | 11740690.0 | 11230710.0 | 11650628.5 |
| dist_nearest_subway      | 0.3414        | 0.2725     | 0.3073       | 0.0127     | 3.0724     | 0.1402     | 0.4425     |
| num_subway_500m          | 1.0           | 1.0        | 0.6348       | 0.0        | 4.0        | 1.0        | 1.0        |
| nearest_subway_ridership | 59446.1738    | 46029.0    | 43107.7023   | 4235.0     | 238671.0   | 28961.0    | 83493.0    |
| subway_ridership_500m    | 60711.2012    | 43283.0    | 60650.2351   | 0.0        | 352395.0   | 18762.0    | 87230.0    |
| num_competing_cafes_500m | 86.5433       | 76.0       | 55.1012      | 1.0        | 280.0      | 45.0       | 117.0      |
| num_restaurants_500m     | 527.8341      | 479.0      | 311.9099     | 6.0        | 1656.0     | 280.0      | 735.0      |
| num_retail_500m          | 431.8972      | 324.0      | 391.6674     | 5.0        | 2886.0     | 192.0      | 518.0      |
| num_convenience_500m     | 30.0837       | 27.0       | 15.5643      | 1.0        | 85.0       | 19.0       | 40.0       |
| avg_income               | 3940506.684   | 3607265.0  | 1091893.1107 | 2152244.0  | 7421305.0  | 3130657.0  | 4846324.5  |
| num_offices              | 45774.8848    | 15114.0    | 63728.5415   | 29.0       | 250897.0   | 3005.0     | 62687.0    |
| living_population        | 39123.6534    | 35698.5    | 21304.2486   | 5870.0     | 98668.0    | 24288.0    | 46455.25   |
| land_price               | 8639839.7357  | 6399509.0  | 6336621.5554 | 1098262.0  | 59244720.0 | 3864306.0  | 13166568.0 |

## 9. dist_nearest_subway 값 범위와 단위 추정

| metric | value  |
| ------ | ------ |
| min    | 0.0017 |
| Q1     | 0.2233 |
| median | 0.3637 |
| Q3     | 0.5493 |
| max    | 3.2369 |

- 단위 기록: 값 범위가 대체로 소수점 단위이며 최댓값도 20 이하이므로 `dist_nearest_subway`는 km 단위일 가능성이 높다고 추정됩니다. 다만 원천 데이터 정의를 확인하기 전까지 확정하지 않습니다.

## 10. 저장된 표

- `reports/tables/missing_values.csv`
- `reports/tables/missing_values_starbucks.csv`
- `reports/tables/basic_stats_all.csv`
- `reports/tables/basic_stats_starbucks.csv`
