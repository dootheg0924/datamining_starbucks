# 서울 전체 카페 EDA 요약

- 입력 파일: `data/final/seoul_cafe_model_features_final.csv`
- 분석 대상: 서울 전체 카페 22,305개, 스타벅스 681개, 비스타벅스 21,624개
- 모델 feature: 18개

## 1. 기본 진단

| dataset       | label | rows  | starbucks_count | non_starbucks_count | feature_missing_cells | nan_reason_marked_rows | duplicate_latlon_rows | district_count |
| ------------- | ----- | ----- | --------------- | ------------------- | --------------------- | ---------------------- | --------------------- | -------------- |
| all           | 전체    | 22305 | 681             | 21624               | 0                     | 726                    | 3594                  | 25             |
| starbucks     | 스타벅스  | 681   | 681             | 0                   | 0                     | 40                     | 0                     | 25             |
| non_starbucks | 비스타벅스 | 21624 | 0               | 21624               | 0                     | 686                    | 3594                  | 25             |

`nan_reason`은 보정 전 결측 사유의 provenance로, feature 결측이 남았다는 뜻은 아닙니다.

| nan_reason          | count | rate  |
| ------------------- | ----- | ----- |
| none                | 21579 | 0.967 |
| 최근접 지하철역 승하차 데이터 없음 | 291   | 0.013 |
| 신설된 행정동             | 217   | 0.01  |
| 강북구 생활인구 코드 불일치     | 194   | 0.009 |
| 스타벅스 행정동코드 조회 실패    | 24    | 0.001 |

## 2. 스타벅스와 비스타벅스 차이

아래 표는 standardized mean difference 절대값이 큰 feature 순서입니다. 양수는 스타벅스가 비스타벅스보다 큰 값, 음수는 작은 값을 뜻합니다.

| feature                         | group                | starbucks_median | non_starbucks_median | median_diff_starbucks_minus_non | smd_starbucks_vs_non | ks_statistic | direction        |
| ------------------------------- | -------------------- | ---------------- | -------------------- | ------------------------------- | -------------------- | ------------ | ---------------- |
| avg_income                      | demographic_economic | 3607265.0        | 3245251.0            | 362014.0                        | 0.382                | 0.175        | starbucks_higher |
| subway_morning_peak_500m        | subway_peak          | 316405.0         | 211499.0             | 104906.0                        | 0.306                | 0.146        | starbucks_higher |
| dist_nearest_subway             | transport            | 0.272            | 0.367                | -0.094                          | -0.299               | 0.178        | starbucks_lower  |
| other_franchise_cafe_count_500m | cafe_competition     | 6.0              | 5.0                  | 1.0                             | 0.293                | 0.134        | starbucks_higher |
| subway_evening_peak_500m        | subway_peak          | 484485.0         | 319255.0             | 165230.0                        | 0.291                | 0.145        | starbucks_higher |
| num_offices                     | demographic_economic | 14874.0          | 5425.0               | 9449.0                          | 0.291                | 0.153        | starbucks_higher |
| num_subway_500m                 | transport            | 1.0              | 1.0                  | 0.0                             | 0.289                | 0.129        | similar_median   |
| subway_lunch_peak_500m          | subway_peak          | 185671.0         | 119374.0             | 66297.0                         | 0.277                | 0.149        | starbucks_higher |

## 3. 전체 카페 분포와 이상치

왜도가 큰 변수는 변환 및 모델 해석에서 특히 주의해야 합니다.

| feature                  | group                | median    | p95        | p99        | max        | skewness | zero_rate |
| ------------------------ | -------------------- | --------- | ---------- | ---------- | ---------- | -------- | --------- |
| land_price               | demographic_economic | 5072899.0 | 17309515.0 | 26360008.0 | 68290599.0 | 3.598    | 0.0       |
| num_retail_500m          | commerce             | 282.0     | 887.8      | 1712.0     | 2890.0     | 3.026    | 0.0       |
| num_offices              | demographic_economic | 5670.0    | 120704.0   | 250897.0   | 250897.0   | 2.714    | 0.0       |
| dist_nearest_subway      | transport            | 0.364     | 1.027      | 1.532      | 3.237      | 2.228    | 0.0       |
| dist_nearest_starbucks   | cafe_competition     | 0.245     | 0.878      | 1.352      | 3.001      | 1.985    | 0.0       |
| subway_lunch_peak_500m   | subway_peak          | 120795.0  | 745022.0   | 1037565.0  | 1722940.0  | 1.846    | 0.293     |
| subway_morning_peak_500m | subway_peak          | 213387.0  | 1090057.0  | 1675579.0  | 2606832.0  | 1.826    | 0.293     |
| subway_evening_peak_500m | subway_peak          | 322074.0  | 1999586.0  | 2661957.0  | 4466485.0  | 1.817    | 0.293     |

IQR 기준 이상치 비율이 큰 변수입니다. 이상치는 제거 대상이 아니라 고밀도 상권이나 특수 입지의 후보로 해석합니다.

| feature                  | group                | iqr_outlier_count | iqr_outlier_rate | skewness |
| ------------------------ | -------------------- | ----------------- | ---------------- | -------- |
| num_offices              | demographic_economic | 2800              | 0.126            | 2.714    |
| num_retail_500m          | commerce             | 1355              | 0.061            | 3.026    |
| land_price               | demographic_economic | 1259              | 0.056            | 3.598    |
| subway_evening_peak_500m | subway_peak          | 1132              | 0.051            | 1.817    |
| nearest_subway_ridership | transport            | 1111              | 0.05             | 1.767    |
| subway_lunch_peak_500m   | subway_peak          | 1084              | 0.049            | 1.846    |
| dist_nearest_subway      | transport            | 1076              | 0.048            | 2.228    |
| dist_nearest_starbucks   | cafe_competition     | 942               | 0.042            | 1.985    |

## 4. 상관과 중복 가능성

전체 카페 기준 변환 후에도 강하게 남는 Pearson 상관쌍입니다.

| feature_a                | feature_b                   | pearson_r | abs_r |
| ------------------------ | --------------------------- | --------- | ----- |
| subway_lunch_peak_500m   | subway_evening_peak_500m    | 0.994     | 0.994 |
| subway_morning_peak_500m | subway_evening_peak_500m    | 0.982     | 0.982 |
| subway_morning_peak_500m | subway_lunch_peak_500m      | 0.968     | 0.968 |
| num_restaurants_500m     | independent_cafe_count_500m | 0.916     | 0.916 |
| num_restaurants_500m     | num_convenience_500m        | 0.848     | 0.848 |
| num_subway_500m          | subway_morning_peak_500m    | 0.779     | 0.779 |
| num_subway_500m          | subway_lunch_peak_500m      | 0.762     | 0.762 |
| num_convenience_500m     | independent_cafe_count_500m | 0.761     | 0.761 |
| num_subway_500m          | subway_evening_peak_500m    | 0.76      | 0.76  |
| num_restaurants_500m     | num_retail_500m             | 0.733     | 0.733 |

## 5. 구별 요약

카페 수 상위 구입니다.

| district | total_cafes | starbucks_count | starbucks_rate |
| -------- | ----------- | --------------- | -------------- |
| 강남구      | 2187        | 102             | 0.047          |
| 마포구      | 1811        | 38              | 0.021          |
| 송파구      | 1341        | 40              | 0.03           |
| 서초구      | 1294        | 59              | 0.046          |
| 종로구      | 1266        | 42              | 0.033          |
| 영등포구     | 1180        | 44              | 0.037          |
| 중구       | 1129        | 54              | 0.048          |
| 강서구      | 1064        | 33              | 0.031          |
| 용산구      | 859         | 24              | 0.028          |
| 성북구      | 795         | 17              | 0.021          |

스타벅스가 5개 이상 있는 구 중 스타벅스 비율이 높은 구입니다.

| district | total_cafes | starbucks_count | starbucks_rate |
| -------- | ----------- | --------------- | -------------- |
| 중구       | 1129        | 54              | 0.048          |
| 강남구      | 2187        | 102             | 0.047          |
| 서초구      | 1294        | 59              | 0.046          |
| 영등포구     | 1180        | 44              | 0.037          |
| 서대문구     | 685         | 24              | 0.035          |
| 종로구      | 1266        | 42              | 0.033          |
| 강서구      | 1064        | 33              | 0.031          |
| 광진구      | 792         | 24              | 0.03           |
| 송파구      | 1341        | 40              | 0.03           |
| 양천구      | 577         | 17              | 0.029          |

## 6. 모델링/클러스터링 시사점

- 비스타벅스는 확정 negative가 아니라 일반 카페 비교군으로만 해석해야 합니다.
- 상권 밀도, 지하철/버스 접근성, 직장인구/지가 변수는 스타벅스와 일반 카페의 차이를 설명하는 핵심 후보입니다.
- 카페 및 상권 count 변수는 서로 강한 상관을 보일 수 있으므로 선형 모델이나 해석 단계에서는 중복성을 확인해야 합니다.
- 전체 카페 분포의 오른쪽 꼬리가 긴 변수는 기존 모델링 흐름과 같이 `log1p` 또는 `sqrt` 변환을 유지하는 것이 타당합니다.

## 7. 산출물

### Tables

- `reports/generated/tables/seoul_cafe_feature_summary_by_group.csv`
- `reports/generated/tables/seoul_cafe_starbucks_comparison.csv`
- `reports/generated/tables/seoul_cafe_iqr_outliers.csv`
- `reports/generated/tables/seoul_cafe_multi_outlier_stores.csv`
- `reports/generated/tables/seoul_cafe_corr_raw_matrix.csv`
- `reports/generated/tables/seoul_cafe_corr_transformed_matrix.csv`
- `reports/generated/tables/seoul_cafe_corr_pairs.csv`
- `reports/generated/tables/seoul_cafe_district_summary.csv`
- `reports/generated/tables/seoul_cafe_basic_diagnostics.csv`
- `reports/generated/tables/seoul_cafe_nan_reason_summary.csv`
- `reports/generated/tables/seoul_cafe_skew_transform_comparison.csv`

### Tracked table snapshots

아래 CSV는 중요한 요약 산출물만 선별해 `reports/archive/tables/`에도 저장하므로 GitHub에 포함됩니다.

- `reports/archive/tables/seoul_cafe_basic_diagnostics.csv`
- `reports/archive/tables/seoul_cafe_corr_pairs.csv`
- `reports/archive/tables/seoul_cafe_district_summary.csv`
- `reports/archive/tables/seoul_cafe_feature_summary_by_group.csv`
- `reports/archive/tables/seoul_cafe_iqr_outliers.csv`
- `reports/archive/tables/seoul_cafe_nan_reason_summary.csv`
- `reports/archive/tables/seoul_cafe_skew_transform_comparison.csv`
- `reports/archive/tables/seoul_cafe_starbucks_comparison.csv`

### Figures

- `histograms`: `reports/generated/figures/eda/seoul_cafe/seoul_cafe_feature_histograms.png`
- `boxplots`: `reports/generated/figures/eda/seoul_cafe/seoul_cafe_feature_boxplots.png`
- `overlay`: `reports/generated/figures/eda/seoul_cafe/starbucks_vs_non_starbucks_overlay.png`
- `raw_corr_heatmap`: `reports/generated/figures/eda/seoul_cafe/seoul_cafe_corr_raw_heatmap.png`
- `transformed_corr_heatmap`: `reports/generated/figures/eda/seoul_cafe/seoul_cafe_corr_transformed_heatmap.png`
- `smd_bar`: `reports/generated/figures/eda/seoul_cafe/starbucks_smd_bar.png`
- `district_chart`: `reports/generated/figures/eda/seoul_cafe/district_cafe_starbucks_rate.png`
