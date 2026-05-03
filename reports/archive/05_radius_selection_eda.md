# 05 Radius Selection EDA

- 생성 시각: 2026-05-04 00:05:25
- 입력 파일: `data/seoul_cafe_master_with_geo_features.csv`
- 분석 대상: `df[df["is_starbucks"] == 1]`
- 스타벅스 매장 수: 681
- 목적: clustering 실행 전 좌표 기반 반경 변수와 중복 변수를 선택
- 처리 원칙: clustering 없음, 결측치 대체 없음, 이상치 제거 없음, 비스타벅스 비교 없음

## 1. 새 변수 기본 통계

| feature                | count | missing_count | mean       | median   | std        | min      | Q1       | Q3       | max      | zero_count | zero_rate | skewness | unique_count | cv       |
| ---------------------- | ----- | ------------- | ---------- | -------- | ---------- | -------- | -------- | -------- | -------- | ---------- | --------- | -------- | ------------ | -------- |
| dist_nearest_bus_stop  | 681   | 0             | 0.074802   | 0.061573 | 0.052088   | 0.002262 | 0.039325 | 0.098051 | 0.350308 | 0          | 0.0       | 1.557569 | 680          | 0.696351 |
| num_bus_stops_100m     | 681   | 0             | 1.723935   | 2.0      | 1.481305   | 0.0      | 1.0      | 2.0      | 7.0      | 162        | 0.237885  | 0.914833 | 8            | 0.859258 |
| num_bus_stops_300m     | 681   | 0             | 9.861968   | 9.0      | 4.596838   | 0.0      | 6.0      | 13.0     | 29.0     | 5          | 0.007342  | 0.693703 | 27           | 0.466118 |
| num_bus_stops_500m     | 681   | 0             | 22.428781  | 21.0     | 9.248275   | 1.0      | 15.0     | 28.0     | 53.0     | 0          | 0.0       | 0.554364 | 48           | 0.41234  |
| cafe_count_300m        | 681   | 0             | 40.217327  | 35.0     | 25.14329   | 0.0      | 21.0     | 57.0     | 115.0    | 2          | 0.002937  | 0.665423 | 105          | 0.625186 |
| cafe_count_500m        | 681   | 0             | 90.005874  | 79.0     | 58.122057  | 1.0      | 46.0     | 123.0    | 291.0    | 0          | 0.0       | 0.969999 | 206          | 0.645758 |
| cafe_count_1000m       | 681   | 0             | 276.279001 | 235.0    | 182.032063 | 5.0      | 145.0    | 348.0    | 858.0    | 0          | 0.0       | 1.228979 | 390          | 0.65887  |
| dist_nearest_starbucks | 681   | 0             | 0.340141   | 0.243029 | 0.294381   | 0.009603 | 0.15324  | 0.438721 | 2.638043 | 0          | 0.0       | 2.456194 | 467          | 0.865468 |

## 2. 버스정류장 반경 선택

| feature            | mean      | median | std      | zero_count | zero_rate | skewness | cv       |
| ------------------ | --------- | ------ | -------- | ---------- | --------- | -------- | -------- |
| num_bus_stops_100m | 1.723935  | 2.0    | 1.481305 | 162        | 0.237885  | 0.914833 | 0.859258 |
| num_bus_stops_300m | 9.861968  | 9.0    | 4.596838 | 5          | 0.007342  | 0.693703 | 0.466118 |
| num_bus_stops_500m | 22.428781 | 21.0   | 9.248275 | 0          | 0.0       | 0.554364 | 0.41234  |

### 버스 반경 간 상관

| feature_1          | feature_2          | pearson  | spearman |
| ------------------ | ------------------ | -------- | -------- |
| num_bus_stops_100m | num_bus_stops_300m | 0.406245 | 0.38139  |
| num_bus_stops_100m | num_bus_stops_500m | 0.344324 | 0.329506 |
| num_bus_stops_300m | num_bus_stops_500m | 0.793404 | 0.786674 |

- `num_bus_stops_100m`: 스타벅스 기준 0값 비율이 높아 가장 가까운 정류장 유무에 치우친다.
- `num_bus_stops_300m`: 0값이 거의 없고 도보 접근권으로 해석 가능하며, 분산도 유지된다.
- `num_bus_stops_500m`: 0값이 없어 안정적이지만 300m와 상관이 높고 생활권 전체를 반영하는 성격이 강하다.
- 판단: 버스 접근성 대표 반경은 `num_bus_stops_300m`를 선택한다. `dist_nearest_bus_stop`은 보조 후보로 유지한다.

## 3. 카페 수 반경 선택

| feature          | mean       | median | std        | zero_count | zero_rate | skewness | cv       |
| ---------------- | ---------- | ------ | ---------- | ---------- | --------- | -------- | -------- |
| cafe_count_300m  | 40.217327  | 35.0   | 25.14329   | 2          | 0.002937  | 0.665423 | 0.625186 |
| cafe_count_500m  | 90.005874  | 79.0   | 58.122057  | 0          | 0.0       | 0.969999 | 0.645758 |
| cafe_count_1000m | 276.279001 | 235.0  | 182.032063 | 0          | 0.0       | 1.228979 | 0.65887  |

### 카페 반경 간 상관

| feature_1       | feature_2        | pearson  | spearman |
| --------------- | ---------------- | -------- | -------- |
| cafe_count_300m | cafe_count_500m  | 0.904901 | 0.91993  |
| cafe_count_300m | cafe_count_1000m | 0.742588 | 0.744315 |
| cafe_count_500m | cafe_count_1000m | 0.900169 | 0.876853 |

- `cafe_count_300m`: 매장 주변 직접 경쟁권으로 해석하기 가장 쉽다.
- `cafe_count_500m`: 기존 `num_competing_cafes_500m`와 거의 중복되어 둘을 동시에 쓰지 않는 것이 좋다.
- `cafe_count_1000m`: 개별 매장 입지보다 지역 상권 전체 규모를 반영한다.
- 판단: clustering 입력 대표 반경은 `cafe_count_300m`를 선택하고, `cafe_count_1000m`는 해석용으로 남긴다.

## 4. 기존 경쟁 카페 변수와 비교

`cafe_count_500m`는 현재 master의 전체 카페 좌표를 기준으로 자기 자신만 제외하고 다시 계산한 변수다. `num_competing_cafes_500m`는 기존 master에 있던 변수로 원천과 세부 정의가 다를 수 있다.

| comparison                                  | pearson  | spearman | mean_existing | mean_new  | mean_new_minus_existing | median_new_minus_existing |
| ------------------------------------------- | -------- | -------- | ------------- | --------- | ----------------------- | ------------------------- |
| num_competing_cafes_500m vs cafe_count_500m | 0.999111 | 0.999403 | 86.543319     | 90.005874 | 3.462555                | 2.0                       |

상관이 매우 높으므로 `cafe_count_500m`와 `num_competing_cafes_500m`를 동시에 clustering feature로 쓰는 것은 피한다. 반경 선택 관점에서는 직접 경쟁권인 `cafe_count_300m`를 새 대표 변수로 쓰고, 기존 `num_competing_cafes_500m`는 비교/해석용으로 남기는 편이 낫다.

## 5. dist_nearest_starbucks 평가

| feature                | mean     | median   | std      | min      | Q1      | Q3       | max      | skewness | cv       |
| ---------------------- | -------- | -------- | -------- | -------- | ------- | -------- | -------- | -------- | -------- |
| dist_nearest_starbucks | 0.340141 | 0.243029 | 0.294381 | 0.009603 | 0.15324 | 0.438721 | 2.638043 | 2.456194 | 0.865468 |

### 상위/하위 10개

| feature                | rank_type | rank | 상호명               | 시군구명 | 도로명주소                                          | value    |
| ---------------------- | --------- | ---- | ----------------- | ---- | ---------------------------------------------- | -------- |
| dist_nearest_starbucks | top       | 1    | 종로평창동             | 종로구  | 서울특별시 종로구 평창12길 3 (평창동)                        | 2.638043 |
| dist_nearest_starbucks | top       | 2    | 청계산입구역            | 서초구  | 서울특별시 서초구 청계산로 203 (신원동)                       | 2.031476 |
| dist_nearest_starbucks | top       | 3    | 홍제역               | 서대문구 | 서울특별시 서대문구 통일로 451 (홍제동)                       | 1.795753 |
| dist_nearest_starbucks | top       | 4    | 강동강일              | 강동구  | 서울특별시 강동구 아리수로93나길 54 (강일동)                    | 1.641649 |
| dist_nearest_starbucks | top       | 5    | 송파위례              | 송파구  | 서울특별시 송파구 위례광장로 230 (장지동, 위례2차아이파크)            | 1.532469 |
| dist_nearest_starbucks | top       | 6    | 서울안토              | 강북구  | 서울특별시 강북구 삼양로 689 (우이동) 지하1층                   | 1.499194 |
| dist_nearest_starbucks | top       | 7    | 덕성여대              | 도봉구  | 서울특별시 도봉구 삼양로 536 (쌍문동)                        | 1.499194 |
| dist_nearest_starbucks | top       | 8    | 이케아강동             | 강동구  | 서울특별시 강동구 고덕비즈밸리로 51 (고덕동)                     | 1.349175 |
| dist_nearest_starbucks | top       | 9    | 신림녹두거리            | 관악구  | 서울특별시 관악구 신림로 99 (신림동, 센터스퀘어 서울대점)             | 1.33492  |
| dist_nearest_starbucks | top       | 10   | 망원한강공원            | 마포구  | 서울특별시 마포구 마포나루길 435 (망원동)                      | 1.312935 |
| dist_nearest_starbucks | bottom    | 1    | 타임스퀘어2F           | 영등포구 | 서울특별시 영등포구 영중로 15 지상2층 (영등포동4가)                | 0.009603 |
| dist_nearest_starbucks | bottom    | 2    | 타임스퀘어B2           | 영등포구 | 서울특별시 영등포구 영중로 15, 지하2층 (영등포동4가)               | 0.009603 |
| dist_nearest_starbucks | bottom    | 3    | 여의도IFC몰(L2)STREET | 영등포구 | 서울특별시 영등포구 국제금융로 10 (여의도동) L2 S15              | 0.028832 |
| dist_nearest_starbucks | bottom    | 4    | 여의도               | 영등포구 | 서울특별시 영등포구 국제금융로2길 28 (여의도동)                   | 0.028832 |
| dist_nearest_starbucks | bottom    | 5    | 잠실대교남단R           | 송파구  | 서울특별시 송파구 송파대로 570 타워 730 1층                   | 0.033167 |
| dist_nearest_starbucks | bottom    | 6    | 잠실역               | 송파구  | 서울특별시 송파구 송파대로 562 (신천동, 웰리스타워,삼성웰리스아파트) 1층,2층 | 0.033167 |
| dist_nearest_starbucks | bottom    | 7    | 리저브광화문            | 종로구  | 서울특별시 종로구 세종대로 178 (세종로)                       | 0.039776 |
| dist_nearest_starbucks | bottom    | 8    | KT광화문웨스트B1F       | 종로구  | 서울특별시 종로구 세종대로 178 (세종로)                       | 0.039776 |
| dist_nearest_starbucks | bottom    | 9    | 자양이마트             | 광진구  | 서울특별시 광진구 아차산로 272 (자양동) 지하1층                  | 0.043211 |
| dist_nearest_starbucks | bottom    | 10   | 건대스타시티            | 광진구  | 서울특별시 광진구 아차산로 262 (자양동)                       | 0.043211 |

상위 매장은 종로평창동, 청계산입구역, 홍제역 등 주변 스타벅스와 거리가 먼 독립 입지로 해석 가능하다. 하위 매장은 타임스퀘어, IFC, 잠실, 광화문 등 한 건물 또는 초근접 상권 내 복수 매장이 있는 도심 밀집형으로 해석된다. 따라서 `dist_nearest_starbucks`는 도심 밀집형과 독립 입지형 cluster를 나누는 데 도움이 되는 변수로 판단한다.

## 6. 새 변수와 기존 변수 상관관계

- correlation table: `reports/tables/radius_feature_correlations.csv`
- heatmap: `reports/figures/radius_selection/radius_feature_correlation_heatmap.png`

| relationship                                   | pearson   | interpretation             |
| ---------------------------------------------- | --------- | -------------------------- |
| dist_nearest_bus_stop vs dist_nearest_subway   | 0.108708  | 버스 접근성과 지하철 접근성이 같은 축인지 확인 |
| dist_nearest_starbucks vs cafe_count_300m      | -0.519572 | 스타벅스 독립성과 직접 카페 밀도 관계      |
| dist_nearest_starbucks vs num_restaurants_500m | -0.515652 | 스타벅스 독립성과 상권 활성도 관계        |
| cafe_count_500m vs num_restaurants_500m        | 0.931357  | 카페 밀도와 음식점 밀도의 중복성         |
| cafe_count_500m vs num_retail_500m             | 0.587343  | 카페 밀도와 소매업 밀도의 중복성         |

## 7. 변수별 최종 recommendation

| feature                  | recommendation              | reason                                                                      |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------- |
| dist_nearest_bus_stop    | candidate                   | 가장 가까운 버스정류장까지 거리라 해석은 명확하지만 num_bus_stops_300m와 함께 쓰면 버스 접근성 축이 과대표현될 수 있음 |
| num_bus_stops_100m       | drop_too_sparse             | 스타벅스 기준 0값 비율이 높아 근접 정류장 유무에 치우침                                            |
| num_bus_stops_300m       | select                      | 0값이 거의 없고 매장 주변 도보권 접근성을 잘 나타내며 500m보다 국지성이 좋음                              |
| num_bus_stops_500m       | drop_too_broad              | 0값이 없고 300m와 상관이 높아 더 넓은 생활권 성격이 강함                                         |
| cafe_count_300m          | select                      | 직접 경쟁권으로 해석하기 쉽고 500m/1000m보다 매장 인접 상권 차이를 더 잘 보존                           |
| cafe_count_500m          | drop_duplicate              | 기존 num_competing_cafes_500m와 거의 동일하며 cafe_count_300m와도 높은 상관                |
| cafe_count_1000m         | drop_too_broad              | 개별 매장 입지보다 지역 상권 규모를 강하게 반영하고 반경별 카페 수와 중복성이 큼                              |
| dist_nearest_starbucks   | select                      | 스타벅스 내부에서 도심 밀집형과 독립 입지형을 구분하는 해석력이 있음                                      |
| dist_nearest_subway      | candidate                   | 지하철 접근성 축으로 기존 후보 유지. 버스 접근성 변수와는 다른 교통 차원을 제공                              |
| num_subway_500m          | use_for_interpretation_only | 이산형이고 IQR이 작아 모델 입력보다는 역세권 여부 해석에 적합                                        |
| subway_ridership_500m    | candidate                   | 역세권 규모를 나타내지만 0값과 왜도가 있어 변환 후 후보로 유지                                        |
| num_competing_cafes_500m | drop_duplicate              | cafe_count_500m와 거의 중복. 반경 대표로 cafe_count_300m를 쓰면 동시에 넣지 않는 편이 좋음          |
| num_restaurants_500m     | candidate                   | 카페 수와 다르면서 상권 활성을 나타내는 기존 상권 후보. 카페 밀도와 상관 확인 후 선택                          |
| num_retail_500m          | use_for_interpretation_only | 상권 규모 변수들과 중복 가능성이 커서 모델 입력보다 유형 설명에 유용                                     |
| num_convenience_500m     | use_for_interpretation_only | 상권 밀도 보조 지표로 해석용 가치가 있으나 음식점/카페 변수와 중복 가능성이 큼                               |

## 8. 제안 geo feature set

| feature_set        | features                                                                                                 | note                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Geo Radius Set 추천안 | dist_nearest_bus_stop, num_bus_stops_300m, cafe_count_300m, dist_nearest_starbucks, num_restaurants_500m | 버스 접근성, 직접 카페 경쟁권, 스타벅스 밀집/독립성, 상권 활성도를 균형 있게 포함 |
| 간결형                | num_bus_stops_300m, cafe_count_300m, dist_nearest_starbucks                                              | 반경 변수만 최소로 넣어 중복을 줄인 구성                          |

## 9. 발표용 반경 선택 논리

버스 접근성은 100m, 300m, 500m 반경을 비교한 결과, 100m는 0값 비율이 높아 희소했고 500m는 모든 스타벅스 매장에 값이 존재해 생활권 전체를 반영하는 경향이 컸다. 300m는 대부분의 매장에서 값이 존재하면서도 매장 간 차이를 유지해, clustering용 버스 접근성 대표 변수로 선택했다. 카페 밀도는 300m, 500m, 1000m가 서로 높은 상관을 보였고, 500m는 기존 경쟁 카페 변수와 거의 중복되었다. 따라서 개별 매장의 직접 경쟁권으로 해석 가능한 300m 카페 수를 대표 변수로 선택하고, 1000m는 지역 상권 규모 해석용으로 남긴다.

## 10. 저장된 시각화

| feature                | figure                                                           |
| ---------------------- | ---------------------------------------------------------------- |
| num_bus_stops_100m     | reports/figures/radius_selection/num_bus_stops_100m_hist.png     |
| num_bus_stops_300m     | reports/figures/radius_selection/num_bus_stops_300m_hist.png     |
| num_bus_stops_500m     | reports/figures/radius_selection/num_bus_stops_500m_hist.png     |
| cafe_count_300m        | reports/figures/radius_selection/cafe_count_300m_hist.png        |
| cafe_count_500m        | reports/figures/radius_selection/cafe_count_500m_hist.png        |
| cafe_count_1000m       | reports/figures/radius_selection/cafe_count_1000m_hist.png       |
| dist_nearest_starbucks | reports/figures/radius_selection/dist_nearest_starbucks_hist.png |

## 11. 저장 산출물

- `reports/05_radius_selection_eda.md`
- `reports/tables/radius_feature_summary_starbucks.csv`
- `reports/tables/radius_feature_correlations.csv`
- `reports/tables/radius_feature_recommendation.csv`
- `reports/tables/radius_feature_top_bottom_starbucks.csv`
- `reports/figures/radius_selection/`
