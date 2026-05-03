# 02 Starbucks-only EDA

- 생성 시각: 2026-05-03 23:33:33
- 입력 파일: `seoul_cafe_master.csv`
- 분석 대상: `df_starbucks = df[df["is_starbucks"] == 1]`
- 스타벅스 매장 수: 681
- 분석 범위: 스타벅스 내부 입지 차이 파악 및 clustering용 기존 변수 후보 정리
- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, clustering 실행 없음, 비스타벅스 비교 없음

## 1. 분석 대상 변수

| feature                  | group   | label              |
| ------------------------ | ------- | ------------------ |
| dist_nearest_subway      | 지하철 접근성 | 가장 가까운 지하철역 거리     |
| num_subway_500m          | 지하철 접근성 | 500m 내 지하철역 수      |
| nearest_subway_ridership | 지하철 접근성 | 가장 가까운 지하철역 승하차 인원 |
| subway_ridership_500m    | 지하철 접근성 | 500m 내 지하철 승하차 인원  |
| num_competing_cafes_500m | 상권 밀도   | 500m 내 경쟁 카페 수     |
| num_restaurants_500m     | 상권 밀도   | 500m 내 음식점 수       |
| num_retail_500m          | 상권 밀도   | 500m 내 소매업 수       |
| num_convenience_500m     | 상권 밀도   | 500m 내 편의점 수       |
| avg_income               | 인구/경제   | 평균 소득              |
| num_offices              | 인구/경제   | 오피스 수              |
| living_population        | 인구/경제   | 생활 인구              |
| land_price               | 인구/경제   | 공시지가/지가            |

## 2. 변수별 기본 통계

아래 통계는 스타벅스 681개 매장만 대상으로 계산했다.

| feature                  | group   | count | missing_count | missing_rate | mean         | median    | std          | min       | Q1        | Q3         | max        | IQR       | zero_count | zero_rate | skewness |
| ------------------------ | ------- | ----- | ------------- | ------------ | ------------ | --------- | ------------ | --------- | --------- | ---------- | ---------- | --------- | ---------- | --------- | -------- |
| dist_nearest_subway      | 지하철 접근성 | 681   | 0             | 0.0          | 0.3414       | 0.2725    | 0.3073       | 0.0127    | 0.1402    | 0.4425     | 3.0724     | 0.3023    | 0          | 0.0       | 3.2893   |
| num_subway_500m          | 지하철 접근성 | 681   | 0             | 0.0          | 1.0          | 1.0       | 0.6348       | 0.0       | 1.0       | 1.0        | 4.0        | 0.0       | 123        | 0.1806    | 0.519    |
| nearest_subway_ridership | 지하철 접근성 | 673   | 8             | 0.0117       | 59446.1738   | 46029.0   | 43107.7023   | 4235.0    | 28961.0   | 83493.0    | 238671.0   | 54532.0   | 0          | 0.0       | 1.3656   |
| subway_ridership_500m    | 지하철 접근성 | 681   | 0             | 0.0          | 60711.2012   | 43283.0   | 60650.2351   | 0.0       | 18762.0   | 87230.0    | 352395.0   | 68468.0   | 125        | 0.1836    | 1.4784   |
| num_competing_cafes_500m | 상권 밀도   | 681   | 0             | 0.0          | 86.5433      | 76.0      | 55.1012      | 1.0       | 45.0      | 117.0      | 280.0      | 72.0      | 0          | 0.0       | 0.9632   |
| num_restaurants_500m     | 상권 밀도   | 681   | 0             | 0.0          | 527.8341     | 479.0     | 311.9099     | 6.0       | 280.0     | 735.0      | 1656.0     | 455.0     | 0          | 0.0       | 0.6997   |
| num_retail_500m          | 상권 밀도   | 681   | 0             | 0.0          | 431.8972     | 324.0     | 391.6674     | 5.0       | 192.0     | 518.0      | 2886.0     | 326.0     | 0          | 0.0       | 2.7129   |
| num_convenience_500m     | 상권 밀도   | 681   | 0             | 0.0          | 30.0837      | 27.0      | 15.5643      | 1.0       | 19.0      | 40.0       | 85.0       | 21.0      | 0          | 0.0       | 0.6173   |
| avg_income               | 인구/경제   | 655   | 26            | 0.0382       | 3940506.684  | 3607265.0 | 1091893.1107 | 2152244.0 | 3130657.0 | 4846324.5  | 7421305.0  | 1715667.5 | 0          | 0.0       | 0.6509   |
| num_offices              | 인구/경제   | 651   | 30            | 0.0441       | 45774.8848   | 15114.0   | 63728.5415   | 29.0      | 3005.0    | 62687.0    | 250897.0   | 59682.0   | 0          | 0.0       | 1.8455   |
| living_population        | 인구/경제   | 652   | 29            | 0.0426       | 39123.6534   | 35698.5   | 21304.2486   | 5870.0    | 24288.0   | 46455.25   | 98668.0    | 22167.25  | 0          | 0.0       | 1.3155   |
| land_price               | 인구/경제   | 681   | 0             | 0.0          | 8639839.7357 | 6399509.0 | 6336621.5554 | 1098262.0 | 3864306.0 | 13166568.0 | 59244720.0 | 9302262.0 | 0          | 0.0       | 1.9313   |

## 3. 거리 변수 단위 기록

`dist_nearest_subway`는 1단계 audit에서 값 범위가 소수점 km 형태로 보여 km 단위일 가능성이 높다고 추정했다. 이번 EDA에서는 원본 컬럼을 덮어쓰지 않고, 그래프 표시용으로만 `dist_nearest_subway_m = dist_nearest_subway * 1000`을 만들어 m 단위로 표시했다. 단위는 여전히 추정이며 원천 데이터 정의 확인 전까지 확정하지 않는다.

## 4. 분포 그래프

각 변수별 histogram과 boxplot을 저장했다. 왜도 절댓값이 1 이상인 변수는 log1p 변환 histogram도 추가했다.

| feature                  | histogram                                                        | boxplot                                                             | log1p_histogram                                                        |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| dist_nearest_subway      | reports/figures/starbucks_only/dist_nearest_subway_hist.png      | reports/figures/starbucks_only/dist_nearest_subway_boxplot.png      | reports/figures/starbucks_only/dist_nearest_subway_log1p_hist.png      |
| num_subway_500m          | reports/figures/starbucks_only/num_subway_500m_hist.png          | reports/figures/starbucks_only/num_subway_500m_boxplot.png          |                                                                        |
| nearest_subway_ridership | reports/figures/starbucks_only/nearest_subway_ridership_hist.png | reports/figures/starbucks_only/nearest_subway_ridership_boxplot.png | reports/figures/starbucks_only/nearest_subway_ridership_log1p_hist.png |
| subway_ridership_500m    | reports/figures/starbucks_only/subway_ridership_500m_hist.png    | reports/figures/starbucks_only/subway_ridership_500m_boxplot.png    | reports/figures/starbucks_only/subway_ridership_500m_log1p_hist.png    |
| num_competing_cafes_500m | reports/figures/starbucks_only/num_competing_cafes_500m_hist.png | reports/figures/starbucks_only/num_competing_cafes_500m_boxplot.png |                                                                        |
| num_restaurants_500m     | reports/figures/starbucks_only/num_restaurants_500m_hist.png     | reports/figures/starbucks_only/num_restaurants_500m_boxplot.png     |                                                                        |
| num_retail_500m          | reports/figures/starbucks_only/num_retail_500m_hist.png          | reports/figures/starbucks_only/num_retail_500m_boxplot.png          | reports/figures/starbucks_only/num_retail_500m_log1p_hist.png          |
| num_convenience_500m     | reports/figures/starbucks_only/num_convenience_500m_hist.png     | reports/figures/starbucks_only/num_convenience_500m_boxplot.png     |                                                                        |
| avg_income               | reports/figures/starbucks_only/avg_income_hist.png               | reports/figures/starbucks_only/avg_income_boxplot.png               |                                                                        |
| num_offices              | reports/figures/starbucks_only/num_offices_hist.png              | reports/figures/starbucks_only/num_offices_boxplot.png              | reports/figures/starbucks_only/num_offices_log1p_hist.png              |
| living_population        | reports/figures/starbucks_only/living_population_hist.png        | reports/figures/starbucks_only/living_population_boxplot.png        | reports/figures/starbucks_only/living_population_log1p_hist.png        |
| land_price               | reports/figures/starbucks_only/land_price_hist.png               | reports/figures/starbucks_only/land_price_boxplot.png               | reports/figures/starbucks_only/land_price_log1p_hist.png               |

## 5. IQR 기준 이상치 요약

이상치는 제거하지 않고, 해석 가능한 매장인지 확인하기 위한 점검 대상으로만 기록했다. 전체 매장 리스트는 `reports/tables/starbucks_only_outliers_iqr.csv`에 저장했다.
`num_subway_500m`는 Q1과 Q3가 모두 1이라 IQR이 0이다. 따라서 IQR 기준에서는 값이 1이 아닌 매장이 모두 이상치로 잡히며, 이 변수는 연속형 변수의 이상치 판단처럼 해석하면 안 된다.

| feature                  | lower_bound | upper_bound | outlier_count | outlier_rate |
| ------------------------ | ----------- | ----------- | ------------- | ------------ |
| avg_income               | 557155.75   | 7419825.75  | 4             | 0.0059       |
| dist_nearest_subway      | -0.3133     | 0.896       | 30            | 0.0441       |
| land_price               | -10089087.0 | 27119961.0  | 11            | 0.0162       |
| living_population        | -8962.875   | 79706.125   | 47            | 0.069        |
| nearest_subway_ridership | -52837.0    | 165291.0    | 20            | 0.0294       |
| num_competing_cafes_500m | -63.0       | 225.0       | 16            | 0.0235       |
| num_convenience_500m     | -12.5       | 71.5        | 5             | 0.0073       |
| num_offices              | -86518.0    | 152210.0    | 56            | 0.0822       |
| num_restaurants_500m     | -402.5      | 1417.5      | 5             | 0.0073       |
| num_retail_500m          | -297.0      | 1007.0      | 52            | 0.0764       |
| num_subway_500m          | 1.0         | 1.0         | 233           | 0.3421       |
| subway_ridership_500m    | -83940.0    | 189932.0    | 32            | 0.047        |

## 6. 상위/하위 10개 매장

각 변수별 상위 10개와 하위 10개 매장은 `reports/tables/starbucks_only_top_bottom_by_feature.csv`에 저장했다. 보고서에는 일부 예시만 표시한다.

| feature             | rank_type | rank | 상호명       | 시군구명 | 도로명주소                                         | value  |
| ------------------- | --------- | ---- | --------- | ---- | --------------------------------------------- | ------ |
| dist_nearest_subway | top       | 1    | 더북한산      | 은평구  | 서울특별시 은평구 대서문길 24-11 (진관동)                    | 3.0724 |
| dist_nearest_subway | top       | 2    | 종로평창동     | 종로구  | 서울특별시 종로구 평창12길 3 (평창동)                       | 2.818  |
| dist_nearest_subway | top       | 3    | 강남세곡      | 강남구  | 서울특별시 강남구 헌릉로569길 18 (세곡동)                    | 2.2068 |
| dist_nearest_subway | top       | 4    | 진관DT      | 은평구  | 서울특별시 은평구 연서로 645 (진관동)                       | 1.9738 |
| dist_nearest_subway | top       | 5    | 강동강일      | 강동구  | 서울특별시 강동구 아리수로93나길 54 (강일동)                   | 1.6374 |
| dist_nearest_subway | top       | 6    | 강남자곡      | 강남구  | 서울특별시 강남구 자곡로 172 (자곡동) 107~109호              | 1.5594 |
| dist_nearest_subway | top       | 7    | 연희DT      | 서대문구 | 서울특별시 서대문구 연희로 144 (연희동)                      | 1.51   |
| dist_nearest_subway | top       | 8    | 송파위례      | 송파구  | 서울특별시 송파구 위례광장로 230 (장지동, 위례2차아이파크)           | 1.4978 |
| dist_nearest_subway | top       | 9    | 서초우면      | 서초구  | 서울특별시 서초구 태봉로 62 (우면동)                        | 1.3562 |
| dist_nearest_subway | top       | 10   | 장안사거리     | 동대문구 | 서울특별시 동대문구 답십리로 267 (장안동)                     | 1.3438 |
| dist_nearest_subway | bottom    | 1    | 금호역       | 성동구  | 서울특별시 성동구 동호로 99 (금호동4가)                      | 0.0127 |
| dist_nearest_subway | bottom    | 2    | 명동남산      | 중구   | 서울특별시 중구 퇴계로 132 (남산동3가)                      | 0.021  |
| dist_nearest_subway | bottom    | 3    | 동대문공원     | 중구   | 서울특별시 중구 장충단로 229 (광희동1가)                     | 0.0211 |
| dist_nearest_subway | bottom    | 4    | 굽은다리역     | 강동구  | 서울특별시 강동구 양재대로 1568 (명일동, 원일타워)               | 0.0265 |
| dist_nearest_subway | bottom    | 5    | 센트럴시티     | 서초구  | 서울특별시 서초구 신반포로 176 (반포동)                      | 0.03   |
| dist_nearest_subway | bottom    | 6    | 군자역       | 광진구  | 서울특별시 광진구 천호대로 548 (군자동) 중앙빌딩                 | 0.0309 |
| dist_nearest_subway | bottom    | 7    | 양재역신분당역사  | 서초구  | 서울특별시 서초구 남부순환로 2585 신분당선 양재역 지하1층 12호(7번출구쪽) | 0.033  |
| dist_nearest_subway | bottom    | 8    | 혜화역       | 종로구  | 서울특별시 종로구 대학로12길 4 (동숭동)1~2층                  | 0.0339 |
| dist_nearest_subway | bottom    | 9    | 숙대입구역     | 용산구  | 서울특별시 용산구 한강대로 291 (갈월동)                      | 0.0345 |
| dist_nearest_subway | bottom    | 10   | 선유도역 1번출구 | 영등포구 | 서울특별시 영등포구 양평로 128 (양평동5가)양평로 128             | 0.0356 |
| num_subway_500m     | top       | 1    | 무교동       | 중구   | 서울특별시 중구 무교로 21 (무교동) 코오롱빌딩 1층                | 4.0    |
| num_subway_500m     | top       | 2    | 인사        | 종로구  | 서울특별시 종로구 인사동길 14 (인사동)                       | 3.0    |
| num_subway_500m     | top       | 3    | 연세세브란스    | 중구   | 서울특별시 중구 통일로 10 (남대문로5가)                      | 3.0    |
| num_subway_500m     | top       | 4    | 서울중앙우체국   | 중구   | 서울특별시 중구 소공로 70 (충무로 1가) 서울 중앙 우체국            | 3.0    |

## 7. 변수 간 상관관계

- Pearson correlation table: `reports/tables/starbucks_only_corr_pearson.csv`
- Spearman correlation table: `reports/tables/starbucks_only_corr_spearman.csv`
- Pearson heatmap: `reports/figures/starbucks_only/correlation_heatmap_pearson.png`
- Spearman heatmap: `reports/figures/starbucks_only/correlation_heatmap_spearman.png`

Spearman 기준 절댓값 0.7 이상인 변수쌍은 다음과 같다.

| feature_1                | feature_2             | spearman_corr |
| ------------------------ | --------------------- | ------------- |
| num_competing_cafes_500m | num_restaurants_500m  | 0.9337        |
| num_restaurants_500m     | num_convenience_500m  | 0.8673        |
| num_competing_cafes_500m | num_convenience_500m  | 0.8337        |
| num_restaurants_500m     | num_retail_500m       | 0.76          |
| num_subway_500m          | subway_ridership_500m | 0.7257        |
| num_retail_500m          | num_convenience_500m  | 0.7121        |
| num_offices              | land_price            | 0.7095        |
| num_competing_cafes_500m | num_retail_500m       | 0.7088        |

### 변수군별 중복성 메모

- 지하철 접근성: `dist_nearest_subway`, `num_subway_500m`, `nearest_subway_ridership`, `subway_ridership_500m`는 접근성, 역 수, 승하차 규모를 서로 다른 관점에서 담는다. `subway_ridership_500m`는 500m 내 역이 없는 경우 0이 될 수 있어 구조적 0값을 주의해야 한다.
- 상권 밀도: `num_competing_cafes_500m`, `num_restaurants_500m`, `num_retail_500m`, `num_convenience_500m`는 같은 반경 내 업종 밀도를 재는 변수라 상호 중복 가능성이 있다. 다음 단계에서는 상관이 높은 변수 조합을 줄이거나 표준화 후 사용해야 한다.
- 인구/경제: `avg_income`, `num_offices`, `living_population`, `land_price`는 결측과 왜도가 함께 존재한다. 특히 `num_offices`, `land_price`는 큰 값 쪽 꼬리가 길어 log1p 변환 후보로 보는 것이 좋다.

## 8. Clustering 관점 변수 평가

| feature                  | group   | recommendation     | rationale                                                              | transform_note                                       |
| ------------------------ | ------- | ------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------- |
| dist_nearest_subway      | 지하철 접근성 | use_with_transform | 왜도 3.29; 방향성이 명확한 접근성 변수                                               | 원본은 km 추정, 시각화만 m 변환; clustering 전 scaling/log 변환 검토 |
| num_subway_500m          | 지하철 접근성 | caution            | 0값 비율 18.06%; IQR이 0이라 IQR 이상치 기준 해석 주의; 해석은 쉽지만 이산형 변수                |                                                      |
| nearest_subway_ridership | 지하철 접근성 | caution            | 결측률 1.17%; 왜도 1.37; 역세권 규모를 직접 반영                                      | log1p 변환 검토                                          |
| subway_ridership_500m    | 지하철 접근성 | caution            | 왜도 1.48; 0값 비율 18.36%; 0값은 500m 내 역 없음의 구조적 값일 수 있음                    | log1p 변환 검토                                          |
| num_competing_cafes_500m | 상권 밀도   | caution            | 중복 가능 변수: num_restaurants_500m (0.93), num_convenience_500m (0.83)     |                                                      |
| num_restaurants_500m     | 상권 밀도   | caution            | 중복 가능 변수: num_competing_cafes_500m (0.93), num_convenience_500m (0.87) |                                                      |
| num_retail_500m          | 상권 밀도   | use_with_transform | 왜도 2.71                                                                | log1p 변환 후 사용 권장                                     |
| num_convenience_500m     | 상권 밀도   | caution            | 중복 가능 변수: num_competing_cafes_500m (0.83), num_restaurants_500m (0.87) |                                                      |
| avg_income               | 인구/경제   | caution            | 결측률 3.82%                                                              | scaling 후 사용 권장                                      |
| num_offices              | 인구/경제   | caution            | 결측률 4.41%; 왜도 1.85                                                     | log1p 변환 후 사용 권장                                     |
| living_population        | 인구/경제   | caution            | 결측률 4.26%; 왜도 1.32                                                     | log1p 변환 검토                                          |
| land_price               | 인구/경제   | use_with_transform | 왜도 1.93                                                                | log1p 변환 후 사용 권장                                     |

## 9. Clustering용 기존 변수 후보 feature set

| feature_set   | features                                                                                                                                                    | note                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| A. 교통 접근성 중심  | dist_nearest_subway, num_subway_500m, nearest_subway_ridership, subway_ridership_500m                                                                       | 역 접근성과 역세권 유동량 차이를 중심으로 스타벅스 입지를 나누는 최소 세트      |
| B. 교통 + 상권 밀도 | dist_nearest_subway, nearest_subway_ridership, subway_ridership_500m, num_competing_cafes_500m, num_restaurants_500m, num_retail_500m, num_convenience_500m | 유동량과 주변 업종 밀도를 함께 반영하되, 상관 높은 변수는 다음 단계에서 축소 검토 |
| C. 균형형        | dist_nearest_subway, nearest_subway_ridership, num_competing_cafes_500m, num_restaurants_500m, avg_income, num_offices, living_population, land_price       | 교통, 상권, 경제/인구를 균형 있게 포함하는 중간발표용 후보 세트           |

이번 단계에서는 위 feature set을 제안만 하며 clustering은 실행하지 않았다.

## 10. 다음 단계에서 추가해야 할 변수

- 버스 접근성 변수
- 후보 반경별 카페 수 변수
- 시간대별 지하철 유동량 변수
- 대학 접근성 변수
- 20-30대 인구 비율
- 직장인/상주인구 대비 생활인구 비율
- 관광/상업 핵심지 여부를 나타내는 공간 태그

## 11. 저장 산출물

- `reports/02_starbucks_only_eda.md`
- `reports/tables/starbucks_only_feature_summary.csv`
- `reports/tables/starbucks_only_outliers_iqr.csv`
- `reports/tables/starbucks_only_top_bottom_by_feature.csv`
- `reports/tables/starbucks_only_corr_pearson.csv`
- `reports/tables/starbucks_only_corr_spearman.csv`
- `reports/tables/starbucks_only_feature_recommendation.csv`
- `reports/figures/starbucks_only/`
