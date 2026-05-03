# 04 Geo Feature Engineering

- 생성 시각: 2026-05-03 23:59:33
- 입력 master: `seoul_cafe_master.csv` (utf-8-sig)
- 입력 버스정류장: `서울시_버스정류소_위치정보.csv` (euc-kr)
- 출력 master: `data/seoul_cafe_master_with_geo_features.csv`
- 처리 범위: `seoul_cafe_master.csv` 전체 행에 좌표 기반 변수 추가
- 거리 계산: Haversine great-circle distance. 거리 단위는 km
- 반경 기준: 100m=0.1km, 300m=0.3km, 500m=0.5km, 1000m=1.0km
- 처리 원칙: 결측치 대체 없음, 이상치 제거 없음, clustering 없음, 지하철 peak/premium/low price 변수 생성 없음

## 1. 생성 변수

| feature                | unit  | definition                         |
| ---------------------- | ----- | ---------------------------------- |
| dist_nearest_bus_stop  | km    | 가장 가까운 버스정류장까지 Haversine 거리        |
| num_bus_stops_100m     | count | 0.1km 반경 내 버스정류장 수                 |
| num_bus_stops_300m     | count | 0.3km 반경 내 버스정류장 수                 |
| num_bus_stops_500m     | count | 0.5km 반경 내 버스정류장 수                 |
| cafe_count_300m        | count | 0.3km 반경 내 전체 카페 수. 자기 자신 제외       |
| cafe_count_500m        | count | 0.5km 반경 내 전체 카페 수. 자기 자신 제외       |
| cafe_count_1000m       | count | 1.0km 반경 내 전체 카페 수. 자기 자신 제외       |
| dist_nearest_starbucks | km    | 가장 가까운 스타벅스까지 거리. 스타벅스 행은 자기 자신 제외 |

## 2. 새 변수 결측치 확인

| feature                | missing_count | missing_rate |
| ---------------------- | ------------- | ------------ |
| dist_nearest_bus_stop  | 0             | 0.0          |
| num_bus_stops_100m     | 0             | 0.0          |
| num_bus_stops_300m     | 0             | 0.0          |
| num_bus_stops_500m     | 0             | 0.0          |
| cafe_count_300m        | 0             | 0.0          |
| cafe_count_500m        | 0             | 0.0          |
| cafe_count_1000m       | 0             | 0.0          |
| dist_nearest_starbucks | 0             | 0.0          |

## 3. 기본 통계: 전체 데이터

| feature                | count | missing_count | missing_rate | mean       | median   | std        | min      | Q1       | Q3       | max      | zero_count | zero_rate | skewness |
| ---------------------- | ----- | ------------- | ------------ | ---------- | -------- | ---------- | -------- | -------- | -------- | -------- | ---------- | --------- | -------- |
| dist_nearest_bus_stop  | 22305 | 0             | 0.0          | 0.093758   | 0.080379 | 0.064685   | 0.000251 | 0.047789 | 0.124952 | 1.252906 | 0          | 0.0       | 2.210972 |
| num_bus_stops_100m     | 22305 | 0             | 0.0          | 1.332571   | 1.0      | 1.399218   | 0.0      | 0.0      | 2.0      | 10.0     | 8407       | 0.376911  | 1.103615 |
| num_bus_stops_300m     | 22305 | 0             | 0.0          | 9.085855   | 9.0      | 4.389171   | 0.0      | 6.0      | 12.0     | 32.0     | 216        | 0.009684  | 0.636079 |
| num_bus_stops_500m     | 22305 | 0             | 0.0          | 22.627348  | 22.0     | 8.738896   | 0.0      | 16.0     | 28.0     | 69.0     | 22         | 0.000986  | 0.480717 |
| cafe_count_300m        | 22305 | 0             | 0.0          | 34.885003  | 28.0     | 25.643777  | 0.0      | 15.0     | 49.0     | 149.0    | 96         | 0.004304  | 1.115471 |
| cafe_count_500m        | 22305 | 0             | 0.0          | 79.708406  | 63.0     | 56.599132  | 0.0      | 37.0     | 108.0    | 298.0    | 23         | 0.001031  | 1.218084 |
| cafe_count_1000m       | 22305 | 0             | 0.0          | 246.943735 | 199.0    | 168.745632 | 0.0      | 132.0    | 304.0    | 862.0    | 3          | 0.000134  | 1.408346 |
| dist_nearest_starbucks | 22305 | 0             | 0.0          | 0.326446   | 0.244757 | 0.286006   | 0.000145 | 0.127046 | 0.444926 | 3.000911 | 0          | 0.0       | 1.985233 |

## 4. 기본 통계: 스타벅스 681개

| feature                | count | missing_count | missing_rate | mean       | median   | std        | min      | Q1       | Q3       | max      | zero_count | zero_rate | skewness |
| ---------------------- | ----- | ------------- | ------------ | ---------- | -------- | ---------- | -------- | -------- | -------- | -------- | ---------- | --------- | -------- |
| dist_nearest_bus_stop  | 681   | 0             | 0.0          | 0.074802   | 0.061573 | 0.052088   | 0.002262 | 0.039325 | 0.098051 | 0.350308 | 0          | 0.0       | 1.557569 |
| num_bus_stops_100m     | 681   | 0             | 0.0          | 1.723935   | 2.0      | 1.481305   | 0.0      | 1.0      | 2.0      | 7.0      | 162        | 0.237885  | 0.914833 |
| num_bus_stops_300m     | 681   | 0             | 0.0          | 9.861968   | 9.0      | 4.596838   | 0.0      | 6.0      | 13.0     | 29.0     | 5          | 0.007342  | 0.693703 |
| num_bus_stops_500m     | 681   | 0             | 0.0          | 22.428781  | 21.0     | 9.248275   | 1.0      | 15.0     | 28.0     | 53.0     | 0          | 0.0       | 0.554364 |
| cafe_count_300m        | 681   | 0             | 0.0          | 40.217327  | 35.0     | 25.14329   | 0.0      | 21.0     | 57.0     | 115.0    | 2          | 0.002937  | 0.665423 |
| cafe_count_500m        | 681   | 0             | 0.0          | 90.005874  | 79.0     | 58.122057  | 1.0      | 46.0     | 123.0    | 291.0    | 0          | 0.0       | 0.969999 |
| cafe_count_1000m       | 681   | 0             | 0.0          | 276.279001 | 235.0    | 182.032063 | 5.0      | 145.0    | 348.0    | 858.0    | 0          | 0.0       | 1.228979 |
| dist_nearest_starbucks | 681   | 0             | 0.0          | 0.340141   | 0.243029 | 0.294381   | 0.009603 | 0.15324  | 0.438721 | 2.638043 | 0          | 0.0       | 2.456194 |

## 5. 스타벅스 기준 검증

### 버스정류장 수 zero rate

| feature            | zero_count | zero_rate |
| ------------------ | ---------- | --------- |
| num_bus_stops_100m | 162        | 0.237885  |
| num_bus_stops_300m | 5          | 0.007342  |
| num_bus_stops_500m | 0          | 0.0       |

### 반경별 카페 수 분포

| feature          | mean       | median | skewness |
| ---------------- | ---------- | ------ | -------- |
| cafe_count_300m  | 40.217327  | 35.0   | 0.665423 |
| cafe_count_500m  | 90.005874  | 79.0   | 0.969999 |
| cafe_count_1000m | 276.279001 | 235.0  | 1.228979 |

### dist_nearest_starbucks 상위/하위 10개

| feature                | rank_type | rank | 상호명               | 시군구명 | 도로명주소                                          | value_km |
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

### 스타벅스 행 중 dist_nearest_starbucks = 0

_No rows._

## 6. 기존 num_competing_cafes_500m와 새 cafe_count_500m 비교

`cafe_count_500m`는 현재 master 파일의 전체 카페 좌표를 사용해 0.5km 반경 내 카페 수를 다시 계산하고 자기 자신만 제외한 변수다. `num_competing_cafes_500m`는 기존 master에 이미 있던 변수로, 생성 원천과 필터 정의가 다를 수 있어 값이 완전히 같다고 가정하지 않았다.

| comparison                                  | pearson_corr_all | spearman_corr_all | pearson_corr_starbucks | spearman_corr_starbucks | mean_difference_all_new_minus_existing | mean_difference_starbucks_new_minus_existing |
| ------------------------------------------- | ---------------- | ----------------- | ---------------------- | ----------------------- | -------------------------------------- | -------------------------------------------- |
| num_competing_cafes_500m vs cafe_count_500m | 0.999203         | 0.999556          | 0.999111               | 0.999403                | 2.747635                               | 3.462555                                     |

## 7. 새 변수 상관관계

- 전체 correlation table: `reports/tables/new_geo_feature_correlations.csv`
- heatmap: `reports/figures/geo_features/new_geo_feature_correlation_heatmap.png`

아래는 버스정류장 count 변수와 카페 count 변수만 발췌한 Pearson correlation이다.

| feature            | num_bus_stops_100m | num_bus_stops_300m | num_bus_stops_500m | cafe_count_300m | cafe_count_500m | cafe_count_1000m |
| ------------------ | ------------------ | ------------------ | ------------------ | --------------- | --------------- | ---------------- |
| num_bus_stops_100m | 1.0                | 0.401781           | 0.273801           | -0.066141       | -0.094376       | -0.10178         |
| num_bus_stops_300m | 0.401781           | 1.0                | 0.743703           | 0.18191         | 0.163971        | 0.127            |
| num_bus_stops_500m | 0.273801           | 0.743703           | 1.0                | 0.229991        | 0.279267        | 0.262463         |
| cafe_count_300m    | -0.066141          | 0.18191            | 0.229991           | 1.0             | 0.917432        | 0.761823         |
| cafe_count_500m    | -0.094376          | 0.163971           | 0.279267           | 0.917432        | 1.0             | 0.897675         |
| cafe_count_1000m   | -0.10178           | 0.127              | 0.262463           | 0.761823        | 0.897675        | 1.0              |

## 8. 저장된 시각화

| feature                | figure                                                       |
| ---------------------- | ------------------------------------------------------------ |
| num_bus_stops_100m     | reports/figures/geo_features/num_bus_stops_100m_hist.png     |
| num_bus_stops_300m     | reports/figures/geo_features/num_bus_stops_300m_hist.png     |
| num_bus_stops_500m     | reports/figures/geo_features/num_bus_stops_500m_hist.png     |
| cafe_count_300m        | reports/figures/geo_features/cafe_count_300m_hist.png        |
| cafe_count_500m        | reports/figures/geo_features/cafe_count_500m_hist.png        |
| cafe_count_1000m       | reports/figures/geo_features/cafe_count_1000m_hist.png       |
| dist_nearest_starbucks | reports/figures/geo_features/dist_nearest_starbucks_hist.png |

## 9. 저장 산출물

- `data/seoul_cafe_master_with_geo_features.csv`
- `reports/04_geo_feature_engineering.md`
- `reports/tables/new_geo_feature_summary_all.csv`
- `reports/tables/new_geo_feature_summary_starbucks.csv`
- `reports/tables/new_geo_feature_correlations.csv`
- `reports/tables/starbucks_new_geo_top_bottom.csv`
- `reports/figures/geo_features/`
