# Analysis Archive Summary

이 문서는 기존 `01`-`07` archive 보고서의 핵심 결정만 압축한 요약본이다. 자세한 재생성 로그와 반복 표/그림은 스크립트를 실행하면 `reports/generated/` 아래에 다시 만들어진다.

## 1. 분석 흐름 요약

| 단계 | 목적 | 핵심 결과 |
| --- | --- | --- |
| 01 Data audit | `seoul_cafe_master.csv` 구조, 결측, 기본 통계 확인 | 원천 master는 카페 전체와 스타벅스 subset 분석이 가능했고, 거리 변수는 km 단위로 해석하는 것이 자연스러웠다. |
| 02 Starbucks-only EDA | 스타벅스 681개 매장 기준 기존 feature 분포 확인 | 일부 변수는 왜도가 커서 모델링 전 log/scaling 검토가 필요하며, 결측이 있는 변수는 별도 처리 방침이 필요하다. |
| 03 Raw data inventory | 추가 feature engineering 가능 데이터 확인 | 버스정류장, 지하철 시간대 승하차, 역사 master를 사용해 geo/traffic feature를 추가할 수 있다고 판단했다. |
| 04 Geo feature engineering | 좌표 기반 버스/카페/스타벅스 거리 feature 생성 | `num_bus_stops_*`, `cafe_count_*`, `dist_nearest_starbucks`를 생성했고 중간 master를 만들었다. |
| 05 Radius selection EDA | 100m/300m/500m/1000m 반경 후보 비교 | 버스 접근성은 `num_bus_stops_300m`, 직접 경쟁권 카페 수는 `cafe_count_300m`가 가장 해석 가능하다고 판단했다. |
| 06 Clustering CSV finalization | clustering 담당자용 slim CSV 생성 | clustering용 보조 산출물이었고, 최종 모델 feature set과는 별도로 다룬다. |
| 07 Model feature finalization v2 | 최종 모델 feature CSV 확정 | 최종 산출물은 `data/seoul_cafe_model_features_v1.csv`와 `data/starbucks_model_features_v1.csv`이며 각각 25개 컬럼을 가진다. |

## 2. 핵심 결정

- 최종 Git 추적 데이터는 모델링 handoff용 CSV 2개만 `data/` 루트에 둔다.
- 좌표 기반 중간 master와 clustering 보조 CSV는 `data/archive/intermediate/`에 생성하고 Git에서는 제외한다.
- 상세 보고서, 반복 통계표, 변수별 histogram/boxplot은 `reports/generated/`에 재생성되도록 하고 Git에서는 제외한다.
- 최종 모델에는 식별/해석 컬럼 7개와 모델 feature 18개를 포함한다.
- `num_bus_stops_300m`는 100m보다 덜 희소하고 500m보다 국지성이 좋아 최종 feature로 포함한다.
- 단순 `cafe_count_300m`는 직접 경쟁권 해석에는 좋지만, 최종 모델에서는 상권 구성을 더 잘 표현하기 위해 500m 카페 유형별 count를 사용한다.
- `subway_ridership_500m`, `num_competing_cafes_500m`, `dist_nearest_bus_stop`, bus/cafe 후보 반경 변수 일부는 최종 output에서 제외한다.
- HTML 지도는 발표/검토용 산출물로 가치가 있어 `archive/html_maps/`에 유지한다.

## 3. 유지하는 근거 산출물

| 구분 | 유지 파일 |
| --- | --- |
| 최종 feature 정의 | `reports/archive/tables/model_feature_v2_columns.csv`, `reports/archive/tables/model_feature_v2_status.csv` |
| 최종 결측/통계 | `reports/archive/tables/model_feature_v2_missing_values_all.csv`, `reports/archive/tables/model_feature_v2_missing_values_starbucks.csv`, `reports/archive/tables/model_feature_v2_summary_starbucks.csv` |
| 신규 feature 근거 | `reports/archive/tables/new_feature_v2_summary.csv`, `reports/archive/tables/cafe_brand_taxonomy_summary.csv`, `reports/archive/tables/subway_peak_station_merge_failures.csv` |
| 반경 선택 근거 | `reports/archive/tables/radius_feature_recommendation.csv`, `reports/archive/tables/radius_feature_correlations.csv` |
| 주요 시각화 | `reports/archive/figures/radius_selection/radius_feature_correlation_heatmap.png`, `reports/archive/figures/geo_features/new_geo_feature_correlation_heatmap.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_pearson.png`, `reports/archive/figures/starbucks_only/correlation_heatmap_spearman.png` |
| 반경 비교 histogram | `reports/archive/figures/radius_selection/num_bus_stops_100m_hist.png`, `reports/archive/figures/radius_selection/num_bus_stops_300m_hist.png`, `reports/archive/figures/radius_selection/num_bus_stops_500m_hist.png`, `reports/archive/figures/radius_selection/cafe_count_300m_hist.png`, `reports/archive/figures/radius_selection/cafe_count_500m_hist.png`, `reports/archive/figures/radius_selection/cafe_count_1000m_hist.png` |

## 4. 재생성 메모

- 원천 데이터는 `rawdata/`에 있어야 하며 Git에는 포함하지 않는다.
- `scripts/04_geo_feature_engineering.py`는 `data/archive/intermediate/seoul_cafe_master_with_geo_features.csv`를 생성한다.
- `scripts/05_radius_selection_eda.py`와 `scripts/06_clustering_csv_finalization.py`는 중간 master를 입력으로 사용한다.
- `scripts/07_model_feature_finalization_v2.py`는 최종 모델 CSV 2개를 `data/` 루트에 다시 생성한다.
- 모든 상세 리포트와 재생성 표/그림은 `reports/generated/` 아래에 생성된다.
