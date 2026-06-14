# 스타벅스 Feature Engineering 요약

이 문서는 외부 인수인계 자료에서 확인한 feature engineering 결정을 현재 최종 CSV 기준으로 정리한 것입니다.

## 산출물

- 입력: `data/final/starbucks_model_features_final.csv`
- 스크립트: `scripts/feature_pipeline/08_starbucks_feature_engineering.py`
- 출력: `data/modeling/starbucks_engineered_features_final.csv`
- 행 수: 스타벅스 681개 매장
- 컬럼 수: 식별/해석용 컬럼 7개 + 파생 feature 16개

## 파생 Feature

| 파생 feature | 원천 feature | 처리 | 이유 |
| --- | --- | --- | --- |
| `log_dist_subway` | `dist_nearest_subway` | `log1p` | 거리 감소 효과와 오른쪽으로 긴 분포를 반영합니다. |
| `subway_count_cat` | `num_subway_500m` | 0 / 1 / 2개 이상 순서형 범주 | 대부분의 매장이 1개 역을 가져 IQR 기반 해석이 제한적입니다. |
| `subway_ridership` | `nearest_subway_ridership` | 원값 유지 | 가장 가까운 지하철역의 절대 규모 자체가 의미 있습니다. |
| `bus_stops_300m` | `num_bus_stops_300m` | 원값 유지 | 버스 접근성을 나타내는 독립 신호입니다. |
| `peak_avg` | 지하철 peak 3개 변수 | 평균 | 출근, 점심, 저녁 peak 변수의 중복성이 높습니다. |
| `restaurants_500m` | `num_restaurants_500m` | 원값 유지 | 주변 음식점 밀도를 나타냅니다. |
| `log_retail_500m` | `num_retail_500m` | `log1p` | 오른쪽으로 긴 분포를 완화합니다. |
| `convenience_500m` | `num_convenience_500m` | 원값 유지 | 생활 편의시설 밀도를 나타냅니다. |
| `indie_cafe_500m` | `independent_cafe_count_500m` | 원값 유지 | 독립 카페 경쟁 및 주변 맥락을 나타냅니다. |
| `low_price_cafe_500m` | `low_price_cafe_count_500m` | 원값 유지 | 저가 프랜차이즈 카페 맥락을 나타냅니다. |
| `franchise_cafe_500m` | `other_franchise_cafe_count_500m` | 원값 유지 | 기타 프랜차이즈 카페 맥락을 나타냅니다. |
| `log_dist_starbucks` | `dist_nearest_starbucks` | `log1p` | 스타벅스 간 거리 감소 효과와 오른쪽으로 긴 분포를 반영합니다. |
| `avg_income` | `avg_income` | 원값 유지 | 지역 구매력의 절대 수준이 의미 있습니다. |
| `offices` | `num_offices` | 원값 유지 | 업무지구 수요 규모를 나타냅니다. |
| `living_pop` | `living_population` | 원값 유지 | 거주 및 생활 인구 규모를 나타냅니다. |
| `land_price` | `land_price` | 원값 유지 | 공시지가의 절대 차이가 입지 수준 해석에 의미 있습니다. |

## 주의사항

- 최종 모델 feature CSV는 feature 결측 보정이 끝난 상태입니다.
- 이 파생 feature set은 모델링 및 클러스터링용 변환을 적용한 파일이며, `data/final/starbucks_model_features_final.csv`를 대체하지 않습니다.
- EDA 표와 그림은 `scripts/eda/`로 재생성되며 `reports/generated/` 아래에 저장됩니다. 해당 폴더는 Git 추적에서 제외합니다.
