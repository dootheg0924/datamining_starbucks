# 분류(Classification) 피처 엔지니어링 정의서

**입력:** `data/final/seoul_cafe_model_features_final.csv` (22,305 = 스벅 681 + 비스벅 21,624)
**산출:** `reports/generated/classification/data/clf_dataset.parquet`
**최종 분류 변수 수:** **15개** (클러스터링 16개 − `log_dist_starbucks`)
**스케일링:** StandardScaler — 각 모델 파이프라인에서 **train fold 기준 fit**(누수 방지). 데이터셋엔 미적용.

> 원칙: 클러스터링 `feature_engineering_notes.md`와 **동일한 중복통합·변환**을 적용. **단 한 가지 차이** = 분류에서는 `log_dist_starbucks`(인근 스벅까지 거리)를 **라벨 누수로 제외**. (음성 정의가 이 변수와 직결 → 학습 시 "스벅에서 멀면 비스벅"을 외움)

---

## 최종 변수 정의 (15개)

| # | 변수명 | 원본 | 처리 | 클러스터링 논의 |
|---|--------|------|------|----------------|
| 1 | `log_dist_subway` | dist_nearest_subway | log(1+x) | 논의6 |
| 2 | `subway_count_cat` | num_subway_500m | 0/1/2+ 순서형 | 논의2 |
| 3 | `subway_ridership` | nearest_subway_ridership | 원본 | 논의6 |
| 4 | `bus_stops_300m` | num_bus_stops_300m | 원본 | 논의3 |
| 5 | `peak_avg` | morning/lunch/evening_peak | 3변수 평균 | 논의1 |
| 6 | `restaurants_500m` | num_restaurants_500m | 원본 | 논의6 |
| 7 | `log_retail_500m` | num_retail_500m | log(1+x) | 논의6 |
| 8 | `convenience_500m` | num_convenience_500m | 원본 | 논의6 |
| 9 | `indie_cafe_500m` | independent_cafe_count_500m | 원본 | 논의6 |
| 10 | `low_price_cafe_500m` | low_price_cafe_count_500m | 원본 | 논의6 |
| 11 | `franchise_cafe_500m` | other_franchise_cafe_count_500m | 원본 | 논의6 |
| 12 | `avg_income` | avg_income | 원본 | 논의4 |
| 13 | `offices` | num_offices | 원본 | 논의5 |
| 14 | `living_pop` | living_population | 원본 | 논의5 |
| 15 | `land_price` | land_price | 원본 | 논의6 |

**분류 제외:** `log_dist_starbucks` (라벨 누수). 단 컬럼 자체는 데이터셋에 보존 — Two-step RN 정의·반경 ablation·EDA 용도. `dist_nearest_starbucks`(원본 km)도 반경 ablation용으로 보존.

---

## 중복 정보 통합 (클러스터링과 동일)

### ① 피크 3변수 → `peak_avg` (평균 통합)
아침·점심·저녁 피크는 상호 r=0.92~0.98로 거의 동일 정보. 3변수 동시 포함 시 같은 신호가 3중 반영 → **평균 1개로 통합**.
```
peak_avg = mean(morning, lunch, evening peak_500m)
```
- 분류 효과: LogReg에서 3개 공선 변수로 흩어지던 계수가 1개로 안정화(해석 트랙 품질↑). 트리는 입력 차원 감소.

### ② 지하철수 → `subway_count_cat` (0/1/2+ 순서형)
원본 num_subway_500m은 Q1=Q3=1로 연속처리 시 34%가 이상치 오판. **역 유무·복수역세권**만 의미 → 3범주.

### ③ 직장인구·생활인구 동시 유지 (`offices`, `living_pop`)
r=0.76 중복이나 오피스형 vs 생활형 상권을 다르게 포착 → 둘 다 유지(클러스터링 논의5와 동일).

---

## 로그 변환 (왜도 > 2.0, 클러스터링 논의6과 동일 기준)

| 변수 | 원본 skew | 변환 | 근거 |
|------|:--------:|------|------|
| `log_dist_subway` | 3.29 | log(1+x) | 거리 감쇠 — 역 거리 효과는 로그 스케일 |
| `log_retail_500m` | 2.71 | log(1+x) | 분포 왜곡 가장 큰 상권변수 |
| ~~log_dist_starbucks~~ | 2.46 | (제외) | **분류 누수로 제외** |

미변환(원본 유지): subway_ridership(1.51)·land_price(1.93)·offices(1.89)·avg_income(0.59) 등 — 절대 격차 자체가 입지 정보거나 분포 양호.

> 스케일에 둔감한 트리계열(GBM/RF)에는 로그변환이 결과를 거의 바꾸지 않으나(단조변환), **클러스터링과의 일관성** 및 **거리/스케일 민감 모델(LogReg·SVM·kNN·MLP)의 안정성**을 위해 동일 적용.

---

## 클러스터링 ↔ 분류 피처셋 비교

| 항목 | 클러스터링 | 분류 |
|---|---|---|
| 변수 수 | 16 | **15** |
| 차이 | — | `log_dist_starbucks` 제외(누수) |
| 통합·변환 | peak평균·subway범주·log(거리/소매/스벅거리) | **동일** (스벅거리만 제외) |
| 스케일링 | StandardScaler(전체 fit) | StandardScaler(**train fold fit**, CV 누수 방지) |
| 단위 | 681 스벅 | 22,305 전체 카페 |

---

## P vs U 판별 신호 (참고, 표준화 평균차 d)

분류 목적에서 양성(스벅)↔비스벅을 가르는 신호 상위:
`avg_income(+0.38)` · `log_dist_subway(−0.35)` · `peak_avg(+0.30)` · `franchise_cafe_500m(+0.29)` · `offices(+0.29)` · `subway_count_cat(+0.29)`.
→ 소득·역세권·유동·프랜차이즈 집적이 핵심 (해석 단계와 일치).

*참고: 클러스터 구분력(η²)과 분류 판별력(d)은 다른 과제 — 예) `avg_income`은 η² 최하(0.072)이나 분류 d 최상(0.38)이라 분류에선 유지.*

---

*데이터마이닝 텀프로젝트 · 분류 피처 엔지니어링 · 2026-06-14*
