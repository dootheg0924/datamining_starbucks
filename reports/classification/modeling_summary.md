# 스타벅스 입점 예측 — Classification 모델링 보고서

> **과목**: 2026-1 데이터마이닝 텀프로젝트
> **작성일**: 2026-06-14
> **범위**: 중간발표 이후 Classification 단계 전 과정 (라벨링·검증·모델비교·PU고도화·통합·해석·시뮬레이션)
> **원칙**: 피처는 **클러스터링과 동일하게 엔지니어링**(피크 3→평균 통합, 지하철수 범주화, 거리·소매 로그변환, StandardScaler) — 단 `log_dist_starbucks`만 라벨 누수로 제외 → **15개**. 정의서 `feature_engineering.md`. 현재 repo에서는 스크립트가 `scripts/06_classification/`, 재생성 산출물이 `reports/generated/classification/`에 저장된다.
> **데이터**: 서울 카페 22,305 = 스타벅스 681(Positive) + 비스타벅스 21,624(Unlabeled)

---

## 0. Executive Summary

| 영역 | 중간발표 시점 | 고도화 후 |
|---|---|---|
| **라벨링** | 비스벅을 일괄 0(부적합) | **PU Learning** — 비스벅 = Unlabeled(미확인). 강제음성 3전략과 정량 비교해 채택 |
| **불균형(1:32)** | 처리방안 미명시 | PU bagging(1:1 부트스트랩 앙상블)로 구조적 해결 + class_weight/undersample/SMOTE 대조 |
| **검증** | 일반 hold-out | **공간 블록 GroupKFold(5)** — random CV와 비교해 과대평가 없음 입증 |
| **모델** | Logistic/Tree/Forest 계획 | 강의 기반 **16개 모델 + one-class 2종** 비교 |
| **PU 방법론** | 개념만 | bagging vs Two-step(RN) vs RP vs EM vs Elkan **5종 실험**, bagging 채택 |
| **클러스터링 연결** | 분리됨 | **Opt1/2/3 전부 구현** — 클러스터링↔분류 라벨공간 공유 |
| **해석** | 없음 | 순열중요도 + LogReg계수 투트랙, 견고성(ρ) 검증 |
| **서울대 적용** | 계획만 | 캠퍼스 시뮬레이션 + OOT 검증 + 타대학 비교 |

**핵심 메시지 3가지**
1. **PU bagging이 최선** — 강제음성·Two-step·RP·EM 어떤 방법도 이기지 못함. AUC 천장 ~0.677은 **모델·방법 무관하게 견고** → "스벅형 입지는 일반 카페와 약하게만 구별된다"(통설 반박).
2. **스벅형 입지의 1위 동인은 지하철 접근성(역세권)** — 소득·생활인구·프랜차이즈집적이 뒤따름. 독립카페 밀집지는 오히려 비스벅적.
3. **서울대 캠퍼스 = 비역세권 '섬'** — 1위 동인(역세권)이 결여되어 스벅 게이트 미달. 정문 밖 서울대입구역(상업활성, 99백분위)에는 이미 스벅 존재.

---

## 1. ⭐ 평가 지표 — 채택·근거·대안 (핵심)

### 1.1 왜 정확도(Accuracy)·F1을 쓰지 않았나

- **극단적 불균형(1:32)**: "전부 비스벅"으로 찍어도 정확도 97% → 정확도는 무의미.
- **PU 설정의 본질**: U(비스벅)에는 *진짜 음성*과 *미입점 유력지(숨은 양성)*가 섞여 있음. 따라서 **진짜 음성 라벨이 없음** → 음성 라벨을 전제하는 precision/recall/F1/MCC/accuracy는 **편향**됨(U 속 양성을 오답으로 처벌).
- 결론: **임계값과 진짜 음성이 필요 없는 '순위(ranking) 기반' 지표**를 주력으로 채택.

### 1.2 채택한 지표

| 지표 | 정의 | 채택 이유 |
|---|---|---|
| **PU-AUC** (P vs U ROC-AUC) | U를 음성처럼 둔 ROC 곡선 면적 | **주력**. 임계값 무관·순위 기반. PU에서 절대값은 과소추정이나 **모델·방법 간 상대비교에 일관**(순위 보존). |
| **AP** (Average Precision, PR곡선 면적) | 정밀도-재현율 곡선 하부면적 | 불균형에서 ROC보다 **양성 검색력**을 민감하게 반영. 상위 순위에 스벅을 얼마나 모으나. |
| **lift@5%** | 상위 5% 내 스벅밀도 / 전체 스벅밀도 | **현업 해석 직결**: "모델 상위 5%를 보면 스벅이 평균의 몇 배". 입지 우선순위 탐색용. |
| **precision@681** | 상위 681개(=실제 양성 수) 중 진짜 스벅 비율 | k=양성수인 precision@k. "681곳 찍으면 몇 개가 실제 스벅"의 직관. |
| **fold AUC 평균±표준편차** | 공간 fold별 AUC의 산포 | **일반화 안정성**(과적합·분산) 점검. |

> 운영 임계값이 필요한 **Opt3 게이팅**에서만 별도로 **Youden's J**(=max(TPR−FPR), OOF 기준 0.534)로 스벅형/비스벅형 절단점을 정함 — 이건 지표가 아니라 *의사결정 임계값* 선택.

### 1.3 클러스터링 검증 지표(고도화 보고서)와의 역할 구분

- 클러스터링(비지도): Silhouette·DB Index·η²(효과크기)·ARI·부트스트랩 안정성 — *군집 분리/안정성*.
- 분류(지도, PU): PU-AUC·AP·lift — *양성 판별/순위*.
- **avg_income 주의**: 클러스터 구분력은 최하(η²=0.072)이나 **스벅↔카페 판별력은 1위**(표준화 평균차 d=0.382) → 과제가 다르므로 **분류에선 유지**(고도화 보고서 D2의 '제거 검토'는 클러스터링 한정).

### 1.4 추가로 쓸 수 있는 대안 지표

| 분류 | 지표 | 쓸 수 있는 상황 |
|---|---|---|
| **PU 전용** | Lee&Liu PU-score = recall²/Pr(예측양성), nnPU risk | 진짜 음성 없이 추정 가능한 PU 성능지표 |
| **확률 품질** | Brier score, calibration curve | 적합도를 '확률'로 해석·보정 품질 평가(현재 score_cal로 일부 제공) |
| **순위 심화** | NDCG, partial-AUC(고특이도 영역), recall@k 곡선 | 상위 후보 우선순위가 핵심일 때 |
| **임계값 기반** | F-β, MCC, Cohen's κ, balanced accuracy | *신뢰 음성을 확보한 뒤*(Two-step RN) 또는 운영점 고정 시 |
| **비용 기반** | expected cost / profit curve | 오입점·기회손실 비용이 정량화될 때 |

---

## 2. 데이터 · 피처 · 분할 (`00_prepare.py`)

- **피처 15개(클러스터링 동일 엔지니어링)**: 교통4(log지하철거리·지하철수범주·역승하차·버스), 유동2(피크평균·생활인구), 상권6(음식점·log소매·편의점·독립카페·저가카페·프랜차이즈카페), 입지3(소득·직장인구·공시지가). 중복통합(피크3→1)·로그(거리/소매)·StandardScaler(train fold fit). 정의서 `분류_피처_엔지니어링.md`.
- **누수 차단**: `dist_nearest_starbucks` 제외 — 음성 정의가 이 변수와 직결되면 라벨 누수. (클러스터링/페르소나엔 사용)
- **공간 블록**: 위경도 0.02° 격자 → `spatial_block`(168블록, 양성 포함 124). GroupKFold(5)의 group.
- **무결성**: 분류 피처 결측 0. 좌표 완전중복 5,528행(같은 건물 다른 카페) — P/U 혼재 0건이라 직접 누수 없음, 공간블록이 동일좌표를 한 fold에 묶어 CV 누수 차단.

---

## 3. 라벨링 5전략 비교 (`01_pu_baselines.py`)

같은 RF base·공간CV에서 음성 처리 전략 비교.

| 전략 | PU-AUC | AP | lift@5% |
|---|---|---|---|
| **pu_bagging** | **0.676** | 0.058 | 2.32 |
| undersample 1:1 | 0.664 | 0.053 | 2.23 |
| weighted(class_weight) | 0.644 | 0.051 | 1.94 |
| pu_elkan | 0.644 | 0.053 | 2.35 |
| SMOTE | 0.615 | 0.044 | 1.65 |

→ **PU(bagging)가 강제음성 전략을 상회** = "미입점을 0으로 두지 마라"(교수 피드백)에 대한 정량 근거. SMOTE 최하(약신호에서 합성양성이 신호를 흐림).

---

## 4. 검증 — 공간 vs 랜덤 CV (`02_cv_comparison.py`)

입지 데이터의 공간 자기상관 → random split 과대평가 우려(피드백)를 정면 검증.

| 전략 | random CV | spatial CV | inflation |
|---|---|---|---|
| pu_bagging | 0.630 | 0.676 | **−0.046** |
| weighted | 0.561 | 0.644 | −0.084 |
| (5전략 전부) | random ≤ spatial | | 음수 |

→ **과대평가 없음(오히려 random이 보수적)**. 이유: **위경도를 피처에서 제외** → 좌표 암기 경로 차단. 두 CV는 다른 능력 측정(random=옆가게와 구분 / spatial=미지 지역 일반화=서울대 과제). **우리 0.677은 신뢰 가능.** (`cv_comparison.png`)

---

## 5. 모델 성능 비교 — 16+2종 (`03_model_zoo.py`)

PU bagging 프레임 위에서 강의 기반 분류기 비교. 검증 spatial CV, 기록 `reports/generated/classification/logs/MODEL_LOG.md`.

| Tier | 모델(강의) | PU-AUC |
|---|---|---|
| 1 (트리앙상블) | **RandomForest(8) 0.675**, GBM(9) 0.673, Bagging(8) 0.672, XGBoost(9) 0.672 |
| 2 | SVM-RBF(12) 0.667, AdaBoost(9) 0.667, DecisionTree(6) 0.665, ExtraTrees(8) 0.661, LightGBM(9) 0.661 |
| 3 (선형/해석) | LogReg(4) 0.660, LDA(5) 0.659 — 엔지니어링 후 상승 |
| 4 | kNN(5) 0.652, MLP(7) 0.645, NaiveBayes(5) 0.628 |
| 실패 | OneClassSVM(13) 0.484, IsolationForest(13) 0.452 (< 0.5) |

**해석**: ① 천장 ~0.67이 **모델 종류와 무관**(신호 한계, 통설 반박). ② 선형(LogReg/LDA)이 트리앙상블에 0.02밖에 안 뒤짐 → **해석트랙으로 거의 무손실**. ③ one-class 실패 = 스벅이 단일 정상덩어리가 아닌 **다중 페르소나**임을 교차확인(판별적 PU가 옳음).

---

## 6. PU 고도화 — 5종 실험 (`04_pu_twostep.py`, `05_pu_twostep_pos.py`)

| 방법 | 1단계 | PU-AUC |
|---|---|---|
| **pu_bagging** | 전체 U 사용 | **0.677** |
| twostep_RP | OOB 상위 U를 의사양성 편입 | 0.672 |
| twostep_EM | OOB 점수로 U 가중(Elkan 복제) | 0.671 |
| twostep_spy | Spy로 신뢰음성(RN) 추출 | 0.646 |
| pu_elkan | c-보정 | 0.639 |
| twostep_dist | 거리기반 RN(대조) | 0.629 |

**결론**: 어떤 고도화도 plain bagging을 못 이김 → **bagging 최종 채택**.
- Spy-RN > 거리RN(0.646>0.629): 데이터기반 RN이 우월. 거리RN은 상권공동화 지역으로 편향(음식점 −0.76 등)을 수치로 입증.
- 그러나 RN(Two-step) < bagging: 약신호·고중첩 문제에선 **경계의 모호한 U가 곧 신호** → U를 버리는 RN은 정보손실.
- **산출물(RP)**: '스벅 없지만 스벅형' 카페 1,065곳 발굴(관악구 29) — 숨은 후보지. (`hidden_candidates_RP.csv`)

---

## 7. 최종 PU 모델 + 적합도 보정 (`06_pu_final.py`)

- **모델**: PU bagging, base=GBM(150tree, depth3), T=30. OOF **PU-AUC 0.677**, fold 0.675±0.022. Elkan c=0.566.
- **적합도 점수 3종**: `score_raw`(bag 평균확률), `score_pct`(서울 카페 백분위, 해석용), `score_cal`(c 보정확률).
- 저장: `reports/generated/classification/models/pu_bagging_gbm.joblib`(신규입력 채점용), `reports/generated/classification/outputs/final_pu_scores.parquet`.

---

## 8. 클러스터링 ↔ 분류 연결 — Opt1/2/3 (`07_personas.py`, `08_integration.py`)

- 기존 k=5 클러스터링 재현(현재 repo canonical 사이즈 47/224/238/73/99) → 전체 카페 페르소나 배정.
- **페르소나별 스벅 침투율**: 오피스고소득 5.8% > 도심초밀집 5.5% > 상업활성 3.5% > 주거생활 3.1% > 비역세권 1.7%.

| 옵션 | 구조 | 출력 |
|---|---|---|
| **Opt1** | 분류 점수 = 적합도, 클러스터링=서술 | 적합도(평균 백분위 페르소나별 단조: 오피스 80>…>비역세권 29) |
| **Opt2** | 6-way soft | `[게이트×페르소나(5), 비스벅]` → "C1 80%/C4 15%/비스벅 5%" |
| **Opt3** | 2단계 게이팅 | 게이트(Youden 0.534) → 최근접 centroid 페르소나 |

- stage2 검증: RF가 kmeans 페르소나를 **acc 0.916/F1 0.910** 복원 → centroid 배정(학습불필요) 정당.

---

## 9. 해석 — 무엇이 스벅형 입지를 만드나 (`11_interpretation.py`)

| 동인(순열중요도 순) | 순열중요도 | LogReg계수 | 방향 |
|---|---|---|---|
| 지하철거리 | 0.053 | −0.18 | 가까울수록 스벅↑ |
| 평균소득 | 0.027 | +0.33 | 높을수록↑ |
| 생활인구 | 0.011 | +0.24 | 많을수록↑ |
| 프랜차이즈카페500m | 0.007 | +0.20 | 집적↑ |
| 버스정류장 | 0.006 | +0.15 | ↑ |
| 독립카페500m | 0.006 | −0.21 | 많을수록↓ |

- **견고성**: 순열중요도 vs |LogReg계수| **Spearman ρ=0.868 (p<0.001)** → 독립적 두 방법이 강하게 합치(피크통합·로그변환으로 공선성 완화되며 0.625→0.868 상승), 해석 견고.
- **핵심**: 역세권이 압도적 1위. 독립카페 골목은 오히려 비스벅적(스벅은 프랜차이즈 집적지 선호). (`feature_interpretation.png`)

---

## 10. 검증·확장 (`09_snu_simulation.py`, `10_validation.py`)

- **(게이트 통과)** 비스벅 7,205곳(33.3%)이 스벅 유력 입지 — 상업활성 다수. (`gate_candidates.csv`)
- **(OOT)** 신규 입점 스벅 8곳 중 **75%**가 인근 평균적합도 게이트 이상(순화동·대치선릉 적중 / 신정·풍납 미적중=주거확장). 단 좌표 근사·proxy 한계.
- **(타대학)** 서울대만 게이트통과 **0%**·비역세권·1km내 카페 12개 = 상권 고립. 연대/고대는 캠퍼스 코어는 낮아도 인접상권(신촌/안암)이 1km내라 스벅 존재.
- **(서울대 시뮬레이션)** 캠퍼스 내부 적합도 **0.48(55백분위)·비역세권·게이트 미통과** vs 서울대입구역 0.52(상업활성, 스벅4). → "왜 서울대만 없나"의 답.

---

## 11. 한계

1. **상권 6피처 재계산 불가**: 원천(소상공인 상가정보) 부재 → 신규 좌표 featurize 불가, 교내 후보지 시뮬은 기존 카페 proxy로 대체.
2. **교내 지점 변별 불가**: 교내 카페 좌표 일부 지오코딩 fallback(동일점 다수) + 행정동 단위 피처 → 캠퍼스 단위 결론만 신뢰.
3. **PU/SCAR 가정**: 스벅이 랜덤 라벨이 아닌(이미 입점 결정) 점 → 적합도는 절대확률보다 상대 위치로 해석.
4. **AUC 천장 0.677**: 성능 향상 레버는 알고리즘이 아닌 피처(향후 과제).
5. **캠퍼스 OOD**: 캠퍼스는 학습 분포 밖일 수 있어 일반화 영역 명시 필요.

---

## 12. 산출물 인덱스 (`scripts/06_classification/`, `reports/generated/classification/`)

**스크립트(파이프라인 순서)**
| 파일 | 단계 |
|---|---|
| `00_prepare.py` | 데이터·피처·공간블록 |
| `01_pu_baselines.py` | 라벨링 5전략 |
| `02_cv_comparison.py` | 공간 vs 랜덤 CV |
| `03_model_zoo.py` | 16+2 모델 비교 |
| `04_pu_twostep.py` | Two-step RN(spy/dist) |
| `05_pu_twostep_pos.py` | RP·EM |
| `06_pu_final.py` | 최종 PU + 적합도 |
| `07_personas.py` | 페르소나 배정 |
| `08_integration.py` | Opt1/2/3 |
| `09_snu_simulation.py` | 서울대 시뮬레이션 |
| `10_validation.py` | 게이트·OOT·타대학 |
| `11_interpretation.py` | 변수 해석 |

**로그(`reports/generated/classification/logs/`)**: MODEL_LOG · PU_LOG · INTEGRATION_LOG · INTERPRETATION_LOG · SNU_REPORT · VALIDATION_REPORT
**모델(`reports/generated/classification/models/`)**: pu_bagging_gbm.joblib · persona_kmeans.joblib
**주요 출력(`reports/generated/classification/outputs/`)**: final_pu_scores · integrated_predictions · model_zoo_comparison · cv_comparison · feature_interpretation · gate_candidates · hidden_candidates_RP · oot_validation · university_comparison · snu_simulation

---

## 13. 검증 체크리스트

- ✅ 라벨링 PU 채택 — 강제음성 3전략 대비 우월 입증
- ✅ 불균형 — bagging(1:1 앙상블) + 대조군
- ✅ 공간 과대평가 없음 — random≤spatial 입증
- ✅ 모델 16+2종 비교 — 천장 0.677 robust
- ✅ PU 5종 비교 — bagging 최종
- ✅ Opt1/2/3 전부 구현 + stage2 검증(acc0.916)
- ✅ 해석 견고성 ρ=0.868
- ✅ OOT 75% 적중 + 타대학 대비 서울대 위치 규명
