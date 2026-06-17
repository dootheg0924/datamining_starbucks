# 서울대학교 스타벅스 입점 예측 프로젝트

> 2026-1학기 서울대학교 데이터마이닝 수업 텀 프로젝트  
> 주제: **“서울대학교에는 왜 스타벅스가 없을까?”**  
> 접근: 서울 스타벅스 입지 패턴을 데이터로 학습하고, 서울대 관악캠퍼스의 스타벅스형 입지 적합도를 정량 평가

## 프로젝트 개요

“스타벅스는 감각적으로 좋은 자리에만 들어간다”는 통설을 데이터마이닝으로 검증하고자 합니다. Starbucks의 자체 상권 분석 시스템처럼, 개별 카페 좌표를 교통, 유동, 상권, 인구·경제 정보를 담은 입지 벡터로 변환하고 서울 전체 카페 공간에서 스타벅스형 입지를 찾고자 합니다.

기본 분석 단위는 행정동이 아니라 **개별 카페 좌표**입니다. 다만 최종 해석에서는 카페 단위의 한계를 보강하기 위해 DBSCAN 기반 **상권 단위**로 한 번 더 검증했습니다. 제안발표와 중간발표에서는 데이터 수집, 초기 feature 설계, 결측·이상치 처리, 스타벅스 입지 유형화를 진행했고, 이후에는 서울 전체 카페 22,305개를 대상으로 스타벅스 입점 적합도 예측과 서울대 관악캠퍼스 검증까지 확장했습니다.

전체 프로젝트 흐름은 다음과 같습니다.

```text
문제 정의
  -> 데이터 수집: 스타벅스, 서울 전체 카페, 교통, 상권, 인구, 경제 데이터 결합
  -> 데이터 정합성 검토: 결측 원인 추적, 이상치 보존·해석
  -> Feature Engineering: 초기 18개 입지 지표를 모델링 가능한 feature set으로 정제
  -> Clustering: 서울 스타벅스 681개 매장의 입지 페르소나 도출
  -> Classification: 서울 전체 카페 22,305개의 스타벅스형 입지 적합도 예측
  -> 검증과 해석: 서울대 시뮬레이션, 신규 입점 검증, 상권 단위 재구성
```

## 핵심 결과

- 서울 카페와 스타벅스 매장 좌표를 교통 접근성, 지하철 유동, 주변 상권, 소득·인구·지가 정보와 결합해 **카페 단위 입지 feature table**을 구축했습니다.
- 결측치는 원인을 추적해 보정하고, 특수 입지·DT·목적지형 매장 같은 이상치는 제거하지 않고 **입지 전략의 일부로 해석**했습니다.
- 초기 18개 입지 지표를 상관 구조와 분포 진단을 통해 정리해, 클러스터링용 16개 feature와 분류용 15개 leakage-free feature를 구성했습니다.
- 서울 스타벅스 681개 매장을 군집화해 **5개 입지 페르소나**를 도출했습니다: 오피스고소득, 상업활성, 주거생활, 도심초밀집, 비역세권.
- 비스타벅스 카페를 확정 음성으로 두지 않고 `Unlabeled`로 보는 **PU learning**을 적용했습니다. 카페 단위 최종 모델은 **PU-AUC 약 0.678**입니다.
- 상권 단위로 분석 단위를 올리면 **PU-AUC 약 0.789**로 개선되며, 서울대 관악캠퍼스는 상권 적합도 기준 하위권으로 나타납니다.
- 종합하면 서울대 캠퍼스 내부는 역세권·상권밀도 조건이 약한 **비역세권형 고립 상권**에 가깝고, 정문 밖 서울대입구역 권역은 오히려 상업활성형 고적합 입지로 해석됩니다.

## 데이터와 산출물

GitHub에는 원천 raw data 대신 최종 재현 artifact만 포함합니다.

| 경로 | 설명 |
|---|---|
| `data/final/seoul_cafe_model_features_final.csv` | 서울 전체 카페 22,305개 기준 최종 feature table |
| `data/final/starbucks_model_features_final.csv` | 스타벅스 681개 매장 subset |
| `data/modeling/starbucks_engineered_features_final.csv` | 스타벅스 전용 모델링·클러스터링 파생 feature set |
| `presentation/` | 제안발표, 중간발표 PDF |
| `reports/` | 최종 보고서와 재현 근거 |
| `scripts/` | 데이터 구축, feature engineering, clustering, classification 재현 코드 |

원천 데이터와 API key 설정은 [`docs/data_sources.md`](docs/data_sources.md)에 정리했습니다.

## 먼저 읽을 문서

1. [`reports/README.md`](reports/README.md)  
   보고서 전체 읽는 순서입니다.

2. [`reports/feature_evidence_summary.md`](reports/feature_evidence_summary.md)  
   최종 feature 구성, 반경 선택, 결측·변환 판단의 핵심 요약입니다.

3. [`reports/starbucks_clustering_enhancement_summary.md`](reports/starbucks_clustering_enhancement_summary.md)  
   KMeans k=5, 대안 알고리즘, 이상치, 비스타벅스 투영을 정리한 클러스터링 보고서입니다.

4. [`reports/classification/01_modeling_summary.md`](reports/classification/01_modeling_summary.md)  
   카페 단위 1차 PU classification 분석입니다. 비스타벅스를 `Unlabeled`로 보는 이유, 분류 피처, 지표, 모델 비교, 카페 단위 한계를 정리했습니다.

5. [`reports/classification/02_district_trust_summary.md`](reports/classification/02_district_trust_summary.md)  
   상권 단위 고도화와 최종 결론 보고서입니다. 카페 단위 한계를 상권 단위로 보강하고, 서울대 결론의 신뢰성을 검증합니다.

## 저장소 구조

```text
data/
  final/
    seoul_cafe_model_features_final.csv
    starbucks_model_features_final.csv
  modeling/
    starbucks_engineered_features_final.csv

docs/
  data_sources.md
  pipeline.md

presentation/
  스타벅스_입점예측_제안발표_vF.pdf
  스타벅스_입점예측_중간발표_vF.pdf

reports/
  README.md
  feature_evidence_summary.md
  starbucks_feature_engineering_summary.md
  starbucks_clustering_enhancement_summary.md
  classification/
    01_modeling_summary.md
    02_district_trust_summary.md
  archive/
    figures/
    tables/
    classification/logs/
  generated/
    # 재생성되는 산출물. Git 추적 제외.

scripts/
  01_source_build/
  02_feature_pipeline/
  03_finalize_features/
  04_eda/
  05_clustering/
  06_classification/
```

## 실행 환경

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

주요 의존성은 `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`입니다. Classification model zoo와 일부 PU baseline 실험에는 `imbalanced-learn`, `lightgbm`, `xgboost`가 필요합니다.

## 재현 방법

전체 재현 순서와 각 스크립트 역할은 [`docs/pipeline.md`](docs/pipeline.md)에 정리되어 있습니다.

핵심 단계만 보면 다음과 같습니다.

```powershell
# clustering 검증 재생성
python scripts/05_clustering/01_validity_and_algorithms.py
python scripts/05_clustering/02_visualization.py
python scripts/05_clustering/03_outlier_and_projection.py
python scripts/05_clustering/04_pdf_deepening.py

# classification 데이터셋 준비와 모델링
python scripts/06_classification/00_prepare.py
python scripts/06_classification/01_pu_baselines.py
python scripts/06_classification/02_cv_comparison.py
python scripts/06_classification/03_model_zoo.py
python scripts/06_classification/06_pu_final.py
python scripts/06_classification/07_personas.py
python scripts/06_classification/08_integration.py
python scripts/06_classification/19_district_final.py
```

스크립트 실행 결과는 기본적으로 `reports/generated/` 아래에 생성되며 Git에는 포함하지 않습니다. 보고서에 사용한 표, 그림, 로그 스냅샷은 `reports/archive/`에 보존했습니다.

서울대학교 2026-1학기 데이터마이닝 텀 프로젝트.
