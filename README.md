# 서울 스타벅스 데이터마이닝 재현 패키지

이 저장소는 서울 카페 및 스타벅스 입지 분석 프로젝트를 제출 가능한 재현 artifact로 정리한 버전입니다. GitHub에는 최종 분석에 필요한 코드, 문서, 최종 CSV만 포함하고, 원천 raw data와 재생성 가능한 중간 산출물은 로컬에서 별도로 관리합니다.

## 먼저 볼 파일

1. `reports/feature_evidence_summary.md`
   - 최종 feature 구성, 반경 선택 근거, 변수별 모델링 주의사항을 요약한 핵심 문서입니다.
2. `data/final/starbucks_model_features_final.csv`
   - 스타벅스 681개 매장 기준 최종 모델 feature입니다.
3. `data/final/seoul_cafe_model_features_final.csv`
   - 서울 카페 22,305개 기준 최종 모델 feature입니다. `nan_reason`은 보정 전 결측 원인을 기록한 provenance 컬럼입니다.
4. `data/modeling/starbucks_engineered_features_final.csv`
   - 스타벅스 전용 모델링 및 클러스터링용 파생 feature set입니다.
5. `reports/starbucks_feature_engineering_summary.md`
   - 스타벅스 전용 파생 feature의 변환 방식과 선택 이유를 정리한 문서입니다.
6. `reports/starbucks_clustering_enhancement_summary.md`
   - KMeans k=5 정당성, 대안 알고리즘, 이상치, 비스타벅스 투영, PDF 심화 검증을 정리한 클러스터링 고도화 보고서입니다.
7. `docs/pipeline.md`
   - 전체 재현 파이프라인과 스크립트 실행 순서를 설명합니다.
8. `docs/data_sources.md`
   - 전체 재현에 필요한 로컬 raw data와 API key 설정을 설명합니다.

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

reports/
  feature_evidence_summary.md
  starbucks_feature_engineering_summary.md
  starbucks_clustering_enhancement_summary.md
  archive/
    analysis_archive_summary.md
    figures/
      clustering/
    html_maps/
    tables/
      clustering/
  generated/
    # 재생성되는 상세 표와 그림. Git 추적 제외.

scripts/
  source_build/
    # 원천 파일 -> rawdata/seoul_cafe_master.csv
  feature_pipeline/
    # feature engineering 및 최종 후보 feature 생성
  finalize_features/
    # 중간 모델 CSV -> data/final/
  eda/
    # 정리된 EDA 진단 스크립트
  clustering/
    # 클러스터링 고도화 분석과 검증 스크립트

presentation/
  *.pdf
```

## 실행 환경

새 가상환경을 만들고 의존성을 설치합니다.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

주요 스크립트는 `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`을 사용합니다. Excel 원천 파일을 점검하는 인벤토리 단계에서는 `openpyxl`이 필요할 수 있습니다.

## 재현 흐름

전체 재현 순서는 `docs/pipeline.md`를 참고하세요. 필요한 원천 파일과 Kakao API key 설정은 `docs/data_sources.md`에 정리되어 있습니다.
