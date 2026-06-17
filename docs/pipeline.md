# 재현 파이프라인

이 저장소는 최종 모델링 artifact는 Git에 포함하고, 원천 데이터와 재생성 가능한 중간 산출물은 제외합니다.

## 폴더 역할

```text
data/
  final/
    seoul_cafe_model_features_final.csv
    starbucks_model_features_final.csv
  modeling/
    starbucks_engineered_features_final.csv

scripts/
  01_source_build/
    원천 파일 -> rawdata/seoul_cafe_master.csv
  02_feature_pipeline/
    master data -> 중간 및 최종 후보 feature
  03_finalize_features/
    중간 모델 CSV -> data/final/
  04_eda/
    진단 표와 그림 -> reports/generated/
  05_clustering/
    스타벅스 파생 feature -> 클러스터링 검증 표와 그림
  06_classification/
    00_prepare.py -> 공통 classification dataset 생성
    01_cafe_level/ -> 카페 단위 1차 PU learning, 페르소나 연결, 서울대 예비 검증
    02_district_level/ -> DBSCAN 상권 단위 PU learning, feature importance, 신규 입점 검증
```

## 전체 재생성 순서

저장소 루트에서 실행합니다. 전체 재현을 하려면 먼저 `rawdata/`에 원천 파일을 준비해야 합니다.

```bash
python scripts/01_source_build/01_kakao_admin_codes.py
python scripts/01_source_build/02_build_cafe_master.py
python scripts/01_source_build/03_build_subway_station_master.py
python scripts/01_source_build/04_add_master_features.py

python scripts/02_feature_pipeline/04_geo_feature_engineering.py
python scripts/02_feature_pipeline/05_radius_selection_eda.py
python scripts/02_feature_pipeline/06_clustering_csv_finalization.py
python scripts/02_feature_pipeline/07_model_feature_finalization_v2.py

python scripts/03_finalize_features/01_repair_feature_missing_values.py
python scripts/03_finalize_features/02_add_nan_reason.py
python scripts/03_finalize_features/03_extract_starbucks_final.py
python scripts/02_feature_pipeline/08_starbucks_feature_engineering.py

python scripts/05_clustering/01_validity_and_algorithms.py
python scripts/05_clustering/02_visualization.py
python scripts/05_clustering/03_outlier_and_projection.py
python scripts/05_clustering/04_pdf_deepening.py

python scripts/06_classification/00_prepare.py
python scripts/06_classification/01_cafe_level/01_pu_baselines.py
python scripts/06_classification/01_cafe_level/02_cv_comparison.py
python scripts/06_classification/01_cafe_level/03_model_zoo.py
python scripts/06_classification/01_cafe_level/04_pu_twostep.py
python scripts/06_classification/01_cafe_level/05_pu_twostep_pos.py
python scripts/06_classification/01_cafe_level/06_pu_final.py
python scripts/06_classification/01_cafe_level/07_personas.py
python scripts/06_classification/01_cafe_level/08_integration.py
python scripts/06_classification/01_cafe_level/09_snu_simulation.py
python scripts/06_classification/01_cafe_level/10_validation.py
python scripts/06_classification/01_cafe_level/11_interpretation.py
python scripts/06_classification/01_cafe_level/12_figures.py
python scripts/06_classification/01_cafe_level/13_pu_advanced.py
python scripts/06_classification/01_cafe_level/14_pu_iterative.py
python scripts/06_classification/01_cafe_level/15_trust_metrics.py
python scripts/06_classification/01_cafe_level/16_metric_curves.py
python scripts/06_classification/02_district_level/17_districts.py
python scripts/06_classification/02_district_level/18_district_advanced.py
python scripts/06_classification/02_district_level/19_district_final.py
python scripts/06_classification/02_district_level/20_district_persona_bridge.py
python scripts/06_classification/02_district_level/21_district_feature_importance.py
python scripts/06_classification/02_district_level/22_new_starbucks_district_projection.py
python scripts/06_classification/02_district_level/23_district_model_selection.py
```

`scripts/01_source_build/01_kakao_admin_codes.py`는 Kakao Local API key가 필요합니다. 자세한 설정은 `docs/data_sources.md`를 참고하세요.

## 필요할 때만 실행하는 점검 스크립트

아래 스크립트는 파이프라인의 핵심 산출물을 만들기 위한 필수 단계라기보다 데이터 상태와 원천 파일을 점검하기 위한 보조 단계입니다.

```bash
python scripts/02_feature_pipeline/01_data_audit.py
python scripts/02_feature_pipeline/02_starbucks_only_eda.py
python scripts/02_feature_pipeline/03_raw_data_inventory.py
```

## EDA 재생성

정리된 EDA 진단 표와 그림은 다음 명령으로 다시 만들 수 있습니다.

```bash
python scripts/04_eda/01_missing_outlier_diagnostics.py
python scripts/04_eda/02_correlation_transform_diagnostics.py
```

재생성된 표와 그림은 `reports/generated/` 아래에 저장되며 Git에는 포함하지 않습니다.

## 클러스터링 고도화 재생성

클러스터링 고도화 스크립트는 `data/modeling/starbucks_engineered_features_final.csv`와 `data/final/seoul_cafe_model_features_final.csv`를 읽어 KMeans k=5 정당성, 대안 알고리즘, 이상치, 비스타벅스 투영, PDF 심화 검증을 재생성합니다.

```bash
python scripts/05_clustering/01_validity_and_algorithms.py
python scripts/05_clustering/02_visualization.py
python scripts/05_clustering/03_outlier_and_projection.py
python scripts/05_clustering/04_pdf_deepening.py
```

기본 출력은 `reports/generated/clustering/` 아래에 저장되며 Git에는 포함하지 않습니다. 보고서에 사용한 현재 스냅샷은 `reports/archive/tables/clustering/`, `reports/archive/figures/clustering/`에 보존합니다.

## Classification 재생성

분류 스크립트는 `data/final/seoul_cafe_model_features_final.csv`를 읽어 `reports/generated/classification/` 아래에 데이터셋, 모델 출력, 그림, 로그를 재생성합니다. 공통 데이터셋은 `reports/generated/classification/data/`에, 카페 단위 1단계 산출물은 `reports/generated/classification/01_cafe_level/`에, 상권 단위 2단계 산출물은 `reports/generated/classification/02_district_level/`에 저장됩니다. 이 폴더는 Git 추적에서 제외되며, 보고서에 사용한 그림/표/로그 스냅샷도 `reports/archive/.../classification/01_cafe_level/`, `reports/archive/.../classification/02_district_level/`로 나눠 보존합니다.

```bash
python scripts/06_classification/00_prepare.py
python scripts/06_classification/01_cafe_level/01_pu_baselines.py
python scripts/06_classification/01_cafe_level/02_cv_comparison.py
python scripts/06_classification/01_cafe_level/03_model_zoo.py
python scripts/06_classification/01_cafe_level/04_pu_twostep.py
python scripts/06_classification/01_cafe_level/05_pu_twostep_pos.py
python scripts/06_classification/01_cafe_level/06_pu_final.py
python scripts/06_classification/01_cafe_level/07_personas.py
python scripts/06_classification/01_cafe_level/08_integration.py
python scripts/06_classification/01_cafe_level/09_snu_simulation.py
python scripts/06_classification/01_cafe_level/10_validation.py
python scripts/06_classification/01_cafe_level/11_interpretation.py
python scripts/06_classification/01_cafe_level/12_figures.py
python scripts/06_classification/01_cafe_level/13_pu_advanced.py
python scripts/06_classification/01_cafe_level/14_pu_iterative.py
python scripts/06_classification/01_cafe_level/15_trust_metrics.py
python scripts/06_classification/01_cafe_level/16_metric_curves.py

python scripts/06_classification/02_district_level/17_districts.py
python scripts/06_classification/02_district_level/18_district_advanced.py
python scripts/06_classification/02_district_level/19_district_final.py
python scripts/06_classification/02_district_level/20_district_persona_bridge.py
python scripts/06_classification/02_district_level/21_district_feature_importance.py
python scripts/06_classification/02_district_level/22_new_starbucks_district_projection.py
python scripts/06_classification/02_district_level/23_district_model_selection.py
```

## 주요 출력 파일

- `data/final/seoul_cafe_model_features_final.csv`: 서울 전체 카페 기준 최종 feature table
- `data/final/starbucks_model_features_final.csv`: 최종 feature table에서 스타벅스만 추출한 subset
- `data/modeling/starbucks_engineered_features_final.csv`: 스타벅스 전용 모델링 및 클러스터링용 파생 feature set
- `reports/starbucks_clustering_enhancement_summary.md`: 클러스터링 고도화 결과와 피드백 반영 요약
- `reports/01_modeling_summary.md`: 카페 단위 1차 PU classification 분석과 상권 단위 보강 필요성
- `reports/02_district_trust_summary.md`: 상권 단위 고도화, 최종 서울대 결론, 신뢰성 검증 요약
