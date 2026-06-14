# Starbucks Data Mining Handoff

서울 스타벅스 입지 분석 프로젝트의 인수인계용 저장소입니다. 이 저장소는 다음 담당자가 최종 feature CSV와 변수 선정 근거를 빠르게 확인할 수 있도록 정리한 버전입니다. 참고하여 다음 진행 상황을 업데이트해주세요.

## What To Open First

1. `reports/feature_evidence_summary.md`
   - 최종 feature별 포함 이유, 통계/시각화 확인 여부, 반경 선택 근거, 제외 변수 이유를 정리한 핵심 문서입니다.
2. `data/starbucks_model_features_final.csv`
   - 스타벅스 681개 매장 기준 최종 모델 feature입니다.
3. `data/seoul_cafe_model_features_final.csv`
   - 서울 카페 22,305개 기준 최종 모델 feature입니다. `nan_reason`은 보정 전 결측 원인을 기록한 provenance 컬럼입니다.
4. `reports/archive/analysis_archive_summary.md`
   - 기존 상세 archive 보고서 7개를 압축한 분석 흐름 요약입니다.

## Repository Structure

```text
data/
  seoul_cafe_model_features_final.csv
  starbucks_model_features_final.csv

reports/
  feature_evidence_summary.md
  archive/
    analysis_archive_summary.md
    figures/
    tables/
  generated/
    # ignored regenerated reports, tables, and figures

scripts/
  source_build/
    # raw source files -> rawdata/seoul_cafe_master.csv
  01_data_audit.py
  02_starbucks_only_eda.py
  03_raw_data_inventory.py
  04_geo_feature_engineering.py
  05_radius_selection_eda.py
  06_clustering_csv_finalization.py
  07_model_feature_finalization_v2.py
  finalize_features/
    # intermediate model CSV -> final CSV

docs/
  data_sources.md

archive/
  html_maps/

presentation/
  *.pdf
```

## Data Policy

Raw data files are not included in GitHub. They are excluded with `.gitignore` because they are source datasets, relatively large, and should be managed separately with their original sources and license information.

Excluded local folders:

- `rawdata/`: source CSV files
- `data/archive/`: intermediate generated data
- `reports/generated/`: regenerated detailed reports, tables, and figures
- `incoming_1/`: original external handoff code, kept locally and not uploaded
- `.venv/`: local Python virtual environment
- `archive/notebooks/`: exploratory notebooks

The final CSV files in `data/` are included because they are the handoff-ready modeling datasets.

## Environment

Create a fresh environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The scripts mainly use `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn`.

## Reproduction Flow

With raw data available locally, the intended flow is:

```bash
python scripts/source_build/01_kakao_admin_codes.py
python scripts/source_build/02_build_cafe_master.py
python scripts/source_build/03_build_subway_station_master.py
python scripts/source_build/04_add_master_features.py

python scripts/04_geo_feature_engineering.py
python scripts/05_radius_selection_eda.py
python scripts/06_clustering_csv_finalization.py
python scripts/07_model_feature_finalization_v2.py

python scripts/finalize_features/01_repair_feature_missing_values.py
python scripts/finalize_features/02_add_nan_reason.py
python scripts/finalize_features/03_extract_starbucks_final.py
```

See `docs/data_sources.md` for required raw files and API-key notes.

## Notes For The Next Person

- The final Seoul-wide CSV has repaired feature missing values, while `nan_reason` preserves why those rows had missing source features before repair.
- The final CSV files do not apply outlier removal, scaling, or log transformation.
- Distance variables are stored in kilometers.
- Radius count variables include the radius in the variable name, such as `num_bus_stops_300m`.
- Some scripts require raw files that are not included in GitHub. Use the reports to identify the original input filenames if full reproduction is needed.
- The most important modeling caveats are summarized in `reports/feature_evidence_summary.md`.
