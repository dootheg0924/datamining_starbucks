# Classification 문서 읽는 순서

이 폴더는 서울 전체 카페 22,305개를 대상으로 스타벅스 입점 적합도를 예측한 Classification 단계 문서입니다. 핵심은 비스타벅스 카페를 확정 음성으로 보지 않고, 아직 입점 여부를 모르는 `Unlabeled`로 두는 PU learning입니다.

## 먼저 읽기

1. [`modeling_summary.md`](modeling_summary.md)
   - Classification 전체 서사와 최종 결론입니다.
   - 문제 정의, PU learning 채택 이유, 모델 비교, 클러스터링 연결, 서울대 시뮬레이션을 한 번에 볼 수 있습니다.

2. [`district_trust_summary.md`](district_trust_summary.md)
   - 상권 단위 보강과 신뢰성 지표를 설명합니다.
   - 카페 하나하나가 아니라 DBSCAN으로 묶은 상권 단위에서 다시 평가해, 서울대가 왜 낮은 적합도인지 더 선명하게 보여줍니다.

## 필요할 때 읽기

3. [`feature_engineering.md`](feature_engineering.md)
   - 분류 모델에 들어간 15개 피처의 정의서입니다.
   - 클러스터링 16개 피처와 거의 같지만, `log_dist_starbucks`는 라벨 누수라 분류 입력에서 제외합니다.

4. [`metric_reference.md`](metric_reference.md)
   - PU-AUC, AP, lift, gains, OOT recall, calibration 같은 평가 지표의 상세 설명입니다.
   - 정확도나 F1을 쓰지 않은 이유가 필요할 때 보면 됩니다.

## 한눈에 보는 구조

| 질문 | 읽을 문서 |
|---|---|
| Classification의 최종 결론이 뭔가? | `modeling_summary.md` |
| 서울대에 스타벅스가 없는 이유를 어떻게 설명했나? | `modeling_summary.md` 10장, `district_trust_summary.md` 1부 |
| 왜 비스타벅스를 음성으로 두지 않았나? | `modeling_summary.md` 1장, 3장 |
| 왜 정확도/F1 대신 PU-AUC와 lift를 썼나? | `metric_reference.md` |
| 어떤 피처가 들어갔고 누수는 어떻게 막았나? | `feature_engineering.md` |
| 보고서 수치의 원 실행 로그는 어디 있나? | `../archive/classification/logs/` |
| 발표용 그림 스냅샷은 어디 있나? | `../archive/figures/classification/` |

## 핵심 용어

| 용어 | 뜻 |
|---|---|
| P | 이미 스타벅스가 있는 카페/입지입니다. 모델에서 Positive로 둡니다. |
| U | 비스타벅스 카페입니다. 부적합 음성이 아니라 아직 스벅 입점 여부를 모르는 Unlabeled로 둡니다. |
| PU learning | Positive와 Unlabeled만으로 “스벅형 입지” 순위를 학습하는 방식입니다. |
| PU-AUC | P가 U보다 높은 점수를 받는 순위 성능입니다. U 안의 숨은 양성 때문에 절대값은 보수적으로 봅니다. |
| lift@k | 모델 상위 k% 후보가 평균 대비 스벅을 몇 배 더 많이 포함하는지 보는 지표입니다. |
| Opt1/2/3 | 분류 점수와 클러스터링 페르소나를 연결하는 세 가지 출력 방식입니다. 보고서에서는 게이팅 기반 Opt3를 주요 서사로 씁니다. |
| 상권 단위 | 개별 카페가 아니라 DBSCAN으로 묶은 카페 밀집 구역을 하나의 분석 단위로 보는 보강 분석입니다. |

## Classification 파이프라인

재생성 스크립트는 [`../../scripts/06_classification/`](../../scripts/06_classification/)에 있습니다. 기본 출력은 Git에 포함하지 않는 `reports/generated/classification/` 아래에 생성됩니다.

```bash
python scripts/06_classification/00_prepare.py
python scripts/06_classification/01_pu_baselines.py
python scripts/06_classification/02_cv_comparison.py
python scripts/06_classification/03_model_zoo.py
python scripts/06_classification/04_pu_twostep.py
python scripts/06_classification/05_pu_twostep_pos.py
python scripts/06_classification/06_pu_final.py
python scripts/06_classification/07_personas.py
python scripts/06_classification/08_integration.py
python scripts/06_classification/09_snu_simulation.py
python scripts/06_classification/10_validation.py
python scripts/06_classification/11_interpretation.py
python scripts/06_classification/12_figures.py
python scripts/06_classification/13_pu_advanced.py
python scripts/06_classification/14_pu_iterative.py
python scripts/06_classification/15_trust_metrics.py
python scripts/06_classification/16_metric_curves.py
python scripts/06_classification/17_districts.py
python scripts/06_classification/18_district_advanced.py
python scripts/06_classification/19_district_final.py
```
