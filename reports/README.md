# Reports 읽는 순서

이 폴더는 최종 보고서와 재현 근거를 보관합니다. 처음 읽을 때는 모든 문서를 순서대로 열 필요 없이 아래 순서만 따르면 됩니다.

## 1. 전체 feature와 데이터 근거

- [`feature_evidence_summary.md`](feature_evidence_summary.md)
  - 최종 feature 구성, 반경 선택, 결측/변환 판단의 핵심 요약입니다.
- [`starbucks_feature_engineering_summary.md`](starbucks_feature_engineering_summary.md)
  - 스타벅스 전용 파생 feature set의 변환 방식과 선택 이유입니다.

## 2. 클러스터링

- [`starbucks_clustering_enhancement_summary.md`](starbucks_clustering_enhancement_summary.md)
  - KMeans k=5, 대안 알고리즘, 이상치, 비스타벅스 투영, PDF 심화 검증을 정리한 최종 클러스터링 보고서입니다.

## 3. Classification

- [`01_modeling_summary.md`](01_modeling_summary.md)
  - 카페 단위 1차 PU classification 분석입니다. 비스타벅스를 `Unlabeled`로 보는 이유, 분류 피처, 지표, 모델 비교, 카페 단위 한계를 정리합니다.
- [`02_district_trust_summary.md`](02_district_trust_summary.md)
  - 상권 단위 고도화와 최종 결론 보고서입니다. 카페 단위 한계를 보강하고 서울대 결론의 신뢰성을 검증합니다.

## 4. Archive와 Generated

- `archive/`
  - 보고서에 사용한 그림, 표, 실행 로그의 보존 스냅샷입니다. 원문 근거를 보려는 경우에만 확인합니다.
- `generated/`
  - 스크립트 실행으로 다시 생성되는 산출물입니다. Git에는 포함하지 않습니다.

## 추천 독서 루트

1. 최종 feature가 궁금하면 `feature_evidence_summary.md`
2. 클러스터링 결론이 궁금하면 `starbucks_clustering_enhancement_summary.md`
3. 분류/입점예측이 궁금하면 `01_modeling_summary.md`
4. 최종 서울대 결론과 신뢰성까지 보려면 `02_district_trust_summary.md`
