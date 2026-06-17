# 상권 단위 모델 선택 및 하이퍼파라미터 비교

> 생성 2026-06-17 21:11:11

- 상권 정의: DBSCAN auto-eps=223m + 노이즈 최근접 흡수
- 상권 수: 170개, 스벅상권: 87개
- 검증: spatial GroupKFold(5), 각 fold train 내부 PU bagging 15회
- 선택 기준: OOF PU-AUC 1순위, lift@5%와 gains@20%를 보조 지표로 확인

## 선택 결과

- Best: **LOGIT_C0.3** (LogisticRegression)
- Best OOF PU-AUC: **0.8024**
- Current RF 기준선 OOF PU-AUC: 0.7899
- Best - current delta: +0.0125
- Best lift@5%: 1.52
- Best gains@20%: 0.356

## Top 10 Candidates

| rank | model_id         | family             | pu_auc | fold_auc_mean | fold_auc_std | lift_at_5 | gains_at_20 |
| ---- | ---------------- | ------------------ | ------ | ------------- | ------------ | --------- | ----------- |
| 1    | LOGIT_C0.3       | LogisticRegression | 0.8024 | 0.8054        | 0.0589       | 1.5198    | 0.3563      |
| 2    | LOGIT_C1         | LogisticRegression | 0.8024 | 0.8011        | 0.0693       | 1.5198    | 0.3563      |
| 3    | LOGIT_C3         | LogisticRegression | 0.8013 | 0.8020        | 0.0689       | 1.5198    | 0.3563      |
| 4    | RF_leaf2_mf0.7   | RandomForest       | 0.7977 | 0.8092        | 0.0558       | 1.9540    | 0.3563      |
| 5    | RF_leaf5_mf0.7   | RandomForest       | 0.7975 | 0.8076        | 0.0590       | 1.9540    | 0.3678      |
| 6    | RF_leaf5_mfsqrt  | RandomForest       | 0.7955 | 0.8048        | 0.0459       | 1.9540    | 0.3563      |
| 7    | RF_leaf1_mf0.7   | RandomForest       | 0.7934 | 0.8064        | 0.0618       | 1.9540    | 0.3563      |
| 8    | RF_leaf2_mfsqrt  | RandomForest       | 0.7899 | 0.8049        | 0.0521       | 1.9540    | 0.3563      |
| 9    | RF_current_leaf2 | RandomForest       | 0.7899 | 0.8056        | 0.0511       | 1.9540    | 0.3563      |
| 10   | RF_leaf1_mfsqrt  | RandomForest       | 0.7892 | 0.8134        | 0.0414       | 1.9540    | 0.3563      |

## Family Best

| model_id          | family             | pu_auc | lift_at_5 | gains_at_20 | params                                                            |
| ----------------- | ------------------ | ------ | --------- | ----------- | ----------------------------------------------------------------- |
| LOGIT_C0.3        | LogisticRegression | 0.8024 | 1.5198    | 0.3563      | {"C": 0.3}                                                        |
| RF_leaf2_mf0.7    | RandomForest       | 0.7977 | 1.9540    | 0.3563      | {"max_features": 0.7, "min_samples_leaf": 2, "n_estimators": 160} |
| ET_leaf5_mf0.7    | ExtraTrees         | 0.7794 | 1.9540    | 0.3678      | {"max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 160} |
| GBM_n140_lr0.1_d2 | GradientBoosting   | 0.7763 | 1.9540    | 0.3563      | {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 140}       |

## 해석

- 기존 19번 상권 최종 모델은 고정 RandomForest 설정을 사용했지만, 이 비교는 상권 단위에서도 모델 family와 hyperparameter 선택 과정을 명시적으로 추가한다.
- PU 구조상 이 값들은 확정 음성 기준의 절대 정확도가 아니라, 스타벅스 상권을 U 상권보다 위에 두는 순위 성능으로 해석해야 한다.
- 최종 채점 모델을 교체할 때는 위 best 후보를 기준으로 19번 scoring 모델을 재학습하고 F16-F22 downstream 산출물을 함께 재생성하면 된다.
