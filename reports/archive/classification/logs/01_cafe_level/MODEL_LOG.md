# Classification 모델 실험 로그

> 자동 생성/누적. 각 실행은 타임스탬프 블록으로 append 됩니다.
> 라벨링=PU(1:1 bagging ×20), 검증=spatial GroupKFold(5), 피처=raw 17개.


---

## 실행 @ 2026-06-14 01:14:06

### 전체 비교표 (PU-AUC 내림차순)

| 모델 | 강의 | type | PU-AUC | fold AUC(평균±std) | AP | lift@5% | prec@681 | sec |
|---|---|---|---|---|---|---|---|---|
| GBM | 09강 | PU-bag | 0.6740 | 0.6734±0.021 | 0.0567 | 2.585 | 0.0808 | 47.8 |
| RandomForest | 08강 | PU-bag | 0.6732 | 0.6726±0.016 | 0.0569 | 2.468 | 0.0793 | 32.1 |
| Bagging | 08강 | PU-bag | 0.6714 | 0.6710±0.019 | 0.0587 | 2.291 | 0.0764 | 13.6 |
| XGBoost | 09강 | PU-bag | 0.6711 | 0.6707±0.020 | 0.0565 | 2.556 | 0.0749 | 11.8 |
| SVM(RBF) | 12강 | PU-bag | 0.6672 | 0.6665±0.021 | 0.0553 | 2.321 | 0.0793 | 35.0 |
| AdaBoost | 09강 | PU-bag | 0.6672 | 0.6651±0.020 | 0.0570 | 2.497 | 0.0852 | 45.6 |
| DecisionTree | 06강 | PU-bag | 0.6654 | 0.6643±0.027 | 0.0592 | 2.526 | 0.0896 | 0.7 |
| ExtraTrees | 08강 | PU-bag | 0.6614 | 0.6611±0.018 | 0.0522 | 2.056 | 0.0602 | 24.6 |
| LightGBM | 09강 | PU-bag | 0.6607 | 0.6602±0.020 | 0.0552 | 2.350 | 0.0808 | 16.5 |
| LogReg | 04강 | PU-bag | 0.6538 | 0.6522±0.027 | 0.0546 | 2.585 | 0.0808 | 0.9 |
| LDA | 05강 | PU-bag | 0.6532 | 0.6515±0.027 | 0.0544 | 2.497 | 0.0793 | 0.5 |
| kNN | 05강 | PU-bag | 0.6436 | 0.6432±0.026 | 0.0481 | 1.763 | 0.0543 | 4.1 |
| MLP | 07강 | PU-bag | 0.6378 | 0.6374±0.013 | 0.0517 | 2.233 | 0.0734 | 93.5 |
| NaiveBayes | 05강 | PU-bag | 0.6251 | 0.6235±0.027 | 0.0455 | 1.704 | 0.0602 | 0.3 |
| OneClassSVM | 13강 | one-class | 0.4838 | 0.4850±0.052 | 0.0313 | 1.087 | 0.0426 | 0.3 |
| IsolationForest | 13강 | one-class | 0.4522 | 0.4547±0.044 | 0.0300 | 0.969 | 0.0323 | 2.1 |

### 모델별 상세

#### GBM  (09강, PU-bag)
- 하이퍼파라미터: `{"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 150, "n_`
- PU-AUC(pooled): **0.6740** | AP: 0.0567 | lift@5%: 2.585 | prec@681: 0.0808
- fold별 AUC: [0.6368, 0.6900, 0.6684, 0.6979, 0.6737]  → 평균 0.6734 ± 0.021
- 소요: 47.8s
- 메모: (1차 기본 하이퍼파라미터)

#### RandomForest  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 2, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null`
- PU-AUC(pooled): **0.6732** | AP: 0.0569 | lift@5%: 2.468 | prec@681: 0.0793
- fold별 AUC: [0.6427, 0.6859, 0.6714, 0.6877, 0.6752]  → 평균 0.6726 ± 0.016
- 소요: 32.1s
- 메모: (1차 기본 하이퍼파라미터)

#### Bagging  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": true, "bootstrap_features": false, "estimator": null, "max_features": 1.0, "max_samples": null, "n_estimators": 50, "n_jobs": -1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}`
- PU-AUC(pooled): **0.6714** | AP: 0.0587 | lift@5%: 2.291 | prec@681: 0.0764
- fold별 AUC: [0.6369, 0.6917, 0.6705, 0.6815, 0.6745]  → 평균 0.6710 ± 0.019
- 소요: 13.6s
- 메모: (1차 기본 하이퍼파라미터)

#### XGBoost  (09강, PU-bag)
- 하이퍼파라미터: `{"objective": "binary:logistic", "base_score": null, "booster": null, "callbacks": null, "colsample_bylevel": null, "colsample_bynode": null, "colsample_bytree": null, "device": null, "early_stopping_rounds": null, "enable_categorical": false, "eval_metric": "logloss", "feature_types": null, "featur`
- PU-AUC(pooled): **0.6711** | AP: 0.0565 | lift@5%: 2.556 | prec@681: 0.0749
- fold별 AUC: [0.6363, 0.6893, 0.6638, 0.6930, 0.6710]  → 평균 0.6707 ± 0.020
- 소요: 11.8s
- 메모: (1차 기본 하이퍼파라미터)

#### SVM(RBF)  (12강, PU-bag)
- 하이퍼파라미터: `{"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": null, "shrinking": true, "tol": 0.001, "verbose": false}`
- PU-AUC(pooled): **0.6672** | AP: 0.0553 | lift@5%: 2.321 | prec@681: 0.0793
- fold별 AUC: [0.6351, 0.6714, 0.6567, 0.6977, 0.6714]  → 평균 0.6665 ± 0.021
- 소요: 35.0s
- 메모: (1차 기본 하이퍼파라미터)

#### AdaBoost  (09강, PU-bag)
- 하이퍼파라미터: `{"estimator": null, "learning_rate": 1.0, "n_estimators": 200, "random_state": 42}`
- PU-AUC(pooled): **0.6672** | AP: 0.0570 | lift@5%: 2.497 | prec@681: 0.0852
- fold별 AUC: [0.6292, 0.6646, 0.6709, 0.6904, 0.6706]  → 평균 0.6651 ± 0.020
- 소요: 45.6s
- 메모: (1차 기본 하이퍼파라미터)

#### DecisionTree  (06강, PU-bag)
- 하이퍼파라미터: `{"ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": 6, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 20, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "random_state": 42, "splitter": "best"}`
- PU-AUC(pooled): **0.6654** | AP: 0.0592 | lift@5%: 2.526 | prec@681: 0.0896
- fold별 AUC: [0.6184, 0.6796, 0.6545, 0.6964, 0.6727]  → 평균 0.6643 ± 0.027
- 소요: 0.7s
- 메모: (1차 기본 하이퍼파라미터)

#### ExtraTrees  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": false, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": nul`
- PU-AUC(pooled): **0.6614** | AP: 0.0522 | lift@5%: 2.056 | prec@681: 0.0602
- fold별 AUC: [0.6283, 0.6792, 0.6568, 0.6737, 0.6672]  → 평균 0.6611 ± 0.018
- 소요: 24.6s
- 메모: (1차 기본 하이퍼파라미터)

#### LightGBM  (09강, PU-bag)
- 하이퍼파라미터: `{"boosting_type": "gbdt", "class_weight": null, "colsample_bytree": 1.0, "importance_type": "split", "learning_rate": 0.1, "max_depth": -1, "min_child_samples": 20, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 200, "n_jobs": -1, "num_leaves": 31, "objective": null, "random_state`
- PU-AUC(pooled): **0.6607** | AP: 0.0552 | lift@5%: 2.350 | prec@681: 0.0808
- fold별 AUC: [0.6291, 0.6869, 0.6497, 0.6732, 0.6618]  → 평균 0.6602 ± 0.020
- 소요: 16.5s
- 메모: (1차 기본 하이퍼파라미터)

#### LogReg  (04강, PU-bag)
- 하이퍼파라미터: `{"C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true, "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 1000, "n_jobs": null, "penalty": "deprecated", "random_state": null, "solver": "lbfgs", "tol": 0.0001, "verbose": 0, "warm_start": false}`
- PU-AUC(pooled): **0.6538** | AP: 0.0546 | lift@5%: 2.585 | prec@681: 0.0808
- fold별 AUC: [0.6052, 0.6546, 0.6500, 0.6886, 0.6626]  → 평균 0.6522 ± 0.027
- 소요: 0.9s
- 메모: (1차 기본 하이퍼파라미터)

#### LDA  (05강, PU-bag)
- 하이퍼파라미터: `{"covariance_estimator": null, "n_components": null, "priors": null, "shrinkage": null, "solver": "svd", "store_covariance": false, "tol": 0.0001}`
- PU-AUC(pooled): **0.6532** | AP: 0.0544 | lift@5%: 2.497 | prec@681: 0.0793
- fold별 AUC: [0.6039, 0.6540, 0.6489, 0.6884, 0.6625]  → 평균 0.6515 ± 0.027
- 소요: 0.5s
- 메모: (1차 기본 하이퍼파라미터)

#### kNN  (05강, PU-bag)
- 하이퍼파라미터: `{"algorithm": "auto", "leaf_size": 30, "metric": "minkowski", "metric_params": null, "n_jobs": null, "n_neighbors": 15, "p": 2, "weights": "uniform"}`
- PU-AUC(pooled): **0.6436** | AP: 0.0481 | lift@5%: 1.763 | prec@681: 0.0543
- fold별 AUC: [0.5985, 0.6536, 0.6355, 0.6793, 0.6489]  → 평균 0.6432 ± 0.026
- 소요: 4.1s
- 메모: (1차 기본 하이퍼파라미터)

#### MLP  (07강, PU-bag)
- 하이퍼파라미터: `{"activation": "relu", "alpha": 0.0001, "batch_size": "auto", "beta_1": 0.9, "beta_2": 0.999, "early_stopping": false, "epsilon": 1e-08, "hidden_layer_sizes": [64, 32], "learning_rate": "constant", "learning_rate_init": 0.001, "max_fun": 15000, "max_iter": 300, "momentum": 0.9, "n_iter_no_change": 1`
- PU-AUC(pooled): **0.6378** | AP: 0.0517 | lift@5%: 2.233 | prec@681: 0.0734
- fold별 AUC: [0.6138, 0.6415, 0.6331, 0.6507, 0.6478]  → 평균 0.6374 ± 0.013
- 소요: 93.5s
- 메모: (1차 기본 하이퍼파라미터)

#### NaiveBayes  (05강, PU-bag)
- 하이퍼파라미터: `{"priors": null, "var_smoothing": 1e-09}`
- PU-AUC(pooled): **0.6251** | AP: 0.0455 | lift@5%: 1.704 | prec@681: 0.0602
- fold별 AUC: [0.5940, 0.6225, 0.5991, 0.6690, 0.6331]  → 평균 0.6235 ± 0.027
- 소요: 0.3s
- 메모: (1차 기본 하이퍼파라미터)

#### OneClassSVM  (13강, one-class)
- 하이퍼파라미터: `rbf, nu=0.2, gamma=scale (P-only)`
- PU-AUC(pooled): **0.4838** | AP: 0.0313 | lift@5%: 1.087 | prec@681: 0.0426
- fold별 AUC: [0.4390, 0.5487, 0.5353, 0.4161, 0.4858]  → 평균 0.4850 ± 0.052
- 소요: 0.3s
- 메모: (1차 기본 하이퍼파라미터)

#### IsolationForest  (13강, one-class)
- 하이퍼파라미터: `n_estimators=300 (P-only)`
- PU-AUC(pooled): **0.4522** | AP: 0.0300 | lift@5%: 0.969 | prec@681: 0.0323
- fold별 AUC: [0.4350, 0.4957, 0.5059, 0.3848, 0.4522]  → 평균 0.4547 ± 0.044
- 소요: 2.1s
- 메모: (1차 기본 하이퍼파라미터)


---

## 실행 @ 2026-06-16 15:11:53

### 전체 비교표 (PU-AUC 내림차순)

| 모델 | 강의 | type | PU-AUC | fold AUC(평균±std) | AP | lift@5% | prec@681 | sec |
|---|---|---|---|---|---|---|---|---|
| RandomForest | 08강 | PU-bag | 0.6751 | 0.6747±0.018 | 0.0571 | 2.585 | 0.0793 | 32.0 |
| GBM | 09강 | PU-bag | 0.6731 | 0.6728±0.020 | 0.0563 | 2.703 | 0.0793 | 43.0 |
| Bagging | 08강 | PU-bag | 0.6723 | 0.6719±0.020 | 0.0586 | 2.321 | 0.0793 | 13.6 |
| XGBoost | 09강 | PU-bag | 0.6721 | 0.6718±0.021 | 0.0561 | 2.468 | 0.0749 | 10.5 |
| SVM(RBF) | 12강 | PU-bag | 0.6711 | 0.6704±0.021 | 0.0574 | 2.614 | 0.0910 | 36.5 |
| AdaBoost | 09강 | PU-bag | 0.6672 | 0.6654±0.021 | 0.0572 | 2.497 | 0.0896 | 42.3 |
| ExtraTrees | 08강 | PU-bag | 0.6658 | 0.6654±0.020 | 0.0525 | 1.880 | 0.0587 | 24.6 |
| DecisionTree | 06강 | PU-bag | 0.6651 | 0.6640±0.026 | 0.0577 | 2.703 | 0.0866 | 0.6 |
| LightGBM | 09강 | PU-bag | 0.6608 | 0.6603±0.019 | 0.0551 | 2.438 | 0.0705 | 14.8 |
| LogReg | 04강 | PU-bag | 0.6598 | 0.6580±0.030 | 0.0567 | 2.379 | 0.0793 | 0.6 |
| LDA | 05강 | PU-bag | 0.6593 | 0.6573±0.031 | 0.0566 | 2.321 | 0.0778 | 0.4 |
| kNN | 05강 | PU-bag | 0.6521 | 0.6516±0.031 | 0.0522 | 2.056 | 0.0617 | 10.6 |
| MLP | 07강 | PU-bag | 0.6454 | 0.6454±0.013 | 0.0525 | 2.174 | 0.0749 | 94.7 |
| NaiveBayes | 05강 | PU-bag | 0.6276 | 0.6260±0.028 | 0.0456 | 1.880 | 0.0573 | 0.3 |
| OneClassSVM | 13강 | one-class | 0.4856 | 0.4865±0.039 | 0.0304 | 0.823 | 0.0308 | 0.3 |
| IsolationForest | 13강 | one-class | 0.4620 | 0.4651±0.044 | 0.0291 | 0.969 | 0.0367 | 2.1 |

### 모델별 상세

#### RandomForest  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 2, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null`
- PU-AUC(pooled): **0.6751** | AP: 0.0571 | lift@5%: 2.585 | prec@681: 0.0793
- fold별 AUC: [0.6402, 0.6917, 0.6749, 0.6893, 0.6774]  → 평균 0.6747 ± 0.018
- 소요: 32.0s
- 메모: (1차 기본 하이퍼파라미터)

#### GBM  (09강, PU-bag)
- 하이퍼파라미터: `{"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 150, "n_`
- PU-AUC(pooled): **0.6731** | AP: 0.0563 | lift@5%: 2.703 | prec@681: 0.0793
- fold별 AUC: [0.6378, 0.6895, 0.6706, 0.6928, 0.6731]  → 평균 0.6728 ± 0.020
- 소요: 43.0s
- 메모: (1차 기본 하이퍼파라미터)

#### Bagging  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": true, "bootstrap_features": false, "estimator": null, "max_features": 1.0, "max_samples": null, "n_estimators": 50, "n_jobs": -1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}`
- PU-AUC(pooled): **0.6723** | AP: 0.0586 | lift@5%: 2.321 | prec@681: 0.0793
- fold별 AUC: [0.6353, 0.6945, 0.6697, 0.6814, 0.6785]  → 평균 0.6719 ± 0.020
- 소요: 13.6s
- 메모: (1차 기본 하이퍼파라미터)

#### XGBoost  (09강, PU-bag)
- 하이퍼파라미터: `{"objective": "binary:logistic", "base_score": null, "booster": null, "callbacks": null, "colsample_bylevel": null, "colsample_bynode": null, "colsample_bytree": null, "device": null, "early_stopping_rounds": null, "enable_categorical": false, "eval_metric": "logloss", "feature_types": null, "featur`
- PU-AUC(pooled): **0.6721** | AP: 0.0561 | lift@5%: 2.468 | prec@681: 0.0749
- fold별 AUC: [0.6355, 0.6899, 0.6684, 0.6930, 0.6721]  → 평균 0.6718 ± 0.021
- 소요: 10.5s
- 메모: (1차 기본 하이퍼파라미터)

#### SVM(RBF)  (12강, PU-bag)
- 하이퍼파라미터: `{"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": null, "shrinking": true, "tol": 0.001, "verbose": false}`
- PU-AUC(pooled): **0.6711** | AP: 0.0574 | lift@5%: 2.614 | prec@681: 0.0910
- fold별 AUC: [0.6379, 0.6786, 0.6609, 0.7030, 0.6716]  → 평균 0.6704 ± 0.021
- 소요: 36.5s
- 메모: (1차 기본 하이퍼파라미터)

#### AdaBoost  (09강, PU-bag)
- 하이퍼파라미터: `{"estimator": null, "learning_rate": 1.0, "n_estimators": 200, "random_state": 42}`
- PU-AUC(pooled): **0.6672** | AP: 0.0572 | lift@5%: 2.497 | prec@681: 0.0896
- fold별 AUC: [0.6278, 0.6674, 0.6713, 0.6908, 0.6699]  → 평균 0.6654 ± 0.021
- 소요: 42.3s
- 메모: (1차 기본 하이퍼파라미터)

#### ExtraTrees  (08강, PU-bag)
- 하이퍼파라미터: `{"bootstrap": false, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": nul`
- PU-AUC(pooled): **0.6658** | AP: 0.0525 | lift@5%: 1.880 | prec@681: 0.0587
- fold별 AUC: [0.6282, 0.6841, 0.6637, 0.6788, 0.6722]  → 평균 0.6654 ± 0.020
- 소요: 24.6s
- 메모: (1차 기본 하이퍼파라미터)

#### DecisionTree  (06강, PU-bag)
- 하이퍼파라미터: `{"ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": 6, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 20, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "random_state": 42, "splitter": "best"}`
- PU-AUC(pooled): **0.6651** | AP: 0.0577 | lift@5%: 2.703 | prec@681: 0.0866
- fold별 AUC: [0.6169, 0.6794, 0.6584, 0.6934, 0.6720]  → 평균 0.6640 ± 0.026
- 소요: 0.6s
- 메모: (1차 기본 하이퍼파라미터)

#### LightGBM  (09강, PU-bag)
- 하이퍼파라미터: `{"boosting_type": "gbdt", "class_weight": null, "colsample_bytree": 1.0, "importance_type": "split", "learning_rate": 0.1, "max_depth": -1, "min_child_samples": 20, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 200, "n_jobs": -1, "num_leaves": 31, "objective": null, "random_state`
- PU-AUC(pooled): **0.6608** | AP: 0.0551 | lift@5%: 2.438 | prec@681: 0.0705
- fold별 AUC: [0.6303, 0.6853, 0.6517, 0.6727, 0.6616]  → 평균 0.6603 ± 0.019
- 소요: 14.8s
- 메모: (1차 기본 하이퍼파라미터)

#### LogReg  (04강, PU-bag)
- 하이퍼파라미터: `{"C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true, "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 1000, "n_jobs": null, "penalty": "deprecated", "random_state": null, "solver": "lbfgs", "tol": 0.0001, "verbose": 0, "warm_start": false}`
- PU-AUC(pooled): **0.6598** | AP: 0.0567 | lift@5%: 2.379 | prec@681: 0.0793
- fold별 AUC: [0.6071, 0.6589, 0.6563, 0.7029, 0.6647]  → 평균 0.6580 ± 0.030
- 소요: 0.6s
- 메모: (1차 기본 하이퍼파라미터)

#### LDA  (05강, PU-bag)
- 하이퍼파라미터: `{"covariance_estimator": null, "n_components": null, "priors": null, "shrinkage": null, "solver": "svd", "store_covariance": false, "tol": 0.0001}`
- PU-AUC(pooled): **0.6593** | AP: 0.0566 | lift@5%: 2.321 | prec@681: 0.0778
- fold별 AUC: [0.6056, 0.6584, 0.6555, 0.7025, 0.6647]  → 평균 0.6573 ± 0.031
- 소요: 0.4s
- 메모: (1차 기본 하이퍼파라미터)

#### kNN  (05강, PU-bag)
- 하이퍼파라미터: `{"algorithm": "auto", "leaf_size": 30, "metric": "minkowski", "metric_params": null, "n_jobs": null, "n_neighbors": 15, "p": 2, "weights": "uniform"}`
- PU-AUC(pooled): **0.6521** | AP: 0.0522 | lift@5%: 2.056 | prec@681: 0.0617
- fold별 AUC: [0.6035, 0.6759, 0.6424, 0.6938, 0.6423]  → 평균 0.6516 ± 0.031
- 소요: 10.6s
- 메모: (1차 기본 하이퍼파라미터)

#### MLP  (07강, PU-bag)
- 하이퍼파라미터: `{"activation": "relu", "alpha": 0.0001, "batch_size": "auto", "beta_1": 0.9, "beta_2": 0.999, "early_stopping": false, "epsilon": 1e-08, "hidden_layer_sizes": [64, 32], "learning_rate": "constant", "learning_rate_init": 0.001, "max_fun": 15000, "max_iter": 300, "momentum": 0.9, "n_iter_no_change": 1`
- PU-AUC(pooled): **0.6454** | AP: 0.0525 | lift@5%: 2.174 | prec@681: 0.0749
- fold별 AUC: [0.6332, 0.6534, 0.6308, 0.6430, 0.6669]  → 평균 0.6454 ± 0.013
- 소요: 94.7s
- 메모: (1차 기본 하이퍼파라미터)

#### NaiveBayes  (05강, PU-bag)
- 하이퍼파라미터: `{"priors": null, "var_smoothing": 1e-09}`
- PU-AUC(pooled): **0.6276** | AP: 0.0456 | lift@5%: 1.880 | prec@681: 0.0573
- fold별 AUC: [0.5926, 0.6255, 0.6039, 0.6740, 0.6338]  → 평균 0.6260 ± 0.028
- 소요: 0.3s
- 메모: (1차 기본 하이퍼파라미터)

#### OneClassSVM  (13강, one-class)
- 하이퍼파라미터: `rbf, nu=0.2, gamma=scale (P-only)`
- PU-AUC(pooled): **0.4856** | AP: 0.0304 | lift@5%: 0.823 | prec@681: 0.0308
- fold별 AUC: [0.4666, 0.5363, 0.5216, 0.4266, 0.4815]  → 평균 0.4865 ± 0.039
- 소요: 0.3s
- 메모: (1차 기본 하이퍼파라미터)

#### IsolationForest  (13강, one-class)
- 하이퍼파라미터: `n_estimators=300 (P-only)`
- PU-AUC(pooled): **0.4620** | AP: 0.0291 | lift@5%: 0.969 | prec@681: 0.0367
- fold별 AUC: [0.4532, 0.5116, 0.5129, 0.3933, 0.4543]  → 평균 0.4651 ± 0.044
- 소요: 2.1s
- 메모: (1차 기본 하이퍼파라미터)

