# -*- coding: utf-8 -*-
"""
[2단계-A] 강의 기반 분류 모델 16종 비교 (PU bagging 프레임 위에서 동일 조건)

라벨링은 1단계에서 채택한 PU 로 고정하고, '분류기'를 바꿔가며 비교한다.
  - 일반 분류기 14종: 각 모델을 PU bagging 의 base learner 로 사용
      (P 전체 + U 1:1 부트스트랩 × T번 앙상블, 매 bag 점수를 순위정규화 후 평균)
  - one-class 2종: positive 만으로 학습(이상탐지), 정상도 점수 사용
검증: spatial GroupKFold(5). 평가: PU-AUC / AP / lift@5% / precision@681.
기록: classification/logs/MODEL_LOG.md 에 모델별 섹션 누적(append).

* 1차 패스는 합리적 기본 하이퍼파라미터. 튜닝은 후속 단계에서 동일 로그에 누적.
"""
import os, time, json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import rankdata

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, IsolationForest)
from sklearn.svm import SVC, OneClassSVM
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RNG = 42
T_BAGS = 20  # PU bagging 반복 수
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
DATA = CLASSIFICATION_DATASET_PATH
OUTDIR = CLASSIFICATION_OUTPUT_DIR
LOGDIR = CLASSIFICATION_LOG_DIR
LOGMD = os.path.join(LOGDIR, "MODEL_LOG.md")

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def need_scale(est):  # 파이프라인에 StandardScaler 끼울지
    return est in {"LogReg", "kNN", "LDA", "MLP", "SVM(RBF)", "OneClassSVM"}

# (lecture, factory, hyperparams-note)
def make(name):
    if name == "LogReg":      return LogisticRegression(max_iter=1000, C=1.0)
    if name == "kNN":         return KNeighborsClassifier(n_neighbors=15)
    if name == "LDA":         return LinearDiscriminantAnalysis()
    if name == "NaiveBayes":  return GaussianNB()
    if name == "DecisionTree":return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RNG)
    if name == "MLP":         return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RNG)
    if name == "RandomForest":return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RNG)
    if name == "Bagging":     return BaggingClassifier(n_estimators=50, n_jobs=-1, random_state=RNG)
    if name == "ExtraTrees":  return ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=RNG)
    if name == "AdaBoost":    return AdaBoostClassifier(n_estimators=200, random_state=RNG)
    if name == "GBM":         return GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RNG)
    if name == "LightGBM":    return LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=-1, random_state=RNG, verbose=-1)
    if name == "XGBoost":     return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                   n_jobs=-1, random_state=RNG, eval_metric="logloss", verbosity=0)
    if name == "SVM(RBF)":    return SVC(kernel="rbf", C=1.0, gamma="scale")  # decision_function 사용
    raise ValueError(name)

CLASSIFIERS = ["LogReg", "kNN", "LDA", "NaiveBayes", "DecisionTree", "MLP",
               "RandomForest", "Bagging", "ExtraTrees", "AdaBoost", "GBM",
               "LightGBM", "XGBoost", "SVM(RBF)"]
ONECLASS = ["OneClassSVM", "IsolationForest"]
LECTURE = {"LogReg": "04", "kNN": "05", "LDA": "05", "NaiveBayes": "05",
           "DecisionTree": "06", "MLP": "07", "RandomForest": "08", "Bagging": "08",
           "ExtraTrees": "08", "AdaBoost": "09", "GBM": "09", "LightGBM": "09",
           "XGBoost": "09", "SVM(RBF)": "12", "OneClassSVM": "13", "IsolationForest": "13"}

# --- 점수 추출 (predict_proba > decision_function) ---------------------------
def get_score(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict(X).astype(float)

def to_rank(v):  # 순위정규화 [0,1] (bag 간 스케일 통일)
    return (rankdata(v) - 1) / (len(v) - 1) if len(v) > 1 else v

# --- 모델별 fold 처리 -------------------------------------------------------
def fit_predict_classifier(name, Xtr, str_tr, Xte, rs):
    pos = np.where(str_tr == 1)[0]; neg = np.where(str_tr == 0)[0]
    sc_te = np.zeros(Xte.shape[0])
    for t in range(T_BAGS):
        u = rs.choice(neg, size=len(pos), replace=True)
        idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
        est = make(name)
        if need_scale(name):
            est = Pipeline([("sc", StandardScaler()), ("m", est)])
        est.fit(Xtr[idx], yy)
        sc_te += to_rank(get_score(est, Xte))
    return sc_te / T_BAGS

def fit_predict_oneclass(name, Xtr, str_tr, Xte, rs):
    pos = np.where(str_tr == 1)[0]
    if name == "OneClassSVM":
        est = Pipeline([("sc", StandardScaler()), ("m", OneClassSVM(kernel="rbf", nu=0.2, gamma="scale"))])
        est.fit(Xtr[pos])
        return est.decision_function(Xte)        # 클수록 정상(=스벅형)
    else:  # IsolationForest
        est = IsolationForest(n_estimators=300, random_state=RNG, n_jobs=-1)
        est.fit(Xtr[pos])
        return est.score_samples(Xte)            # 클수록 정상

# --- 평가 -------------------------------------------------------------------
def lift_at(y, s, frac=0.05):
    n = max(1, int(len(s) * frac)); o = np.argsort(-s)[:n]
    return y[o].mean() / y.mean()

def prec_at(y, s, k):
    o = np.argsort(-s)[:k]; return y[o].sum() / k

# --- 메인 -------------------------------------------------------------------
def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    groups = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, groups))

    all_models = CLASSIFIERS + ONECLASS
    oof = {m: np.full(len(s), np.nan) for m in all_models}
    perfold = {m: [] for m in all_models}
    timing = {}

    for m in all_models:
        t0 = time.time()
        for fold, (tr, te) in enumerate(splits):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])  # 공통 스케일(트리엔 무해)
            rs = np.random.RandomState(RNG + fold)
            if m in ONECLASS:
                pred = fit_predict_oneclass(m, Xtr, s[tr], Xte, rs)
            else:
                pred = fit_predict_classifier(m, Xtr, s[tr], Xte, rs)
            oof[m][te] = pred
            if 0 < s[te].sum() < len(te):
                perfold[m].append(roc_auc_score(s[te], pred))
        timing[m] = time.time() - t0
        pf = np.array(perfold[m])
        print(f"  {m:16s} PU-AUC(pooled)={roc_auc_score(s, oof[m]):.4f}  "
              f"fold={pf.mean():.4f}±{pf.std():.3f}  ({timing[m]:.1f}s)")

    # 집계표
    rows = []
    for m in all_models:
        pf = np.array(perfold[m])
        rows.append({"model": m, "lecture": LECTURE[m], "type": ("one-class" if m in ONECLASS else "PU-bag"),
                     "PU_AUC": roc_auc_score(s, oof[m]), "AP": average_precision_score(s, oof[m]),
                     "fold_AUC_mean": pf.mean(), "fold_AUC_std": pf.std(),
                     "lift@5%": lift_at(s, oof[m]), "prec@681": prec_at(s, oof[m], int(s.sum())),
                     "sec": timing[m]})
    res = pd.DataFrame(rows).sort_values("PU_AUC", ascending=False)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("[2단계-A] 모델 16종 비교 (PU bagging / spatial GroupKFold5)")
    print("=" * 90)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 저장
    res.to_csv(os.path.join(OUTDIR, "model_zoo_comparison.csv"), index=False, encoding="utf-8-sig")
    sdf = df[["name", "brand", "is_starbucks", "sigungu", "spatial_block"]].copy()
    for m in all_models: sdf[f"score_{m}"] = oof[m]
    sdf.to_parquet(os.path.join(OUTDIR, "model_zoo_scores.parquet"), index=False)

    write_log(res, perfold)
    print(f"\n[저장] outputs/model_zoo_comparison.csv, model_zoo_scores.parquet")
    print(f"[기록] logs/MODEL_LOG.md (모델별 섹션 누적)")

# --- 로그 md 작성 (append) --------------------------------------------------
def write_log(res, perfold):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_exists = os.path.exists(LOGMD)
    lines = []
    if not header_exists:
        lines += ["# Classification 모델 실험 로그",
                  "",
                  "> 자동 생성/누적. 각 실행은 타임스탬프 블록으로 append 됩니다.",
                  "> 라벨링=PU(1:1 bagging ×%d), 검증=spatial GroupKFold(5), 피처=raw 17개." % T_BAGS,
                  ""]
    lines += ["", "---", "", f"## 실행 @ {ts}", "",
              "### 전체 비교표 (PU-AUC 내림차순)", "",
              "| 모델 | 강의 | type | PU-AUC | fold AUC(평균±std) | AP | lift@5% | prec@681 | sec |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append("| %s | %s강 | %s | %.4f | %.4f±%.3f | %.4f | %.3f | %.4f | %.1f |" % (
            r["model"], r["lecture"], r["type"], r["PU_AUC"], r["fold_AUC_mean"],
            r["fold_AUC_std"], r["AP"], r["lift@5%"], r["prec@681"], r["sec"]))
    # 모델별 상세 섹션
    lines += ["", "### 모델별 상세", ""]
    for _, r in res.iterrows():
        m = r["model"]
        pf = ", ".join(f"{a:.4f}" for a in perfold[m])
        lines += [f"#### {m}  ({r['lecture']}강, {r['type']})",
                  f"- 하이퍼파라미터: `{describe_params(m)}`",
                  f"- PU-AUC(pooled): **{r['PU_AUC']:.4f}** | AP: {r['AP']:.4f} | lift@5%: {r['lift@5%']:.3f} | prec@681: {r['prec@681']:.4f}",
                  f"- fold별 AUC: [{pf}]  → 평균 {r['fold_AUC_mean']:.4f} ± {r['fold_AUC_std']:.3f}",
                  f"- 소요: {r['sec']:.1f}s",
                  "- 메모: (1차 기본 하이퍼파라미터)", ""]
    with open(LOGMD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def describe_params(m):
    if m in ("OneClassSVM", "IsolationForest"):
        return {"OneClassSVM": "rbf, nu=0.2, gamma=scale (P-only)",
                "IsolationForest": "n_estimators=300 (P-only)"}[m]
    est = make(m)
    return json.dumps(est.get_params(), default=str)[:300]

if __name__ == "__main__":
    main()


