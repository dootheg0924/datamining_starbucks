# -*- coding: utf-8 -*-
"""
[1.5단계] 공간 블록 CV vs 랜덤 CV 격차 증명 (피드백 #3 정면 대응)

입지 데이터는 공간 자기상관이 강해, 일반 random split 은 같은 상권의
'이웃 카페'가 train/test 양쪽에 들어가 성능을 과대평가한다.
동일 5전략을 두 CV 체계로 돌려 그 격차(inflation)를 수치로 보인다.

  - Random   : StratifiedKFold(5)            (관행적 분할)
  - Spatial  : GroupKFold(5) on spatial_block (이웃 누수 차단)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# 모듈명이 숫자로 시작해 일반 import 불가 → 파일 경로로 로드
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pu1", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_pu_baselines.py"))
pu1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(pu1)

CLF_FEATURES = pu1.CLF_FEATURES
STRATS = pu1.STRATS
RNG = pu1.RNG
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


def lift_at(s_true, score, frac=0.05):
    n = max(1, int(len(score) * frac))
    order = np.argsort(-score)[:n]
    return s_true[order].mean() / s_true.mean()


def run_cv(X, s, splits):
    """OOF 점수 + fold별 AUC(스케일 왜곡 배제용) 둘 다 반환."""
    oof = {k: np.full(len(s), np.nan) for k in STRATS}
    perfold = {k: [] for k in STRATS}
    for fold, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        rs = np.random.RandomState(RNG + fold)
        for name, fn in STRATS.items():
            sco = fn(Xtr, s[tr], Xte, rs)
            oof[name][te] = sco
            if 0 < s[te].sum() < len(te):
                perfold[name].append(roc_auc_score(s[te], sco))  # fold 내부 AUC
    return oof, perfold


def summarize(oof, s, scheme):
    rows = []
    for name in STRATS:
        sc = oof[name]
        rows.append({"scheme": scheme, "strategy": name,
                     "PU_AUC": roc_auc_score(s, sc),
                     "AP": average_precision_score(s, sc),
                     "lift@5%": lift_at(s, sc)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float)
    s = df["s"].to_numpy(int)
    groups = df["spatial_block"].to_numpy()

    print("[Spatial] GroupKFold(5) on spatial_block ...")
    sp, sp_pf = run_cv(X, s, list(GroupKFold(5).split(X, s, groups)))
    print("[Random]  StratifiedKFold(5) ...")
    rd, rd_pf = run_cv(X, s, list(StratifiedKFold(5, shuffle=True, random_state=RNG).split(X, s)))

    res = pd.concat([summarize(rd, s, "random"), summarize(sp, s, "spatial")], ignore_index=True)
    piv = res.pivot(index="strategy", columns="scheme", values="PU_AUC")
    piv["infl(pooled)"] = piv["random"] - piv["spatial"]
    piv = piv.sort_values("spatial", ascending=False)

    print("\n" + "=" * 72)
    print("[A] Pooled OOF PU-AUC  (fold 스케일 차이에 민감)")
    print("=" * 72)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    # fold별 AUC 평균±표준편차 (스케일 왜곡 배제, 더 엄밀)
    print("\n" + "=" * 72)
    print("[B] Fold별 AUC 평균±표준편차  (엄밀 비교) — random 이 높으면 과대평가")
    print("=" * 72)
    pf_rows = []
    for name in STRATS:
        r, sp_ = np.array(rd_pf[name]), np.array(sp_pf[name])
        pf_rows.append({"strategy": name,
                        "random": f"{r.mean():.4f}±{r.std():.3f}",
                        "spatial": f"{sp_.mean():.4f}±{sp_.std():.3f}",
                        "infl(mean)": f"{r.mean()-sp_.mean():+.4f}"})
    pf = pd.DataFrame(pf_rows).set_index("strategy").loc[piv.index]
    print(pf.to_string())
    print("\n[전체 지표]")
    print(res.pivot_table(index="strategy", columns="scheme",
                          values=["PU_AUC", "AP", "lift@5%"]).to_string(float_format=lambda x: f"{x:.4f}"))

    res.to_csv(os.path.join(OUTDIR, "cv_comparison.csv"), index=False, encoding="utf-8-sig")
    print("\n[저장] outputs/cv_comparison.csv")


if __name__ == "__main__":
    main()


