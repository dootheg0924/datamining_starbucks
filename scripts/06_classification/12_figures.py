# -*- coding: utf-8 -*-
"""
[발표용 그림] 우리가 수행한 Classification 분석 결과 시각화
출력: reports/generated/classification/figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
OUT = CLASSIFICATION_OUTPUT_DIR
FIG = CLASSIFICATION_FIGURE_DIR
for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    try: plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
GREEN, GRAY, RED, NAVY = "#1f7a3f", "#bbbbbb", "#c0392b", "#2c3e50"

def savefig(fn):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fn), dpi=140, bbox_inches="tight"); plt.close()
    print("  저장:", fn)

# F1 — 라벨링 5전략
def f1():
    d = pd.read_csv(os.path.join(OUT, "strategy_comparison.csv")).sort_values("PU_AUC")
    nm = {"pu_bagging": "PU bagging", "undersample": "Undersample", "weighted": "Class-weight",
          "pu_elkan": "Elkan-Noto", "smote": "SMOTE"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    c = [GREEN if s == "pu_bagging" else GRAY for s in d.strategy]
    ax.barh([nm.get(s, s) for s in d.strategy], d.PU_AUC, color=c)
    for i, v in enumerate(d.PU_AUC): ax.text(v + .002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, ls="--", c=RED, lw=1); ax.set_xlim(0.5, 0.72)
    ax.set_xlabel("PU-AUC (spatial CV)"); ax.set_title("F1. 라벨링 5전략 비교 — PU가 강제음성보다 우월")
    savefig("F1_labeling_strategies.png")

# F2 — 공간 vs 랜덤 CV
def f2():
    d = pd.read_csv(os.path.join(OUT, "cv_comparison.csv"))
    p = d.pivot(index="strategy", columns="scheme", values="PU_AUC").sort_values("spatial")
    y = np.arange(len(p)); h = .38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(y - h/2, p["random"], h, label="Random CV", color=GRAY)
    ax.barh(y + h/2, p["spatial"], h, label="Spatial Block CV", color=GREEN)
    ax.set_yticks(y); ax.set_yticklabels(p.index); ax.axvline(0.5, ls="--", c=RED, lw=1)
    ax.set_xlim(0.5, 0.72); ax.set_xlabel("PU-AUC")
    ax.set_title("F2. 공간 vs 랜덤 CV — 과대평가 없음 (random ≤ spatial)"); ax.legend()
    savefig("F2_cv_comparison.png")

# F3 — 모델 16+2종
def f3():
    d = pd.read_csv(os.path.join(OUT, "model_zoo_comparison.csv")).sort_values("PU_AUC")
    c = [RED if t == "one-class" else (GREEN if m in ("GBM", "RandomForest") else NAVY)
         for m, t in zip(d.model, d.type)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d.model, d.PU_AUC, color=c)
    for i, v in enumerate(d.PU_AUC): ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, ls="--", c=RED, lw=1); ax.set_xlim(0.4, 0.72)
    ax.set_xlabel("PU-AUC (spatial CV)")
    ax.set_title("F3. 분류기 16+2종 비교 — 천장 ~0.677 (모델 무관), one-class 실패")
    savefig("F3_model_zoo.png")

# F4 — PU 고도화 5종
def f4():
    a = pd.read_csv(os.path.join(OUT, "pu_twostep_comparison.csv"))
    b = pd.read_csv(os.path.join(OUT, "pu_twostep_pos_comparison.csv"))
    vals = {}
    for _, r in pd.concat([a, b]).iterrows():
        vals[r["method"]] = max(vals.get(r["method"], 0), r["PU_AUC"])
    nm = {"pu_bagging": "PU bagging\n(전체 U)", "twostep_RP": "Two-step RP\n(의사양성)",
          "twostep_EM": "Two-step EM\n(soft가중)", "twostep_spy": "Two-step RN\n(Spy)",
          "pu_elkan": "Elkan-Noto", "twostep_dist": "Two-step RN\n(거리,대조)"}
    s = pd.Series(vals).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    c = [GREEN if k == "pu_bagging" else GRAY for k in s.index]
    ax.barh([nm.get(k, k) for k in s.index], s.values, color=c)
    for i, v in enumerate(s.values): ax.text(v + .002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.6, 0.69); ax.set_xlabel("PU-AUC")
    ax.set_title("F4. PU 고도화 5종 — 어떤 방법도 plain bagging 못 이김")
    savefig("F4_pu_methods.png")

# F5 — 페르소나별 스벅 침투율
def f5():
    d = pd.read_parquet(os.path.join(OUT, "cafe_personas.parquet"))
    t = d.groupby("persona_name").agg(전체=("name", "size"), 스벅=("is_starbucks", "sum"))
    t["침투율"] = t["스벅"] / t["전체"] * 100
    t = t.sort_values("침투율", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(t.index, t["침투율"], color=GREEN)
    for i, v in enumerate(t["침투율"]): ax.text(v + .05, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("스타벅스 침투율 (구역 내 스벅 / 전체 카페, %)")
    ax.set_title("F5. 페르소나별 스벅 침투율 — 오피스고소득·도심초밀집 선호")
    savefig("F5_persona_penetration.png")

# F6 — 적합도 분포 + 서울대 위치
def f6():
    d = pd.read_parquet(os.path.join(OUT, "integrated_predictions.parquet"))
    camp = 0.480  # 캠퍼스 전형 적합도
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(d.loc[d.is_starbucks == 0, "score_raw"], bins=60, color=GRAY, alpha=.8, label="비스타벅스")
    ax.hist(d.loc[d.is_starbucks == 1, "score_raw"], bins=60, color=GREEN, alpha=.7, label="스타벅스")
    ax.axvline(0.549, ls="--", c=NAVY, lw=1.3, label="게이트 0.549")
    ax.axvline(camp, ls="-", c=RED, lw=2, label="서울대 캠퍼스 0.48")
    ax.set_xlabel("스벅 적합도 (score_raw)"); ax.set_ylabel("카페 수")
    ax.set_title("F6. 적합도 분포와 서울대 캠퍼스 위치 (게이트 미달)"); ax.legend()
    savefig("F6_score_distribution_SNU.png")

# F7 — 대학 비교 (적합도 vs 게이트통과율)
def f7():
    d = pd.read_csv(os.path.join(OUT, "university_comparison.csv"))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, r in d.iterrows():
        snu = r["대학"].startswith("서울대")
        ax.scatter(r["적합도평균"], r["게이트통과율"] * 100, s=120 if snu else 70,
                   c=RED if snu else NAVY, zorder=3 if snu else 2)
        ax.annotate(r["대학"], (r["적합도평균"], r["게이트통과율"] * 100),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("캠퍼스 1km 평균 적합도"); ax.set_ylabel("게이트 통과율 (%)")
    ax.set_title("F7. 서울 주요 대학 비교 — 서울대만 게이트 통과율 0%")
    savefig("F7_university_comparison.png")

# F8 — 변수 해석 (순열중요도 + 방향)
def f8():
    r = pd.read_csv(os.path.join(OUT, "feature_interpretation.csv")).sort_values("perm_imp")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].barh(r["한글"], r["perm_imp"], color=GREEN)
    ax[0].set_title("F8a. 순열중요도 (GBM) — 지하철거리 압도적"); ax[0].set_xlabel("PU-AUC 하락폭")
    c = [RED if v > 0 else NAVY for v in r["logreg_coef"]]
    ax[1].barh(r["한글"], r["logreg_coef"], color=c); ax[1].axvline(0, c="k", lw=.8)
    ax[1].set_title("F8b. LogReg 계수 (빨강=스벅↑/남색=스벅↓)"); ax[1].set_xlabel("표준화 계수")
    savefig("F8_interpretation.png")

# F9 — OOT 신규 스벅 검증
def f9():
    d = pd.read_csv(os.path.join(OUT, "oot_validation.csv")).sort_values("적합도평균")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c = [GREEN if v >= 0.549 else GRAY for v in d["적합도평균"]]
    ax.barh(d["신규매장"], d["적합도평균"], color=c)
    for i, v in enumerate(d["적합도평균"]): ax.text(v + .005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(0.549, ls="--", c=NAVY, lw=1.2, label="게이트 0.549")
    ax.set_xlabel("인근 카페 평균 적합도"); ax.set_title("F9. OOT 검증 — 신규 입점 스벅 인근 적합도 (62% 적중)")
    ax.legend()
    savefig("F9_oot_validation.png")

if __name__ == "__main__":
    print("[발표용 그림 생성]")
    for fn in [f1, f2, f3, f4, f5, f6, f7, f8, f9]:
        try: fn()
        except Exception as e: print("  ERR", fn.__name__, e)
    print("완료 →", FIG)

