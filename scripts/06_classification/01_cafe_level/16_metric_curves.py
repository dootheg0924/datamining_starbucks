# -*- coding: utf-8 -*-
"""
[지표 보강] 입지추천 운영 곡선 — lift / precision@k / cumulative gains / calibration
"왜 이 지표인가": 목적이 '상위 후보 추천'이므로 정확도가 아니라 상위 k 집중도(lift)·
정밀도(precision@k)·회수율(gains)·확률신뢰(calibration)가 핵심.
입력: outputs/final_pu_scores.parquet (OOF, 공정)  출력: figures/F15, outputs/operating_table.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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
OUT = CLASSIFICATION_OUTPUT_DIR; FIG = CLASSIFICATION_FIGURE_DIR
for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    try: plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
GREEN, GRAY, RED, NAVY = "#1f7a3f", "#bbbbbb", "#c0392b", "#2c3e50"

def main():
    df = pd.read_parquet(os.path.join(OUT, "final_pu_scores.parquet"))
    s = df["is_starbucks"].to_numpy(int)
    sc = df["score_raw"].to_numpy(float)
    cal_sc = df["score_cal"].to_numpy(float)
    N, P = len(s), int(s.sum())
    base = P / N                                  # 무작위 기대 양성률

    order = np.argsort(-sc); s_sorted = s[order]
    cum_pos = np.cumsum(s_sorted)
    ks = np.arange(1, N + 1)
    precision = cum_pos / ks                      # precision@k (PU → 하한)
    recall = cum_pos / P                          # cumulative gains
    lift = precision / base
    frac = ks / N * 100

    # 핵심 지점 표
    rows = []
    for f in [1, 5, 10, 20, 30, 50]:
        k = max(1, int(N * f / 100))
        rows.append({"top%": f, "k": k, "precision@k": cum_pos[k-1]/k,
                     "recall(gains)": cum_pos[k-1]/P, "lift": (cum_pos[k-1]/k)/base})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("=" * 60); print("[운영 곡선 핵심 지점]  (base rate=%.4f, AUC=%.3f)" % (base, roc_auc_score(s, sc)))
    print("=" * 60)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    tab.to_csv(os.path.join(OUT, "operating_table.csv"), index=False, encoding="utf-8-sig")

    # PU 신뢰도: score 십분위(1=하위~10=상위) → 실제 라벨양성률
    # (진짜 음성 부재 → 대각선 calibration 부당. 단조 증가=순위 의미보존만 확인)
    db = pd.DataFrame({"p": sc, "s": s})
    db["dec"] = pd.qcut(db["p"], 10, labels=False, duplicates="drop") + 1
    cal = db.groupby("dec").agg(obs=("s", "mean"))

    # ── 그림 2x2 ──
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # lift
    ax[0,0].plot(frac, lift, color=GREEN); ax[0,0].axhline(1, ls="--", c=GRAY)
    ax[0,0].set_title("F15a. Lift 곡선 (상위 k% 스벅 집중도)"); ax[0,0].set_xlabel("상위 k%"); ax[0,0].set_ylabel("lift")
    ax[0,0].set_xlim(0, 100)
    # precision@k
    ax[0,1].plot(frac, precision, color=NAVY); ax[0,1].axhline(base, ls="--", c=GRAY, label=f"base={base:.3f}")
    ax[0,1].set_title("F15b. Precision@k (PU 하한)"); ax[0,1].set_xlabel("상위 k%"); ax[0,1].set_ylabel("precision"); ax[0,1].legend()
    ax[0,1].set_xlim(0, 100)
    # cumulative gains
    ax[1,0].plot(frac, recall*100, color=GREEN, label="모델")
    ax[1,0].plot([0,100],[0,100], ls="--", c=GRAY, label="무작위")
    ax[1,0].set_title("F15c. Cumulative Gains (상위 k%가 스벅 회수율)"); ax[1,0].set_xlabel("상위 k%"); ax[1,0].set_ylabel("recall %"); ax[1,0].legend()
    # PU 신뢰도(순위 의미보존): 십분위↑ → 실제 스벅률↑
    ax[1,1].plot(cal.index, cal["obs"]*100, "o-", color=GREEN)
    ax[1,1].axhline(base*100, ls="--", c=GRAY, label=f"base={base*100:.1f}%")
    ax[1,1].set_title("F15d. 점수 신뢰도 (십분위↑→실제 스벅률↑)"); ax[1,1].set_xlabel("score 십분위(10=상위)"); ax[1,1].set_ylabel("실제 라벨 스벅률 %"); ax[1,1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F15_operating_curves.png"), dpi=130); plt.close()
    print("\n[저장] figures/F15_operating_curves.png, outputs/operating_table.csv")
    print("주의: U에 숨은 양성 존재 → precision@k·calibration은 보수적 하한(실제는 더 높음).")

if __name__ == "__main__":
    main()


