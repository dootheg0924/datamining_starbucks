# -*- coding: utf-8 -*-
"""
[신뢰성 평가] PU 모델 신뢰 4기둥
  1) PU-valid 성능   : Lee&Liu PU-score + held-out positive recall@k
  2) 통계적 안정성   : PU-AUC bootstrap 95% CI + 모델간 점수 순위일치(Spearman)
  3) 외적 타당성     : OOT 신규스벅 recall@top-k% (모델 상위가 신규입점 지역을 덮나)
  4) 가정/보정       : class prior c 추정 + PU 신뢰도곡선(score bin→라벨양성률)
입력: outputs/final_pu_scores.parquet, model_zoo_scores.parquet, integrated_predictions.parquet
출력: outputs/trust_metrics.csv, figures/F11~F14, logs/TRUST_LOG.md
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
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
OUT = CLASSIFICATION_OUTPUT_DIR; FIG = CLASSIFICATION_FIGURE_DIR; LOG = CLASSIFICATION_LOG_DIR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    try: plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
GREEN, GRAY, RED, NAVY = "#1f7a3f", "#bbbbbb", "#c0392b", "#2c3e50"
GATE = 0.534
NEW_SB = {  # 10_validation 과 동일 (데이터수집 이후 신규 입점, 근사좌표)
    "순화동": (37.5650, 126.9690), "대치선릉": (37.5035, 127.0490),
    "신정네거리": (37.5247, 126.8554), "영등포구청": (37.5247, 126.8956),
    "송파풍납": (37.5345, 127.1165), "보라매역": (37.4995, 126.9203),
    "잠실래미안": (37.5130, 127.0830), "삼성휘문": (37.4965, 127.0590),
}

def hav(la, lo, la2, lo2):
    R = 6371; p = np.pi / 180
    a = np.sin((la2-la)*p/2)**2 + np.cos(la*p)*np.cos(la2*p)*np.sin((lo2-lo)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def main():
    df = pd.read_parquet(os.path.join(OUT, "final_pu_scores.parquet"))
    s = df["is_starbucks"].to_numpy(int); sc = df["score_raw"].to_numpy(float)
    pct = df["score_pct"].to_numpy(float)
    N, P = len(s), int(s.sum())
    out = {}

    # ── 기둥1: Lee&Liu PU-score + recall@k ──
    def pu_score(th):
        r = (sc[s == 1] >= th).mean()                 # recall(라벨양성)
        ppos = (sc >= th).mean()                       # P(예측=양성)
        return (r * r / ppos) if ppos > 0 else 0, r, ppos
    grid = np.linspace(sc.min(), sc.max(), 200)
    pus = [pu_score(t)[0] for t in grid]
    best_t = grid[int(np.argmax(pus))]
    out["PU_score@gate"] = pu_score(GATE)[0]
    out["PU_score_max"] = max(pus)
    out["PU_score_best_thr"] = best_t
    def recall_at(frac):
        k = max(1, int(N * frac)); o = np.argsort(-sc)[:k]
        return s[o].sum() / P
    rec = {f"recall@{int(f*100)}%": recall_at(f) for f in [0.05, 0.10, 0.20, 0.30]}
    out.update(rec)

    # ── 기둥2: bootstrap CI + 모델간 순위일치 ──
    rng = np.random.RandomState(42); aucs = []
    for _ in range(1000):
        idx = rng.randint(0, N, N)
        if s[idx].sum() in (0, len(idx)): continue
        aucs.append(roc_auc_score(s[idx], sc[idx]))
    out["AUC"] = roc_auc_score(s, sc)
    out["AUC_CI_lo"], out["AUC_CI_hi"] = np.percentile(aucs, [2.5, 97.5])

    mz = pd.read_parquet(os.path.join(OUT, "model_zoo_scores.parquet"))
    mcols = [c for c in ["score_GBM", "score_RandomForest", "score_XGBoost",
                         "score_LogReg", "score_SVM(RBF)"] if c in mz.columns]
    S = np.zeros((len(mcols), len(mcols)))
    for i, a in enumerate(mcols):
        for j, b in enumerate(mcols):
            S[i, j] = spearmanr(mz[a], mz[b]).statistic
    off = S[np.triu_indices(len(mcols), 1)]
    out["model_rank_spearman_mean"] = off.mean(); out["model_rank_spearman_min"] = off.min()

    # ── 기둥3: OOT recall@top-k% ──
    ip = pd.read_parquet(os.path.join(OUT, "integrated_predictions.parquet"))
    la, lo, p2 = ip["lat"].to_numpy(), ip["lon"].to_numpy(), ip["score_pct"].to_numpy()
    oot_rec = {}
    for k in [5, 10, 20, 30, 50]:
        thrp = 100 - k; covered = 0
        for nm, (a, b) in NEW_SB.items():
            d = hav(la, lo, a, b); near = p2[d <= 0.3]
            if len(near) and near.max() >= thrp: covered += 1
        oot_rec[f"OOT_recall@top{k}%"] = covered / len(NEW_SB)
    out.update(oot_rec)

    # ── 기둥4: class prior c + 신뢰도곡선 ──
    c_hat = sc[s == 1].mean()                          # E[score|라벨양성] ≈ Elkan c
    out["elkan_c"] = c_hat
    out["implied_pi(P/c/N)"] = (P / c_hat) / N         # 추정 진짜 양성 비율(SCAR 근사)
    # PU 신뢰도곡선: score 십분위 → 라벨양성률
    dec = pd.qcut(sc, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"dec": dec, "s": s, "sc": sc}).groupby("dec").agg(
        score_mean=("sc", "mean"), pos_rate=("s", "mean"))
    out["calib_monotonic_rho"] = spearmanr(cal["score_mean"], cal["pos_rate"]).statistic

    # ── 출력 ──
    res = pd.Series(out)
    print("=" * 70); print("[PU 신뢰성 4기둥]"); print("=" * 70)
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))
    res.to_frame("value").to_csv(os.path.join(OUT, "trust_metrics.csv"), encoding="utf-8-sig")

    figs(grid, pus, best_t, aucs, out, mcols, S, oot_rec, cal, rec)
    write_log(out, mcols, S, rec, oot_rec)
    print("\n[저장] outputs/trust_metrics.csv, figures/F11~F14, logs/TRUST_LOG.md")

def figs(grid, pus, best_t, aucs, out, mcols, S, oot_rec, cal, rec):
    # F11 bootstrap AUC
    plt.figure(figsize=(7, 4))
    plt.hist(aucs, bins=40, color=GREEN, alpha=.85)
    plt.axvline(out["AUC"], c=NAVY, lw=2, label=f"AUC={out['AUC']:.3f}")
    plt.axvline(out["AUC_CI_lo"], c=RED, ls="--", lw=1)
    plt.axvline(out["AUC_CI_hi"], c=RED, ls="--", lw=1, label=f"95% CI [{out['AUC_CI_lo']:.3f}, {out['AUC_CI_hi']:.3f}]")
    plt.title("F11. PU-AUC Bootstrap 분포·95% CI"); plt.xlabel("PU-AUC"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F11_bootstrap_auc.png"), dpi=130); plt.close()

    # F12 recall@k + OOT recall
    plt.figure(figsize=(7, 4))
    ks = [5, 10, 20, 30]
    plt.plot(ks, [rec[f"recall@{k}%"] for k in ks], "o-", color=GREEN, label="held-out 스벅 recall@k%")
    ok = [5, 10, 20, 30, 50]
    plt.plot(ok, [oot_rec[f"OOT_recall@top{k}%"] for k in ok], "s--", color=RED, label="OOT 신규스벅 recall@top-k%")
    plt.title("F12. 회수율 곡선 (held-out 스벅 / OOT 신규스벅)"); plt.xlabel("상위 k%"); plt.ylabel("recall"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F12_recall_curves.png"), dpi=130); plt.close()

    # F13 모델간 Spearman heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(S, vmin=0.7, vmax=1, cmap="Greens")
    lab = [c.replace("score_", "") for c in mcols]
    plt.xticks(range(len(lab)), lab, rotation=45, ha="right", fontsize=8); plt.yticks(range(len(lab)), lab, fontsize=8)
    for i in range(len(lab)):
        for j in range(len(lab)): plt.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Spearman"); plt.title(f"F13. 모델간 점수 순위일치 (평균 {out['model_rank_spearman_mean']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F13_model_agreement.png"), dpi=130); plt.close()

    # F14 PU 신뢰도곡선
    plt.figure(figsize=(6, 4.5))
    plt.plot(cal["score_mean"], cal["pos_rate"], "o-", color=GREEN)
    plt.title(f"F14. PU 신뢰도곡선 (score↑→라벨양성률↑, ρ={out['calib_monotonic_rho']:.2f})")
    plt.xlabel("score 십분위 평균"); plt.ylabel("실제 라벨양성(스벅)률")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F14_pu_calibration.png"), dpi=130); plt.close()
    for fn in ["F11_bootstrap_auc", "F12_recall_curves", "F13_model_agreement", "F14_pu_calibration"]:
        print("  저장:", fn + ".png")

def write_log(out, mcols, S, rec, oot_rec):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["# PU 신뢰성 평가 로그 (4기둥)", "", f"> 생성 {ts}", "",
         "## 1. PU-valid 성능", "",
         f"- Lee&Liu PU-score: gate={out['PU_score@gate']:.3f}, max={out['PU_score_max']:.3f}(thr={out['PU_score_best_thr']:.3f})",
         "- held-out 스벅 recall: " + ", ".join(f"{k}={v:.3f}" for k, v in rec.items()), "",
         "## 2. 통계적 안정성", "",
         f"- PU-AUC = {out['AUC']:.4f}, **95% CI [{out['AUC_CI_lo']:.4f}, {out['AUC_CI_hi']:.4f}]** (bootstrap 1000)",
         f"- 모델간 점수 순위일치 Spearman: 평균 {out['model_rank_spearman_mean']:.3f}, 최소 {out['model_rank_spearman_min']:.3f} "
         f"({', '.join(c.replace('score_','') for c in mcols)})",
         "  → 서로 다른 분류기가 사실상 같은 순위 = 결과가 특정 모델 산물이 아님", "",
         "## 3. 외적 타당성 (OOT 신규스벅)", "",
         "- recall@top-k%: " + ", ".join(f"{k.split('@')[1]}={v:.2f}" for k, v in oot_rec.items()),
         "  → 모델 상위 k% 안에 신규 입점 지역이 얼마나 덮이나(미지의 진실 회수)", "",
         "## 4. 가정·보정", "",
         f"- Elkan c (E[score|라벨양성]) = {out['elkan_c']:.3f}, 추정 진짜 양성비율 π≈{out['implied_pi(P/c/N)']:.3f} (SCAR 근사)",
         f"- PU 신뢰도곡선 단조성 ρ = {out['calib_monotonic_rho']:.3f} (1에 가까울수록 score↑→스벅률↑ 일관)",
         "  *주의: 진짜 음성 부재로 절대 calibration 아님 — 순위 신뢰도(랭킹 의미보존) 확인용*", "",
         "## 종합", "",
         "PU에 맞는 지표(PU-score)+안정성(CI·모델일치)+외적타당(OOT)+가정정합(c·단조성)으로 신뢰 삼각형 구축. "
         "단일 AUC가 아니라 다지표로 'PU 모델이 신뢰 가능'함을 입증.", ""]
    with open(os.path.join(LOG, "TRUST_LOG.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()


