# -*- coding: utf-8 -*-
"""
[해석] 무엇이 '스벅형 입지'를 만드는가 — 투트랙 + 견고성 검증

  성능트랙 : GBM PU-bagging 의 순열중요도(permutation importance)
             = 피처를 섞었을 때 PU-AUC 하락폭 (모델이 실제로 의존하는 정도)
  해석트랙 : LogReg(balanced) PU-bagging 표준화 계수 = 방향(+/-)과 크기
  견고성   : 두 랭킹의 Spearman 상관 + fold 간 부호 안정성
검증: spatial GroupKFold(5). 피처 17개(raw, dist_nearest_starbucks 제외).
출력: outputs/feature_interpretation.csv, feature_interpretation.png, logs/INTERPRETATION_LOG.md
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RNG = 42
T_BAGS = 20
N_PERM = 3
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

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]
KOR = {  # 발표용 한글 라벨
    "log_dist_subway": "지하철거리(log)", "subway_count_cat": "지하철수(범주)",
    "subway_ridership": "역승하차", "bus_stops_300m": "버스정류장",
    "peak_avg": "피크평균", "restaurants_500m": "음식점",
    "log_retail_500m": "소매(log)", "convenience_500m": "편의점",
    "indie_cafe_500m": "독립카페", "low_price_cafe_500m": "저가카페",
    "franchise_cafe_500m": "프랜차이즈카페", "avg_income": "평균소득",
    "offices": "직장인구", "living_pop": "생활인구", "land_price": "공시지가",
}

def gbm():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RNG)

def fit_bag(est_fn, Xtr, s_tr, rs, T=T_BAGS):
    pos = np.where(s_tr == 1)[0]; neg = np.where(s_tr == 0)[0]
    models = []
    for _ in range(T):
        u = rs.choice(neg, size=len(pos), replace=len(neg) < len(pos))
        idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
        models.append(est_fn().fit(Xtr[idx], yy))
    return models

def bag_pred(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    groups = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, groups))
    F = len(CLF_FEATURES)

    perm = np.zeros((5, F)); coefs = np.zeros((5, F)); gbm_imp = np.zeros((5, F))
    for k, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        s_tr, s_te = s[tr], s[te]
        rs = np.random.RandomState(RNG + k)

        # 성능트랙: GBM bagging
        gmodels = fit_bag(gbm, Xtr, s_tr, rs)
        base = roc_auc_score(s_te, bag_pred(gmodels, Xte))
        gbm_imp[k] = np.mean([m.feature_importances_ for m in gmodels], axis=0)
        for j in range(F):
            drops = []
            for _ in range(N_PERM):
                Xp = Xte.copy(); rs.shuffle(Xp[:, j])
                drops.append(base - roc_auc_score(s_te, bag_pred(gmodels, Xp)))
            perm[k, j] = np.mean(drops)

        # 해석트랙: LogReg balanced bagging 표준화계수
        lmodels = fit_bag(lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                          Xtr, s_tr, np.random.RandomState(RNG + 100 + k))
        coefs[k] = np.mean([m.coef_[0] for m in lmodels], axis=0)
        print(f"[fold {k}] base AUC={base:.4f}")

    res = pd.DataFrame({
        "feature": CLF_FEATURES,
        "한글": [KOR[f] for f in CLF_FEATURES],
        "perm_imp": perm.mean(0), "perm_imp_std": perm.std(0),
        "gbm_imp": gbm_imp.mean(0),
        "logreg_coef": coefs.mean(0), "logreg_coef_std": coefs.std(0),
    })
    res["dir"] = np.where(res["logreg_coef"] > 0, "양(+, 스벅↑)", "음(-, 스벅↓)")
    # fold 부호 안정성
    res["coef_sign_stable"] = [(np.sign(coefs[:, j]) == np.sign(coefs[:, j].mean())).mean() for j in range(F)]
    res = res.sort_values("perm_imp", ascending=False).reset_index(drop=True)

    rho, p = spearmanr(res["perm_imp"], res["logreg_coef"].abs())
    pd.set_option("display.width", 200)
    print("\n" + "=" * 84)
    print("[해석] 스벅형 입지를 만드는 변수 (순열중요도 내림차순)")
    print("=" * 84)
    print(res[["한글", "perm_imp", "gbm_imp", "logreg_coef", "dir", "coef_sign_stable"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[견고성] GBM순열중요도 vs |LogReg계수| Spearman ρ={rho:.3f} (p={p:.3f})")
    print("  → 두 독립적 방법이 같은 변수를 중요하다고 보면 해석이 견고함")

    res.to_csv(os.path.join(OUTDIR, "feature_interpretation.csv"), index=False, encoding="utf-8-sig")
    make_fig(res)
    write_log(res, rho, p)
    print("\n[저장] outputs/feature_interpretation.csv, feature_interpretation.png, logs/INTERPRETATION_LOG.md")

def make_fig(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
        try: plt.rcParams["font.family"] = f; break
        except Exception: pass
    plt.rcParams["axes.unicode_minus"] = False
    r = res.sort_values("perm_imp")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].barh(r["한글"], r["perm_imp"], color="#1f7a3f")
    ax[0].set_title("순열중요도 (GBM, PU-AUC 하락폭)"); ax[0].set_xlabel("중요도")
    colors = ["#c0392b" if c > 0 else "#2c3e50" for c in r["logreg_coef"]]
    ax[1].barh(r["한글"], r["logreg_coef"], color=colors)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_title("LogReg 표준화계수 (빨강=스벅↑ / 남색=스벅↓)"); ax[1].set_xlabel("계수")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "feature_interpretation.png"), dpi=130)

def write_log(res, rho, p):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top = res.head(6)
    L = ["# 변수 해석 로그 — 무엇이 스벅형 입지를 만드는가", "", f"> 생성 {ts}",
         "> 성능트랙=GBM 순열중요도, 해석트랙=LogReg(balanced) 표준화계수, spatial CV(5)", "",
         f"## 견고성: GBM중요도 vs |LogReg계수| Spearman ρ={rho:.3f} (p={p:.3f})", "",
         "## 변수별 (순열중요도 순)", "",
         "| 변수 | 순열중요도 | GBM중요도 | LogReg계수 | 방향 | 부호안정성 |",
         "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append("| %s | %.4f | %.4f | %+.4f | %s | %.0f%% |" % (
            r["한글"], r["perm_imp"], r["gbm_imp"], r["logreg_coef"], r["dir"], r["coef_sign_stable"]*100))
    L += ["", "## 핵심 해석", "",
          "- 스벅형 입지 상위 동인: " + ", ".join(top["한글"].tolist()),
          "- 양(+) 방향(스벅↑): " + ", ".join(res[res.logreg_coef > 0]["한글"].tolist()),
          "- 음(-) 방향(스벅↓): " + ", ".join(res[res.logreg_coef < 0]["한글"].tolist()),
          "", "메모: (방향 해석·통설 검증 연결은 여기에)", ""]
    with open(os.path.join(LOGDIR, "INTERPRETATION_LOG.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()


