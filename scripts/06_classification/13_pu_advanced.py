# -*- coding: utf-8 -*-
"""
[PU 고도화 2] ② 페르소나별 PU 모델 5개  ③ 스벅 반경 제외 ablation

② within-persona PU: "그 페르소나 안에서 스벅이 고르는 자리는 무엇이 다른가"
   - 양성=페르소나 k의 스벅, 음성=같은 페르소나의 비스벅(쉬운 페르소나간 신호 제거)
   - base=LogReg(소표본 안정+해석). 산출물=페르소나별 표준화계수(동인). AUC는 fold CI로 정직하게.
③ ablation: 메인 GBM PU-bagging에서 음성 풀을 스벅 반경(0/100/200m) 밖으로만 제한.
   - dist_nearest_starbucks는 '샘플선택'에만 사용(피처 아님 → 누수 아님). 평가는 전체 동일.
검증: spatial GroupKFold(5). 기록: logs/PU_LOG.md (append)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RNG = 42
T = 30
HERE = os.path.dirname(os.path.abspath(__file__))
from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
DATA = CLASSIFICATION_DATASET_PATH
OUT = CLASSIFICATION_OUTPUT_DIR
LOGDIR = CLASSIFICATION_LOG_DIR
PERSONA = {0: "오피스고소득", 1: "상업활성", 2: "주거생활", 3: "도심초밀집", 4: "비역세권"}

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]
KOR = {"log_dist_subway": "지하철거리", "subway_count_cat": "지하철수", "subway_ridership": "역승하차",
       "bus_stops_300m": "버스정류장", "peak_avg": "피크평균",
       "restaurants_500m": "음식점", "log_retail_500m": "소매", "convenience_500m": "편의점",
       "indie_cafe_500m": "독립카페", "low_price_cafe_500m": "저가카페", "franchise_cafe_500m": "프랜차이즈카페",
       "avg_income": "평균소득", "offices": "직장인구", "living_pop": "생활인구", "land_price": "공시지가"}

def load():
    df = pd.read_parquet(DATA)
    pe = pd.read_parquet(os.path.join(OUT, "cafe_personas.parquet")).reset_index(drop=True)
    df = df.reset_index(drop=True)
    assert (df["name"].values == pe["name"].values).all()
    df["persona"] = pe["persona"].values
    return df

# ── ② 페르소나별 within-persona PU (LogReg base) ──────────────────────────
def per_persona(df):
    print("=" * 80)
    print("② 페르소나별 PU 모델 (within-persona: 같은 페르소나 내 스벅 vs 비스벅)")
    print("=" * 80)
    rows, drivers = [], {}
    for k in range(5):
        sub = df[df.persona == k].reset_index(drop=True)
        X = sub[CLF_FEATURES].to_numpy(float); s = sub["s"].to_numpy(int)
        g = sub["spatial_block"].to_numpy()
        npos = int(s.sum())
        ng = pd.Series(g[s == 1]).nunique()
        nsplit = max(2, min(5, ng))
        oof = np.full(len(s), np.nan); coefs = []
        for tr, te in GroupKFold(nsplit).split(X, s, g):
            if s[tr].sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            pos = np.where(s[tr] == 1)[0]; neg = np.where(s[tr] == 0)[0]
            rs = np.random.RandomState(RNG); pred = np.zeros(len(te)); cf = []
            for _ in range(T):
                u = rs.choice(neg, len(pos), replace=len(neg) < len(pos))
                idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
                m = LogisticRegression(max_iter=2000).fit(Xtr[idx], yy)
                pred += m.predict_proba(Xte)[:, 1]; cf.append(m.coef_[0])
            oof[te] = pred / T; coefs.append(np.mean(cf, axis=0))
        # fold별 auc
        aucs = []
        for tr, te in GroupKFold(nsplit).split(X, s, g):
            if 0 < s[te].sum() < len(te) and not np.isnan(oof[te]).any():
                aucs.append(roc_auc_score(s[te], oof[te]))
        mean_coef = np.mean(coefs, axis=0)
        drivers[k] = mean_coef
        rows.append({"persona": PERSONA[k], "스벅": npos, "비스벅": int((s == 0).sum()),
                     "AUC_mean": np.mean(aucs) if aucs else np.nan,
                     "AUC_std": np.std(aucs) if aucs else np.nan, "n_fold": len(aucs)})
        # top drivers
        order = np.argsort(-np.abs(mean_coef))[:5]
        tops = ", ".join(f"{KOR[CLF_FEATURES[j]]}({mean_coef[j]:+.2f})" for j in order)
        print(f"\n[{PERSONA[k]}] 스벅 {npos} / AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} (n_fold={len(aucs)})")
        print(f"   주요 동인: {tops}")
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # 동인 테이블 저장
    dd = pd.DataFrame({KOR[f]: [drivers[k][i] for k in range(5)] for i, f in enumerate(CLF_FEATURES)},
                      index=[PERSONA[k] for k in range(5)]).T
    dd.to_csv(os.path.join(OUT, "per_persona_drivers.csv"), encoding="utf-8-sig")
    res.to_csv(os.path.join(OUT, "per_persona_auc.csv"), index=False, encoding="utf-8-sig")
    return res, dd

# ── ③ 스벅 반경 제외 ablation (GBM base) ──────────────────────────────────
def gbm():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RNG)

def ablation(df):
    print("\n" + "=" * 80)
    print("③ 스벅 반경 제외 ablation (음성 풀을 스벅 반경 밖으로만 제한)")
    print("=" * 80)
    X = df[CLF_FEATURES].to_numpy(float); s = df["s"].to_numpy(int)
    dist = df["dist_nearest_starbucks"].to_numpy(float)
    g = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, g))
    rows = []
    for thr in [0.0, 0.1, 0.2]:
        oof = np.full(len(s), np.nan)
        excl_total = 0
        for tr, te in splits:
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            s_tr = s[tr]; d_tr = dist[tr]
            pos = np.where(s_tr == 1)[0]
            neg = np.where((s_tr == 0) & (d_tr > thr))[0]   # 스벅 반경 밖 음성만
            excl_total += int(((s_tr == 0) & (d_tr <= thr)).sum())
            rs = np.random.RandomState(RNG); pred = np.zeros(len(te))
            for _ in range(T):
                u = rs.choice(neg, len(pos), replace=len(neg) < len(pos))
                idx = np.concatenate([pos, u]); yy = np.r_[np.ones(len(pos)), np.zeros(len(u))]
                m = gbm().fit(Xtr[idx], yy); pred += m.predict_proba(Xte)[:, 1]
            oof[te] = pred / T
        auc = roc_auc_score(s, oof)
        n = max(1, int(len(oof) * 0.05)); o = np.argsort(-oof)[:n]
        lift = s[o].mean() / s.mean()
        rows.append({"제외반경": f"{int(thr*1000)}m" if thr else "없음(baseline)",
                     "제외U수(avg/fold)": excl_total // 5, "PU_AUC": auc, "lift@5%": lift})
        print(f"  반경 {int(thr*1000)}m 제외: PU-AUC={auc:.4f}, lift@5%={lift:.3f}, "
              f"제외 U≈{excl_total//5}/fold")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ablation_sb_radius.csv"), index=False, encoding="utf-8-sig")
    return res

def write_log(per_res, drivers, abl):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["", "---", "", f"## 실행 @ {ts} — ② 페르소나별 PU + ③ 반경제외 ablation", "",
         "### ② 페르소나별 within-persona PU (base=LogReg, spatial CV)", "",
         "| 페르소나 | 스벅 | 비스벅 | AUC(fold 평균±std) | n_fold |", "|---|---|---|---|---|"]
    for _, r in per_res.iterrows():
        L.append("| %s | %d | %d | %.3f±%.3f | %d |" % (
            r["persona"], r["스벅"], r["비스벅"], r["AUC_mean"], r["AUC_std"], r["n_fold"]))
    L += ["", "*소표본(C0=47 등)이라 AUC는 참고용(넓은 CI). 핵심은 아래 페르소나별 동인.*", "",
          "### 페르소나별 주요 동인 (표준화 LogReg 계수, +면 스벅↑)", "",
          "| 변수 | " + " | ".join(drivers.columns) + " |",
          "|" + "---|" * (len(drivers.columns) + 1)]
    for feat, row in drivers.iterrows():
        L.append("| %s | %s |" % (feat, " | ".join(f"{v:+.2f}" for v in row.values)))
    L += ["", "### ③ 스벅 반경 제외 ablation (음성 풀 제한, GBM)", "",
          "| 제외반경 | 제외U(≈/fold) | PU-AUC | lift@5% |", "|---|---|---|---|"]
    for _, r in abl.iterrows():
        L.append("| %s | %d | %.4f | %.3f |" % (r["제외반경"], r["제외U수(avg/fold)"], r["PU_AUC"], r["lift@5%"]))
    L += ["", "메모: 페르소나별 동인 차이 해석 / 반경제외 효과(개선 여부) 결론은 여기에.", ""]
    with open(os.path.join(LOGDIR, "PU_LOG.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    df = load()
    per_res, drivers = per_persona(df)
    abl = ablation(df)
    write_log(per_res, drivers, abl)
    print("\n[저장] outputs/per_persona_drivers.csv, per_persona_auc.csv, ablation_sb_radius.csv")
    print("[기록] logs/PU_LOG.md")

