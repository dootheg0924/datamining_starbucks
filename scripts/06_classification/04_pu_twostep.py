# -*- coding: utf-8 -*-
"""
[PU 고도화] Two-step PU with Reliable Negatives (RN)

실험 설계(controlled): 음성 샘플링 '풀'만 바꿔 RN 효과를 격리한다.
  - pu_bagging    : 음성을 전체 U 에서 부트스트랩            (RN 미사용, 기존)
  - twostep_spy   : 음성을 Spy-RN 에서만 부트스트랩          (RN = 데이터기반)
  - twostep_dist  : 음성을 거리기반 RN 에서만 부트스트랩      (RN = 휴리스틱 대조군)
  - pu_elkan      : Elkan-Noto c-보정                        (참고 비교)
모든 분류기 base = RandomForest, 검증 = spatial GroupKFold(5).

RN 추출은 각 train fold 내부에서만 수행(test 누수 차단).
  · Spy-RN  : P의 15%를 스파이로 U에 심어 P vs U(+spy) 분류 → 스파이 점수의
              하위 SPY_PCT 분위를 임계값으로, 그보다 낮은 U 를 RN 으로.
  · dist-RN : dist_nearest_starbucks 가 가장 큰 U 를 Spy-RN 과 동일 개수만큼.
              (dist 변수는 RN 정의에만 사용, 학습 피처로는 미사용)

기록: classification/logs/PU_LOG.md (append, RN 진단 포함)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = 42
T_BAGS = 30
SPY_FRAC = 0.15      # P 중 스파이 비율
SPY_PCT = 10         # 스파이 점수의 하위 백분위 → RN 임계값
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
OUTDIR = CLASSIFICATION_OUTPUT_DIR
LOGDIR = CLASSIFICATION_LOG_DIR
PULOG = os.path.join(LOGDIR, "PU_LOG.md")

CLF_FEATURES = [
    "log_dist_subway", "subway_count_cat", "subway_ridership", "bus_stops_300m",
    "peak_avg", "restaurants_500m", "log_retail_500m", "convenience_500m",
    "indie_cafe_500m", "low_price_cafe_500m", "franchise_cafe_500m",
    "avg_income", "offices", "living_pop", "land_price",
]

def rf(n=100, balanced=False):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=2, n_jobs=-1,
                                  random_state=RNG,
                                  class_weight=("balanced" if balanced else None))

# --- RN 추출 ----------------------------------------------------------------
def spy_reliable_negatives(Xtr, s, rs):
    """Spy 기법. 반환: U_train 내 RN 마스크(불리언), 진단 dict."""
    pos = np.where(s == 1)[0]; U = np.where(s == 0)[0]
    n_spy = max(1, int(len(pos) * SPY_FRAC))
    spy = rs.choice(pos, size=n_spy, replace=False)
    pos_rest = np.setdiff1d(pos, spy)
    # P_rest(=1) vs U+spy(=0). spy 단계는 '부드러운 확률'을 주는 LogReg 사용
    # (RF는 약신호에서 순수음성 leaf 때문에 확률을 정확히 0으로 줘 임계값이 붕괴).
    # LogReg는 개별점 메모리제이션이 없어 in-sample 점수도 안전.
    idx = np.concatenate([pos_rest, U, spy])
    yy = np.r_[np.ones(len(pos_rest)), np.zeros(len(U) + len(spy))]
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr[idx], yy)
    U_score = clf.predict_proba(Xtr[U])[:, 1]
    spy_score = clf.predict_proba(Xtr[spy])[:, 1]
    thr = np.percentile(spy_score, SPY_PCT)
    rn_mask = U_score < thr            # 스파이 하위분위보다 더 음성스러운 U
    diag = dict(n_spy=n_spy, thr=float(thr),
                n_RN=int(rn_mask.sum()), U_total=len(U),
                rn_ratio=float(rn_mask.mean()))
    return U, rn_mask, diag

def dist_reliable_negatives(s, dist, n_target):
    """거리기반 RN: dist_nearest_starbucks 큰 U 상위 n_target."""
    U = np.where(s == 0)[0]
    order = np.argsort(-dist[U])       # 먼 순
    sel = np.zeros(len(U), dtype=bool); sel[order[:n_target]] = True
    return U, sel

# --- 풀에서 부트스트랩 PU 배깅 (음성 풀만 다름) -----------------------------
def bagging_from_pool(Xtr, pos_idx, neg_pool_idx, Xte, rs, T=T_BAGS):
    if len(neg_pool_idx) == 0:
        return np.full(Xte.shape[0], 0.5)
    acc = np.zeros(Xte.shape[0])
    for _ in range(T):
        u = rs.choice(neg_pool_idx, size=len(pos_idx),
                      replace=len(neg_pool_idx) < len(pos_idx))
        idx = np.concatenate([pos_idx, u])
        yy = np.r_[np.ones(len(pos_idx)), np.zeros(len(u))]
        m = rf(100).fit(Xtr[idx], yy)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / T

def elkan(Xtr, s, Xte):
    m = rf(150).fit(Xtr, s)
    pos = np.where(s == 1)[0]
    c = max(m.predict_proba(Xtr[pos])[:, 1].mean(), 1e-6)
    return np.clip(m.predict_proba(Xte)[:, 1] / c, 0, 1)

# --- 평가 -------------------------------------------------------------------
def lift_at(y, sc, frac=0.05):
    n = max(1, int(len(sc) * frac)); o = np.argsort(-sc)[:n]
    return y[o].mean() / y.mean()
def prec_at(y, sc, k):
    o = np.argsort(-sc)[:k]; return y[o].sum() / k

METHODS = ["pu_bagging", "twostep_spy", "twostep_dist", "pu_elkan"]

def main():
    df = pd.read_parquet(DATA)
    X = df[CLF_FEATURES].to_numpy(float)
    s = df["s"].to_numpy(int)
    dist = df["dist_nearest_starbucks"].to_numpy(float)  # RN 정의 전용
    groups = df["spatial_block"].to_numpy()
    splits = list(GroupKFold(5).split(X, s, groups))

    oof = {m: np.full(len(s), np.nan) for m in METHODS}
    perfold = {m: [] for m in METHODS}
    rn_diag = []           # fold별 RN 진단
    rn_profiles = {"spy": [], "dist": []}   # RN 피처 프로파일(표준화 평균)

    feat_mean, feat_std = X.mean(0), X.std(0) + 1e-9

    for fold, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        str_ = s[tr]; rs = np.random.RandomState(RNG + fold)
        pos_tr = np.where(str_ == 1)[0]
        U_tr = np.where(str_ == 0)[0]

        # --- RN 추출 (train 내부) ---
        _, spy_mask, diag = spy_reliable_negatives(Xtr, str_, rs)
        spy_rn = U_tr[spy_mask]
        _, dist_mask = dist_reliable_negatives(str_, dist[tr], n_target=int(spy_mask.sum()))
        dist_rn = U_tr[dist_mask]
        # 중복도(Jaccard)
        a, b = set(spy_rn.tolist()), set(dist_rn.tolist())
        jac = len(a & b) / len(a | b) if (a | b) else 0.0
        diag.update(fold=fold, n_dist_RN=len(dist_rn), jaccard_spy_dist=jac,
                    train_P=len(pos_tr), train_U=len(U_tr))
        rn_diag.append(diag)
        # RN 피처 프로파일(원척도 표준화 평균)
        rn_profiles["spy"].append((X[tr][spy_rn] - feat_mean) / feat_std)
        rn_profiles["dist"].append((X[tr][dist_rn] - feat_mean) / feat_std)

        # --- 4개 방법 예측 ---
        oof["pu_bagging"][te] = bagging_from_pool(Xtr, pos_tr, U_tr, Xte, rs)
        oof["twostep_spy"][te] = bagging_from_pool(Xtr, pos_tr, spy_rn, Xte, rs)
        oof["twostep_dist"][te] = bagging_from_pool(Xtr, pos_tr, dist_rn, Xte, rs)
        oof["pu_elkan"][te] = elkan(Xtr, str_, Xte)

        for m in METHODS:
            if 0 < s[te].sum() < len(te):
                perfold[m].append(roc_auc_score(s[te], oof[m][te]))
        print(f"[fold {fold}] spy-RN={diag['n_RN']}/{len(U_tr)} "
              f"(thr={diag['thr']:.3f}) dist-RN={len(dist_rn)} jaccard={jac:.3f}")

    # 집계
    rows = []
    for m in METHODS:
        pf = np.array(perfold[m])
        rows.append({"method": m, "PU_AUC": roc_auc_score(s, oof[m]),
                     "AP": average_precision_score(s, oof[m]),
                     "fold_AUC_mean": pf.mean(), "fold_AUC_std": pf.std(),
                     "lift@5%": lift_at(s, oof[m]), "prec@681": prec_at(s, oof[m], int(s.sum()))})
    res = pd.DataFrame(rows).sort_values("PU_AUC", ascending=False)
    diag_df = pd.DataFrame(rn_diag)

    # RN 피처 프로파일 평균(전 fold 통합)
    prof = pd.DataFrame({
        "feature": CLF_FEATURES,
        "spy_RN": np.vstack(rn_profiles["spy"]).mean(0),
        "dist_RN": np.vstack(rn_profiles["dist"]).mean(0),
    })

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("[PU 고도화] Two-step RN 비교 (spatial GroupKFold5)")
    print("=" * 78)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[RN 진단 - fold별]")
    print(diag_df[["fold", "n_RN", "rn_ratio", "thr", "n_dist_RN", "jaccard_spy_dist"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n[RN 피처 프로파일] (표준화 평균; 음수=전체평균보다 낮음)")
    print(prof.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # 저장
    res.to_csv(os.path.join(OUTDIR, "pu_twostep_comparison.csv"), index=False, encoding="utf-8-sig")
    prof.to_csv(os.path.join(OUTDIR, "rn_feature_profile.csv"), index=False, encoding="utf-8-sig")
    write_log(res, diag_df, prof)
    make_profile_fig(prof)
    print("\n[저장] outputs/pu_twostep_comparison.csv, rn_feature_profile.csv, rn_profile.png")
    print("[기록] logs/PU_LOG.md")

def make_profile_fig(prof):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
        try: plt.rcParams["font.family"] = f; break
        except Exception: pass
    plt.rcParams["axes.unicode_minus"] = False
    p = prof.sort_values("spy_RN")
    y = np.arange(len(p)); h = 0.4
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - h/2, p["spy_RN"], h, label="Spy-RN", color="#1f7a3f")
    ax.barh(y + h/2, p["dist_RN"], h, label="거리기반 RN", color="#bbbbbb")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(p["feature"], fontsize=8)
    ax.set_xlabel("표준화 평균 (0 = 전체 카페 평균)")
    ax.set_title("Reliable Negative 피처 프로파일: Spy-RN vs 거리기반 RN")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "rn_profile.png"), dpi=130)

def write_log(res, diag_df, prof):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = not os.path.exists(PULOG)
    L = []
    if new:
        L += ["# PU Learning 고도화 로그", "",
              "> Two-step(RN) vs bagging vs Elkan. 검증=spatial GroupKFold(5).",
              f"> 설정: base=RandomForest, T_BAGS={T_BAGS}, SPY_FRAC={SPY_FRAC}, SPY_PCT={SPY_PCT}.", ""]
    L += ["", "---", "", f"## 실행 @ {ts}", "",
          f"설정: T_BAGS={T_BAGS}, SPY_FRAC={SPY_FRAC}, SPY_PCT={SPY_PCT}", "",
          "### 방법 비교 (PU-AUC 내림차순)", "",
          "| 방법 | PU-AUC | fold AUC(평균±std) | AP | lift@5% | prec@681 |",
          "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append("| %s | %.4f | %.4f±%.3f | %.4f | %.3f | %.4f |" % (
            r["method"], r["PU_AUC"], r["fold_AUC_mean"], r["fold_AUC_std"],
            r["AP"], r["lift@5%"], r["prec@681"]))
    L += ["", "### RN 추출 진단 (fold별)", "",
          "| fold | spy-RN 수 | RN 비율 | 임계값 | dist-RN 수 | Jaccard(spy∩dist) |",
          "|---|---|---|---|---|---|"]
    for _, r in diag_df.iterrows():
        L.append("| %d | %d | %.3f | %.3f | %d | %.3f |" % (
            r["fold"], r["n_RN"], r["rn_ratio"], r["thr"], r["n_dist_RN"], r["jaccard_spy_dist"]))
    L += ["", "### RN 피처 프로파일 (표준화 평균, 음수=전체 카페 평균 미만)", "",
          "| 피처 | Spy-RN | 거리기반 RN |", "|---|---|---|"]
    for _, r in prof.iterrows():
        L.append("| %s | %+.2f | %+.2f |" % (r["feature"], r["spy_RN"], r["dist_RN"]))
    L += ["", "메모: (Spy-RN과 거리기반 RN의 차이/편향 해석은 여기에 추가)", ""]
    with open(PULOG, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

