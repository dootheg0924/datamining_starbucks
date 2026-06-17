# -*- coding: utf-8 -*-
"""
[2단계] Clustering ↔ Classification 연결: Opt1 / Opt2 / Opt3 동시 구현

입력: outputs/final_pu_scores.parquet (PU 게이트 점수, OOF)
      outputs/cafe_personas.parquet  (페르소나 + centroid 거리)
출력: outputs/integrated_predictions.parquet (카페별 세 옵션 결과)
      logs/INTEGRATION_LOG.md

  Opt1 (Decoupled)   : 적합도 = PU score. 페르소나는 서술적 맥락.
  Opt2 (Multi-class) : 6-way soft = [gate×persona_softmax(5), 1-gate]
                       → "C1 80% / C4 15% / 비스벅 5%" 식 풍부한 출력.
  Opt3 (Two-stage)   : stage1 PU 게이트(Youden J 임계값) →
                       stage2 최근접 centroid 페르소나(학습 불필요).
검증: Opt2/3의 stage2(페르소나 분류) 신뢰성 = 681 스벅에 대해
      RF 다중분류가 kmeans 배정을 얼마나 복원하는지(spatial CV).
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
)
OUTDIR = CLASSIFICATION_OUTPUT_DIR
LOGDIR = CLASSIFICATION_LOG_DIR
LOG = os.path.join(LOGDIR, "INTEGRATION_LOG.md")
PERSONA_NAMES = {0: "오피스고소득", 1: "상업활성", 2: "주거생활", 3: "도심초밀집", 4: "비역세권"}

def softmax_neg(D, temp=1.0):
    z = -D / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def main():
    pu = pd.read_parquet(os.path.join(OUTDIR, "final_pu_scores.parquet")).reset_index(drop=True)
    pe = pd.read_parquet(os.path.join(OUTDIR, "cafe_personas.parquet")).reset_index(drop=True)
    # 두 파일 모두 clf_dataset 와 동일 행 순서 → 위치 기준 결합(중복좌표 머지 폭증 방지).
    assert len(pu) == len(pe), "행 수 불일치"
    assert (pu["name"].values == pe["name"].values).all() and \
           np.allclose(pu["lat"].values, pe["lat"].values), "행 정렬 불일치"
    df = pu.copy()
    for col in ["persona", "persona_name"] + [f"dist_C{k}" for k in range(5)]:
        df[col] = pe[col].values
    assert df["persona"].notna().all(), "페르소나 결합 실패 행 존재"
    s = df["is_starbucks"].to_numpy(int)

    # ── Opt3 게이트 임계값: OOF score_raw 기준 Youden J ──
    fpr, tpr, thr = roc_curve(s, df["score_raw"])
    j = tpr - fpr
    gate_thr = float(thr[np.argmax(j)])
    df["opt3_gate"] = (df["score_raw"] >= gate_thr).astype(int)

    # ── 페르소나 softmax (centroid 거리 기반) ──
    D = df[[f"dist_C{k}" for k in range(5)]].to_numpy(float)
    P = softmax_neg(D, temp=1.0)            # (N,5) 페르소나 소속확률
    nearest = df["persona"].to_numpy(int)

    # ── Opt1: 적합도 = PU 백분위 ──
    df["opt1_fit_pct"] = df["score_pct"]

    # ── Opt2: 6-way soft = [gate*persona, 1-gate] ──
    gate = df["score_raw"].to_numpy(float)   # 0~1 게이트(스벅형 확률 근사)
    for k in range(5):
        df[f"opt2_pC{k}"] = gate * P[:, k]
    df["opt2_p_nonSB"] = 1 - gate

    # ── Opt3: 게이트 + 최근접 페르소나 ──
    df["opt3_persona"] = nearest
    df["opt3_persona_name"] = df["persona_name"]
    df["opt3_persona_conf"] = P[np.arange(len(P)), nearest]

    # ── stage2 신뢰성 검증: RF 다중분류가 kmeans 배정 복원하는가(681 스벅, spatial CV) ──
    sb = df[df.is_starbucks == 1].reset_index(drop=True)
    from joblib import load
    pj = load(os.path.join(HERE, "models", "persona_kmeans.joblib"))
    feats_src = pd.read_parquet(os.path.join(HERE, "data", "clf_dataset.parquet"))
    # 16 클러스터 피처 재생성(스벅만)
    import importlib.util
    spec = importlib.util.spec_from_file_location("p7", os.path.join(HERE, "07_personas.py"))
    p7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p7)
    Xc_all = p7.build_cluster_features(feats_src)
    sb_mask = feats_src["is_starbucks"].to_numpy() == 1
    Xc_sb = pj["scaler"].transform(Xc_all[sb_mask].values)
    raw_y_sb = pj["kmeans"].predict(Xc_sb)
    raw_to_canonical = pj.get("raw_to_canonical")
    if raw_to_canonical is None:
        y_sb = raw_y_sb
    else:
        y_sb = np.array([raw_to_canonical[int(label)] for label in raw_y_sb], dtype=int)
    grp_sb = feats_src.loc[sb_mask, "spatial_block"].to_numpy()
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    yhat = cross_val_predict(clf, Xc_sb, y_sb, cv=GroupKFold(5), groups=grp_sb)
    acc = accuracy_score(y_sb, yhat); f1m = f1_score(y_sb, yhat, average="macro")

    # ── 요약 출력 ──
    pd.set_option("display.width", 200)
    print("=" * 74)
    print("[2단계] Opt1/2/3 통합 결과")
    print("=" * 74)
    print(f"Opt3 게이트 임계값(Youden J on OOF) = {gate_thr:.4f}  "
          f"→ 게이트 통과: {int(df['opt3_gate'].sum())} / {len(df)} "
          f"(스벅 중 통과율 {df.loc[s==1,'opt3_gate'].mean():.1%})")
    print(f"\n[Opt1] 페르소나별 평균 적합도 백분위:")
    print(df.groupby("persona_name")["opt1_fit_pct"].mean().round(1).sort_values(ascending=False).to_string())
    print(f"\n[stage2 검증] RF 다중분류로 kmeans 페르소나 복원: acc={acc:.3f}, macroF1={f1m:.3f}")
    print("  (높을수록 최근접-centroid 배정이 안정적 = Opt3 stage2 학습 불필요 정당화)")

    # 예시: 비스벅 고적합 + 관악구
    print("\n[Opt2 예시] 관악구 카페 적합도 상위 5:")
    gw = df[(df.sigungu == "관악구")].sort_values("score_raw", ascending=False).head(5)
    for _, r in gw.iterrows():
        top = max(range(5), key=lambda k: r[f"opt2_pC{k}"])
        print("  %-18s 적합도%.0f%%  → %s %.0f%% / 비스벅 %.0f%%" % (
            r["name"][:18], r["score_raw"]*100, PERSONA_NAMES[top],
            r[f"opt2_pC{top}"]*100, r["opt2_p_nonSB"]*100))

    df.to_parquet(os.path.join(OUTDIR, "integrated_predictions.parquet"), index=False)
    write_log(gate_thr, df, s, acc, f1m, confusion_matrix(y_sb, yhat))
    print("\n[저장] outputs/integrated_predictions.parquet")
    print("[기록] logs/INTEGRATION_LOG.md")

def write_log(gate_thr, df, s, acc, f1m, cm):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fit_by = df.groupby("persona_name")["opt1_fit_pct"].mean().round(1).sort_values(ascending=False)
    L = [] if os.path.exists(LOG) else ["# 2단계 통합(Opt1/2/3) 로그", ""]
    L += ["", "---", "", f"## 실행 @ {ts}", "",
          "### Opt1 (Decoupled) — 페르소나별 평균 적합도 백분위", "",
          "| 페르소나 | 평균 적합도 pct |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in fit_by.items()]
    L += ["", "### Opt3 (Two-stage 게이팅)", "",
          f"- stage1 게이트 임계값(Youden J, OOF score_raw): **{gate_thr:.4f}**",
          f"- 게이트 통과: {int(df['opt3_gate'].sum())}/{len(df)} "
          f"(스벅 통과율 {df.loc[s==1,'opt3_gate'].mean():.1%}, 비스벅 통과율 {df.loc[s==0,'opt3_gate'].mean():.1%})",
          "- stage2: 최근접 centroid 페르소나(학습 불필요)", "",
          "### Opt2 (Multi-class soft) — 출력 형식", "",
          "- 각 카페: `[opt2_pC0..pC4, opt2_p_nonSB]` = gate×persona_softmax + (1-gate)",
          "", "### stage2 신뢰성 (RF가 kmeans 페르소나 복원, 681 스벅 spatial CV)", "",
          f"- accuracy={acc:.3f}, macro-F1={f1m:.3f}",
          f"- confusion matrix:\n```\n{cm}\n```",
          "  → 높으면 페르소나가 피처로 잘 분리됨 = Opt3 stage2(centroid) 정당화.", "",
          "메모: (세 옵션 중 발표 메인 선택, 서울대 적용 해석은 여기에)", ""]
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

