# -*- coding: utf-8 -*-
"""
[검증/확장] (1) 게이트 통과 카페  (2) out-of-time 신규스벅  (3) 타대학 비교

(1) Opt3 게이트 통과한 '비스벅' 카페 = 모델이 본 스벅 유력 입지.
(2) 데이터수집 이후 신규 입점 스벅 8곳의 좌표(역/지역 기준 근사)를 잡아,
    그 지점 인근(≤0.3km) 기존 카페의 적합도로 '사전 예측 적중' 검증.
    * 신규스벅 자체 featurize 불가 → 인근 카페 proxy. 좌표는 근사값(한계).
(3) 서울 주요 대학 캠퍼스(≤1.0km) 카페 적합도 비교 — 서울대 위치 맥락화.
출력: outputs/gate_candidates.csv, outputs/oot_validation.csv,
      outputs/university_comparison.csv, logs/VALIDATION_REPORT.md
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

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
GATE_THR = 0.534

def hav(la, lo, la2, lo2):
    R = 6371; p = np.pi / 180
    a = np.sin((la2 - la) * p / 2) ** 2 + np.cos(la * p) * np.cos(la2 * p) * np.sin((lo2 - lo) * p / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# (2) 신규 입점 스벅 (데이터수집 이후) — 좌표는 역/지역 기준 근사값
NEW_SB = {
    "순화동경창궁앞": (37.5650, 126.9690),
    "대치선릉점": (37.5035, 127.0490),
    "신정네거리역1번출구": (37.5247, 126.8554),
    "영등포구청역5호선": (37.5247, 126.8956),
    "송파풍납점": (37.5345, 127.1165),
    "보라매역점": (37.4995, 126.9203),
    "잠실래미안아이파크": (37.5130, 127.0830),
    "삼성휘문점": (37.4965, 127.0590),
}
# (3) 서울 주요 대학 캠퍼스 중심(근사)
UNIV = {
    "서울대(관악)": (37.4591, 126.9520),
    "연세대(신촌)": (37.5665, 126.9388),
    "고려대(안암)": (37.5894, 127.0327),
    "한양대(왕십리)": (37.5559, 127.0438),
    "홍익대(상수)": (37.5511, 126.9250),
    "중앙대(흑석)": (37.5051, 126.9571),
    "경희대(회기)": (37.5970, 127.0517),
    "이화여대(대현)": (37.5620, 126.9469),
    "건국대(화양)": (37.5403, 127.0793),
    "서강대(신촌)": (37.5511, 126.9410),
}

def main():
    df = pd.read_parquet(os.path.join(OUTDIR, "integrated_predictions.parquet"))
    rep = []

    # ───────── (1) 게이트 통과 카페 ─────────
    nonsb = df[df.is_starbucks == 0]
    passed = nonsb[nonsb.score_raw >= GATE_THR].sort_values("score_raw", ascending=False)
    print("=" * 76)
    print(f"(1) Opt3 게이트 통과 비스벅 카페: {len(passed)} / {len(nonsb)} ({len(passed)/len(nonsb):.1%})")
    print("=" * 76)
    print("[페르소나별 통과 수]")
    print(passed.persona_name.value_counts().to_string())
    print("\n[시군구별 통과 수 top10]")
    print(passed.sigungu.value_counts().head(10).to_string())
    print("\n[적합도 top15 — 가장 스벅 유력 입지]")
    print(passed.head(15)[["name", "sigungu", "score_raw", "score_pct", "persona_name"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    passed[["name", "sigungu", "lat", "lon", "score_raw", "score_pct", "persona_name"]].to_csv(
        os.path.join(OUTDIR, "gate_candidates.csv"), index=False, encoding="utf-8-sig")

    # ───────── (2) out-of-time 신규 스벅 검증 ─────────
    print("\n" + "=" * 76)
    print("(2) out-of-time 검증 — 신규 입점 스벅 인근(≤0.3km) 카페 적합도")
    print("=" * 76)
    oot = []
    for nm, (la, lo) in NEW_SB.items():
        d = hav(df.lat, df.lon, la, lo)
        near = df[d <= 0.3]
        if len(near) == 0:
            near = df[d <= 0.5]
        oot.append({"신규매장": nm, "인근카페수": len(near),
                    "적합도평균": near.score_raw.mean(),
                    "백분위평균": near.score_pct.mean(),
                    "게이트통과율": (near.score_raw >= GATE_THR).mean(),
                    "최빈페르소나": near.persona_name.mode().iloc[0] if len(near) else "-"})
    oot = pd.DataFrame(oot)
    print(oot.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    hit = (oot["적합도평균"] >= GATE_THR).mean()
    print(f"\n→ 신규매장 {len(oot)}곳 중 인근 평균적합도가 게이트 이상: {hit:.0%} "
          f"(높을수록 '사전 예측 적중')")
    oot.to_csv(os.path.join(OUTDIR, "oot_validation.csv"), index=False, encoding="utf-8-sig")

    # ───────── (3) 타대학 비교 ─────────
    print("\n" + "=" * 76)
    print("(3) 서울 주요 대학 캠퍼스(≤1.0km) 적합도 비교")
    print("=" * 76)
    uni = []
    for nm, (la, lo) in UNIV.items():
        d = hav(df.lat, df.lon, la, lo)
        z = df[d <= 1.0]
        if len(z) == 0:
            continue
        uni.append({"대학": nm, "카페수": len(z), "스벅수": int(z.is_starbucks.sum()),
                    "적합도평균": z.score_raw.mean(), "백분위평균": z.score_pct.mean(),
                    "게이트통과율": (z.score_raw >= GATE_THR).mean(),
                    "최빈페르소나": z.persona_name.mode().iloc[0]})
    uni = pd.DataFrame(uni).sort_values("적합도평균", ascending=False)
    print(uni.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    uni.to_csv(os.path.join(OUTDIR, "university_comparison.csv"), index=False, encoding="utf-8-sig")

    write_report(passed, nonsb, oot, hit, uni)
    print("\n[저장] gate_candidates.csv, oot_validation.csv, university_comparison.csv, logs/VALIDATION_REPORT.md")

def write_report(passed, nonsb, oot, hit, uni):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["# 검증·확장 분석 리포트", "", f"> 생성 {ts}", "",
         "## (1) 게이트 통과 비스벅 카페 = 스벅 유력 입지", "",
         f"- 통과 {len(passed)}/{len(nonsb)} ({len(passed)/len(nonsb):.1%})",
         "- 페르소나별: " + ", ".join(f"{k} {v}" for k, v in passed.persona_name.value_counts().items()),
         "- 시군구 top5: " + ", ".join(f"{k}({v})" for k, v in passed.sigungu.value_counts().head(5).items()),
         "", "적합도 top10:", "",
         "| 카페 | 시군구 | 적합도 | 백분위 | 페르소나 |", "|---|---|---|---|---|"]
    for _, r in passed.head(10).iterrows():
        L.append("| %s | %s | %.3f | %.0f | %s |" % (r["name"], r["sigungu"], r["score_raw"], r["score_pct"], r["persona_name"]))
    L += ["", "## (2) out-of-time 검증 (신규 입점 스벅 인근 카페 적합도)", "",
          "> 신규매장 좌표는 역/지역 기준 근사값, 인근 카페 proxy (한계).", "",
          "| 신규매장 | 인근카페 | 적합도평균 | 백분위 | 게이트통과율 | 최빈페르소나 |",
          "|---|---|---|---|---|---|"]
    for _, r in oot.iterrows():
        L.append("| %s | %d | %.3f | %.0f | %.0f%% | %s |" % (
            r["신규매장"], r["인근카페수"], r["적합도평균"], r["백분위평균"], r["게이트통과율"]*100, r["최빈페르소나"]))
    L += ["", f"→ 신규매장 인근 평균적합도가 게이트 이상인 비율: **{hit:.0%}** (사전 예측 적중도)", "",
          "## (3) 서울 주요 대학 캠퍼스 비교 (≤1.0km)", "",
          "| 대학 | 카페수 | 스벅수 | 적합도평균 | 백분위 | 게이트통과율 | 최빈페르소나 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in uni.iterrows():
        L.append("| %s | %d | %d | %.3f | %.0f | %.0f%% | %s |" % (
            r["대학"], r["카페수"], r["스벅수"], r["적합도평균"], r["백분위평균"], r["게이트통과율"]*100, r["최빈페르소나"]))
    L += ["", "메모: 서울대의 상대적 위치(적합도/게이트통과율 순위)와 페르소나를 타대학과 대조해 해석.", ""]
    with open(os.path.join(HERE, "logs", "VALIDATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

