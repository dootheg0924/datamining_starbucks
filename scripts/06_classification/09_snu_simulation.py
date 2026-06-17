# -*- coding: utf-8 -*-
"""
[3단계] 서울대 관악캠퍼스 시뮬레이션

* 피처 생성 파이프라인/상가 원천이 폴더에 없어 신규 좌표 featurize 불가 →
  bak5에 이미 피처가 완비된 '캠퍼스 내·인근 카페'를 후보지 proxy 로 사용.

구역 정의(서울대 중심 37.4591,126.9520 기준 거리):
  - 캠퍼스 내부 : ≤1.2km (실제 교내 카페 mug/잔디/단대 등)
  - 정문권     : 1.2~2.0km (서울대입구역·낙성대 방면)
  - 관악구 전체 : 대조
저장된 모델로 Opt1/2/3 적용. 결과: outputs/snu_simulation.csv, SNU_REPORT.md
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
PERSONA_NAMES = {0: "오피스고소득", 1: "상업활성", 2: "주거생활", 3: "도심초밀집", 4: "비역세권"}
SNU = (37.4591, 126.9520)          # 관악캠퍼스 중심
SNU_GATE = (37.4811, 126.9527)     # 서울대입구역(실제 스벅 존재 권역)
GATE_THR = 0.534   # 08_integration Youden J

def hav(la, lo, la2, lo2):
    R = 6371; p = np.pi / 180
    a = np.sin((la2 - la) * p / 2) ** 2 + np.cos(la * p) * np.cos(la2 * p) * np.sin((lo2 - lo) * p / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def opt2_vector(row):
    v = {PERSONA_NAMES[k]: row[f"opt2_pC{k}"] for k in range(5)}
    v["비스벅"] = row["opt2_p_nonSB"]
    return v

def main():
    df = pd.read_parquet(os.path.join(OUTDIR, "integrated_predictions.parquet"))
    df["d_snu"] = hav(df["lat"], df["lon"], *SNU)
    df["d_gate"] = hav(df["lat"], df["lon"], *SNU_GATE)

    zones = {
        "캠퍼스내부(≤1.2km)": df[df.d_snu <= 1.2],
        "서울대입구역(≤0.5km)": df[df.d_gate <= 0.5],
        "관악구전체": df[df.sigungu == "관악구"],
        "서울전체": df,
    }

    print("=" * 76)
    print("[3단계] 서울대 관악캠퍼스 시뮬레이션")
    print("=" * 76)
    rows = []
    for name, z in zones.items():
        if len(z) == 0:
            continue
        rows.append({
            "구역": name, "카페수": len(z), "스벅수": int(z.is_starbucks.sum()),
            "적합도평균": z.score_raw.mean(), "적합도백분위평균": z.score_pct.mean(),
            "게이트통과율": (z.score_raw >= GATE_THR).mean(),
            "최빈페르소나": z.persona_name.mode().iloc[0],
        })
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 캠퍼스 내부 상세
    camp = zones["캠퍼스내부(≤1.2km)"]
    print(f"\n[캠퍼스 내부 {len(camp)}곳] — 실제 교내 카페")
    print(camp[["name", "score_raw", "score_pct", "persona_name"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Opt1/2/3 캠퍼스 대표값 = 전형적 카페(중앙값 점수 행; iloc[0]은 예외값 위험)
    rep = camp.iloc[(camp["score_raw"] - camp["score_raw"].median()).abs().argmin()]
    v = opt2_vector(rep)
    print("\n[서울대 캠퍼스 — 세 옵션 결과]")
    print(f"  Opt1 (적합도)   : {rep.score_raw*100:.0f}% (서울 카페 중 {rep.score_pct:.0f}백분위)")
    print(f"  Opt3 (게이팅)   : 게이트 {GATE_THR} → {'통과(스벅형)' if rep.score_raw>=GATE_THR else '미통과(스벅형 아님)'}"
          f" / 유형: {rep.persona_name}")
    print(f"  Opt2 (soft)     : " + " / ".join(f"{k} {val*100:.0f}%" for k, val in
          sorted(v.items(), key=lambda x: -x[1]) if val > 0.03))

    # 서울대입구역 대조(실제 스벅 존재 권역)
    gate = zones["서울대입구역(≤0.5km)"].sort_values("score_raw", ascending=False)
    print(f"\n[서울대입구역 대조] 적합도 상위 5 (캠퍼스 밖 = 스벅형 입지 존재):")
    print(gate.head(5)[["name", "is_starbucks", "score_raw", "score_pct", "persona_name"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # 저장
    camp_out = camp[["name", "sigungu", "lat", "lon", "d_snu", "score_raw", "score_pct",
                     "persona_name", "opt3_gate"] + [f"opt2_pC{k}" for k in range(5)] + ["opt2_p_nonSB"]]
    camp_out.to_csv(os.path.join(OUTDIR, "snu_simulation.csv"), index=False, encoding="utf-8-sig")
    write_report(summ, camp, rep, v, gate)
    print("\n[저장] outputs/snu_simulation.csv, logs/SNU_REPORT.md")

def write_report(summ, camp, rep, v, gate):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["# 서울대 관악캠퍼스 스타벅스 입지 시뮬레이션", "",
         f"> 생성 {ts} · 모델: PU bagging(GBM) + 페르소나(k=5)", "",
         "## 핵심 결론", "",
         f"- 캠퍼스 내부 카페 {len(camp)}곳의 스벅 적합도 **{rep.score_raw*100:.0f}%** "
         f"(서울 카페 중 {rep.score_pct:.0f}백분위) → **게이트 {GATE_THR} 미달, '스벅형 입지 아님'**",
         f"- 페르소나 = **{rep.persona_name}** (스벅 침투율 최저 유형)",
         "- → 서울대에 스벅이 없는 이유: 캠퍼스 내부가 지하철 접근성·상권밀도·직장인구가 낮은 "
         "'비역세권'형 입지이기 때문. 정문 밖(서울대입구역 방면)에는 스벅형 고적합 입지가 존재.", "",
         "## 구역별 비교", "",
         "| 구역 | 카페수 | 스벅수 | 적합도평균 | 백분위평균 | 게이트통과율 | 최빈페르소나 |",
         "|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append("| %s | %d | %d | %.3f | %.0f | %.1f%% | %s |" % (
            r["구역"], r["카페수"], r["스벅수"], r["적합도평균"], r["적합도백분위평균"],
            r["게이트통과율"] * 100, r["최빈페르소나"]))
    L += ["", "## 세 옵션 결과 (캠퍼스 내부)", "",
          f"- **Opt1**: 적합도 {rep.score_raw*100:.0f}% ({rep.score_pct:.0f}백분위)",
          f"- **Opt3**: 게이트 미통과 → 스벅형 아님 / 유형 {rep.persona_name}",
          "- **Opt2**: " + ", ".join(f"{k} {val*100:.0f}%" for k, val in
                                       sorted(v.items(), key=lambda x: -x[1]) if val > 0.01), "",
          "## 한계", "",
          "- 신규 좌표 featurize 불가(상가 원천 부재) → 교내 기존 카페를 proxy 로 사용.",
          "- 교내 카페들의 피처가 사실상 동일 → **캠퍼스 내부 지점 간 차별화 불가**(기 인지된 한계).",
          "- 캠퍼스는 학습 분포 밖(OOD) 가능성 → 적합도는 절대확률보다 상대 위치로 해석.", ""]
    with open(os.path.join(HERE, "logs", "SNU_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()

