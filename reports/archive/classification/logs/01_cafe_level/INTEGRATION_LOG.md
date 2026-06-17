# 2단계 통합(Opt1/2/3) 로그


---

## 실행 @ 2026-06-14 01:46:28

### Opt1 (Decoupled) — 페르소나별 평균 적합도 백분위

| 페르소나 | 평균 적합도 pct |
|---|---|
| 오피스프리미엄 | 80.2 |
| 초번화가 | 76.1 |
| 번화가 | 62.1 |
| 일반상권 | 48.2 |
| 비역세권 | 29.3 |

### Opt3 (Two-stage 게이팅)

- stage1 게이트 임계값(Youden J, OOF score_raw): **0.5490**
- 게이트 통과: 7207/22621 (스벅 통과율 57.7%, 비스벅 통과율 31.1%)
- stage2: 최근접 centroid 페르소나(학습 불필요)

### Opt2 (Multi-class soft) — 출력 형식

- 각 카페: `[opt2_pC0..pC4, opt2_p_nonSB]` = gate×persona_softmax + (1-gate)

### stage2 신뢰성 (RF가 kmeans 페르소나 복원, 681 스벅 spatial CV)

- accuracy=0.916, macro-F1=0.910
- confusion matrix:
```
[[ 40   3   2   0   2]
 [  0 211  12   1   3]
 [  0  11 224   0   0]
 [  1  17   0  55   0]
 [  0   2   3   0  94]]
```
  → 높으면 페르소나가 피처로 잘 분리됨 = Opt3 stage2(centroid) 정당화.

메모: (세 옵션 중 발표 메인 선택, 서울대 적용 해석은 여기에)


---

## 실행 @ 2026-06-14 01:47:02

### Opt1 (Decoupled) — 페르소나별 평균 적합도 백분위

| 페르소나 | 평균 적합도 pct |
|---|---|
| 오피스프리미엄 | 79.7 |
| 초번화가 | 76.0 |
| 번화가 | 62.0 |
| 일반상권 | 48.2 |
| 비역세권 | 29.3 |

### Opt3 (Two-stage 게이팅)

- stage1 게이트 임계값(Youden J, OOF score_raw): **0.5490**
- 게이트 통과: 7073/22305 (스벅 통과율 57.7%, 비스벅 통과율 30.9%)
- stage2: 최근접 centroid 페르소나(학습 불필요)

### Opt2 (Multi-class soft) — 출력 형식

- 각 카페: `[opt2_pC0..pC4, opt2_p_nonSB]` = gate×persona_softmax + (1-gate)

### stage2 신뢰성 (RF가 kmeans 페르소나 복원, 681 스벅 spatial CV)

- accuracy=0.916, macro-F1=0.910
- confusion matrix:
```
[[ 40   3   2   0   2]
 [  0 211  12   1   3]
 [  0  11 224   0   0]
 [  1  17   0  55   0]
 [  0   2   3   0  94]]
```
  → 높으면 페르소나가 피처로 잘 분리됨 = Opt3 stage2(centroid) 정당화.

메모: (세 옵션 중 발표 메인 선택, 서울대 적용 해석은 여기에)


---

## 실행 @ 2026-06-14 20:24:25

### Opt1 (Decoupled) — 페르소나별 평균 적합도 백분위

| 페르소나 | 평균 적합도 pct |
|---|---|
| 오피스고소득 | 79.7 |
| 도심초밀집 | 76.0 |
| 상업활성 | 62.0 |
| 주거생활 | 48.2 |
| 비역세권 | 29.3 |

### Opt3 (Two-stage 게이팅)

- stage1 게이트 임계값(Youden J, OOF score_raw): **0.5490**
- 게이트 통과: 7073/22305 (스벅 통과율 57.7%, 비스벅 통과율 30.9%)
- stage2: 최근접 centroid 페르소나(학습 불필요)

### Opt2 (Multi-class soft) — 출력 형식

- 각 카페: `[opt2_pC0..pC4, opt2_p_nonSB]` = gate×persona_softmax + (1-gate)

### stage2 신뢰성 (RF가 kmeans 페르소나 복원, 681 스벅 spatial CV)

- accuracy=0.916, macro-F1=0.910
- confusion matrix:
```
[[ 40   3   2   0   2]
 [  0 211  12   1   3]
 [  0  11 224   0   0]
 [  1  17   0  55   0]
 [  0   2   3   0  94]]
```
  → 높으면 페르소나가 피처로 잘 분리됨 = Opt3 stage2(centroid) 정당화.

메모: (세 옵션 중 발표 메인 선택, 서울대 적용 해석은 여기에)


---

## 실행 @ 2026-06-16 15:17:42

### Opt1 (Decoupled) — 페르소나별 평균 적합도 백분위

| 페르소나 | 평균 적합도 pct |
|---|---|
| 오피스고소득 | 79.9 |
| 도심초밀집 | 76.5 |
| 상업활성 | 61.9 |
| 주거생활 | 48.1 |
| 비역세권 | 29.4 |

### Opt3 (Two-stage 게이팅)

- stage1 게이트 임계값(Youden J, OOF score_raw): **0.5339**
- 게이트 통과: 7618/22305 (스벅 통과율 60.1%, 비스벅 통과율 33.3%)
- stage2: 최근접 centroid 페르소나(학습 불필요)

### Opt2 (Multi-class soft) — 출력 형식

- 각 카페: `[opt2_pC0..pC4, opt2_p_nonSB]` = gate×persona_softmax + (1-gate)

### stage2 신뢰성 (RF가 kmeans 페르소나 복원, 681 스벅 spatial CV)

- accuracy=0.916, macro-F1=0.910
- confusion matrix:
```
[[ 40   3   2   0   2]
 [  0 211  12   1   3]
 [  0  11 224   0   0]
 [  1  17   0  55   0]
 [  0   2   3   0  94]]
```
  → 높으면 페르소나가 피처로 잘 분리됨 = Opt3 stage2(centroid) 정당화.

메모: (세 옵션 중 발표 메인 선택, 서울대 적용 해석은 여기에)

