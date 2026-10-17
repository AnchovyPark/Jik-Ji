# VLA-Cache on TPU — 실험 기록

static-graph 가속기(TPU v4, PyTorch/XLA)에서 VLA-Cache를 돌릴 때 bucket 방식이 얼마나 비효율적인지 단계별로 측정한다. 결과는 단계가 끝날 때마다 아래에 누적한다.

## 환경

| 항목 | 값 |
|---|---|
| 가속기 | Google TPU v4-8 (chip 4개, chip당 32 GiB HBM), chip 1개 = process 1개 |
| SW | torch 2.6.0, torch_xla 2.6.0, `torch.compile(backend="openxla")` |
| 모델 | `openvla/openvla-7b-finetuned-libero-spatial`, bf16 |
| 데이터셋 | LIBERO-Spatial (task 10개), 원본 eval 전처리 그대로 (JPEG, lanczos 224, center crop 0.9) |
| VLA-Cache 설정 | 원본 OpenVLA 설정: static top-k 130, τ 0.996, task-relevant top-120, pruning layer [2, 6, 9, 11] |
| prompt layout | BOS 1 + vision 256 + text slot 48 = **305 slot** (text는 pad로 채움; 실제 prompt 길이 ~292) |

graph 구성: `vision` (고정), `prefill(rows=(n_2, n_6, n_9, n_11))` (row 수별 1개), `decode` (고정). KV는 고정 크기 buffer에 두고, 재사용 token은 이전 frame의 K/V를 그대로 읽는다.

## 0. 구현 검증 (2026-10-16)

| 검사 | 결과 |
|---|---|
| cache off: HF OpenVLA vs 우리 구현 (CPU fp32) | action token 7개 일치 |
| cache off: HF (CPU bf16) vs 우리 구현 (TPU bf16) | 일치 |
| cache on: 원본 VLA-Cache 코드 (transformers fork, CPU fp32) vs 우리 구현 (CPU fp32, exact row) | 6 frame 모두 action 일치 (reuse frame 5개 포함) |

host 오버헤드 수정: timm vision tower의 fused SDPA가 XLA에서 block마다 host sync를 일으켜 step당 graph가 53개로 쪼개졌다. 이를 끄고 `torch.compile`로 graph를 cache하자 prefill이 **1,046 ms → 32 ms**로 줄었다.

합성 frame stream 20개 기준 (chip 1개):

| mode | 컴파일 | prefill | decode (7 token) |
|---|---|---|---|
| off (full prefill) | 1회 | 32.4 ms | 105.6 ms |
| static (bucket 32) | 2회, 이후 0 | 29.5 ms | 105.7 ms |
| dynamic (원본 semantics) | 새 row 수마다 1회, 회당 ~30 s (dynamo + XLA) | 30.9 ms (컴파일 없을 때) | — |

관찰: TPU에서는 prefill(30 ms)보다 decode(105 ms)가 step 시간을 지배한다. VLA-Cache가 줄이는 쪽은 prefill뿐이다.

## 1. LIBERO-Spatial trace (2026-10-17)

cache off로 closed-loop을 돌리면서, step마다 원본 VLA-Cache가 계획했을 정확한 row 수(bucket 없음)를 기록했다. 실행은 full prefill이므로, 다음 step의 계획에 쓰이는 attention 통계는 full prefill에서 나온 값이다. 실제 cache run과는 조금 다를 수 있다.

- 범위: task 10개 × episode 2개(init state 0, 1) = 20 episode, 3,140 step (계획이 있는 step 3,120). TPU chip 4개로 병렬 실행, 약 4.5분.
- 원본 데이터: `results/traces/*.npz` (frame + plan, git 밖). 요약: `reports/trace_stats.json`

| 항목 | mean | p10 | p50 | p90 | min / max |
|---|---|---|---|---|---|
| static patch (τ 0.996 통과, top-k 130 cap) | 130 | 130 | 130 | 130 | 130 / 130 |
| 재사용 token (static − task-relevant) | 74.1 | 67 | 73 | 82 | 60 / 97 |
| 마지막 segment에서 빠진 row | 69.9 | 63 | 69 | 78 | 57 / 92 |

pruning layer별 row 수 (305 slot 기준, pad 13개 포함):

| layer | rows mean | p10 | p90 | 서로 다른 값 |
|---|---|---|---|---|
| 2 | 262.8 | 257 | 268 | 26 |
| 6 | 235.1 | 227 | 242 | 36 |
| 9 | 235.1 | 227 | 242 | 36 |
| 11 | 235.1 | 227 | 242 | 36 |

layer별 재사용 비율(entropy schedule)의 평균은 [0.576, 0.950, 0.831, 0.755], 표준편차는 0.01~0.02다.

**관찰**
1. 고정 camera인 LIBERO에서는 τ 0.996을 넘는 patch가 항상 130개(top-k cap)보다 많다. 재사용 수는 사실상 `130 − |static ∩ attention top-120|`으로 정해진다. 즉 frame 변화가 아니라 cap과 task-relevance가 재사용량을 결정한다.
2. 실제로 효과가 있는 pruning 지점은 **2개(layer 2, 6)**뿐이다. layer 6의 비율(0.95)이 가장 커서, 뒤 지점(9, 11)은 "이전보다 많이 빼야만 적용" 규칙에 걸려 항상 skip된다.
3. 비율이 거의 상수(std 0.02)다. adaptive schedule을 offline 고정 schedule로 바꿔도 손실이 거의 없을 것으로 예상된다(§12.1 가설 확인).
4. row tuple이 **3,120 step 동안 219종류**다. 앞 10% step에서 이미 81개, 50%에서 164개가 나온다. 원본처럼 동적으로 돌리면 graph 219개를 compile해야 한다(회당 ~10~30 s).
5. 성공률: 11/20(55%). OpenVLA 논문의 LIBERO-Spatial 84.7%보다 낮다. 표본이 작지만 차이가 크므로, 시뮬레이터 버전(mujoco 3.2.3 / robosuite 1.4.1 / CPU EGL 렌더링) 차이를 점검해야 한다. **열린 이슈.**
