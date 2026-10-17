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
