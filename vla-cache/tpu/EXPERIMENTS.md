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

## 2. TPU prefill latency vs row 수 (2026-10-17)

`bench_rows.py`로 측정했다. chip 1개, graph마다 30회 반복한 median이며 vision은 제외했다. 데이터: `reports/bench_rows.json`, `reports/bench_tuples.json`

**(a) row 수를 바꿔 가며 측정 (layer 0–1은 305 row, layer 2–31은 n row, gather 1회)**

| rows | 160 | 176 | 192 | 208 | 224 | 240 | 256 | 272 | 288 | 304 | 305 | full (pruning graph 없음) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms | 20.4 | 21.0 | 21.6 | 22.0 | 21.9 | 22.6 | 23.0 | 24.9 | 25.6 | 26.3 | 27.4 | 27.1 |

**(b) 실제 VLA-Cache graph 구조로 측정 (pruning 지점 4곳에서 매번 gather), 29개 tuple**

| row tuple (layer 2, 6, 9, 11) | ms | | row tuple | ms |
|---|---|---|---|---|
| full (pruning 없음) | 27.05 | | 256, 224, 224, 224 | 21.96 |
| 305, 256, 256, 256 | 24.06 | | 256, 240, 240, 240 | 22.69 |
| 288, 256, 256, 256 | 23.60 | | 256, 256, 256, 256 | 23.01 |
| 272, 240, 240, 240 | 23.16 | | 288, 224, 224, 224 | 22.71 |
| exact tuple 20개 (258~267, 231~239) | 23.14 ~ 23.89 | | | |

새 shape를 처음 호출할 때 드는 비용(dynamo trace + XLA compile + 1회 실행)은 **shape당 28~34 s**이고, XLA compile만 따지면 8~10 s다.

**관찰**
1. **row를 줄인 만큼 빨라지지 않는다.** row를 47% 줄여도(305→160) latency는 25%만 준다. 300 token 근처에서 7B prefill은 weight 읽기(13.5 GB ÷ 1.2 TB/s ≈ 11 ms)에 묶여 memory-bound에 가깝다. ridge point는 v4 기준 ~230 FLOP/byte다.
2. **latency가 계단형이다.** 8 row 차이는 측정 noise 수준이다. 정렬되지 않은 exact row 수(예: 262/236)는 정렬된 bucket shape(288/256)보다 빠르지 않다.
3. pruning 지점마다 gather하는 비용이 있어, 단일 gather 곡선(a)으로 만든 모델은 실제 절감을 ~20% 과대평가한다. 3단계에서는 (b) 실측값을 쓰고, 측정하지 않은 tuple만 0.80으로 보정한 모델을 썼다.

## 3. bucket 정책의 비효율 (2026-10-17)

1단계 trace(3,120 step)의 각 step을 bucket 정책별 row tuple로 바꾸고, 2단계 latency를 붙였다. `analyze_buckets.py buckets`, 결과: `reports/bucket_report.json`

- **ideal** = exact row 수에 compile 비용 0 (도달 불가능한 하한).
- **dynamic 상각**은 처음 보는 shape마다 28.6 s를 내는 online runtime이 3,120 step 동안 내는 평균이다.

| bucket | graph 수 | prefill (ms) | ideal 대비 | full 대비 | 낭비 row (layer 평균) | compile 총합 | dynamic 상각 (ms/step) | 실측 비율 |
|---|---|---|---|---|---|---|---|---|
| 1 (원본 그대로) | **219** | 23.54 | 0 | −13.0% | 0 | **6,270 s** | **2,033** | 31% |
| 4 | 27 | 23.49 | −0.2% | −13.2% | 1.5 | 773 s | 271 | 32% |
| 8 | 12 | 23.52 | −0.1% | −13.0% | 3.5 | 344 s | 134 | 23% |
| **16** | **6** | **23.15** | **−1.6%** | **−14.4%** | 6.8 | 172 s | 78 | 100% |
| 32 | 4 | 23.52 | −0.1% | −13.1% | 19.9 | 115 s | 60 | 100% |
| 64 | 2 | 23.99 | +1.9% | −11.3% | 23.4 | 57 s | 42 | 100% |

("실측 비율"은 step 중 latency를 직접 잰 tuple의 비율이다. 나머지는 보정 모델 값이다.)

**결론**
1. **bucket으로 인한 비효율은 사실상 0이다.** bucket 16~32는 ideal과 같거나 오히려 빠르다(정렬된 shape 효과). bucket 64도 +1.9%에 그친다. static graph로 옮겨서 잃는 prefill 성능은 없다.
2. **진짜 비효율은 dynamic compile이다.** 원본처럼 row 수를 그대로 쓰면 graph 219개, compile 6,270 s(1.7시간)가 든다. 3,120 step에 상각하면 step당 2 s로 prefill(23.5 ms)의 ~86배다. graph가 점점 포화하더라도 처음 보는 shape마다 ~30 s씩 멈추므로 실시간 제어에는 쓸 수 없다. 원본 그대로 NPU에 올리면 어떻게 되는지에 대한 정량 답이다(H2).
3. **TPU에서 VLA-Cache 자체의 이득이 작다.** prefill은 −13~14%(27.1 → 23.2 ms)지만, step 전체(vision ~3.6 + prefill + decode 105.6 ms ≈ 136 ms)로 보면 **−2.8%**다. 짧은 prompt(~300 token)의 prefill은 memory-bound에 가깝고, decode가 step을 지배한다.
4. 함의: KV reuse가 의미 있으려면 prefill이 compute-bound여야 한다. 원래 목표 workload(주행, 6 cam × 576 ≈ 3,500 token prefill)가 그런 경우다. OpenVLA-LIBERO는 static-graph 이식이 맞게 동작하는지 검증하는 용도로는 충분하지만, 이득을 보여주기에는 맞지 않는다.

**주의**: bucket 1/4/8 수치는 31% 이하만 실측이고 나머지는 보정 모델이다. latency는 합성 이미지로 쟀다(prefill latency는 이미지 내용과 무관하다).

## 4. 성공률이 낮았던 원인: MuJoCo 버전 (2026-10-17)

1단계에서 성공률이 11/20(55%)으로 OpenVLA 논문(84.7%)보다 낮았다. 정책 출력은 HF 원본과 token 단위로 같으므로(§0) 시뮬레이터 쪽을 의심했다.

- 실패 episode의 frame을 보면, 팔이 그릇까지 가서 gripper를 닫지만 그릇이 들리지 않는다. 그 뒤로는 빈 gripper로 접시 위를 맴돈다. 정책이 엉뚱한 곳으로 가는 게 아니라 grasp가 미끄러지는 패턴이다.
- 같은 20 episode(task 10 × init state 0, 1)를 MuJoCo 버전만 바꿔 다시 돌렸다. robosuite 1.4.1, 정책·전처리는 동일하다.

| MuJoCo | 성공 | 비고 |
|---|---|---|
| 3.2.3 (pip 최신, 1단계에서 사용) | 11/20 (55%) | 3.2.3에서만 실패한 episode 6개: (0,1) (1,1) (4,1) (7,0) (7,1) (9,1) |
| **3.1.6** | **16/20 (80%)** | OpenVLA 공개 시점(2024-06)의 pip 최신 |
| 2.3.7 | 15/20 (75%) | |

**결론**: MuJoCo 3.2.x에서 LIBERO grasp가 깨진다. 기본 시뮬레이터를 **MuJoCo 3.1.6**(`~/venvs/libero-mj316`)으로 바꿨다. 3.1.6과 2.3.7의 차이(episode 4개가 서로 엇갈림)는 20 episode 규모에서 noise 범위다.

1~3단계 결과에 미치는 영향: trace는 3.2.3 궤적에서 뽑았다. 하지만 재사용 token 수는 cap과 task-relevance가 결정하고(§1 관찰 1), bucket 분석은 shape 분포만 쓰므로 결론은 그대로다.
