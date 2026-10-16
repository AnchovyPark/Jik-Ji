# Jik-Ji

**Static-graph NPU 위에서 frame 간 KV reuse로 driving VLM을 SLO 안에 서빙하는 연구 프로젝트**

## 이름

직지(直指)는 현존하는 가장 오래된 금속활자 인쇄본이다. 금속활자는 활자를 한 번 주조해 두고, 판마다 재조판해 다시 찍는다.
이 프로젝트도 같은 구조다. graph는 미리 compile해 고정해 두고(주조된 활자), frame마다 바뀐 부분만 다시 계산하고 나머지 KV는 재배치해 재사용한다(재조판).

## 배경

- Driving VLM/VLA offloading은 uplink bandwidth 때문에 MEC(Multi-access Edge Computing)에 놓이고, MEC는 75~150 W passive 단일 slot 가속기만 수용한다.
- 이 전력 class에서는 NPU(FuriosaAI RNGD 등)가 GPU(T4, L4)보다 spec상 앞서지만, 7B model × ~3,500 token prefill은 server budget(200~250 ms) 안에 빠듯하다.
- 연속 frame 간 중복을 KV reuse로 줄이면 budget 안에 들어오지만, 이 dynamic 기법은 GPU runtime에서는 쉽고 static/AOT compile NPU에서는 막혀 있다.

## 접근

- reuse / recompute 결정을 static graph 안에서 실행한다: 고정 budget bucket, gather, RoPE 재회전, 외부 KV 입출력.
- 하나의 Python plug-in이 reuse 결정을 담당하고, NPU vendor backend(Furiosa SDK, Neuron, TensorRT 등)에는 얇은 adapter로 연결한다.
- 평가 기준: 상한선 = GPU + vLLM, 하한선 = NPU vendor runtime.

## 대상

- Hardware: FuriosaAI RNGD (주), AWS Inferentia2 / TPU (generality)
- Model: Qwen2.5-VL-7B, Qwen3-VL-8B, LLaVA-1.5-7B
- Workload: nuScenes 6-cam scene replay, DriveLM / NuScenes-QA

## 현재 상태

- [x] 선행연구 분석: VLA-Cache, VLN-Cache
- [ ] 선행연구 분석: FlashDrive, CacheBlend, KVCOMM 등
- [ ] nuScenes frame 간 reuse 가능 비율 측정 (H1)
- [ ] NPU vendor runtime에서 dynamic reuse 비용 측정 (H2)
- [ ] static-graph reuse plug-in 구현 및 평가 (H3)
