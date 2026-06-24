# TurboQuant KV 캐시 압축 for llama.cpp

> [TurboQuant (ICLR 2026, Google DeepMind)](https://arxiv.org/abs/2504.19874) 구현 — Walsh-Hadamard Transform + Lloyd-Max 양자화 + QJL 보정 기반 KV 캐시 압축

[🇬🇧 English](README.md)

### 📄 논문

- 📗 [**TurboQuant 구현: 프로덕션 LLM 추론 엔진에서의 3비트 KV 캐시 압축**](paper/turboquant_impl_ko.pdf) — 원본 TBQ v1 구현 논문 (WHT + Lloyd-Max + QJL)
- 📕 [English — TurboQuant Impl](paper/turboquant_impl.pdf)

### 🆕 v1.9.0 — DeepSeek-V4-Flash on 2× DGX Spark: vLLM급 서빙, llama.cpp에서

> **vLLM에만 있던 기능 — 멀티노드 TP, 연속 배칭, 스펙 디코딩 — 을 llama.cpp에서, 작은 박스 두 대로.**

| 기능 | 스펙 |
|---|---|
| 🌐 멀티노드 텐서 병렬 | **2노드 over RoCE/RDMA** — 느린 RPC 백엔드 아님 |
| 📜 컨텍스트 길이 | **1,048,576** (풀 1M) |
| 🗜️ KV 캐시 @ 1M | **~9.6 GB** — TurboQuant 3-bit (TBQ3) |
| ⚡ 스펙 디코딩 | **MTP — 16.5 t/s @ 1M**, plain 능가 |
| 🔀 연속 배칭 | 멀티슬롯 (`-np 2`) → ~1.4× aggregate (~22.8 t/s @ 1M) |
| 💻 하드웨어 | 2× DGX Spark (GB10, 128 GB) — 데이터센터 GPU 불필요 |

**실측 처리량** — 단일 스트림, 2× DGX Spark over RoCE:

| 모델 | ctx | 모드 | gen t/s |
|---|---|---|---|
| Q4 | **1M** | **2노드 TP + MTP + TBQ3 KV** | **16.5** ⚡ |
| Q4 | 1M | 2노드 TP plain + TBQ3 KV | 15.5 |
| FP4-native | 8K | 2노드 TP + MTP | 16.6 |
| FP4-native | 8K | 2노드 TP plain | ~15 |
| IQ2 | 8K | 단일 박스 (TP 없음) | 16.9 |

> **1M 컨텍스트에서 MTP가 plain을 능가** — 메모리 바운드 MoE에서도 실제로 이득 보는 스펙 디코딩, 두 박스에 걸쳐.

**구현한 것:**

- **풀 1M 토큰 컨텍스트를 ~9.6 GB에서 — 큰 GPU 하나가 아니라 두 박스에 걸쳐.** TurboQuant 3-bit KV 압축이 100만 토큰 KV 캐시를 **~9.6 GB**로 줄여, 전체 컨텍스트가 두 머신에 쪼개져 들어간다. **거기서 MTP 스펙 디코딩이 16.5 t/s — plain보다 빠름** — 이 규모에서 처음 가능해진 풀스피드 1M 컨텍스트 스펙 디코딩.
- **신규 `GGML_TYPE_F8_E4M3_B128` 양자화 타입, end-to-end.** FP8 dense weight를 위한 전체 기계장치: 블록 구조(128× E4M3 값 + E8M0 거듭제곱 블록 스케일 1개), **비트-정확 CPU 코덱**(E4M3FN 디코드 — subnormal/NaN/±448 saturation + E8M0 `2^(e−127)` 스케일), CUDA dequant(cuBLAS GEMM 경로용 `to_fp16`)와 CUDA **MMVQ GEMV** 커널. MMVQ는 `__shared__`에 적재한 256-엔트리 LUT를 기본 사용(스칼라 dequant와 비트-동일); `int8 dp4a` 근사는 있지만 **env로 OFF**(`GGML_CUDA_F8_APPROX_DP4A`, 손실 있음). CPU↔CUDA↔LUT 디코드 테이블은 256개 코드 전부 일치 검증. 타입 슬롯 확보를 위해 `GGML_TYPE_TBQ3_0`을 `42 → 65`로 이전(그에 의존하던 range-guard도 수정 — 아래 감사 참조).
- **재양자화 없이 FP4/FP8-native 로딩, 로드타임 어댑터로.** 146GB `nsparks` 체크포인트(F8_E4M3 dense + MXFP4 expert, `ftype 41`)를 그대로 서빙. 어댑터는 `ftype 41`에서만 켜져(**업스트림 모델은 모든 키 required 유지**) ~21개 nsparks 텐서명을 우리 `deepseek4` 스키마로 통역(`compressor→compress`, `attn_kv→attn_kv_latent`, `output_hc→hc_head`, `.weight` 접미사 제거)하고 nsparks가 빠뜨린 메타데이터 키(`output_lora_rank`, `n_hc`, `hc_sinkhorn`, 43개 `compress_ratios`, `compress_rope_freq_base`, …)를 아키텍처 기본값으로 채웁니다.
- **MTP를 FP4 파일에 구워 넣음(사이드-shard 아님).** 단독 MTP 헤드(arch `deepseek4_mtp_support`, `mtp.0.*`)는 `tok_embd`/`output`이 없어 단독 draft로 못 뜹니다. `turboquant/ds4_fp4_bake_mtp.py`가 mmap으로 스트리밍(150GB를 RAM에 안 올림)해 결합 GGUF 1개 = 모든 FP4 텐서 바이트-동일 + `blk.43.nextn.*`로 리네임된 MTP 헤드 + `nextn_predict_layers=1`, `file_type 41` 유지(FP4 어댑터 계속 적용). 그 후 `--spec-type draft-mtp`가 바로 소비.
- **`DSV4_VERIFY_REUSE`가 잠금 해제 열쇠 — v1.8.0의 정직한 반전.** v1.8.0에선 이 verify-graph-reuse 인프라가 **OFF**였습니다(그래프 재빌드가 async GPU와 오버랩돼 이득 ~0). FP4 + MTP 스택에선 상황이 역전 — MTP 첫 실측이 **2.27 t/s, `graphs reused = 0`**(매 스펙 라운드가 풀 DSV4 그래프 재빌드, ~1초/라운드)이고 *그 재빌드*가 이제 병목입니다. `DSV4_VERIFY_REUSE`를 켜면 재사용이 살아나(`0 → 100+/요청`) → **16.6 t/s, MTP 없는 baseline(~15)을 능가**. perplexity 민감(256-pad 뷰가 flash-attn fp 누적 순서를 교란)이라 게이트 유지하며, 여기선 무손실 검증(greedy `France→Paris`/`수도→서울`, 한글 자모 분해 멀티턴 repro, 양박스 안정).
- **2-노드 TP + 멀티슬롯 + MTP, 합성.** 두 Spark에 걸친 SPMD 텐서 병렬(출력 미러, NCCL all-reduce), 멀티슬롯 디코드 배칭 경로(`DSV4_MULTISLOT`, 동시 요청 → **~1.4× aggregate** — 2-box Q4 @ 1M 실측: 단일 ~16 t/s → 2동시 ~22.8 t/s), MTP가 전부 함께 돕니다. 가능케 한 수정은 메타-백엔드의 **graph-scoped 메타데이터 재빌드** — 멀티슬롯 + MTP verify 그래프가 TP split을 넘어 더는 오염되지 않습니다.
- **긴 컨텍스트 그래프-arena 크래시 수정.** DSV4 chunk 압축기가 긴 멀티턴 prefill에서 `O(context)`개 그래프 객체를 만들어 메타데이터 arena를 고갈시킴(N턴 후 `"괭"` 크래시). `DSV4_BATCHED_COMPRESSOR`가 chunk 압축기를 고정 `O(1)` op 수로 재작성(carry-in / bulk-pool / carry-out, 언롤된 recurrence와 수치 동일); sched-context 분리(`max_splits` cap, 21GB → ~1.3GB 메타데이터)와 `-ub 256`이 모든 arena를 유계로 유지.
- **푸시 전 코드 감사(병렬 심층 리뷰 4건 + 수동 검증).** 릴리스 전 잠복 **critical 2건**을 잡아 수정: (1) 신규 배치 압축기의 carry-out이 carry-in 블록이 유일한 완료 블록일 때(`n_full==0`, 예: 비정렬 `ratio==128` prefill chunk) **음수 텐서 오프셋**을 읽음; (2) `TBQ3_0 42→65` 이전으로 세 개의 `>=TBQ3_0 && <=TBQP4_4` range-guard(`65..61` = 항상 false)가 조용히 깨져 TBQ KV 캐시의 GB10 SoC-freeze 보호를 무력화 → `==TBQ3_0 || (TBQ4_0..TBQP4_4)`로 복원. 추가로 레이어×토큰마다 도는 `stderr` 디버그 flood를 런치 스크립트에서 제거. F8 코덱 / sched cap / 로더 alias / 굽기 툴은 감사 통과(조치 불필요).

#### 성능 특성

단일 스트림 디코드는 **~16.6 t/s**(MTP+reuse) vs **~15**(plain) — FP4 자체는 Q4 대비 디코드 속도 레버가 *아닙니다*(디코드는 메모리-바운드 GEMV; 네이티브 FP4 텐서코어는 GEMM/prefill엔 도움, 토큰당 GEMV엔 아님). 그래서 여기서의 승리는 quant가 아니라 **그래프 재사용으로 살려낸 MTP**입니다. 구조적 천장은 v1.8.0이 기록한 것과 동일 — DSV4 디코드는 LPDDR 대역폭 바운드(matmul ~52%, 데이터 셔플 ~19%, 어텐션 ~2.3%)이고 MoE 투기는 distinct-expert 바이트 증가로 상한이 걸립니다. 즉 MTP의 plain 대비 우위는 설계상 완만하고, 진짜 결과는 **풀 TP + 멀티슬롯 + MTP 스택이 네이티브 FP4 체크포인트 위에서 안정적·무손실로** parity-이상 처리량으로 돈다는 것입니다. 멀티슬롯은 *단일* MTP 스트림을 배칭하지 않습니다(`pure_decode=0`, recurrent-state 롤백 ⇒ `split_seq`); ~1.4×(~22.8 t/s vs 단일 ~16, 2-box Q4 @ 1M 실측)는 동시 부하에서의 aggregate. `DSV4_MULTISLOT=1` 필요 — 이 env 없이 `-np 2`만 주면 생짜 contending 경로라 단일보다 느림.

#### 사용법

```bash
# 1) MTP 헤드를 FP4 체크포인트에 굽기 → 결합 GGUF 1개 (~5분, 재양자화 없음)
python3 turboquant/ds4_fp4_bake_mtp.py \
  DeepSeek-V4-Flash-FP4-FP8-native.gguf \
  DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  DeepSeek-V4-Flash-FP4-FP8-native-MTP.gguf      # --dry-run 으로 메타데이터만 검증 가능

# 2) 결합 GGUF를 두 번째 Spark로 미러 (4-NIC 병렬 복사; 자기 클러스터에 맞게 INTERFACES 편집)
python3 turboquant/scp_dgx_spark.py ~/Models/DeepSeek-V4-Flash-GGUF/FP4

# 3) 2-노드 TP + 멀티슬롯 + MTP 스택 서빙
bash fp4ctl.sh start                 # start | stop | restart | status  (pid-only kill, GPU 회수 대기)

# raw 2-박스 런치 (마스터 = 박스 A 10.0.1.1, 슬레이브 = 박스 B 10.0.1.2):
#   bash run_tp_MASTER_DSV4-FP4-MTP.sh   # http://0.0.0.0:8080 서빙
#   bash run_tp_SLAVE_DSV4-FP4-MTP.sh    # 팔로워, HTTP 없음
```

런치 스크립트에 박힌 핵심 env: `DSV4_MULTISLOT=1`(동시 슬롯 배칭), `DSV4_VERIFY_REUSE=1`(MTP verify-graph 재사용 — 속도 열쇠), `DSV4_BATCHED_COMPRESSOR=1`(긴 컨텍스트 안전), 그리고 `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0`. 마스터와 슬레이브는 동일 env로 돌아야 함(SPMD).

> ⚠️ **2노드 prefill이 갑자기 10–30× 느려졌다? 코드 말고 GPUDirect RDMA부터 봐라.**
> 같은 프롬프트 처리가 ~1–2초에서 ~30초로 뛰면 거의 항상 NCCL **GPUDirect RDMA** 경로가 깨진 거지 모델 문제가 아니다. DGX Spark **OTA 시스템 업데이트가 `rdma-core`를 Ubuntu 기본 `50.0`으로 조용히 다운그레이드**할 수 있는데, 그 `libmlx5`엔 NCCL이 GPU 메모리를 직접 RDMA 등록하는 데 쓰는 `mlx5dv_reg_dmabuf_mr@MLX5_1.25` 심볼이 없다. 그러면 NCCL이 레이어마다 all-reduce를 GPU→host→NIC로 스테이징해서 **prefill을 박살낸다**(decode는 all-reduce가 작아 거의 영향 없음). 우리는 이거 때문에 커널을 몇 시간 팠다 — 하지 마라.
>
> **진단** — `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET`로 기동. 깨진 상태:
> `dlvsym failed on mlx5dv_reg_dmabuf_mr ... undefined symbol ... MLX5_1.25` + `GPU Direct RDMA Disabled for HCA`.
>
> **수정** — NVIDIA DOCA `rdma-core`를 **양쪽 박스**에 설치(Ubuntu repo엔 50.0뿐, DOCA는 `MLX5_1.25/26/27` 가진 2604.x 제공):
> ```bash
> B=https://linux.mellanox.com/public/repo/doca/latest/ubuntu24.04/arm64-sbsa
> curl -sL $B/doca_keyring.gpg | sudo tee /usr/share/keyrings/doca.gpg >/dev/null
> echo "deb [signed-by=/usr/share/keyrings/doca.gpg] $B/ ./" | sudo tee /etc/apt/sources.list.d/doca.list
> sudo apt update && sudo apt install -y --no-install-recommends rdma-core ibverbs-providers libibverbs1
> sudo apt-mark hold rdma-core ibverbs-providers libibverbs1   # 다음 OTA가 또 다운그레이드 못 하게 hold
> ```
> **확인:** `nm -D /usr/lib/aarch64-linux-gnu/libmlx5.so* | grep MLX5_1.25`로 심볼 뜨고, NCCL 로그에 `GPU Direct RDMA Enabled`. (`nvidia-peermem`은 GB10 open 드라이버선 로드 실패 — dmabuf가 정석.)
>
> **⚠️ 두 박스가 *완전히 같은* rdma-core/libmlx5 버전이어야 한다.** 두 Spark가 불일치하면 — 한쪽만 업그레이드했거나 OTA가 한쪽만 건드리면 — NCCL이 노드 간 공통 RDMA 전송을 못 정하고 **조용히 TCP 소켓으로 폴백**한다. 각 박스의 GPUDirect가 로컬로는 멀쩡해 보여도 *같은* prefill 급락이 난다. 그러니 항상 **양쪽 박스**에 동일 버전을 `apt install` + `apt-mark hold` 하고, 시스템 업데이트 후엔 모델 탓하기 전에 `nm -D … | grep MLX5_1.25`로 양쪽 다 재확인.

> 🛠️ **개발 기간.** 주말 프로젝트가 아닙니다. FP4 + 2-노드 TP + 멀티슬롯 + MTP 스택은 거의 쉬지 않은 ~5일 스프린트(**2026년 6/18–22, ~46커밋**)에 들어왔고, 그것은 **~2주 DeepSeek-V4-Flash 마라톤**(**~140 DSV4 관련 커밋**, 6/9–22 거의 매일)의 끝자락입니다. 개발자 1명, DGX Spark 2대, 잠 거의 없이 — 타입 포팅, 로더 어댑터, 굽기 파이프라인, 긴 컨텍스트 크래시 추적, graphs-reused-0 → verify-reuse 돌파, 푸시 전 전수 감사까지 전부 수작업.


---

### 🆕 v1.8.0 — DeepSeek-V4-Flash 풀 CUDA 포팅 + MTP 셀프-스펙 디코딩

**DeepSeek-V4-Flash(`deepseek4`)가 TurboQuant 포크에서 end-to-end로 돌아갑니다 — CSA/HCA 압축 attention, hyper-connection, DSA lightning indexer, phase-uniform decode graph(CUDA-graph capture)까지 포함, GB10에서 ds4 레퍼런스 엔진 parity(13.4 t/s). 그 위에 antirez의 사이드 GGUF로부터 MTP 셀프-스펙 디코딩, 그리고 global attention 레이어의 tbq3 KV까지.**

**환경:** NVIDIA DGX Spark (GB10, 128GB) · DeepSeek-V4-Flash-IQ2_XS-XL (82GB, antirez 계보) · ctx=16384 · greedy.

| 모드 | gen t/s | 비고 |
|------|---------|------|
| baseline (f16 KV) | 13.4 | ds4 엔진(13.75)의 97% |
| + MTP (`--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2`) | **15.5 (자유 텍스트) / 17.0 (규칙적 텍스트)** | **+15% / +27%**, accept 98–100% |
| + MTP + **tbq3 KV** (프로덕션) | **~16–20** (수용률 기반, 아래 참조) | accept 96–100%; tbq3 = KV 메모리 절약(동등 품질), DSV4에서 속도 레버는 아님 |

- **MTP 헤드는 별도 GGUF로 배포됨** — `turboquant/ds4_mtp_to_shard.py`가 `mtp.0.*` 텐서를 `blk.43.nextn.*`로 리네임해 **3번째 split shard**로 변환하고 shard1의 `split.count`를 in-place 패치(6바이트, `--revert` 지원)합니다. 82GB 메인 shard 재양자화 불필요.
- **draft 헤드의 hidden 입력 = full hyper-connection 상태** (`n_embd_h = n_hc·n_embd = 16384`). 새 `llama_model_n_embd_h()` API로 pre-norm 버퍼/MTP 배치 폭/draft-mtp impl에 배관. **p_min 게이트 필수**: 게이트 없이는 accept 69%로 떨어져 verify+체크포인트 비용이 이득을 역전(baseline보다 느려짐).
- **`-ctk tbq3 -ctv tbq3`가 DSV4에서 동작** — global(ratio==0) 레이어가 TBQ3_0 @ head_dim 512 (GLM-4.7-Flash 512 커널 재활용), SWA·compressed 사이드 캐시는 품질 정책상 f16 유지. dim≥1 양자화 CUDA concat 추가.
- **버그픽스:** DSV4 compressed-KV state 복원이 raw 호스트 read를 사용 — ON_DEVICE 체크포인트(스펙 롤백)에서 스트림 어긋남. `read_tensor`로 수정.

#### 성능 특성 (GB10 실측, 프로덕션 설정)

프로덕션 설정은 `-ctk tbq3 -ctv tbq3 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2`. 실측 디코드는 **~16–20 tok/s이고 본질적으로 들쭉날쭉**합니다 — 불안정이 아니라 *수용률 기반*: `t/s = (MTP 라운드당 수용 토큰) / (라운드 시간)`. 라운드 시간은 거의 일정(**~25 ms**, 타깃 verify GPU 패스가 지배)이고, 라운드당 토큰 수가 draft 수용률(실측 **96–100%**)에 따라 1→3으로 출렁입니다. 수용률은 현재 텍스트의 예측가능성에 비례 — 예측 쉬운 구간(리스트·흔한 구문)은 ~20 t/s, 새로운 내용(숫자·고유명사)은 ~16으로 하락. essay 집계(프롬프트 eval 포함)는 ~16, 순간 피크는 ~20–23. 이 폭은 투기적 디코딩의 본질이지 회귀가 아닙니다.

전체 op별 GPU 프로파일(`DSV4_KERNEL_PROF`)은 **DSV4 디코드가 메모리-바운드**(GB10 LPDDR 대역폭 한계 근처)임을 보여줍니다:

| GPU op 클래스 | 비중 | 헤드룸 |
|---|---|---|
| MoE expert matvec (IQ2_XS, 256 expert) + dense 어텐션 투영 (Q8_0) | **~52%** | LPDDR 한계 근처 — gate+up 융합해도 런치 오버헤드만 절약, **읽는 바이트는 동일** |
| 데이터 셔플 (CPY / CONT / GET_ROWS / SET_ROWS / CONCAT — 압축기 state recurrence + hyper-connection expand) | **~19%** | 진짜 헤드룸이나 대부분 구조적 (각 op이 작고 필요) |
| Flash-attention (D=512 압축 KV) | **~2.3%** | 무시 — 압축 KV 설계가 잘 작동 |

matmul은 매 토큰 LPDDR에서 양자화 weight를 읽습니다. 그 벽 너머는 오직 바이트를 덜 읽는 것(더 낮은 quant / expert 수 = 품질 트레이드오프)뿐, 커널 융합이 아닙니다. 그래프 재빌드는 **임계경로 밖**으로 측정됨 — async GPU 컴퓨트와 오버랩(graphs-reused ON vs OFF: 19.83 vs 19.80 t/s) — 그래서 이번 릴리스의 **verify-graph phase-uniform 재사용 인프라는 정확하지만 기본 OFF**(`DSV4_VERIFY_REUSE`; raw 압축 뷰에서 chunk 경로와 비트-동일, 재사용에 필요한 256-pad 뷰가 flash-attn fp 누적 순서만 교란 → perplexity 게이트 필요, 단독 이득 ~0 t/s)입니다. 정직한 결론: **DSV4는 이 quant에서 GB10의 실용적 컴퓨트 한계에 있고, 현실적 최적화 여지는 한 자릿수 %이며 그것도 matmul이 아닌 ~19% 데이터 셔플에 있습니다.**

### v1.7.0 — TriAttention 통합 + attn_rot_k 중복 회전 제거

**AMX3_1 하이브리드 K 캐시에 TriAttention 토큰 가지치기 — dequant-free pre-RoPE polar 스코어링 + 물리 eviction. 전 TBQ/TBQP/AMX 인코더의 외부 attn_rot_k 의존 제거 (중복 Hadamard 제거).**

> ⚠️ **Breaking change — TBQX3_1 폐기.** v1.6.0의 polar-only 3.625 bpw 포맷 (TBQX3_1 / `tbqx3`)은 **v1.7.0에서 삭제**됩니다. 그 polar `(r, φ)` 아이디어는 이제 AMX3_1의 Part B로 흡수되었고, Part A는 WHT로 FA attention을 담당합니다 — 즉 v1.6.0에서 품질 향상에 기여했던 "polar" 경로는 그대로 살아있지만, 독립 K 캐시 포맷으로서의 TBQX3_1은 사라집니다. `--cache-type-k tbqx3` 사용하던 스크립트는 `--cache-type-k amx3`로 교체해주세요.

**환경:** NVIDIA DGX Spark (GB10, 128GB) · CUDA 12.8 · Qwen3-14B Q4_K_M · ctx=40960 · temp=0.

#### 압축률 스토리 (2.37배로 보이지만 실제로는 더)

| 축 | Before | After | 이득 |
|------|--------|-------|------|
| **원시 블록 크기** (128 원소) | f16 → 256 B | AMX3_1 → 108 B | **2.37×** |
| **슬롯당 살아있는 토큰** (budget=128, 50% retention) | 2500 전부 생존 | ~128 alive, 2372 evicted | **~2×** |
| **실효 토큰 압축률** | 1× | — | **~4.74×** (동급 attention 품질 기준) |

AMX3_1 단독으로도 이미 2.37× 패킹 이득입니다. TriAttention은 **별도의 두 번째 축** — "레이어별로 매 attention 스텝에 실제로 기여하는 슬롯은 ~128개뿐"이라고 보는 거라, 나머지 ~95%의 할당된 KV 캐시는 추론 시점에는 죽은 무게로 취급 가능. 두 축을 곱하면 f16 ctx=N 품질 목표가 대략 `N/4.7` 메모리에 들어감. 물리 할당량은 `-c` 값 그대로 — 줄어드는 건 매 스텝 attention이 건드려야 하는 양입니다.

#### TriAttention — 새 CLI 플래그 (`-ctk amx3` K 캐시 필요)

| 플래그 | 기본값 | 의미 |
|------|---------|---------|
| `--triattention FILE` | — | TRIA v2 칼리브레이션 파일 (`calibrate_ref.py` 출력) |
| `--tri-budget N` | 0 (off) | 레이어별 Top-B 유지 슬롯 수 |
| `--tri-interval N` | 128 | N 토큰마다 스코어링 트리거 (논문 β) |
| `--tri-keep-first N` | 4 | Attention sink — 앞 N 슬롯 무조건 유지 (프롬프트 헤더 보호용 32 추천) |

#### AMX3_1 하이브리드 K 블록 (108 B, 6.75 bpw raw)

| Part | 크기 | 용도 |
|------|------|---------|
| Part A (WHT) | 50 B (`d_wht` + `qs[48]`) | FA 디코드 — tbq3_1 동치 |
| Part B (polar) | 58 B (`d_r` + `qr[24]` + `qphi[32]`) | TriAttention 스코어링 — pre-RoPE `(r, φ)` |

Part B는 스코어링 커널이 GPU에서 직접 소비 — dequant-and-copy 없음, CPU 왕복 없음.

#### Needle-in-haystack 회수 (Qwen3-14B · budget=128 · interval=128)

프롬프트: "비밀번호는 **다람쥐7429** 이다. [1000자 요약 요청] … 마지막에 비밀번호 정확히 적어라."

| 설정 | 생성 토큰 | 종료 | 비밀번호 회수 | 관찰 동작 |
|--------|------------------|--------|-----------------|-------------------|
| TriAttention 수정 전 | 2500 | length | ❌ 언급 없음 | "다름, 다름, 다름…" 무한 반복 |
| keep_first=4 | 575 | stop | ❌ "다섯" 로 환각 | 첫 음절 부분 환각 |
| **keep_first=32** | **366** | **stop** | **✅ "다람쥐7429"** | 정답 회수 + 자연 종료 |

#### head_dim별 한국어 + 스코어링 커널 검증

네 경우 모두 시작 로그에 `attn_rot_k = 0 / attn_rot_v = 0` 확인 (외부 Hadamard 완전 비활성; 내부 WHT가 회전 담당):

| head_dim | 모델 (gguf) | 해석된 KV 타입 | 한국어 8행 시 | 스코어링 파이프라인 |
|---|---|---|---|---|
| 64 | gpt-oss-20b-MXFP4 | `tbqp3_3` (double WHT) | ✅ 자연스러움 | 해당 없음 (TriAttention은 head_dim=128 전용) |
| **128** | **Qwen3-14B Q4_K_M** | **`tbq3_1` / `amx3_1`** | **✅ + needle 성공** | ✅ 3-kernel + eviction + sink 검증 |
| 256 | Qwen3.5-9B Q8_0 | `tbq3_0` | ✅ 자연스러움 | 해당 없음 |
| 576 (MLA) | GLM-4.7-Flash | `tbq3_4` (512-WHT + rope 64 passthrough) | ✅ 자연스러움 | 해당 없음 |

#### 성능 (Qwen3-14B Q4_K_M, AMX3_1 하이브리드)

| 지표 | 값 | f16 기준선 대비 |
|--------|-------|-----------------|
| Prompt 처리량 | 694 tok/s | ~700 (≈동등) |
| Decode 처리량 | **20.7 tok/s** | ~21 (≈동등) |
| 트리거당 오버헤드 | ~120 ms (40 레이어 × 3 ms) | 128-토큰 decode window의 ≈2% |
| 실효 kept / n_kv | ~130 / 2500 (95% evicted) | — |

> 매 트리거마다 슬롯의 95%를 물리적으로 0 처리하는 상태에서도 품질 유지 — TriAttention 논문 주장을 로컬에서 재현.

#### 핵심 설계 결정 (TriAttention이 여기서 실제로 동작하는 이유)

1. **Dequant-free 스코어링.** Part B가 스코어링 공식이 필요로 하는 pre-RoPE `(kxc, kyc)` 페어로 직접 복원됨. 그림자 fp16 K 버퍼 없음, D2H 복사 없음 — 3-kernel 파이프라인 (raw · z-norm · aggregate)이 양자화 블록을 제자리에서 읽음.
2. **영구성 있는 물리 eviction.** Evict된 슬롯은 `d_wht` (Part A → FA attention → 0)와 `d_r` (Part B → 다음 스코어링 → 0) **둘 다** zero. d_wht만 zero하면 "유령" 슬롯이 다음 트리거 Top-B에 재선택되어 budget을 조용히 까먹음; 둘 다 zero해야 eviction이 영구적.
3. **Attention sink는 필수, 선택 아님.** `--tri-keep-first 0`으로 돌리면 decode가 ~300 토큰 내에 "다름, 다름…" 토큰 반복 루프로 붕괴. Sink는 StreamingLLM 스타일 해결책: softmax가 잔여 확률 질량을 버릴 곳이 있도록 앞 N 슬롯을 강제 유지. `4`는 chat template 커버, `32`는 주입된 지시사항 포함 프롬프트 헤더 전체를 커버.
4. **외부 attn_rot_k는 중복 회전이었음.** 모든 TBQ/TBQP/AMX 인코더는 이미 내부 `tbq_signs` 부호 뒤집기 + butterfly WHT를 실행; fattn-vec도 Q 쪽에 동일한 WHT 적용. llama-kv-cache가 인코더 전에 적용하던 Hadamard matmul은 중복 두 번째 회전 — Parseval에 의해 Q와 K 사이에서 상쇄되므로 출력은 변하지 않지만, 매 토큰마다 추가 matmul 1회 + 약간의 반올림 손실을 지불. v1.7.0에서 내부-WHT 타입 23종 전부를 `attn_rot_k`에서 제외; head_dim = 64 / 128 / 256 / 576에서 한국어 산문 품질 유지 확인.

#### 권장 설정

Qwen3-14B (head_dim=128, AMX3_1 지원):
```
--cache-type-k amx3 --cache-type-v amxv3 \
--triattention calib/qwen3_14b.bin --tri-budget 128 --tri-keep-first 32
```

다른 head_dim (64/256/576) — TriAttention 미지원, attn_rot_k 정리는 자동 적용:
```
--cache-type-k tbq3 --cache-type-v tbq3     # head_dim에 따라 _2/_0/_4 자동 해석
```

> ⚠️ **범위 — AMX3_1은 head_dim=128 필요.** Qwen3.6-35B (K=256 비대칭), GLM MLA 576, gpt-oss 계열은 아키텍처별 AMX 변종이 나오기 전까지 **TriAttention 후보 아님**. `attn_rot_k` 정리는 head_dim 무관하게 전역 적용.

#### 크레딧

- **TriAttention 알고리즘** — Mao et al., *"Tri-attention: Tail-token saliency via trigonometric scoring on pre-RoPE keys"*, [arxiv 2604.04921](https://arxiv.org/abs/2604.04921) (2026).
- **Python 레퍼런스 포트** — [`domvox/triattention-ggml`](https://github.com/domvox/triattention-ggml). TRIA v2 바이너리 칼리브레이션 포맷, 스코어링 수식, 헤드별 통계 추출을 이 레퍼런스에서 이식. CUDA 포트, AMX3_1 하이브리드 K 캐시 통합, llama-kv-cache / llama-context / fattn-vec 쪽 물리 eviction + attention-sink 배선은 이번 릴리즈에서 신규.

#### 약간의 비하인드

v1.6.0 초기 작업 때 저희는 "polar derotation"을 독자적으로 스케치했습니다 — K를 `(r, φ_content)`로 저장하고 위치를 대수적으로 벗겨내는 방식, 나름대로 우리만의 작은 아이디어라 생각했죠. 배포도 했고 논문도 썼고, 이론적 프레이밍이 완전히 맞지 않는다는 걸 깨닫고 논문은 철회했습니다 (`eab1d2ad1` — RoPE 하에서 pair 구조 + content-only WHT + 정확한 Q·K 보존이 동시에 성립할 수 없음). 그러다 TriAttention 논문을 읽다가 발견했죠 — 우리가 고안한 그 polar 분해가 이미 누군가의 score 공식 안에 들어있었다는 것을. 용도는 완전히 달랐습니다: 저장이 아니라 토큰 중요도 측정. 겹침이 알고 보니 선물이었습니다 — AMX3_1 하이브리드 블록이 TriAttention 스코어링 수식에 거의 그대로 맞아 떨어졌거든요, 왜냐하면 Part B가 **바로** 그 논문이 원하던 pre-RoPE polar 페어였으니까요. 독립 재발견은 겸손해지는 일이지만, 덕분에 이 통합 작업이 원래 들 시간보다 훨씬 짧아졌습니다.

솔직하게 짚고 넘어갈 부분도 있습니다: domvox Python 레퍼런스는 스코어링 트리거마다 GPU→CPU 왕복을 치릅니다 — K를 dequantize해서 호스트로 끌어내리고, Python으로 스코어 계산 후 마스크를 다시 올리는 구조. β=128, 40 레이어 기준으로 이게 14B 모델에서 지배적 지연 요인이 됩니다. 저희 CUDA 포트는 이 병목을 완전히 제거했습니다: Part B가 스코어링 수식이 요구하는 레이아웃 그대로이기 때문에 3개의 스코어링 커널이 양자화 블록을 제자리에서 읽고, 히스토그램 Top-B와 eviction 커널이 같은 스트림에서 실행되며, 트리거 중 PCIe를 건너는 건 수 KB 포지션 배열 하나뿐입니다. 트리거 오버헤드가 128-토큰 decode window의 ~2%로 떨어짐 (호스트 왕복이었다면 트리거당 몇 초). 대신 대가가 있습니다: AMX3_1은 Part A (attention용 WHT)와 Part B (스코어링용 polar)를 나란히 저장하기 때문에 블록이 108 B — tbq3_1 단독 50 B의 두 배가 조금 넘습니다. head_dim=128에서 같은 슬롯 수 기준 원시 K 캐시 풋프린트가 2배 넘게 커진다는 뜻입니다. TriAttention이 트리거당 95% 슬롯을 evict해서 메모리를 돌려주긴 하지만 (실제 *살아있는* working set은 tbq3_1보다 작아짐) **할당된** KV 버퍼 자체는 더 큽니다. 속도 병목 대신 바이트를 지불한 거죠 — 규모에서는 맞는 거래지만 솔직하게 명시할 가치가 있습니다.

---

### v1.6.0 — Polar Derotate + Tangent Residual (TBQX3_1, Qwen3-14B) — 폐기됨, v1.7.0 참조

**새 K 캐시 포맷: polar 좌표 저장 + content/position 분리 + 해석적 tangent residual 보정. 수학 추론에서 f16 를 압도하면서 한국어 산문 품질 유지.**

**환경:** NVIDIA DGX Spark (GB10, 128GB) · CUDA 12.8 · 모델: Qwen3-14B Q4_K_M · ctx=40960 · temp=0

**메모리 사용량 (ctx=40960, 40 layers):**

| 설정 | K 버퍼 | V 버퍼 | 총 KV | 압축률 |
|------|--------|--------|-------|--------|
| f16/f16 | 3200 MiB | 3200 MiB | 6400 MiB | 1.0x |
| **tbqx3/tbq3** | **725 MiB** | 625 MiB | **1350 MiB** | **4.74x** |
| tbq3/tbq3 | 625 MiB | 625 MiB | 1250 MiB | 5.12x |

**속도 (decode, 동일 프롬프트):**

| 설정 | t/s | vs f16 |
|------|-----|--------|
| f16/f16 | 24–25 | 1.00x |
| **tbqx3/tbq3** | **21–22** | **0.87x** |
| tbq3/tbq3 | 21–22 | 0.87x |

**수학 정확도 (35문제, seed=1234, temp=0):**

| 설정 | Math (/35) | % | vs f16 |
|------|------------|---|--------|
| **tbqx3/tbq3** | **13/35** | **37.1%** | **+8%** |
| f16/f16 | 12/35 | 34.3% | — |
| tbq3/tbq3 | 10/35 | 28.6% | −17% |

> **tbqx3/tbq3 는 4.74배 압축하면서 수학에서 f16 를 이김**. 기존 tbq3/tbq3 는 f16 대비 17% 열세.

**TBQX3_1 블록 포맷 (3.625 bpw, head_dim=128):**

| 필드 | 크기 | 용도 |
|------|------|------|
| `d_r` | 2 B | Rayleigh σ (half) |
| `qr[24]` | 24 B | 3비트 r 인덱스 (Rayleigh Lloyd-Max, 64 페어) |
| `qphi[24]` | 24 B | 3비트 φ_content 인덱스 (uniform, 64 페어) |
| `qtan[8]` | 8 B | 페어당 1비트 접선 부호 |
| **총** | **58 B** | **3.625 bpw** |

**핵심 아이디어:**

1. **Polar derotate (content/position 분리)**: K 를 `(r, φ_content) = (r, φ_post − pos·freq_i)` 로 페어당 저장. content 는 position-invariant. 읽기 시점에 `pos·freq_i` 로 re-rotation. Attention 이 content 기하를 직접 봄 (content·position 얽힘 없이).
2. **r 에 Rayleigh Lloyd-Max 적용**: 복소 가우시안 페어의 크기 `r = √(x² + y²)` 는 Rayleigh 분포. Rayleigh quantile 경계로부터 8-레벨 Lloyd-Max 코드북 해석적으로 도출 (캘리브레이션 불필요, 모델 독립적).
3. **Tangent Residual (drift 픽스)**: 3비트 uniform φ 양자화 오차 ±π/8. 이로 인한 K perturbation 의 거의 전부가 접선 방향 `(-sin φ, cos φ)` 에 위치. 1비트로 `Δφ` 부호 인코딩, 크기 `r · π/16` 은 해석적 (half-cell). 학습 無, 이미 계산된 sin/cos 재활용 — 페어당 FMA 2개 추가로 끝. φ 오차 절반 (22.5° → 11.25°), 희소 토큰 drift (키릴 문자 오염) 제거.
4. **Content 경로에 WHT 없음**: polar 구조 자체가 각도 정보를 보존. 페어 간 WHT 는 RoPE 페어 구조를 파괴. TBQX3_1 은 WHT 를 완전히 생략한 최초의 TBQ 변종.

**Qwen3-14B 및 기타 head_dim=128 RoPE 모델 권장 설정:**
```
--cache-type-k tbqx3 --cache-type-v tbq3
```

> ⚠️ **알려진 한계 — VEC 커널만 구현됨.** TBQX3_1 은 현재 `fattn-vec` 경로에만 구현되어 있음. 그 결과 Qwen3-14B 디코드 처리량이 f16 의 약 0.87× 수준에 머무름. `fattn-mma` (Tensor Core) 커널로 이식하면 full 속도가 회복되지만 **아직 구현 전** — 다음 릴리즈의 최대 미결 과제.

---

### v1.5.3 — Double WHT Per-Head for head_dim=64 (GPT-OSS 120B)

**Cross-head WHT 폐기. Double WHT per-head (S1→WHT64→S2→WHT64)로 교체. QJL 1비트 보정 재활성화 — 멀티턴 안정성에 필수.**

**환경:** NVIDIA DGX Spark (GB10, 128GB) · CUDA 12.8 · 모델: GPT-OSS 120B (MXFP4)

**수학 정확도 (35문제, temp=0):**

| 설정 | K 캐시 | V 캐시 | 수학 (/35) | 한국어 | 멀티턴 |
|------|--------|--------|-----------|--------|--------|
| f16/f16 | f16 | f16 | **35/35** | ✅ | ✅ |
| **tbq4/tbq3** | tbq4_2 | tbq3_2 | **35/35** | ✅ | ✅ |
| tbqp3/tbq3 | tbqp3_3 | tbq3_2 | ❌ (행렬) | ✅ | ✅ (9턴+) |

> **권장:** head_dim=64 모델에서는 `--cache-type-k tbq4 --cache-type-v tbq3` 사용.
> 4비트 K가 f16 동급 수학 정확도(35/35) 달성. 3비트 V 압축과 조합.
> 3비트 K(tbqp3)는 한국어 대화와 멀티턴은 지원하나 행렬 연산 정확도 부족.

**v1.5.3 핵심 변경:**

1. **Double WHT per-head (D=64)**: Cross-head WHT(512-point, $H_8 \otimes H_{64}$) 폐기 — Q-K 도메인 불일치. S1→WHT64→S2→WHT64 double WHT per-head로 교체. 첨도(kurtosis) 0.375→0.047 (근사 가우시안).
2. **D=64 K에 QJL 재활성화**: v1.5.2에서 제거했던 QJL을 복원. 1비트 QJL 보정이 멀티턴 안정성에 필수. QJL 없으면 3-4턴에서 반복 루프. QJL 있으면(TBQP3_3) 9턴+ 검증.
3. **TBQ_TUNING D=64 인스턴스**: D=64 모든 K/V 조합을 TBQ_TUNING 빌드에 추가 (tbqp3_3, tbq3_3, tbq4_2, f16, q8_0 × tbq3_2/f16).

---

### v1.5.2 — PPL 21%→6%, 정밀도 수정, 결정적 커널

**Flash attention 커널의 치명적 정밀도 손실 수정. 3비트 KV 캐시가 결정적(deterministic)으로 동작하며 f16 대비 PPL 1.06배 달성.**

**환경:** NVIDIA DGX Spark (GB10, 128GB VRAM) · CUDA 12.8 · 모델: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL

**메모리 & 압축률 (262K ctx, Gemma 4 26B MoE):**

| 설정 | Global KV | SWA KV | 전체 KV | 압축률 |
|------|-----------|--------|---------|--------|
| f16/f16 | 5,120 MiB | 300 MiB (f16) | 5,420 MiB | 1.0x |
| **tbqp3/tbq3** | 990 MiB (K:500 + V:490) | 300 MiB (f16) | **1,290 MiB** | **4.2x** |

**PPL 벤치마크 (wikitext-2-raw, ctx=2048):**

| 설정 | PPL | 대 f16 |
|------|-----|--------|
| f16/f16 | 419.8 | 1.00x |
| **tbqp3/tbq3** | **445.7** | **1.06x** |

**수학 정확도 (262K ctx, temp=0, 35문제 × 10회):**

| 설정 | 10회 (/35) | 평균 | 최고 |
|------|-----------|------|------|
| **tbqp3/tbq3** | 19,23,18,22,18,20,19,16,19,17 | **19.1** | **23** |
| f16/f16 | 19,20,21,20,20,21,21,19,20,20 | 20.1 | 21 |

> **4.2배 압축, PPL 6% 차이, f16 동급 수학.** 최고점 23은 f16 최고 21을 초과.

**핵심 변경:**

1. **Flash Attention 정밀도 수정 (half2→float)**: upstream fattn-vec이 half2(V_DOT2)를 VKQ 누적 및 KQ 공유 메모리에 사용. 이 정밀도 손실이 512-point IWHT 버터플라이에 의해 증폭되어 비결정적 MoE expert 라우팅 유발. 70개 TBQ V 템플릿 인스턴스에 float 누적 경로 강제(`#undef V_DOT2_F32_F16_AVAILABLE`). **PPL: 454.7 → 445.7 (1.08x → 1.06x).**
2. **V IWHT Float Staging**: reduce 단계에서 `__shared__ half KQ[]`에 half 값을 썼지만, IWHT가 `float*`로 읽음 — 타입 불일치가 버터플라이에 의해 증폭. float 레지스터 staging + `__syncthreads()` 배리어로 수정.
3. **동적 Attention Sharpening**: α(N) = 1 + c × √(ln N), c는 MMSE/EVT 이론에서 도출. 컨텍스트 크기에 적응: 생성 시 작은 α, 긴 컨텍스트 prefill 시 큰 α. 클램프 불필요.
4. **V Rotation 버그 수정 (attn_rot_v=0)**: attn_rot이 V에도 적용되었으나 IWHT 디코드에는 역 rotation이 없음. K rotation은 안전 (Q·K 내적에서 상쇄).
5. **Per-block Norm (TBQ3 D=512)**: 512-WHT 후 256-half 단위 독립 norm. TBQP3은 글로벌 norm 유지 (QJL이 cross-block WHT 사용).
6. **1.15x V 핵 제거**: 원칙적 attention sharpening으로 대체.
7. **tbq4_0 D=512 OOB 읽기 수정**.
8. **tbq4/tbqp4 D=512 WHT 도메인 불일치 수정**: K 인코딩이 256-point WHT를 사용하는데 Q는 512-point WHT 사용. 512-point 인코드 함수 추가.
9. **Double WHT per-head (head_dim=64)**: Cross-head 512-point WHT 폐기 (Q-K 도메인 불일치). S1→WHT64→S2→WHT64 double WHT per-head로 교체. 첨도 0.375→0.047 (근사 가우시안). Q와 K가 같은 도메인 — IWHT 불필요.
10. **head_dim=64 K: TBQP3_3 (QJL 활성화)**: QJL 1비트 보정이 멀티턴 안정성에 필수 (7턴+ 검증). K가 자동으로 TBQP3_3에 매핑 (2비트 Lloyd-Max + 1비트 QJL + double WHT).
11. **동적 MMSE softening (head_dim=64)**: α(N) = SQNR/(SQNR + √(ln N/ln N₀)), sharpening의 반대 — SQNR이 낮을 때 과신을 줄임.

### Attention Sharpening — 이론 & 동적 α 공식

K의 양자화 노이즈가 attention 점수에 분산을 추가하여 softmax 분포를 평탄화합니다. sharpening 계수 α가 이를 보상합니다:

**동적 (현재 구현):**
```
α(N) = 1 + c × √(ln N)
c = 1/(2 × SQNR_eff × √(ln N₀))
```
N = 현재 KV 토큰 수 (런타임), N₀ = 2048 (기준 컨텍스트 크기).

√(ln N) 항은 **극값 이론(Extreme Value Theory)**에서 유래 — N개 경쟁 토큰 중 최대 노이즈의 기대값이 √(2 ln N)으로 증가하여, 잘못된 토큰이 softmax 질량을 훔칠 확률을 높입니다. 공식이 자연적으로 적응: 자기회귀 생성(작은 N) 시 작은 α, 긴 컨텍스트 prefill(큰 N) 시 큰 α. 클램프 불필요.

**TBQ3과 TBQP3의 α가 다른 이유:**

| 타입 | 구조 | Attention 점수 노이즈 | 유효 SQNR | α 계수 |
|------|------|---------------------|-----------|--------|
| **TBQ3** | 3비트 Lloyd-Max (8레벨) | 깨끗한 요소별 노이즈만 | 31.2 | 0.016 |
| **TBQP3** | 2비트 MSE + 1비트 QJL | MSE 노이즈 + QJL 랜덤 프로젝션 노이즈 | 13.8 | 0.036 |
| **TBQ4** | 4비트 Lloyd-Max (16레벨) | 최소 노이즈 | 56.2 | 0.009 |
| **TBQP4** | 3비트 MSE + 1비트 QJL | MSE 노이즈 + QJL 프로젝션 노이즈 | 24.5 | 0.020 |

TBQ3이 요소별 복원 오차가 더 낮지만(8레벨), TBQP3의 QJL 보정은 attention 점수에 **두 번째 노이즈원**을 추가합니다: 1비트 랜덤 프로젝션 `d_qjl × Σ(Q_wht2[i] × sign[i])`. QJL이 K 복원 MSE를 줄이지만(V 용도에 좋음), attention 점수 분산을 증가시킵니다(softmax에 나쁨). 이것이 **TBQP3이 더 나은 복원 품질에도 불구하고 TBQ3보다 강한 sharpening이 필요한 이유**입니다.

**컨텍스트 크기별 동적 α:**

| ctx N | TBQ3 | TBQP3 | TBQ4 | TBQP4 |
|-------|------|-------|------|-------|
| 256 | 1.012 | 1.027 | 1.007 | 1.015 |
| 512 | 1.013 | 1.030 | 1.007 | 1.017 |
| 1024 | 1.015 | 1.033 | 1.008 | 1.018 |
| **2048** | **1.016** | **1.036** | **1.009** | **1.020** |
| 4096 | 1.017 | 1.038 | 1.009 | 1.021 |
| 8192 | 1.018 | 1.041 | 1.010 | 1.023 |
| 32768 | 1.019 | 1.044 | 1.011 | 1.025 |
| 131072 | 1.020 | 1.047 | 1.011 | 1.026 |
| **262144** | **1.021** | **1.048** | **1.012** | **1.027** |

범위가 좁습니다(1.007–1.048). √(ln N)이 매우 느리게 증가하기 때문입니다. 128배 컨텍스트 증가(2048→262144)에도 α 변화는 ~0.012뿐. 고정 상수를 실용적 1차 근사로 사용할 수 있음을 검증합니다.

```bash
# 권장 설정 (f16 동급, 4.2배 압축)
./llama-server -m ~/Models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 1 \
    --cont-batching --jinja \
    --reasoning off --reasoning-budget 0 --reasoning-format none \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k tbqp3 --cache-type-v tbq3 \
    --temp 0 --host 0.0.0.0 --port 8888
```
> SWA K+V는 자동으로 f16으로 업그레이드됩니다. 추가 설정 불필요.

**지원 K/V 조합:**

약어(`tbq3`, `tbqp3` 등)를 사용하세요 — 내부 접미사(`_0`, `_1` 등)는 head dimension에 따라 자동 매핑됩니다.

| `--cache-type-k` \ `--cache-type-v` | f16 | q8_0 | q4_0 | tbq3 | tbq4 |
|--------------------------------------|-----|------|------|------|------|
| **f16** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **q8_0** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **q4_0** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **tbq3** | ✅ | ✅ | — | ✅ | ✅ |
| **tbq4** | ✅ | ✅ | — | ✅ | ✅ |
| **tbqp3** | ✅ | ✅ | — | ✅ | ✅ |
| **tbqp4** | ✅ | ✅ | — | ✅ | ✅ |

> **`tbqp` 타입은 K 전용입니다.** `--cache-type-v`에 `tbqp3`/`tbqp4`를 사용하지 마세요 — QJL 보정은 Q·K 내적에 작동하며 V 캐시에는 의미가 없습니다. 권장: `--cache-type-k tbqp3 --cache-type-v tbq3`.

---

### v1.5.1 — f16 동급 달성: SWA f16 Bypass + V 512-WHT + QJL D=512 복원

**3비트 KV 캐시(tbq3/tbq3)로 f16을 넘어서는 품질 달성. SWA f16 bypass가 핵심 브레이크스루.**

**벤치마크 (Gemma 4 26B-A4B-it MoE, UD-Q4_K_XL, DGX Spark GB10, 262K ctx, temp=0):**

| 설정 | K 캐시 | V 캐시 | attn_rot_k | Global KV | SWA KV | 수학 정확도 (10회) | 평균 | 압축 |
|------|--------|--------|-----------|-----------|--------|------------------|------|------|
| **tbqp3/tbq3** | tbqp3 | tbq3 | OFF(자동) | 990 MiB | 300 MiB(f16) | 37,38,40,38,38,36,37,36,37,37 | **37.4** | **4.2x** |
| **tbq3/tbq3** | tbq3 | tbq3 | ON | 980 MiB | 300 MiB(f16) | 39,39,37,37,38,35,35,39,35,36 | **37.0** | **4.2x** |
| f16/f16 | f16 | f16 | OFF | 5120 MiB | 300 MiB(f16) | 37,36,36,36,36,38,36,38,37,36 | 36.6 | 1.0x |

> **tbqp3/tbq3이 f16을 초과 (37.4 > 36.6).** 최고점 40/65는 f16 최고(38)보다 높습니다.

**핵심 기술:**

1. **SWA KV f16 Bypass**: SWA 캐시는 작지만(~300 MiB) 25개 레이어로 전체 품질을 지배. SWA K+V를 자동 f16 업그레이드하여 SWA 양자화 노이즈 완전 제거 — 이전 모든 최적화 시도를 가린 숨은 원흉.
2. **V 512-WHT + 512-IWHT**: V 캐시에 K와 동일한 512-point WHT 적용 (부호 반전 + 9단계 버터플라이 + 글로벌 norm). Attention 출력 시 512-point IWHT (128 쓰레드 × 4 요소, warp shuffle + 공유 메모리 버터플라이).
3. **QJL D=512 복원**: 이전에 "D=512에서 비효과적"으로 제거됨 — SWA 노이즈가 QJL 개선을 가리고 있었음. SWA f16 bypass와 함께 tbqp3 K가 tbq3 K보다 우수 (37.4 > 37.0). TBQP에서 attn_rot 자동 비활성화 (삼중 rotation 방지).

---

### v1.5.0 — Upstream Rebase + Gemma 4 지원

**최신 upstream llama.cpp(b7ad48ebd)에 완전 rebase + Gemma 4 TBQ KV 캐시 지원.**

기존에는 upstream llama.cpp와 별도 히스토리로 관리하여 동기화마다 수백 커밋을 수동 패치해야 했습니다. v1.5.0부터 포크가 upstream 커밋 히스토리를 공유하여 `git merge upstream/master` 한 번으로 동기화 가능.

**변경 사항:**
- 모든 TBQ/TBQP 코드를 최신 upstream llama.cpp(b7ad48ebd)에 3-way 머지
- **Gemma 4 지원**: hybrid head_dim=512(global) + 256(SWA) 아키텍처 완전 지원
- Deepseek MLA 512x512 MMA 설정 포함 (upstream 추가분)
- bf16 flash attention vec 커널 지원 (upstream 추가분)
- V-less 캐시, stream-k FA 등 upstream 최적화 포함
- 향후 upstream 동기화를 위한 적절한 포크 구조 확립

**Gemma 4 핵심 기술:**
1. **SWA 캐시 타입 자동 리매핑**: global(head_dim=512)과 SWA(head_dim=256)가 다를 때, SWA 캐시에 올바른 TBQ 서브타입 자동 할당
2. **가변 GQA 지원**: Gemma 4는 레이어별 head_count_kv(16/4)가 다름. WHT rotation은 헤드별 동작이므로 가변 GQA 안전 — `attn_rot_k` 활성화 조건 업데이트
3. **D=512 단일 패스 WHT + 글로벌 norm**: K 양자화에 512-point WHT 한 패스 적용(9단계 버터플라이). 두 256-블록이 같은 글로벌 norm 공유. Q 전처리도 512-point WHT 사용(128 쓰레드 × 4 요소). V IWHT는 256-블록 독립 처리(V는 블록별 norm)
4. **op_params를 통한 head_dim**: head_dim을 `op_params[0]`으로 set_rows 커널에 전달하여 D=512와 D=256 정확히 구분

**벤치마크 (Gemma 4 31B-it Dense, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| 캐시 | GPU 메모리 | 압축 | PP t/s | TG t/s | PPL (wiki, 2K) | 수학 정확도 | 파울리 |
|------|-----------|------|--------|--------|----------------|------------|--------|
| f16/f16 | 41,500 MiB | 1.0x | 152.9 | 10.1 | 309.7 | 42/65 (64.6%) | PASS |
| **tbq3/tbq3** | **23,215 MiB** | **1.8x** | 112.9 | 9.0 | 212.1 | **32/65 (49.2%)** | **PASS** |

**벤치마크 (Gemma 4 26B-A4B MoE, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| 캐시 | GPU 메모리 | 압축 | TG t/s | 수학 정확도 | 파울리 |
|------|-----------|------|--------|------------|--------|
| f16/f16 | 5,720 MiB | 1.0x | 56.4 | 37/65 (56.9%) | PASS |
| **tbq3/tbq3** | **1,106 MiB** | **5.2x** | 41.1 | **30/65 (46.2%)** | **PASS** |

> **참고:** Gemma 4는 hybrid SWA 아키텍처로, non-SWA 10 layers(head_dim=512) + SWA 50 layers(head_dim=256)입니다. SWA 캐시는 sliding window 크기(1536 cells)로 제한되어, Dense 모델의 압축률(1.8x)은 MoE 모델(5.2x)보다 낮습니다.
>
> **참고:** 수학 정확도는 [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py) 벤치마크(2x2/3x3 행렬곱, 스칼라 연산, filler 0-2000t)로 측정. 65개 테스트 중 f16과 TBQ는 **55개(31B Dense) / 58개(26B MoE)에서 동일한 결과**. 차이가 발생한 케이스도 작은 수치 오차(예: 160→150, 519→509)로 garbage 출력이 아닙니다. 파울리 테스트(과학자 이름 한국어 번역)는 정상 통과.
>
> **⚠️ D=512 제한:** Gemma 4의 head_dim=512는 TurboQuant 논문의 검증 범위(head_dim=128)를 초과합니다. QJL 1비트 보정(TBQP)은 D=512에서 작동하지 않음 — **TBQ(MSE 전용)만 지원**. 8가지 QJL 변형 테스트(글로벌/블록별 norm, L2/RMS 감마, 상관/독립 부호 패턴) 모두 품질 저하. TBQP(QJL)는 head_dim≤256 모델에서 정상 동작.
>
> **✅ v1.5.1 업데이트:** D=512 QJL 제한은 v1.5.1에서 해결되었습니다. SWA f16 bypass와 함께 사용하면 TBQP3(QJL) D=512가 정상 동작하며 tbq3보다 우수합니다. 자세한 내용은 위의 v1.5.1 섹션을 참조하세요.

<details>
<summary>빌드 및 실행 방법</summary>

**빌드:**
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON \
      -DGGML_BLAS=ON \
      -DGGML_CCACHE=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-lpthread -lm" \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      ..

make -j12
```

**서버 실행 (f16 baseline):**
```bash
./llama-server -m ~/Models/gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 2 \
    --cont-batching --jinja --reasoning-format auto \
    --chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "medium"}' \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k f16 --cache-type-v f16 \
    --top-k 20 --temp 0.6 --top-p 0.95 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port 8888
```

**서버 실행 (TurboQuant):**
```bash
./llama-server -m ~/Models/gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 2 \
    --cont-batching --jinja --reasoning-format auto \
    --chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "medium"}' \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k tbq3 --cache-type-v tbq3 \
    --top-k 20 --temp 0.6 --top-p 0.95 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port 8889
```
</details>

**호환성:** v1.4.2의 모든 기능(MMA 텐서 코어, QJL 스칼라 보정, MLA 비대칭) 완전 보존. 기존 모델 벤치마크 변동 없음.

---

### v1.4.2 — MMA Tensor Core 가속 + QJL Scalar Correction

**TBQ/TBQP MMA 텐서 코어 가속: TG 속도 30→49 t/s (+63%).**

GLM-4.7-Flash(MLA, K=576/V=512) 비대칭 모델용 MMA 텐서 코어 attention 가속. vec 커널 대비 최대 1.6배 TG 속도 향상.

**핵심 기능:**

1. **MMA K spatial dequant**: raw TBQ/TBQP 블록 → IWHT → spatial f16 → 텐서 코어. 워프 셔플 최적화로 cooperative IWHT `__syncthreads` 8→4 감소.
2. **QJL scalar correction**: TBQP 1비트 QJL 보정을 전체 MMA 패스 대신 경량 스칼라 연산으로 처리. raw K 블록에서 부호 비트 + dq 직접 읽기. QJL의 두 번째 부호 기저(signs2) 정확히 처리.
3. **V = K view spatial**: K를 spatial 도메인으로 디양자화하면 V(= K 뷰)도 자동 spatial. 출력 IWHT 완전 제거.
4. **적절한 커널 시그니처**: `fattn_kernel_t`에 `raw_K_data`, `raw_K_stride`, `Q_wht2_data`, `Q_wht2_stride` 확장. 포인터 핵 없음.
5. **Fused Q WHT12**: Q->data를 직접 읽어 Q_wht1(중간값) + Q_wht2(QJL용) 계산. cudaMemcpy 없음.

**벤치마크 (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| 캐시 | KV MiB | 비율 | TG t/s | 대 v1.4.1 | 파울리 |
|------|--------|------|--------|-----------|--------|
| f16/f16 | 10,469 | 1.0x | 67.5 | — | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | **49.7** | **+55%** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | **42.8** | **+36%** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | ~49 | +47% | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | ~42 | +45% | PASS |

> **참고:** TBQP(QJL 포함)가 TBQ보다 느린 이유: QJL이 Q_wht2 사전계산을 위한 두 번째 WHT 변환(signs2) + 토큰별 스칼라 보정 오버헤드가 필요. 트레이드오프는 내적 정확도 개선(더 나은 PPL).

---

### v1.4.1 — GLM-4.7-Flash (MLA) 비대칭 K/V 지원

**MLA 아키텍처 모델 완전 TurboQuant 지원: GLM-4.7-Flash, DeepSeek-V2/V3.**

MLA(Multi-head Latent Attention)는 비대칭 K=concat(latent[512], rope[64])=576 차원과 V=latent[512] 차원을 가집니다. 세 가지 핵심 기술:

1. **D_V 템플릿 파라미터**: vec 커널이 K/Q 차원(D=576)과 V 차원(D_V=512)을 분리. VKQ 배열, combine stride, IWHT 패스, 출력 쓰기 모두 D_V 사용. 대칭 모델은 기본 D_V=D(영향 없음).
2. **RoPE f16 패스스루**: _4 블록 구조가 서브블록 3(rope 64)을 WHT+양자화 대신 raw f16으로 저장. RoPE norm(~10.49)이 latent norm(~0.13)보다 ~80배 큼 — 양자화 오차가 attention 점수를 지배. Q 전처리와 내적이 서브블록 3을 직접 f16으로 처리.
3. **MLA V-as-K-view**: MLA absorption에서 V는 K 캐시의 뷰(같은 타입). TBQP V 디양자화는 MSE centroid만 사용(QJL은 K·Q 내적 보정용, V 복원용 아님). IWHT는 256+256 투 패스 실행(rope 패스 건너뜀).

**벤치마크 (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| 캐시 | KV MiB | 압축 | PP t/s | TG t/s | PPL | 파울리 |
|------|--------|------|--------|--------|-----|--------|
| f16/f16 | 10,469 | 1.0x | 73.0 | 60.3 | 5.998 | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | 68.2 | 32.0 | **6.836** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | 66.8 | 31.5 | **6.586** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | 67.2 | 33.4 | — | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | 65.8 | 28.9 | — | PASS |

> **참고:** MLA 압축률(3.5x)이 표준 모델(5.2x)보다 낮은 이유: MLA가 이미 KV를 256차원 latent로 압축 — 원래 캐시가 이미 작음. 7.5GB 절약(10,469→2,981 MiB)은 여전히 의미 있음.
>
> **참고:** TG 속도(31.5 대 60.3 t/s): TBQ는 vec 커널(스칼라 WHT 내적) 사용, f16은 MMA 커널(텐서 코어) 사용. TBQ 즉석 디양자화를 MMA에 추가하면 텐서 코어 가속 가능 — 향후 작업.

**수정된 버그:**

| 버그 | 원인 | 수정 |
|------|------|------|
| GLM tbqp3/tbq3 크래시 (v1.4.0) | `get_best_fattn_kernel`이 D=576 TBQ에 NONE 반환 | D=576+V=512 TBQ vec 커널 라우팅 추가 |
| 대칭 dispatch가 비대칭 가림 | `FATTN_VEC_CASE`가 V->ne[0] 미확인 → DV=576 실행 | ASYM 케이스를 대칭 앞에 배치 |
| RoPE 양자화 → garbage | rope norm(~10.49) >> latent norm(~0.13), 양자화 오차가 attention 지배 | 서브블록 3을 f16 패스스루로 변경 |
| TBQP V dequant + QJL → garbage | QJL은 K·Q 내적만 보정, V 복원용 아님 | V dequant에서 QJL 제거(MSE만) |
| Q WHT 서브블록 레이스 컨디션 | 서브블록 저장과 다음 WHT 사이 `__syncthreads` 누락 | `__syncthreads` 배리어 추가 |

---

### v1.3.0 — Bulletproof head_dim 감지 + 치명적 버그 수정

**벤치마크 (Qwen3-30B-A3B Q4_K_M, DGX Spark GB10, head_dim=128):**

| 설정 | PPL | 대 F16 | 비고 |
|------|-----|--------|------|
| f16/f16 | 6.26 | baseline | |
| **tbqp4/tbq4** | **6.70** | **+7.1%** | +Direct Sign 보정 (head_dim=128) |
| tbq4/tbq4 | 6.73 | +7.5% | MSE 전용 |
| **tbqp3/tbq3** | **7.91** | **+26.3%** | +Direct Sign 보정 (head_dim=128) |
| tbq3/tbq3 | 8.49 | +35.6% | MSE 전용 |

> **참고:** TBQP 잔차 보정 방식은 head_dim에 따라 다릅니다:
> - **head_dim=256**: QJL (원 논문, SRHT 기반)
> - **head_dim=128/64**: Direct Sign (QJL 분산 문제에 대한 수정 — 4.3배 낮은 분산)
>
> 위 PPL 수치는 **MoE 모델**(토큰당 3.7B active params) 기준. F16 baseline 6.26은 아키텍처 고유값으로 TurboQuant 문제가 아닙니다.

**해결된 이슈:**

| 이슈 | 보고자 | 상태 |
|------|--------|------|
| Phi-4, DeepSeek 자동 head_dim 감지 실패 | @fritolays | 수정 — P1→P5 캐스케이드 |
| turbo4-K PPL 폭발 (Cydonia-24B에서 18,202) | @TheTom | 수정 — head_dim 오감지 |
| GLM head_dim=576, Qwen3-4B head_dim=80 | @fritolays, @sztlink | 수정 — pow2 검사 + q8_0 폴백 |
| Qwen3.5-27B에서 "////" 출력 | @modderBUG | v1.3.0에서 재현 불가 |
| llama-bench TBQ 타입 미인식 | @sztlink | 수정 — 16 타입 + 4 약어 |
| Windows OpenSSL DLL 의존성 | @sztlink | 수정 — 독립 빌드 |

**기타 개선:**
- llama-bench: -ctk/-ctv를 통한 전체 TBQ/TBQP 타입 지원
- 미지원 head_dim: q8_0으로 폴백하여 압축 유지
- 독립 빌드: 외부 DLL 의존성 없음

<details>
<summary>🔧 head_dim 감지 기술 세부사항 (P1→P5 우선순위 캐스케이드)</summary>

이전에는 GGUF `{arch}.attention.key_length` 메타데이터만 읽었음 — 대부분의 모델(Phi-4, DeepSeek, Gemma, Mistral)이 이를 저장하지 않아 TurboQuant가 조용히 비활성화됨.

현재 6개 감지 신호를 엄격한 우선순위 캐스케이드로 사용:
- **P1**: `attention.key_length` (100% — GGUF 권위)
- **P2**: `attention.key_length_mla` (100% — DeepSeek V2 등 MLA 모델)
- **P3**: `attention.key_length_swa` (100% — Gemma 2/3 등 SWA 모델)
- **P4**: `attention.value_length` (95% — 교차 검증)
- **P5**: `n_embd / n_head` (70% — 폴백, MoE에서 오류 가능)

신호 간 교차 검증 + 진단 로깅:
```
TurboQuant head_dim signals — key=128 val=128 computed=64 mla_k=0 mla_v=0 swa_k=0
[P1✓ P5✗] key_length=128 but n_embd/n_head=64 — using P1
```

**n_embd/n_head가 많은 모델에서 틀림:**

| 모델 | n_embd/n_head | 실제 head_dim | P1 없이 |
|------|---------------|---------------|---------|
| Qwen3-30B-A3B (MoE) | 64 | **128** | 잘못된 WHT 블록 → garbage |
| Qwen3.5-27B | 213 | **256** | TurboQuant 비활성화 |

Power-of-2 검증으로 WHT 비호환 차원(head_dim=80, 576)을 조기 포착.
</details>

---

### v1.2.0 — 자동 head_dim 매핑 + head_dim=64 품질 수정 + V Cross-Head WHT

**자동 head_dim 매핑 — 접미사 번호 불필요:**
```bash
# v1.2.0: tbq3/tbqp3만 사용, head_dim 기반 자동 선택
--cache-type-k tbqp3 --cache-type-v tbq3
```
- `head_dim=256`: K=tbqp3_0 (QJL), V=tbq3_0 → **5.2배 압축**
- `head_dim=128`: K=tbqp3_1 (Direct Sign), V=tbq3_1 → **5.0배 압축**
- `head_dim=64`: K=**q8_0** (자동 폴백), V=tbq3_2 → **2.7배 압축**

**head_dim=64 품질 문제 발견 및 수정:**

과학자 이름 테스트(독일어→한국어 표기)에서 근본적 WHT 한계 발견:
- PPL은 개선(2195 < 4008)되었지만 생성이 깨짐(attention smoothing artifact)
- TurboQuant 논문은 head_dim=128만 검증 — d=64에서 CLT 수렴 실패
- **수정: K 캐시를 자동으로 q8_0 폴백**(WHT 완전 우회) + V는 WHT 유지(값의 가중합은 노이즈에 더 관대)

| 모델 | head_dim | 설정 | KV 메모리 | 압축 | PPL (2K) | 프롬프트 t/s | 생성 t/s | 파울리 |
|:-----|:---------|:-----|----------:|-----:|---------:|-----------:|--------:|:---:|
| GPT OSS 120B | 64 | F16/F16 | 4,608 MiB | 1.0x | 2413 | 133 | 47 | ✅ |
| GPT OSS 120B | 64 | **q8_0/tbq3 (자동)** | **1,692 MiB** | **2.7x** | **1925** | **145** | **46** | **✅** |
| GPT OSS 20B | 64 | F16/F16 | 3,072 MiB | 1.0x | 4008 | 412 | 74 | ✅ |
| GPT OSS 20B | 64 | **q8_0/tbq3 (자동)** | **1,128 MiB** | **2.7x** | **2649** | **421** | **75** | **✅** |
| Qwen3.5-122B | 256 | F16/F16 | 6,144 MiB | 1.0x | — | 91 | 23 | ✅ |
| Qwen3.5-122B | 256 | **tbqp3/tbq3 (자동)** | **1,188 MiB** | **5.2x** | **—** | **102** | **22** | **✅** |

**입력 검증 — 모든 잘못된 조합 안전 처리:**
- V에 TBQP(QJL) → TBQ로 자동 다운그레이드
- 잘못된 _N 접미사 → head_dim 기반 자동 수정 또는 폴백
- 미지원 head_dim → q8_0/f16 폴백 + 경고

#### 왜 "파울리 테스트"인가? — PPL이 거짓말할 때

**문제:** Perplexity(PPL)는 KV 캐시 양자화 평가의 표준 지표입니다. 낮은 PPL = 더 좋다, 맞죠? 작은 head dimension에서 WHT 기반 양자화에 이것이 **위험할 정도로 오해를 불러일으킨다**는 것을 발견했습니다.

cross-head WHT 구현(8 헤드 × head_dim=64 = 512-element WHT)으로, f16보다 *더 좋아 보이는* PPL 수치를 달성했습니다:

| 설정 | PPL (2K) | 생성 품질 |
|:-----|:---------|:---------|
| F16/F16 (기준) | 4008 | ✅ 정상 |
| cross-head WHT 3비트 | **2195** (더 좋아 보임!) | ❌ 완전 고장 |

**PPL이 45% 개선되었지만 — 모델이 이름조차 제대로 쓰지 못했습니다.** WHT 양자화 노이즈가 attention smoothing으로 작용: 극단적 잘못된 예측을 줄이고(평균 surprise 감소 = 낮은 PPL) 단일 올바른 토큰을 선택하는 데 필요한 날카로운 attention 피크를 파괴합니다(생성 고장).

**테스트:** 이를 포착하는 지표가 필요했습니다. 독일 과학자 이름의 한국어 표준 표기를 선택 — 구체적으로 "Wolfgang Pauli" → "볼프강 파울리".

이것이 벤치마크로 효과적인 이유:
- **문화적 특수성:** 한국어에는 외국 이름 표기에 대한 국가 공식 표준([외래어 표기법](https://kornorms.korean.go.kr/))이 있습니다. "Wolfgang Pauli"는 반드시 "볼프강 파울리"(bol-peu-gang pa-ul-li)로 표기해야 합니다.
- **LLM에 극도로 어려움:** 여러 표기가 허용되는 영/중/일어와 달리, 한국어는 이름당 **정확히 하나의 정답**만 있습니다. 독일어 음소를 한국어 음절에 매핑하는 표준화된 음운 규칙으로 결정됩니다.
- **Attention 정밀도에 민감:** 다음절 한국어 표기를 맞추려면 모델이 원래 이름에 대해 정확한 토큰별 attention을 유지해야 합니다. 양자화 노이즈로 인한 attention 흐림은 즉시 잘못된 음절을 생성합니다.
- **검증 기준:** 정답은 한국 중·고등학교 과학 교과서에 나오는 것과 일치 — 객관적이고 검증 가능한 정답.

**모든 설정에서의 결과 (head_dim=64):**

| K 타입 | 비트 | "Wolfgang Pauli" → | PPL |
|:-------|:-----|:-------------------|:----|
| F16 | 16 | 볼프강 파울리 ✅ | 4008 |
| q8_0 | 8 | 볼프강 파울리 ✅ | — |
| tbq4_0 (4비트 WHT) | 4 | 볼프강 **파우리** ⚠️ (한 음절 오류) | — |
| tbq3_0 (3비트 WHT) | 3 | **파이브라스** ❌ (무의미) | — |
| tbqp3_0 (3비트 + QJL) | 3 | **er** ❌ (한국어도 아님) | — |
| tbqp3_3 (cross-head) | 3 | **2.** ❌ | **2195** (오해의 소지가 있는 "좋은" 수치) |

"최고" PPL(2195)을 달성한 cross-head 설정이 최악의 생성 출력을 냈습니다. **PPL과 생성 품질이 역상관 관계였습니다.**

**근본 원인:** TurboQuant 논문(ICLR 2026)은 head_dim=128 모델(Gemma, Mistral, Llama-3.1-8B)에서만 검증. head_dim=64에서는 WHT가 필요로 하는 중심극한정리(CLT) 수렴이 불충분 — 좌표가 정확한 스칼라 양자화에 필요한 가우시안에 충분히 근사하지 못합니다.

**해결책:** head_dim=64에서 K 캐시를 자동으로 q8_0 폴백(WHT 완전 우회), V 캐시는 WHT 유지(값 가중합은 노이즈에 더 관대). 파울리 테스트를 통과하면서 2.7배 압축 유지.

**교훈:** KV 캐시 양자화는 항상 PPL이 아닌 생성 품질 테스트로 검증하세요. Attention 분포를 평활화하는 방법은 PPL을 개선하면서 모델의 정확한 출력 생성 능력을 파괴할 수 있습니다.

> **참고: Cross-Head WHT 코드에 대해**
>
> v1.2.0에는 head_dim=64 모델용으로 개발된 cross-head WHT 구현이 포함되어 있습니다: 8 KV 헤드를 512-element WHT로 그룹화, 크로네커 분해 H_512 = H_8 ⊗ H_64, V cross-head 스코어링 등. 이 코드는 head_dim=64에서 PPL을 개선했지만 생성 품질을 보존하지 못해 **자동 매핑에서 사용되지 않습니다**. 그러나 향후 연구를 위해 의도적으로 코드를 유지합니다: head_dim>=128에서의 cross-head 실험(예: 1024-element WHT), 대안적 방법(KITTY, learned rotation)과의 비교, 또는 더 나은 CLT 수렴을 달성할 수 있는 새로운 접근법. 관련 타입: `_3` 접미사(tbq3_3, tbq4_3, tbqp3_3, tbqp4_3).

---

### v1.1.0 — head_dim 64/128 지원 + Direct Sign 잔차 보정

**다중 head_dim 지원:**
- `head_dim=256`: 기존 (Qwen3.5, Qwen3-Next) — QJL 잔차 보정
- `head_dim=128`: **신규** (Llama, Qwen3, Mistral, MiniMax, 대부분 모델) — Direct Sign 보정
- `head_dim=64`: **신규** (gpt-oss, 소형 모델) — Direct Sign 보정
- 자동 감지 — 사용자 CLI 변경 없음 (`--cache-type-k tbqp3_0` 모든 모델에서 동작)

**Direct Sign — 논문 QJL 대비 4.3배 낮은 분산:**

논문의 QJL은 잔차 보정에 SRHT 랜덤 프로젝션을 사용하지만, d≤128에서는 프로젝션 노이즈가 보정 이득을 초과. Direct Sign은 `sign(residual)`을 직접 저장:
- 4.3배 분산 감소: `(1-2/π)/(π/2) = 0.23`
- 두 번째 WHT 불필요 → 더 빠른 쿼리 전처리
- d=256에서는 QJL이 우수하여 유지 (하이브리드 전략)

#### head_dim=128 벤치마크 (Qwen3-30B-A3B Q4_K_M, 2K 컨텍스트, DGX Spark GB10)

| KV 설정 | PPL | 속도 (t/s) | KV 크기 | 압축 |
|:--------|----:|----------:|--------:|-----:|
| f16 + f16 (기준) | 6.69 | 87.8 | 192 MiB | 1.0x |
| q8_0 + q8_0 | 6.68 | 84.3 | 102 MiB | 1.9x |
| q4_0 + q4_0 | 7.33 | 85.0 | 54 MiB | 3.6x |
| tbq4_0 + tbq4_0 | 7.02 | 68.6 | 50 MiB | 3.9x |
| tbq4_0 + tbq3_0 | 7.19 | 68.1 | 44 MiB | 4.4x |
| **tbqp4_0 + tbq3_0 (Direct Sign)** | **7.08** | **63.6** | **44 MiB** | **4.3x** |
| tbqp3_0 + tbq3_0 (Direct Sign) | 7.95 | 65.3 | 38 MiB | 5.0x |

#### Direct Sign 대 QJL 비교 (head_dim=128, TBQP3/TBQ3)

| 방법 | PPL | 비고 |
|:-----|----:|:-----|
| QJL (논문) | 11.04 | d=128에서 프로젝션 노이즈 지배 |
| **Direct Sign (v1.1.0)** | **7.95** | **PPL 3.09 감소** |

#### 버그 수정

- `__syncthreads()` 레이스 컨디션: ncols=2(프롬프트 평가)에서 쿼리 WHT 공유 메모리 손상 — PPL이 2000+로 폭발하지만 토큰 생성(ncols=1)은 정상으로 보임

---

### v1.0.0 벤치마크 (head_dim=256)

**핵심 결과: `tbqp3_0 + tbq3_0` = 5.2배 압축 + FP16보다 낮은 PPL + 12% 더 빠름**

### 벤치마크 환경

- **모델**: Qwen3.5-35B-A3B Q4_K_M (19.71 GiB)
- **시스템**: NVIDIA DGX Spark, GB10 GPU, 128GB 통합 메모리, CUDA 13.0
- **데이터셋**: wikitext-2-raw (test set)

### 종합 성능표

| KV 설정 | KV 메모리 | 압축 | PPL (2K) | PPL (8K) | 속도 (8K) |
|:--------|----------:|-----:|---------:|---------:|----------:|
| f16 + f16 (기준) | 5,120 MiB | 1.0x | 4.678 | 6.829 | 51.9 t/s |
| q8_0 + q8_0 | 2,720 MiB | 1.9x | 4.679 | 6.806 | 50.1 t/s |
| tbq3_0 + tbq3_0 | 980 MiB | 5.2x | 4.756 | 6.963 | 63.5 t/s |
| **tbqp3_0 + tbq3_0** | **990 MiB** | **5.2x** | **4.672** | **6.850** | **58.3 t/s** |

### 핵심 발견

```
메모리:   5,120 → 990 MiB  (81% 절약)
PPL@2K:   4.678 → 4.672   (FP16보다 우수!)
PPL@8K:   6.829 → 6.850   (+0.3% 차이)
속도@8K:  51.9  → 58.3 t/s (+12% 더 빠름)
```

### 컨텍스트 길이별 Perplexity

| KV 설정 | 2K | 4K | 8K | 32K |
|:--------|---:|---:|---:|----:|
| f16 + f16 | 4.678 | 6.591 | 6.829 | 6.151 |
| q8_0 + q8_0 | 4.679 | 6.585 | 6.806 | 6.144 |
| tbq3_0 + tbq3_0 | 4.756 | 6.736 | 6.963 | 6.201 |
| **tbqp3_0 + tbq3_0** | **4.672** | **6.683** | **6.850** | **6.273** |

### 컨텍스트 길이별 토큰 생성 속도 (t/s)

| KV 설정 | 2K | 4K | 8K | 32K |
|:--------|---:|---:|---:|----:|
| f16 + f16 | 51.1 | 52.5 | 51.9 | 51.6 |
| q8_0 + q8_0 | — | 52.3 | 50.1 | — |
| tbq3_0 + tbq3_0 | 66.9 | 66.4 | 63.5 | 66.3 |
| **tbqp3_0 + tbq3_0** | **63.9** | **63.0** | **58.3** | **63.3** |

### QJL 보정 효과

QJL (Quantized Johnson-Lindenstrauss) = 논문의 TurboQuant_prod. Key에 1비트 잔차 보정 추가.

| 컨텍스트 | tbq3_0 (QJL 없음) | tbqp3_0 (QJL 포함) | 개선 |
|:---------|---:|---:|:---:|
| 2K | 4.756 | **4.672** | **-0.084** |
| 4K | 6.736 | **6.683** | **-0.053** |
| 8K | 6.963 | **6.850** | **-0.113** |
| 32K | 6.201 | 6.273 | +0.072 |

---

## 참고문헌

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google DeepMind)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 기반 프레임워크

## 라이선스

이 구현은 llama.cpp 프로젝트 라이선스(MIT)를 따릅니다.
