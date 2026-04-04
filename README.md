# TurboQuant KV Cache Compression for llama.cpp

> Implementation of [TurboQuant (ICLR 2026, Google DeepMind)](https://arxiv.org/abs/2504.19874) — KV cache compression via Walsh-Hadamard Transform + Lloyd-Max quantization with QJL correction

<details>
<summary>🇰🇷 한국어</summary>

## 개요

Google DeepMind의 TurboQuant 논문을 llama.cpp에 구현했습니다. KV 캐시를 3~4비트로 압축하여 메모리를 최대 5.2배 절약하면서 FP16 수준의 품질을 유지합니다.

### 🆕 v1.5.0 — Upstream Rebase + Gemma 4 지원

**llama.cpp 최신 upstream(b7ad48ebd)에 완전 리베이스 + Gemma 4 TBQ KV 캐시 지원.**

기존에는 upstream llama.cpp와 별도 히스토리로 관리되어, 최신 기능을 반영하려면 수백 커밋을 수동 패치해야 했습니다. v1.5.0부터 upstream 커밋 히스토리를 공유하는 구조로 전환하여, `git merge upstream/master` 한 줄로 최신 동기화가 가능합니다.

**변경 사항:**
- upstream llama.cpp 최신 커밋(b7ad48ebd)에 3-way merge로 전체 TBQ/TBQP 코드 적용
- **Gemma 4 지원**: head_dim=512(global) + 256(SWA) 혼합 아키텍처 완전 지원
- Deepseek MLA 512x512 MMA config 반영 (upstream 추가분)
- bf16 flash attention vec 커널 지원 (upstream 추가분)
- V-less cache, stream-k FA 등 upstream 최적화 포함
- `git merge upstream/master`로 향후 동기화 가능한 fork 구조 확립

**Gemma 4 핵심 기술:**
1. **SWA 캐시 타입 자동 재매핑**: global(head_dim=512)과 SWA(head_dim=256)가 다른 경우, SWA 캐시에 올바른 TBQ 서브타입을 자동 할당
2. **가변 GQA 대응**: Gemma 4는 레이어마다 head_count_kv가 다름(16/4). WHT rotation이 head 단위로 동작하므로 가변 GQA에서도 안전 — `attn_rot_k` 활성화 조건 수정
3. **D=512 vec 커널 2-pass WHT**: 512차원 Q를 256-block × 2로 나눠 각각 WHT 처리. IWHT도 동일 방식

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

> **참고:** Gemma 4는 hybrid SWA 아키텍처로, non-SWA 10 layers (head_dim=512) + SWA 50 layers (head_dim=256)입니다. SWA 캐시는 sliding window 크기(1536 cells)로 제한되어, Dense 모델의 압축률(1.8x)은 MLA 모델(5.2x)보다 낮습니다. MoE 변형은 레이어가 적어 **5.2x 압축** 달성.
>
> **참고:** 수학 정확도는 [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py) 벤치마크(2x2/3x3 행렬곱, 스칼라 연산, filler 0-2000t)로 측정. 65개 테스트 중 f16과 TBQ는 **55개(31B Dense) / 58개(26B MoE)에서 동일한 결과**를 보였으며, 차이가 발생한 케이스도 작은 수치 오차(예: 160→150, 519→509)로 garbage 출력이 아닙니다. 파울리 테스트(과학자 이름 한국어 번역)는 정상 통과.

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

**호환성:** 기존 v1.4.2의 모든 기능(MMA 텐서코어, QJL scalar correction, MLA 비대칭) 유지. 기존 모델 벤치마크 동일.

---

### v1.4.2 — MMA Tensor Core 가속 + QJL Scalar Correction

**TBQ/TBQP MMA 텐서코어 가속으로 토큰 생성 속도 30→49 t/s (+63%).**

GLM-4.7-Flash (MLA, K=576/V=512) 비대칭 모델에서 MMA 텐서코어를 활용한 KV 캐시 attention 가속. vec 커널 대비 최대 1.6배 TG 속도 향상.

**핵심 기술:**

1. **MMA K spatial dequant**: TBQ/TBQP raw blocks → IWHT → spatial f16 → tensor core. Warp shuffle 최적화로 cooperative IWHT의 `__syncthreads` 8→4회 감소.
2. **QJL scalar correction**: TBQP의 QJL 1-bit 보정을 full MMA pass 대신 경량 scalar 연산으로 수행. K의 raw block에서 sign bits + dq를 직접 읽어 `dq × Σ(Q_wht2[i] × sign[i])` 계산. QJL의 두 번째 sign basis(signs2)를 올바르게 처리.
3. **V = K view spatial**: K를 spatial domain으로 dequant하므로 V(= K view)도 자동으로 spatial. Output IWHT 완전 제거.
4. **정식 커널 시그니처**: `fattn_kernel_t`에 `raw_K_data`, `raw_K_stride`, `Q_wht2_data`, `Q_wht2_stride` 파라미터 추가. hack 없는 정석 구현.
5. **Fused Q WHT12**: Q->data에서 직접 읽어 Q_wht1(중간값) + Q_wht2(QJL용) 계산. cudaMemcpy 제거.

**벤치마크 (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| 캐시 | KV MiB | 압축 | TG t/s | vs v1.4.1 | 파울리 |
|------|--------|------|--------|-----------|--------|
| f16/f16 | 10,469 | 1.0x | 67.5 | — | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | **49.7** | **+55%** (32→49.7) | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | **42.8** | **+36%** (31.5→42.8) | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | ~49 | +47% | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | ~42 | +45% | PASS |

> **참고:** TBQP(QJL 보정)가 TBQ보다 느린 이유: QJL은 별도 sign basis(signs2)로 WHT 변환 후 scalar correction을 수행하므로 Q_wht2 precompute + per-token scalar correction 오버헤드. 대신 dot product 정확도가 향상되어 PPL 개선.

---

### v1.4.1 — GLM-4.7-Flash (MLA) 비대칭 K/V 지원

**GLM-4.7-Flash, DeepSeek-V2/V3 등 MLA 아키텍처 모델에서 TurboQuant 완전 지원.**

MLA(Multi-head Latent Attention)는 K=concat(latent[512], rope[64])=576차원, V=latent[512]차원의 비대칭 구조입니다. 세 가지 핵심 기술로 해결:

1. **D_V 템플릿 파라미터**: vec 커널이 K/Q 차원(D=576)과 V 차원(D_V=512)을 분리 처리. VKQ 배열, combine stride, IWHT 패스 수, 출력 쓰기 모두 D_V 기준. 기존 대칭 모델은 D_V=D 기본값으로 영향 없음.
2. **RoPE f16 passthrough**: _4 블록 구조체에서 마지막 64차원(rope)을 WHT+양자화 대신 f16으로 직접 저장. RoPE 값의 norm이 latent 대비 ~80배 커서(10.49 vs 0.13), 어떤 비트 수의 양자화든 오차가 attention score를 지배하는 문제 해결. Q 전처리와 dot product도 sub-block 3을 f16 직접 연산으로 처리.
3. **MLA V-as-K-view 지원**: MLA absorption 최적화에서 V는 K 캐시의 view(같은 타입). TBQP V dequantize는 MSE centroid만 사용(QJL correction은 K·Q dot product 보정 전용이므로 V 재구성에 미적용). IWHT는 256+256 2패스(rope 64 패스 스킵).

**벤치마크 (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| 캐시 | KV MiB | 압축 | PP t/s | TG t/s | PPL | 파울리 |
|------|--------|------|--------|--------|-----|--------|
| f16/f16 | 10,469 | 1.0x | 73.0 | 60.3 | 5.998 | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | 68.2 | 32.0 | **6.836** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | 66.8 | 31.5 | **6.586** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | 67.2 | 33.4 | — | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | 65.8 | 28.9 | — | PASS |

> **참고:** MLA 모델의 압축률(3.5x)이 일반 모델(5.2x)보다 낮은 이유: MLA는 이미 KV를 256-dim 잠재 표현으로 압축하므로, 원래 캐시가 작습니다. 7.5GB 절약(10,469→2,981 MiB)은 실질적으로 큰 차이입니다.
>
> **참고:** TG 속도(31.5 t/s vs f16 60.3 t/s): TBQ는 vec 커널(스칼라 WHT dot product), f16은 MMA 커널(텐서코어 행렬곱)을 사용합니다. MMA 커널에 TBQ on-the-fly dequantize를 추가하면 텐서코어 활용 가능 — 향후 과제.

**해결된 버그:**

| 버그 | 원인 | 수정 |
|------|------|------|
| GLM tbqp3/tbq3 크래시 (v1.4.0) | `get_best_fattn_kernel`이 D=576 TBQ를 NONE 반환 | D=576+V=512 TBQ vec 커널 라우팅 추가 |
| 대칭 dispatch가 비대칭을 먹음 | `FATTN_VEC_CASE`가 V->ne[0] 미확인 → DV=576으로 잘못 launch | ASYM 케이스를 대칭 앞에 배치 |
| RoPE 양자화 → garbage 출력 | rope norm(~10.49) >> latent norm(~0.13), 양자화 오차가 attention 지배 | sub-block 3를 f16 passthrough로 변경 |
| TBQP V dequant에 QJL 적용 → garbage | QJL은 K·Q dot product 보정 전용, V 재구성에 부적합 | V dequant에서 QJL 제거 (MSE only) |
| Q WHT sub-block 간 race condition | sub-block 1 저장 후 `__syncthreads` 누락 → sub-block 2 WHT가 Q_wht 오염 가능 | `__syncthreads` barrier 추가 |

---

### v1.3.0 — Bulletproof head_dim Detection + Critical Bug Fixes

**벤치마크 (Qwen3-30B-A3B Q4_K_M, DGX Spark GB10, head_dim=128):**

| 설정 | PPL | vs F16 | 비고 |
|------|-----|--------|------|
| f16/f16 | 6.26 | 기준 | |
| **tbqp4/tbq4** | **6.70** | **+7.1%** | +Direct Sign 보정 (head_dim=128) |
| tbq4/tbq4 | 6.73 | +7.5% | MSE only |
| **tbqp3/tbq3** | **7.91** | **+26.3%** | +Direct Sign 보정 (head_dim=128) |
| tbq3/tbq3 | 8.49 | +35.6% | MSE only |

> **참고:** TBQP의 잔차 보정 방식은 head_dim에 따라 다릅니다:
> - head_dim=256: **QJL** (논문 원본, SRHT 기반)
> - head_dim=128/64: **Direct Sign** (QJL 분산 문제 해결, 분산 4.3배 감소)
>
> 위 수치는 **MoE 모델** (토큰당 3.7B 활성 파라미터)입니다. F16 기준 PPL 6.26 자체가 MoE 특성이며, TurboQuant 이슈가 아닙니다.

**해결된 이슈:**

| 이슈 | 보고자 | 상태 |
|------|--------|------|
| Phi-4, DeepSeek 자동 head_dim 감지 실패 | @fritolays | 수정 — P1→P5 캐스케이드 |
| turbo4-K PPL 폭발 (Cydonia-24B에서 18,202) | @TheTom | 수정 — head_dim 오감지가 원인 |
| GLM head_dim=576, Qwen3-4B head_dim=80 | @fritolays, @sztlink | 수정 — pow2 검사 + q8_0 폴백 |
| Qwen3.5-27B-UD에서 "////" 출력 | @modderBUG | v1.3.0에서 재현 불가 |
| llama-bench TBQ 타입 미지원 | @sztlink | 수정 — 16개 타입 + 4개 약어 |
| Windows OpenSSL DLL 의존성 | @sztlink | 수정 — 독립 빌드 추가 |

**기타 개선:**
- llama-bench: -ctk/-ctv로 TBQ/TBQP 타입 완전 지원
- 미지원 head_dim: q8_0으로 폴백하여 압축 유지
- 독립 빌드(standalone): 외부 DLL 의존성 없음

<details>
<summary>🔧 head_dim 감지 기술 상세 (P1→P5 Priority Cascade)</summary>

기존에는 GGUF `{arch}.attention.key_length` 메타데이터만 읽었으나, 대부분의 모델(Phi-4, DeepSeek, Gemma, Mistral)은 이 값을 저장하지 않아 TurboQuant가 조용히 비활성화되는 문제가 있었습니다.

이제 6개 감지 신호를 엄격한 우선순위 캐스케이드로 사용:
- **P1**: `attention.key_length` (100% — GGUF 공식값)
- **P2**: `attention.key_length_mla` (100% — DeepSeek V2 등 MLA 모델)
- **P3**: `attention.key_length_swa` (100% — Gemma 2/3 등 SWA 모델)
- **P4**: `attention.value_length` (95% — 교차 검증)
- **P5**: `n_embd / n_head` (70% — 폴백, MoE에서 오류 가능)

신호 간 교차 검증 + 진단 로깅:
```
TurboQuant head_dim signals — key=128 val=128 computed=64 mla_k=0 mla_v=0 swa_k=0
[P1✓ P5✗] key_length=128 but n_embd/n_head=64 — using P1
```

**n_embd/n_head가 틀리는 실제 사례:**

| 모델 | n_embd/n_head | 실제 head_dim | P5만 사용시 |
|------|---------------|---------------|-------------|
| Qwen3-30B-A3B (MoE) | 64 | **128** | 잘못된 WHT 블록 → 쓰레기 출력 |
| Qwen3.5-27B | 213 | **256** | TurboQuant 비활성 |

Power-of-2 검증으로 WHT 비호환 차원(head_dim=80, 576) 조기 감지.
</details>

---

### v1.2.0 — Auto head_dim Mapping + head_dim=64 Quality Fix + V Cross-Head WHT

**자동 head_dim 매핑 — 사용자가 숫자 접미사를 알 필요 없음:**
```bash
# v1.2.0: 그냥 tbq3/tbqp3 입력하면 head_dim에 따라 자동 선택
--cache-type-k tbqp3 --cache-type-v tbq3
```
- `head_dim=256`: K=tbqp3_0 (QJL), V=tbq3_0 → **5.2x 압축**
- `head_dim=128`: K=tbqp3_1 (Direct Sign), V=tbq3_1 → **5.0x 압축**
- `head_dim=64`: K=**q8_0** (자동 fallback), V=tbq3_2 → **2.7x 압축**

**head_dim=64 품질 문제 발견 및 해결:**

과학자 이름 테스트(독일어→한국어 표준 표기)에서 WHT 양자화의 근본적 한계를 발견:
- PPL은 개선되어도(2195 < 4008) 실제 생성은 깨지는 현상 (attention smoothing)
- TurboQuant 논문은 head_dim=128만 검증 — head_dim=64는 CLT 수렴 부족
- **해결: K cache를 q8_0으로 자동 fallback** + V는 WHT 유지 (V는 가중합이라 노이즈에 관대)

| Model | head_dim | Config | KV Memory | Compress | PPL (2K) | Prompt t/s | Gen t/s | Pauli Test |
|:------|:---------|:-------|----------:|---------:|---------:|---------:|--------:|:---:|
| GPT OSS 120B | 64 | F16/F16 | 4,608 MiB | 1.0x | 2413 | 133 | 47 | ✅ |
| GPT OSS 120B | 64 | **q8_0/tbq3 (auto)** | **1,692 MiB** | **2.7x** | **1925** | **145** | **46** | **✅** |
| GPT OSS 20B | 64 | F16/F16 | 3,072 MiB | 1.0x | 4008 | 412 | 74 | ✅ |
| GPT OSS 20B | 64 | **q8_0/tbq3 (auto)** | **1,128 MiB** | **2.7x** | **2649** | **421** | **75** | **✅** |
| Qwen3.5-122B | 256 | F16/F16 | 6,144 MiB | 1.0x | — | 91 | 23 | ✅ |
| Qwen3.5-122B | 256 | **tbqp3/tbq3 (auto)** | **1,188 MiB** | **5.2x** | **—** | **102** | **22** | **✅** |

**입력 검증 — 잘못된 조합도 크래시 없이 안전 처리:**
- V에 TBQP(QJL) 지정 → 자동으로 TBQ로 다운그레이드
- 잘못된 _N 접미사 → head_dim에 맞게 자동 수정 또는 fallback
- 미지원 head_dim → q8_0/f16 fallback + 경고

#### 왜 "파울리 테스트"인가? — PPL이 거짓말하는 경우

**문제:** Perplexity(PPL)는 KV 캐시 양자화 품질의 표준 지표입니다. 낮을수록 좋다고 알려져 있지만, head_dim이 작은 경우 WHT 기반 양자화에서 **심각하게 오해를 유발**합니다.

Cross-head WHT(8개 head를 묶어 512-element WHT)를 구현했을 때, F16보다 *더 좋아 보이는* PPL 수치를 달성했습니다:

| 설정 | PPL (2K) | 생성 품질 |
|:-----|:---------|:---------|
| F16/F16 (기준) | 4008 | ✅ 정확 |
| cross-head WHT 3-bit | **2195** (더 좋아 보임!) | ❌ 완전히 깨짐 |

**PPL이 45% 개선**되었지만 모델은 이름조차 제대로 출력하지 못했습니다. WHT 양자화 노이즈가 attention을 평활화(smoothing)하여 극단적인 오답 예측을 줄이면서(평균 surprise 감소 = PPL 하락), 정답 토큰에 집중하는 날카로운 attention peak은 파괴합니다(생성 실패).

**테스트 설계:** 이를 감지하기 위해 독일 과학자 이름의 한국어 표준 표기를 벤치마크로 선정했습니다. 구체적으로 "Wolfgang Pauli" → "볼프강 파울리".

이 테스트가 효과적인 이유:
- **문화적 특수성:** 한국어에는 외래어 이름에 대한 국가 공식 표준([외래어 표기법](https://kornorms.korean.go.kr/))이 존재합니다. "Wolfgang Pauli"는 반드시 "볼프강 파울리"로만 표기되어야 하며, "볼프강 파우리"나 "워워우 파울리" 등은 오답입니다.
- **LLM에게 극도로 어려운 과제:** 영어/중국어/일본어에서는 여러 음차가 허용되지만, 한국어는 이름당 **정확히 하나의 정답**만 존재합니다. 독일어 음소를 한국어 음절에 대응하는 표준 음운 규칙에 의해 결정됩니다.
- **attention 정밀도에 민감:** 다음절 한국어 음차를 정확히 출력하려면 원어 이름에 대한 정밀한 토큰별 attention이 필요합니다. 양자화 노이즈로 인한 attention 흐림은 즉시 잘못된 음절로 나타납니다.
- **검증 기준:** 정답은 한국 중고등학교 과학 교과서에 수록된 표기와 일치하는지로 판단 — 객관적이고 검증 가능한 기준입니다.

**head_dim=64에서 전체 설정별 결과:**

| K 타입 | 비트 | "Wolfgang Pauli" → | PPL |
|:-------|:----|:-------------------|:----|
| F16 | 16 | 볼프강 파울리 ✅ | 4008 |
| q8_0 | 8 | 볼프강 파울리 ✅ | — |
| tbq4_0 (4-bit WHT) | 4 | 볼프강 **파우리** ⚠️ (1음절 오류) | — |
| tbq3_0 (3-bit WHT) | 3 | **파이브라스** ❌ (의미 없음) | — |
| tbqp3_0 (3-bit + QJL) | 3 | **er** ❌ (한국어도 아님) | — |
| tbqp3_3 (cross-head) | 3 | **2.** ❌ | **2195** (오해를 유발하는 "좋은" 수치) |

"최고" PPL(2195)을 달성한 cross-head 설정이 가장 나쁜 생성 결과를 보였습니다. **PPL과 생성 품질이 역상관 관계.**

**근본 원인:** TurboQuant 논문(ICLR 2026)은 head_dim=128 모델(Gemma, Mistral, Llama-3.1-8B)만 검증했습니다. head_dim=64에서는 WHT가 요구하는 중심극한정리(CLT) 수렴이 불충분하여, 좌표가 가우시안에 충분히 근사하지 않아 정확한 스칼라 양자화가 불가능합니다.

**해결:** head_dim=64에서는 K 캐시를 자동으로 q8_0으로 fallback(WHT 우회), V 캐시만 WHT 유지(가중합이므로 노이즈에 관대). 파울리 테스트 통과 + 2.7배 압축 달성.

**교훈:** KV 캐시 양자화는 반드시 PPL뿐 아니라 실제 생성 품질 테스트로 검증해야 합니다. attention 분포를 평활화하는 방법은 PPL을 개선하면서도 모델의 정밀한 출력 능력을 파괴할 수 있습니다.

> **참고: Cross-Head WHT 코드에 대하여**
>
> v1.2.0에는 head_dim=64 모델을 위해 개발한 cross-head WHT 구현(8개 KV head를 묶어 512-element WHT, Kronecker 분해 H_512 = H_8 ⊗ H_64, V cross-head scoring 등)이 포함되어 있습니다. 이 코드는 head_dim=64에서 PPL은 개선했지만 실제 생성 품질을 보존하지 못해 현재 **자동 매핑에서 사용되지 않습니다**. 그러나 head_dim=128 이상에서의 cross-head 실험(예: 1024-element WHT), 다른 변환 기법(KITTY, learned rotation 등)과의 비교 연구, 또는 향후 CLT 수렴이 개선되는 새로운 접근법에서 재활용될 가치가 있으므로 **의도적으로 코드를 유지**하고 있습니다. 관련 타입: `_3` suffix (tbq3_3, tbq4_3, tbqp3_3, tbqp4_3).

---

### v1.1.0 — head_dim 64/128 지원 + Direct Sign 잔차 보정

**멀티 head_dim 지원:**
- `head_dim=256`: 기존 (Qwen3.5, Qwen3-Next) — QJL 잔차 보정
- `head_dim=128`: **신규** (Llama, Qwen3, Mistral, MiniMax, 대부분 모델) — Direct Sign 잔차 보정
- `head_dim=64`: **신규** (gpt-oss, 소형 모델) — Direct Sign 잔차 보정
- 자동 감지 — 사용자 CLI 변경 없음 (`--cache-type-k tbqp3_0` 그대로)

**Direct Sign — 논문 QJL 대비 4.3배 낮은 분산:**

논문의 QJL은 SRHT 랜덤 프로젝션으로 잔차를 보정하지만, d≤128에서 프로젝션 잡음이 보정 효과를 초과합니다. Direct Sign은 `sign(residual)`을 직접 저장하여:
- 분산 4.3배 감소: `(1-2/π)/(π/2) = 0.23`
- 두 번째 WHT 불필요 → 속도 향상
- d=256에서는 QJL 유지 (QJL이 더 우수)

#### head_dim=128 벤치마크 (Qwen3-30B-A3B Q4_K_M, 2K context, DGX Spark GB10)

| K/V 설정 | PPL | 속도 (t/s) | KV 크기 | 압축률 |
|:---------|----:|---------:|-------:|------:|
| f16 + f16 (baseline) | 6.69 | 87.8 | 192 MiB | 1.0x |
| q8_0 + q8_0 | 6.68 | 84.3 | 102 MiB | 1.9x |
| q4_0 + q4_0 | 7.33 | 85.0 | 54 MiB | 3.6x |
| tbq4_0 + tbq4_0 | 7.02 | 68.6 | 50 MiB | 3.9x |
| tbq4_0 + tbq3_0 | 7.19 | 68.1 | 44 MiB | 4.4x |
| **tbqp4_0 + tbq3_0 (Direct Sign)** | **7.08** | **63.6** | **44 MiB** | **4.3x** |
| tbqp3_0 + tbq3_0 (Direct Sign) | 7.95 | 65.3 | 38 MiB | 5.0x |

#### Direct Sign vs QJL 비교 (head_dim=128, TBQP3/TBQ3)

| 방식 | PPL | 비고 |
|:-----|----:|:-----|
| QJL (논문 원본) | 11.04 | d=128에서 프로젝션 잡음 폭발 |
| **Direct Sign (v1.1.0)** | **7.95** | **PPL 3.09 감소** |

#### 버그 수정

- `__syncthreads()` race condition: ncols=2(prompt eval)에서 쿼리 WHT shared memory 오염 → PPL 2000+ 폭발, 토큰 생성(ncols=1)은 정상이라 발견 어려운 버그

---

### v1.0.0 벤치마크 (head_dim=256)

**핵심 결과: `tbqp3_0 + tbq3_0` = 5.2x 압축 + F16보다 낮은 PPL + 12% 빠른 속도**

### 벤치마크 환경

- **Model**: Qwen3.5-35B-A3B Q4_K_M (19.71 GiB)
- **System**: NVIDIA DGX Spark, GB10 GPU, 128GB Unified Memory, CUDA 13.0
- **Dataset**: wikitext-2-raw (test set)

### 종합 성능표

| KV 설정 | KV 메모리 | 압축률 | PPL (2K) | PPL (8K) | 속도 (8K) |
|:--------|----------:|-------:|---------:|---------:|----------:|
| f16 + f16 (baseline) | 5,120 MiB | 1.0x | 4.678 | 6.829 | 51.9 t/s |
| q8_0 + q8_0 | 2,720 MiB | 1.9x | 4.679 | 6.806 | 50.1 t/s |
| tbq3_0 + tbq3_0 | 980 MiB | 5.2x | 4.756 | 6.963 | 63.5 t/s |
| **tbqp3_0 + tbq3_0** | **990 MiB** | **5.2x** | **4.672** | **6.850** | **58.3 t/s** |

### 핵심 발견

```
메모리:  5,120 → 990 MiB  (81% 절약)
PPL@2K:  4.678 → 4.672   (F16보다 좋음!)
PPL@8K:  6.829 → 6.850   (+0.3% 차이)
속도@8K: 51.9  → 58.3 t/s (+12% 빠름)
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

QJL (Quantized Johnson-Lindenstrauss) = 논문의 TurboQuant_prod. Key에 1-bit 잔차 보정 추가.

| Context | tbq3_0 (QJL 없음) | tbqp3_0 (QJL 있음) | 개선 |
|:--------|---:|---:|:---:|
| 2K | 4.756 | **4.672** | **-0.084** |
| 4K | 6.736 | **6.683** | **-0.053** |
| 8K | 6.963 | **6.850** | **-0.113** |
| 32K | 6.201 | 6.273 | +0.072 |

### 사용법

```bash
# 빌드
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# 추천: QJL Key + TBQ Value (5.2x 압축, F16급 품질)
./build/bin/llama-cli -m model.gguf -ngl 99 \
  --cache-type-k tbqp3 --cache-type-v tbq3

# 속도 우선 (5.2x 압축, 최고 속도)
./build/bin/llama-cli -m model.gguf -ngl 99 \
  --cache-type-k tbq3_0 --cache-type-v tbq3_0
```

### 지원 KV 조합

✅ = 정상 동작 (CUDA 가속) / ❌ = llama.cpp 제한

| K \ V | f16 | q8_0 | tbq3_0 | tbq4_0 |
|:------|:---:|:----:|:------:|:------:|
| f16 | ✅ | ❌ | ❌ | ❌ |
| q8_0 | ✅ | ✅ | ❌ | ❌ |
| tbq3_0 | ✅ | ✅ | ✅ | ✅ |
| tbq4_0 | ✅ | ✅ | ✅ | ✅ |
| tbqp3_0 | ✅ | ✅ | ✅ | ✅ |
| tbqp4_0 | ✅ | ✅ | ✅ | ✅ |

### 타입 설명

| 타입 | 설명 | bpw |
|:-----|:-----|----:|
| tbq3_0 | TurboQuant 3-bit (WHT + Lloyd-Max) | 3.06 |
| tbq4_0 | TurboQuant 4-bit (WHT + Lloyd-Max) | 4.06 |
| tbqp3_0 | TurboQuant_prod 3-bit (2-bit LM + 1-bit QJL) | 3.12 |
| tbqp4_0 | TurboQuant_prod 4-bit (3-bit LM + 1-bit QJL) | 4.12 |

</details>

<details open>
<summary>🇺🇸 English</summary>

## Overview

This is an implementation of Google DeepMind's TurboQuant paper in llama.cpp. It compresses KV cache to 3-4 bits, achieving up to 5.2x memory savings while maintaining FP16-level quality.

### 🆕 v1.5.0 — Upstream Rebase + Gemma 4 Support

**Full rebase on latest upstream llama.cpp (b7ad48ebd) + Gemma 4 TBQ KV cache support.**

Previously maintained as a separate history from upstream llama.cpp, requiring manual patching of hundreds of commits for each sync. Starting with v1.5.0, the fork shares upstream commit history, enabling single-command sync via `git merge upstream/master`.

**Changes:**
- 3-way merged all TBQ/TBQP code onto latest upstream llama.cpp (b7ad48ebd)
- **Gemma 4 support**: hybrid head_dim=512 (global) + 256 (SWA) architecture fully supported
- Deepseek MLA 512x512 MMA config included (upstream addition)
- bf16 flash attention vec kernel support (upstream addition)
- V-less cache, stream-k FA, and other upstream optimizations included
- Established proper fork structure for easy future upstream sync

**Gemma 4 key techniques:**
1. **SWA cache type auto-remapping**: When global (head_dim=512) and SWA (head_dim=256) differ, the SWA cache is automatically assigned the correct TBQ sub-type
2. **Variable GQA support**: Gemma 4 has per-layer head_count_kv (16/4). WHT rotation operates per-head, so variable GQA is safe — `attn_rot_k` activation condition updated
3. **D=512 vec kernel 2-pass WHT**: 512-dim Q is split into 256-block × 2, each WHT'd separately. IWHT uses the same approach

**Benchmark (Gemma 4 31B-it Dense, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| Cache | GPU Memory | Compress | PP t/s | TG t/s | PPL (wiki, 2K) | Math Accuracy | Pauli |
|-------|-----------|----------|--------|--------|----------------|---------------|-------|
| f16/f16 | 41,500 MiB | 1.0x | 152.9 | 10.1 | 309.7 | 42/65 (64.6%) | PASS |
| **tbq3/tbq3** | **23,215 MiB** | **1.8x** | 112.9 | 9.0 | 212.1 | **32/65 (49.2%)** | **PASS** |

**Benchmark (Gemma 4 26B-A4B MoE, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| Cache | GPU Memory | Compress | TG t/s | Math Accuracy | Pauli |
|-------|-----------|----------|--------|---------------|-------|
| f16/f16 | 5,720 MiB | 1.0x | 56.4 | 37/65 (56.9%) | PASS |
| **tbq3/tbq3** | **1,106 MiB** | **5.2x** | 41.1 | **30/65 (46.2%)** | **PASS** |

> **Note:** Gemma 4 is a hybrid SWA architecture with non-SWA 10 layers (head_dim=512) + SWA 50 layers (head_dim=256). SWA cache is limited to the sliding window size (1536 cells), so Dense compression (1.8x) is lower than MLA models (5.2x). The MoE variant achieves **5.2x compression**.
>
> **Note:** Math accuracy measured using [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py) (2x2/3x3 matrix multiplication, scalar arithmetic, filler 0-2000t). Out of 65 tests, f16 and TBQ **agreed on 55 (31B Dense) / 58 (26B MoE) cases**. Divergent cases show small numerical deviations (e.g. `160→150`, `519→509`), not garbage output. Pauli test (scientist name recall, German→Korean) passes with coherent, diverse output.

<details>
<summary>Build & Run</summary>

**Build:**
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

**Server (f16 baseline):**
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

**Server (TurboQuant):**
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

**Compatibility:** All v1.4.2 features (MMA tensor core, QJL scalar correction, MLA asymmetric) fully preserved. Existing model benchmarks unchanged.

---

### v1.4.2 — MMA Tensor Core Acceleration + QJL Scalar Correction

**TBQ/TBQP MMA tensor core acceleration: TG speed 30→49 t/s (+63%).**

MMA tensor core attention acceleration for GLM-4.7-Flash (MLA, K=576/V=512) asymmetric models. Up to 1.6x TG speed improvement over vec kernel.

**Key features:**

1. **MMA K spatial dequant**: raw TBQ/TBQP blocks → IWHT → spatial f16 → tensor core. Warp shuffle optimization reduces cooperative IWHT `__syncthreads` from 8 to 4.
2. **QJL scalar correction**: TBQP 1-bit QJL correction via lightweight scalar ops instead of full MMA pass. Reads sign bits + dq directly from raw K blocks. Correctly handles QJL's second sign basis (signs2).
3. **V = K view spatial**: K dequanted to spatial domain, V (= K view) automatically spatial. Output IWHT completely eliminated.
4. **Proper kernel signature**: `fattn_kernel_t` extended with `raw_K_data`, `raw_K_stride`, `Q_wht2_data`, `Q_wht2_stride`. No pointer hacks.
5. **Fused Q WHT12**: Reads Q->data directly, computes Q_wht1 (intermediate) + Q_wht2 (for QJL). No cudaMemcpy.

**Benchmarks (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| Cache | KV MiB | Ratio | TG t/s | vs v1.4.1 | Pauli |
|-------|--------|-------|--------|-----------|-------|
| f16/f16 | 10,469 | 1.0x | 67.5 | — | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | **49.7** | **+55%** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | **42.8** | **+36%** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | ~49 | +47% | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | ~42 | +45% | PASS |

> **Note:** TBQP (with QJL) is slower than TBQ because QJL requires a second WHT transform (signs2) for Q_wht2 precomputation + per-token scalar correction overhead. The tradeoff is improved dot product accuracy (better PPL).

---

### v1.4.1 — GLM-4.7-Flash (MLA) Asymmetric K/V Support

**Full TurboQuant support for MLA architecture models: GLM-4.7-Flash, DeepSeek-V2/V3.**

MLA (Multi-head Latent Attention) has asymmetric K=concat(latent[512], rope[64])=576 dims and V=latent[512] dims. Three key techniques:

1. **D_V template parameter**: vec kernel separates K/Q dim (D=576) from V dim (D_V=512). VKQ arrays, combine stride, IWHT passes, output writes all use D_V. Symmetric models default D_V=D (zero impact).
2. **RoPE f16 passthrough**: _4 block structs store sub-block 3 (rope 64) as raw f16 instead of WHT+quantized. RoPE norm (~10.49) is ~80x larger than latent (~0.13) — any quantization error dominates attention scores. Q preprocessing and dot product handle sub-block 3 as direct f16.
3. **MLA V-as-K-view**: In MLA absorption, V is a view of K cache (same type). TBQP V dequantize uses MSE centroid only (QJL is for K·Q dot product correction, not V reconstruction). IWHT runs 256+256 two-pass (rope pass skipped).

**Benchmark (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| Cache | KV MiB | Compress | PP t/s | TG t/s | PPL | Pauli |
|-------|--------|----------|--------|--------|-----|-------|
| f16/f16 | 10,469 | 1.0x | 73.0 | 60.3 | 5.998 | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | 68.2 | 32.0 | **6.836** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | 66.8 | 31.5 | **6.586** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | 67.2 | 33.4 | — | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | 65.8 | 28.9 | — | PASS |

> **Note:** MLA compression ratio (3.5x) is lower than standard models (5.2x) because MLA already compresses KV to a 256-dim latent — the original cache is already small. The 7.5GB savings (10,469→2,981 MiB) is still significant.
>
> **Note:** TG speed (31.5 vs 60.3 t/s): TBQ uses the vec kernel (scalar WHT dot product), f16 uses the MMA kernel (tensor cores). Adding TBQ on-the-fly dequantize to MMA would enable tensor core acceleration — future work.

**Bugs fixed:**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| GLM tbqp3/tbq3 crash (v1.4.0) | `get_best_fattn_kernel` returned NONE for D=576 TBQ | Added D=576+V=512 TBQ vec kernel routing |
| Symmetric dispatch shadows asymmetric | `FATTN_VEC_CASE` doesn't check V->ne[0] → launches DV=576 | ASYM cases placed before symmetric |
| RoPE quantization → garbage | rope norm(~10.49) >> latent norm(~0.13), quant error dominates attention | Sub-block 3 changed to f16 passthrough |
| TBQP V dequant with QJL → garbage | QJL corrects K·Q dot products only, not V reconstruction | Removed QJL from V dequant (MSE only) |
| Q WHT sub-block race condition | Missing `__syncthreads` between sub-block storage and next WHT | Added `__syncthreads` barriers |

---

### v1.3.0 — Bulletproof head_dim Detection + Critical Bug Fixes

**Benchmark (Qwen3-30B-A3B Q4_K_M, DGX Spark GB10, head_dim=128):**

| Config | PPL | vs F16 | Note |
|--------|-----|--------|------|
| f16/f16 | 6.26 | baseline | |
| **tbqp4/tbq4** | **6.70** | **+7.1%** | +Direct Sign correction (head_dim=128) |
| tbq4/tbq4 | 6.73 | +7.5% | MSE only |
| **tbqp3/tbq3** | **7.91** | **+26.3%** | +Direct Sign correction (head_dim=128) |
| tbq3/tbq3 | 8.49 | +35.6% | MSE only |

> **Note:** TBQP residual correction method varies by head_dim:
> - **head_dim=256**: QJL (original paper, SRHT-based)
> - **head_dim=128/64**: Direct Sign (our fix for QJL variance issue — 4.3x lower variance)
>
> PPL numbers above are from a **MoE model** (3.7B active params/token). F16 baseline 6.26 is inherent to the architecture, not a TurboQuant issue.

**Issues Resolved:**

| Issue | Reporter | Status |
|-------|----------|--------|
| Phi-4, DeepSeek auto head_dim detection failure | @fritolays | Fixed — P1→P5 cascade |
| turbo4-K PPL explosion (18,202 on Cydonia-24B) | @TheTom | Fixed — was head_dim misdetection |
| GLM head_dim=576, Qwen3-4B head_dim=80 | @fritolays, @sztlink | Fixed — pow2 check + q8_0 fallback |
| "////" output on Qwen3.5-27B-UD | @modderBUG | Not reproducible on v1.3.0 |
| llama-bench TBQ types not accepted | @sztlink | Fixed — 16 types + 4 shorthands |
| Windows OpenSSL DLL dependency | @sztlink | Fixed — standalone builds |

**Other Improvements:**
- llama-bench: full TBQ/TBQP type support via -ctk/-ctv
- Unsupported head_dim: falls back to q8_0, preserving compression
- Standalone builds: no external DLL dependencies

<details>
<summary>🔧 head_dim Detection Technical Details (P1→P5 Priority Cascade)</summary>

Previously only read GGUF `{arch}.attention.key_length` metadata — most models (Phi-4, DeepSeek, Gemma, Mistral) don't store this, causing TurboQuant to silently disable.

Now uses 6 detection signals with strict priority cascade:
- **P1**: `attention.key_length` (100% — GGUF authoritative)
- **P2**: `attention.key_length_mla` (100% — MLA models like DeepSeek V2)
- **P3**: `attention.key_length_swa` (100% — SWA models like Gemma 2/3)
- **P4**: `attention.value_length` (95% — cross-check)
- **P5**: `n_embd / n_head` (70% — fallback, can be wrong for MoE)

Cross-validation between signals with diagnostic logging:
```
TurboQuant head_dim signals — key=128 val=128 computed=64 mla_k=0 mla_v=0 swa_k=0
[P1✓ P5✗] key_length=128 but n_embd/n_head=64 — using P1
```

**n_embd/n_head is WRONG for many models:**

| Model | n_embd/n_head | Actual head_dim | Without P1 |
|-------|---------------|-----------------|------------|
| Qwen3-30B-A3B (MoE) | 64 | **128** | Wrong WHT block → garbage |
| Qwen3.5-27B | 213 | **256** | TurboQuant disabled |

Power-of-2 validation catches non-WHT-compatible dimensions (head_dim=80, 576) early.
</details>

---

### v1.2.0 — Auto head_dim Mapping + head_dim=64 Quality Fix + V Cross-Head WHT

**Automatic head_dim mapping — no suffix numbers needed:**
```bash
# v1.2.0: just use tbq3/tbqp3, auto-selects based on head_dim
--cache-type-k tbqp3 --cache-type-v tbq3
```
- `head_dim=256`: K=tbqp3_0 (QJL), V=tbq3_0 → **5.2x compression**
- `head_dim=128`: K=tbqp3_1 (Direct Sign), V=tbq3_1 → **5.0x compression**
- `head_dim=64`: K=**q8_0** (auto fallback), V=tbq3_2 → **2.7x compression**

**head_dim=64 quality issue discovered and fixed:**

Scientist name test (German→Korean transliteration) revealed a fundamental WHT limitation:
- PPL improved (2195 < 4008) but generation broke (attention smoothing artifact)
- TurboQuant paper only validated head_dim=128 — CLT convergence fails at d=64
- **Fix: K cache auto-falls back to q8_0** + V keeps WHT (values tolerate noise in weighted sums)

| Model | head_dim | Config | KV Memory | Compress | PPL (2K) | Prompt t/s | Gen t/s | Pauli Test |
|:------|:---------|:-------|----------:|---------:|---------:|---------:|--------:|:---:|
| GPT OSS 120B | 64 | F16/F16 | 4,608 MiB | 1.0x | 2413 | 133 | 47 | ✅ |
| GPT OSS 120B | 64 | **q8_0/tbq3 (auto)** | **1,692 MiB** | **2.7x** | **1925** | **145** | **46** | **✅** |
| GPT OSS 20B | 64 | F16/F16 | 3,072 MiB | 1.0x | 4008 | 412 | 74 | ✅ |
| GPT OSS 20B | 64 | **q8_0/tbq3 (auto)** | **1,128 MiB** | **2.7x** | **2649** | **421** | **75** | **✅** |
| Qwen3.5-122B | 256 | F16/F16 | 6,144 MiB | 1.0x | — | 91 | 23 | ✅ |
| Qwen3.5-122B | 256 | **tbqp3/tbq3 (auto)** | **1,188 MiB** | **5.2x** | **—** | **102** | **22** | **✅** |

**Input validation — safe handling of all invalid combinations:**
- TBQP (QJL) for V → auto-downgrade to TBQ
- Wrong _N suffix → auto-correct or fallback based on head_dim
- Unsupported head_dim → q8_0/f16 fallback with warning

#### Why the "Pauli Test"? — When PPL Lies

**The problem:** Perplexity (PPL) is the standard metric for evaluating KV cache quantization. Lower PPL = better, right? We discovered this is **dangerously misleading** for WHT-based quantization at small head dimensions.

With our cross-head WHT implementation (512-element WHT across 8 heads, head_dim=64), we achieved PPL numbers that looked *better* than FP16:

| Config | PPL (2K) | Generation Quality |
|:-------|:---------|:-------------------|
| F16/F16 (baseline) | 4008 | ✅ Correct |
| cross-head WHT 3-bit | **2195** (looks better!) | ❌ Completely broken |

**PPL improved by 45% — but the model couldn't even spell names correctly.** The WHT quantization noise acts as attention smoothing: it reduces extreme wrong predictions (lowering average surprise = lower PPL) while destroying the sharp attention peaks needed to pick the single correct token (breaking generation).

**The test:** We needed a metric that catches this. We chose German scientist names translated to Korean standard notation — specifically "Wolfgang Pauli" → "볼프강 파울리".

Why this works as a benchmark:
- **Cultural specificity:** Korean has an official national standard ([외래어 표기법](https://kornorms.korean.go.kr/)) for transliterating foreign names. "Wolfgang Pauli" must be rendered as "볼프강 파울리" (bol-peu-gang pa-ul-li) — not "볼프강 파우리" or "워워우 파울리" or any other variant.
- **Extreme difficulty for LLMs:** Unlike English/Chinese/Japanese where multiple transliterations are acceptable, Korean has exactly ONE correct answer per name. This is determined by standardized phonological rules that map German phonemes to Korean syllables.
- **Sensitivity to attention precision:** Getting multi-syllable Korean transliteration correct requires the model to maintain precise token-by-token attention over the source name. Any attention blurring from quantization noise immediately produces wrong syllables.
- **Validation standard:** The correct answer matches what appears in Korean middle/high school science textbooks — an objective, verifiable ground truth.

**What we found across all configurations (head_dim=64):**

| K type | bits | "Wolfgang Pauli" → | PPL |
|:-------|:-----|:-------------------|:----|
| F16 | 16 | 볼프강 파울리 ✅ | 4008 |
| q8_0 | 8 | 볼프강 파울리 ✅ | — |
| tbq4_0 (4-bit WHT) | 4 | 볼프강 **파우리** ⚠️ (one syllable wrong) | — |
| tbq3_0 (3-bit WHT) | 3 | **파이브라스** ❌ (nonsense) | — |
| tbqp3_0 (3-bit + QJL) | 3 | **er** ❌ (not even Korean) | — |
| tbqp3_3 (cross-head) | 3 | **2.** ❌ | **2195** (misleadingly "good") |

The cross-head configuration that achieved the "best" PPL (2195) produced the worst generation output. **PPL and generation quality were inversely correlated.**

**Root cause:** The TurboQuant paper (ICLR 2026) only validated on head_dim=128 models (Gemma, Mistral, Llama-3.1-8B). At head_dim=64, the Central Limit Theorem convergence required by WHT is insufficient — coordinates don't approximate Gaussian well enough for accurate scalar quantization.

**Our solution:** For head_dim=64, K cache automatically falls back to q8_0 (bypassing WHT entirely), while V cache keeps WHT (value weighted sums are more noise-tolerant). This passed the Pauli test while maintaining 2.7x compression.

**Lesson learned:** Always validate KV cache quantization with generation-quality tests, not just PPL. A method that smooths attention distributions can improve PPL while destroying the model's ability to generate precise outputs.

> **Note: About the Cross-Head WHT code**
>
> v1.2.0 includes cross-head WHT implementations developed for head_dim=64 models: 8 KV heads grouped into a 512-element WHT, Kronecker decomposition H_512 = H_8 ⊗ H_64, V cross-head scoring, etc. This code improved PPL at head_dim=64 but failed to preserve generation quality, so it is **not used in auto-mapping**. However, we intentionally retain the code for future research: cross-head experiments at head_dim>=128 (e.g., 1024-element WHT), comparison with alternative methods (KITTY, learned rotation), or new approaches that may achieve better CLT convergence. Related types: `_3` suffix (tbq3_3, tbq4_3, tbqp3_3, tbqp4_3).

---

### v1.1.0 — head_dim 64/128 Support + Direct Sign Residual Correction

**Multi head_dim support:**
- `head_dim=256`: existing (Qwen3.5, Qwen3-Next) — QJL residual correction
- `head_dim=128`: **new** (Llama, Qwen3, Mistral, MiniMax, most models) — Direct Sign correction
- `head_dim=64`: **new** (gpt-oss, smaller models) — Direct Sign correction
- Auto-detected — user CLI unchanged (`--cache-type-k tbqp3_0` works for all)

**Direct Sign — 4.3x lower variance than paper's QJL:**

The paper's QJL uses SRHT random projection for residual correction, but at d≤128 the projection noise exceeds the correction benefit. Direct Sign stores `sign(residual)` directly:
- 4.3x variance reduction: `(1-2/π)/(π/2) = 0.23`
- No second WHT needed → faster query preprocessing
- QJL retained for d=256 where it excels (hybrid strategy)

#### head_dim=128 Benchmark (Qwen3-30B-A3B Q4_K_M, 2K context, DGX Spark GB10)

| KV Config | PPL | Speed (t/s) | KV Size | Compression |
|:----------|----:|----------:|--------:|------------:|
| f16 + f16 (baseline) | 6.69 | 87.8 | 192 MiB | 1.0x |
| q8_0 + q8_0 | 6.68 | 84.3 | 102 MiB | 1.9x |
| q4_0 + q4_0 | 7.33 | 85.0 | 54 MiB | 3.6x |
| tbq4_0 + tbq4_0 | 7.02 | 68.6 | 50 MiB | 3.9x |
| tbq4_0 + tbq3_0 | 7.19 | 68.1 | 44 MiB | 4.4x |
| **tbqp4_0 + tbq3_0 (Direct Sign)** | **7.08** | **63.6** | **44 MiB** | **4.3x** |
| tbqp3_0 + tbq3_0 (Direct Sign) | 7.95 | 65.3 | 38 MiB | 5.0x |

#### Direct Sign vs QJL Comparison (head_dim=128, TBQP3/TBQ3)

| Method | PPL | Note |
|:-------|----:|:-----|
| QJL (paper) | 11.04 | Projection noise dominates at d=128 |
| **Direct Sign (v1.1.0)** | **7.95** | **PPL reduced by 3.09** |

#### Bug Fix

- `__syncthreads()` race condition: query WHT shared memory corruption at ncols=2 (prompt eval) — PPL exploded to 2000+ while token generation (ncols=1) appeared normal

---

### v1.0.0 Benchmark (head_dim=256)

**Key result: `tbqp3_0 + tbq3_0` = 5.2x compression + lower PPL than FP16 + 12% faster**

### Benchmark Setup

- **Model**: Qwen3.5-35B-A3B Q4_K_M (19.71 GiB)
- **System**: NVIDIA DGX Spark, GB10 GPU, 128GB Unified Memory, CUDA 13.0
- **Dataset**: wikitext-2-raw (test set)

### Performance Summary

| KV Config | KV Memory | Compression | PPL (2K) | PPL (8K) | Speed (8K) |
|:----------|----------:|------------:|---------:|---------:|-----------:|
| f16 + f16 (baseline) | 5,120 MiB | 1.0x | 4.678 | 6.829 | 51.9 t/s |
| q8_0 + q8_0 | 2,720 MiB | 1.9x | 4.679 | 6.806 | 50.1 t/s |
| tbq3_0 + tbq3_0 | 980 MiB | 5.2x | 4.756 | 6.963 | 63.5 t/s |
| **tbqp3_0 + tbq3_0** | **990 MiB** | **5.2x** | **4.672** | **6.850** | **58.3 t/s** |

### Key Findings

```
Memory:   5,120 → 990 MiB  (81% savings)
PPL@2K:   4.678 → 4.672   (better than FP16!)
PPL@8K:   6.829 → 6.850   (+0.3% difference)
Speed@8K: 51.9  → 58.3 t/s (+12% faster)
```

### Perplexity by Context Length

| KV Config | 2K | 4K | 8K | 32K |
|:----------|---:|---:|---:|----:|
| f16 + f16 | 4.678 | 6.591 | 6.829 | 6.151 |
| q8_0 + q8_0 | 4.679 | 6.585 | 6.806 | 6.144 |
| tbq3_0 + tbq3_0 | 4.756 | 6.736 | 6.963 | 6.201 |
| **tbqp3_0 + tbq3_0** | **4.672** | **6.683** | **6.850** | **6.273** |

### Token Generation Speed (t/s)

| KV Config | 2K | 4K | 8K | 32K |
|:----------|---:|---:|---:|----:|
| f16 + f16 | 51.1 | 52.5 | 51.9 | 51.6 |
| q8_0 + q8_0 | — | 52.3 | 50.1 | — |
| tbq3_0 + tbq3_0 | 66.9 | 66.4 | 63.5 | 66.3 |
| **tbqp3_0 + tbq3_0** | **63.9** | **63.0** | **58.3** | **63.3** |

### QJL Correction Effect

QJL (Quantized Johnson-Lindenstrauss) = paper's TurboQuant_prod. Adds 1-bit residual correction to Key.

| Context | tbq3_0 (no QJL) | tbqp3_0 (with QJL) | Improvement |
|:--------|---:|---:|:---:|
| 2K | 4.756 | **4.672** | **-0.084** |
| 4K | 6.736 | **6.683** | **-0.053** |
| 8K | 6.963 | **6.850** | **-0.113** |
| 32K | 6.201 | 6.273 | +0.072 |

### Usage

```bash
# Build
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Recommended: QJL Key + TBQ Value (5.2x compression, FP16-level quality)
./build/bin/llama-cli -m model.gguf -ngl 99 \
  --cache-type-k tbqp3 --cache-type-v tbq3

# Speed priority (5.2x compression, fastest)
./build/bin/llama-cli -m model.gguf -ngl 99 \
  --cache-type-k tbq3_0 --cache-type-v tbq3_0
```

### Supported KV Combinations

✅ = Working (CUDA accelerated) / ❌ = llama.cpp limitation

| K \ V | f16 | q8_0 | tbq3_0 | tbq4_0 |
|:------|:---:|:----:|:------:|:------:|
| f16 | ✅ | ❌ | ❌ | ❌ |
| q8_0 | ✅ | ✅ | ❌ | ❌ |
| tbq3_0 | ✅ | ✅ | ✅ | ✅ |
| tbq4_0 | ✅ | ✅ | ✅ | ✅ |
| tbqp3_0 | ✅ | ✅ | ✅ | ✅ |
| tbqp4_0 | ✅ | ✅ | ✅ | ✅ |

### Type Descriptions

| Type | Description | bpw |
|:-----|:-----------|----:|
| tbq3_0 | TurboQuant 3-bit (WHT + Lloyd-Max) | 3.06 |
| tbq4_0 | TurboQuant 4-bit (WHT + Lloyd-Max) | 4.06 |
| tbqp3_0 | TurboQuant_prod 3-bit (2-bit LM + 1-bit QJL) | 3.12 |
| tbqp4_0 | TurboQuant_prod 4-bit (3-bit LM + 1-bit QJL) | 4.12 |

</details>

---

## Applying the Patch / 패치 적용

<details>
<summary>🇰🇷 한국어</summary>

이 구현은 llama.cpp 커밋 [`f5d1c4179`](https://github.com/ggml-org/llama.cpp/commit/f5d1c4179) 기준입니다.

```bash
# 방법 1: 이 저장소를 직접 빌드
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# 방법 2: 동일 커밋의 llama.cpp에 패치 적용
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout f5d1c4179
git apply /path/to/turboquant/turboquant_kv_cache.patch
```

### 수정된 파일 목록

| 파일 | 변경 내용 |
|:-----|:---------|
| `ggml/include/ggml.h` | TBQ 타입 enum 추가 (41-44) |
| `ggml/src/ggml-common.h` | 블록 구조체 정의 |
| `ggml/src/ggml.c` | 타입 traits 등록 |
| `common/arg.cpp` | KV 캐시 타입 CLI 인자 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | SET_ROWS 지원 등록 |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | KV 양자화 함수 (SET_ROWS) |
| `ggml/src/ggml-cuda/set-rows.cu` | SET_ROWS dispatch |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Fused attention 연산 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Flash Attention vec 커널 |
| `ggml/src/ggml-cuda/fattn.cu` | FA dispatch + 조합 검증 |
| + 16 template instances | FA 커널 인스턴스화 |

</details>

<details open>
<summary>🇺🇸 English</summary>

This implementation is based on llama.cpp commit [`f5d1c4179`](https://github.com/ggml-org/llama.cpp/commit/f5d1c4179).

```bash
# Option 1: Build this repository directly
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Option 2: Apply patch to matching llama.cpp commit
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout f5d1c4179
git apply /path/to/turboquant/turboquant_kv_cache.patch
```

### Modified Files

| File | Changes |
|:-----|:--------|
| `ggml/include/ggml.h` | TBQ type enums (41-44) |
| `ggml/src/ggml-common.h` | Block structure definitions |
| `ggml/src/ggml.c` | Type traits registration |
| `common/arg.cpp` | KV cache type CLI arguments |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | SET_ROWS support registration |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | KV quantization functions (SET_ROWS) |
| `ggml/src/ggml-cuda/set-rows.cu` | SET_ROWS dispatch |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Fused attention operations |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Flash Attention vec kernel |
| `ggml/src/ggml-cuda/fattn.cu` | FA dispatch + combination validation |
| + 16 template instances | FA kernel instantiations |

</details>

---

## Technical Details / 기술 상세

<details>
<summary>🇰🇷 한국어</summary>

### 구현 아키텍처

TurboQuant는 KV 캐시 압축에 최적화된 양자화 기법입니다:

1. **Walsh-Hadamard Transform (WHT)**: 벡터를 회전하여 가우시안 분포로 변환
2. **Lloyd-Max 양자화**: N(0,1) 분포에 최적인 스칼라 양자화기 적용
3. **QJL 보정** (tbqp 타입): 1-bit 잔차 보정으로 비편향 내적 추정

#### Key 캐시 처리 흐름

```
[쓰기] 새 Key → WHT(Key) → Lloyd-Max 양자화 → 3-bit 인덱스 저장
                                              → 잔차 계산 → SRHT → QJL 1-bit 부호 저장 (tbqp만)

[읽기] Query × Key 내적:
  WHT(Query) × centroid[key_idx] × scale  (MSE 점수)
  + WHT₂(Query) × sign[qjl_idx] × scale₂  (QJL 보정, tbqp만)
```

#### Value 캐시 처리 흐름

```
[쓰기] 새 Value → WHT(Value) → Lloyd-Max 양자화 → 인덱스 저장

[읽기] softmax(QK) × V:
  WHT 도메인에서 centroid lookup → 가중합 계산 → IWHT(결과)
  (IWHT는 선형 연산이므로 가중합 후 한 번만 적용)
```

#### 핵심 설계 결정

- **Q8_1 비사용**: WHT가 양자화 오차를 16배 증폭 → F32 활성화 사용
- **직렬 WHT**: Flash Attention은 128 스레드, WHT는 256 원소 → 스레드 0이 직렬 처리
- **Fused 커널**: Key 역양자화 없이 WHT(Query)×centroid 직접 계산
- **V IWHT**: softmax×V 가중합 후 FP16 하이브리드 IWHT (shfl + shared memory)
- **독립 QJL 부호**: MSE용 WHT와 QJL용 SRHT에 서로 다른 부호 패턴 사용 (상관 잡음 방지)

### 왜 KV 캐시에만 적용하는가?

| | 가중치 양자화 | KV 캐시 압축 |
|:--|:--|:--|
| WHT 오버헤드 | 매 matmul마다 (느림) | Query당 1회 (무시 가능) |
| 효과 | Q4_K_M보다 느림 | F16보다 빠르고 5.2x 절약 |
| 논문 핵심 | 부수적 | **핵심 기여** |

가중치 양자화는 이미 Q4_K_M 등 고도로 최적화된 기법이 있어 TurboQuant의 이점이 없습니다. KV 캐시는 컨텍스트 길이에 비례하여 증가하므로 압축 효과가 극대화됩니다.

</details>

<details open>
<summary>🇺🇸 English</summary>

### Implementation Architecture

TurboQuant is a quantization technique optimized for KV cache compression:

1. **Walsh-Hadamard Transform (WHT)**: Rotates vectors to approximate Gaussian distribution
2. **Lloyd-Max quantization**: Applies optimal scalar quantizer for N(0,1) distribution
3. **QJL correction** (tbqp types): 1-bit residual correction for unbiased inner product estimation

#### Key Cache Pipeline

```
[Write] New Key → WHT(Key) → Lloyd-Max quantize → store 3-bit indices
                                                 → compute residual → SRHT → store QJL 1-bit signs (tbqp only)

[Read]  Query × Key dot product:
  WHT(Query) × centroid[key_idx] × scale  (MSE score)
  + WHT₂(Query) × sign[qjl_idx] × scale₂  (QJL correction, tbqp only)
```

#### Value Cache Pipeline

```
[Write] New Value → WHT(Value) → Lloyd-Max quantize → store indices

[Read]  softmax(QK) × V:
  Centroid lookup in WHT domain → weighted sum → IWHT(result)
  (IWHT is linear, so applied once after weighted sum)
```

#### Key Design Decisions

- **No Q8_1**: WHT amplifies quantization error 16x → use F32 activations
- **Serial WHT**: Flash Attention has 128 threads, WHT needs 256 elements → thread 0 does serial WHT
- **Fused kernel**: Compute WHT(Query)×centroid directly without Key dequantization
- **V IWHT**: FP16 hybrid IWHT (shfl + shared memory) after softmax×V weighted sum
- **Independent QJL signs**: Different sign patterns for MSE WHT and QJL SRHT (prevents correlated noise)

### Why KV Cache Only?

| | Weight Quantization | KV Cache Compression |
|:--|:--|:--|
| WHT overhead | Every matmul (slow) | Once per query (negligible) |
| Effect | Slower than Q4_K_M | Faster than FP16 + 5.2x savings |
| Paper focus | Secondary | **Core contribution** |

Weight quantization already has highly optimized methods (Q4_K_M, etc.) where TurboQuant offers no advantage. KV cache grows linearly with context length, maximizing the compression benefit.

</details>

---

## Development History / 개발 히스토리

<details>
<summary>🇰🇷 한국어</summary>

### 구현 과정

이 프로젝트는 논문을 처음부터 llama.cpp의 CUDA 커널로 구현한 작업입니다. 주요 단계:

#### Phase 1: 가중치 양자화 (시행착오)

처음에는 논문의 가중치 양자화를 구현했으나, WHT 오버헤드가 매 matmul마다 발생하여 Q4_K_M (72 t/s) 대비 훨씬 느린 속도 (23 t/s)를 보였습니다. 이를 통해 **TurboQuant의 핵심은 KV 캐시 압축**임을 확인하고 방향을 전환했습니다.

#### Phase 2: KV 캐시 압축 (핵심 구현)

- Flash Attention vec 커널에 TBQ Key/Value 지원 추가
- Fused attention: Key 역양자화 없이 WHT(Query) × centroid 직접 계산
- Value: WHT 도메인 저장 + softmax×V 후 IWHT 적용

#### Phase 3: QJL 보정 (TurboQuant_prod)

- tbqp3_0/tbqp4_0 타입 추가: (b-1)-bit Lloyd-Max + 1-bit QJL
- 독립 SRHT (다른 부호 패턴)로 비편향 내적 추정
- PPL이 F16보다 좋아지는 결과 달성

### 해결한 주요 버그들

| 문제 | 원인 | 해결 |
|:-----|:-----|:-----|
| PPL 폭발 | 1/√d 스케일링이 WHT 후 centroid 분포와 불일치 | 불필요한 스케일링 제거 |
| Q8_1 오차 증폭 | WHT가 int8 양자화 오차를 16배 증폭 | F32 활성화 사용 |
| Warp reduction 데드락 | 8 스레드만 shfl_down 호출, 나머지 24 스레드 미참여 | 직렬 합산으로 변경 |
| nthreads(128) < D(256) | FA 128 스레드로 256원소 WHT 불가 | 스레드 0 직렬 WHT |
| j 루프 syncthreads 데드락 | 일부 warp가 루프 이탈 후 syncthreads 미스 | 1열씩 처리로 변경 |
| MMA/tile 커널 라우팅 | TBQ가 존재하지 않는 MMA 커널로 라우팅 | VEC 커널 조기 반환 |
| 템플릿 인스턴스 누락 | 교차 비트 조합 인스턴스 미생성 → GGML_ABORT | 16개 전체 인스턴스 생성 |
| Q_ds OOB 접근 | TBQP가 Q_ds[0..3] 접근하나 배열 크기 2 | 동적 Q_ds_size (TBQP=4) |
| QJL 32K 속도 저하 | Key당 내적 2회 → 계산량 2배 | float 곱셈 제거, 부호 반전으로 최적화 |
| QJL 품질 저하 | MSE용 WHT와 동일 부호 사용 (상관 잡음) | 독립 qjl_signs 패턴 사용 |

### 파일별 변경 개요 (906줄 추가, 6줄 수정)

총 10개 파일 수정 + 16개 신규 파일:

```
cpy-utils.cuh    +265  KV 캐시 양자화 (WHT + Lloyd-Max + QJL)
fattn-common.cuh +274  Fused attention 연산 (vec_dot_KQ, dequantize_V)
fattn-vec.cuh    +208  Query WHT 전처리, V IWHT 후처리
fattn.cu          +51  FA dispatch + 조합 검증
set-rows.cu       +40  SET_ROWS TBQ dispatch
ggml-common.h     +36  블록 구조체 정의
ggml.c            +24  타입 traits
ggml.h             +6  타입 enum
ggml-cuda.cu       +4  SET_ROWS 지원
arg.cpp            +4  CLI 인자
```

</details>

<details open>
<summary>🇺🇸 English</summary>

### Implementation Journey

This project implements the TurboQuant paper from scratch as CUDA kernels in llama.cpp. Key phases:

#### Phase 1: Weight Quantization (Trial and Error)

Initially implemented the paper's weight quantization, but WHT overhead per matmul made it much slower (23 t/s) than Q4_K_M (72 t/s). This confirmed that **TurboQuant's core value is KV cache compression**, and we pivoted accordingly.

#### Phase 2: KV Cache Compression (Core Implementation)

- Added TBQ Key/Value support to Flash Attention vec kernel
- Fused attention: compute WHT(Query) × centroid directly without Key dequantization
- Value: WHT-domain storage + IWHT after softmax×V

#### Phase 3: QJL Correction (TurboQuant_prod)

- Added tbqp3_0/tbqp4_0 types: (b-1)-bit Lloyd-Max + 1-bit QJL
- Independent SRHT (different sign patterns) for unbiased inner product estimation
- Achieved PPL better than FP16

### Major Bugs Resolved

| Problem | Root Cause | Fix |
|:--------|:-----------|:----|
| PPL explosion | 1/√d scaling mismatched centroid distribution after WHT | Removed unnecessary scaling |
| Q8_1 error amplification | WHT amplifies int8 quantization error 16x | Use F32 activations |
| Warp reduction deadlock | Only 8 threads call shfl_down, other 24 don't participate | Changed to serial summation |
| nthreads(128) < D(256) | FA has 128 threads, WHT needs 256 elements | Serial WHT on thread 0 |
| j-loop syncthreads deadlock | Some warps exit loop early, miss syncthreads | Process one column at a time |
| MMA/tile kernel routing | TBQ routed to non-existent MMA kernel | Early VEC kernel return |
| Missing template instances | Cross-bit combinations not instantiated → GGML_ABORT | Created all 16 instances |
| Q_ds OOB access | TBQP accesses Q_ds[0..3] but array size was 2 | Dynamic Q_ds_size (TBQP=4) |
| QJL slow at 32K | Two dot products per key doubled computation | Replaced float multiply with sign-flip |
| QJL quality degradation | Same WHT signs for MSE and QJL (correlated noise) | Independent qjl_signs pattern |

### Changes Overview (906 lines added, 6 modified)

Total 10 files modified + 16 new files:

```
cpy-utils.cuh    +265  KV cache quantization (WHT + Lloyd-Max + QJL)
fattn-common.cuh +274  Fused attention ops (vec_dot_KQ, dequantize_V)
fattn-vec.cuh    +208  Query WHT preprocessing, V IWHT postprocessing
fattn.cu          +51  FA dispatch + combination validation
set-rows.cu       +40  SET_ROWS TBQ dispatch
ggml-common.h     +36  Block structure definitions
ggml.c            +24  Type traits
ggml.h             +6  Type enums
ggml-cuda.cu       +4  SET_ROWS support
arg.cpp            +4  CLI arguments
```

</details>

---

## References / 참조

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google DeepMind)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Base framework
- **Base commit**: [`f5d1c4179`](https://github.com/ggml-org/llama.cpp/commit/f5d1c4179)

## License

This implementation follows the llama.cpp project license (MIT).
