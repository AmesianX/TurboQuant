# TurboQuant Side Buffer 개발 로그 (2026-04-11)

## 목표
TBQP3_3 (3-bit K cache)의 수학 정밀도 문제 해결. f16 K와 동일한 품질 달성.

## 배경
- TBQP3_3: 2-bit Lloyd-Max + 1-bit QJL + double WHT per-head (D=64)
- 한글 대화: ✅ 성공 (QJL 필수 — 없으면 한글 완전 붕괴)
- 수학 (행렬 읽기/계산): ❌ 실패 (입력 숫자를 정확히 못 읽음)
- f16 K: 35/35 math bench 완벽
- TBQ4 (4-bit): 35/35 math bench 완벽

## 핵심 아이디어
양자화 전 원본 K를 f16으로 가로채서 별도 버퍼(side buffer)에 저장.
attention이 양자화된 K 대신 side buffer의 f16 K를 읽음.

---

## 실험 과정 (시간순)

### Phase 1: WHT 도메인 f16 side buffer (prompt only)

**구현**: prompt-k-side.cuh에 double WHT + f16 저장 커널 생성.
set-rows에서 ne01 > 1 (prefill)일 때만 write_side 호출.
fattn-vec.cuh에서 side buffer 읽기 경로 추가 (수제 dot product).

**결과**:
- "안녕?" 한글: ✅
- 2×2 행렬 에코 (118 토큰): ✅ 완벽
- B 행렬 단독 (109 토큰): ✅ 완벽
- 4개 행렬 동시 (195 토큰): ❌ B의 28→0 오류

**분석**: WHT 도메인에서 f16 저장 시 정밀도 손실. WHT가 값을 최대 64배 증폭 → f16 절삭 오차도 증폭.

---

### Phase 2: Raw f16 side buffer (WHT 제거)

**변경**: k_double_wht_f32_to_f16 → k_f32_to_f16 (WHT 없이 raw f32→f16 변환).
dot product에서 raw Q × raw K_f16 사용, scale 별도 적용.

**결과**: WHT 도메인보다 개선되었으나 긴 프롬프트에서 여전히 B 오류.

---

### Phase 3: Generation 토큰도 side buffer에 저장

**변경**: ne01 > 1 조건 제거, generation (ne01=1)도 write_side 호출.
in_gen 플래그 제거 (매 generation 토큰마다 reset 일으키는 버그 발견 및 수정).

**발견한 버그**: in_gen=true가 매 generation 토큰마다 설정 → 다음 토큰의 write_side에서 n_tokens=0 리셋 → prompt 데이터 전부 소실.

**결과**: generation 토큰 포함 후에도 B 오류 지속. softmax에서 generation K의 TBQP3_3 노이즈가 간섭한다고 판단 → 전체 f16으로 전환.

---

### Phase 4: SQNR 기반 dynamic sharpening

**구현**: α(N) = 1 + f_gen × √(ln N / ln N₀) / SQNR.
f_gen = (N - prompt_len) / N (generation 토큰 비율).
SQNR 값: TBQP3_3=8.35, TBQ3_3=30.25, TBQP4_3=49.0, TBQ4_3=99.0.

**결과**: 효과 미미. 근본 원인이 sharpening이 아니었음.

---

### Phase 5: 수제 dot product 디버깅

**시도들**:
1. scale pre-multiply (f16 커널 방식): 효과 없음
2. *D /D 왕복 제거: 효과 없음
3. Q WHT 스킵 (side buffer full 시): 오히려 전체 깨짐 (__syncthreads 누락)
4. Q_side_reg 별도 선언 + vec_dot_f16 직접 호출: 빈 출력 (레지스터 레이아웃 변경으로 모든 커널 영향)

**핵심 발견**: 수제 dot product가 수학적으로 더 정확하지만, f16 커널과 미세하게 다른 결과 → 64 레이어 누적 시 차이 발생.

---

### Phase 6: f16 커널 redirect

**구현**: fattn.cu에서 TBQP3_3 + side buffer 활성 시:
1. K->type = F16로 변경
2. K->data = side buffer 포인터
3. K->nb[0,1,2,3] = f16 레이아웃 stride
4. f16 커널 (ggml_cuda_flash_attn_ext_vec_case<64, F16, TBQ3_2>) 직접 dispatch
5. 실행 후 원래 값 복원

**시도 1: Head-major 레이아웃**
- write_side를 head-major로 변경
- nb[1] = per-head token stride, nb[2] = cap × head stride
- 결과: A 깨짐 (61→11, 77→1)

**시도 2: Token-major 레이아웃 (원래대로)**
- nb[1] = 1024 (all-heads token stride), nb[2] = 128 (head stride)
- 결과: A,B,C 완벽, D 128→8 오류

**시도 3: ne[1] = side_n (191)**
- 결과: A,B,C 완벽, D 128→8 오류 (일관적)

**시도 4: ne[1] = 256 (패딩)**
- 결과: A 깨짐 (192~255 위치에 mask가 안 걸리는 것으로 추정)

---

### Phase 7: 디버그 비교 — 근본 원인 발견

**K 값 비교 (KDATA 디버그)**:
```
tbqp3_3 side buffer tok100: [5.0781, 3.5195, -4.5234, -5.1094]
f16 KV cache tok100:        [-0.3289, -0.1147, -0.5571, 0.3875]
```
→ **완전히 다른 값!** 같은 token 100인데 데이터가 다름.

**원인**: write_side가 src1 인덱스를 무시하고 **순차 저장** (0, 1, 2, ...).
set-rows는 src1 인덱스로 KV 캐시 **특정 위치**에 저장.
→ 위치 불일치로 attention이 엉뚱한 토큰의 K를 읽음.

**수정**: k_f32_to_f16_indexed 커널 — src1[i_token] 위치에 저장.
```cuda
const int dst_row = (int)indices[i_token];
dst[dst_row * ne00 + i_elem] = __float2half(src[i_token * s01 + i_elem]);
```

---

### Phase 8: 최종 성공

**결과**: 4개 행렬 100% 완벽 + math bench 35/35 + 멀티턴 성공.

```
A = [[19, 7, 33], [42, 88, 12], [5, 61, 77]]       ✅
B = [[3, 156, 28], [27, 5, 94], [66, 13, 41]]       ✅
C = [[201, 8], [15, 399], [72, 50]]                  ✅
D = [[1024, 777, 333, 111], [256, 512, 128, 64]]     ✅
```

---

## 최종 코드 구조

### prompt-k-side.cuh
- `prompt_k_side_t`: per-layer side buffer (buf, n_tokens, capacity, ne00)
- `get_side_map()`: K cache pointer → side buffer 매핑 (mutex 보호)
- `k_f32_to_f16_indexed`: src1 인덱스 기반 f32→f16 변환 커널
- `write_side()`: set-rows에서 호출, src1 전달, force_reset(ne01>10)
- `get_prompt_k_ptr()`, `get_prompt_k_stride()`: attention에서 side buffer 조회

### set-rows.cu (TBQP3_3 블록)
```cpp
set_rows_cuda_xhead_v1<...>(src0_d, src1_d, ...);  // TBQP3_3 양자화
turboquant::write_side(dst->data, src0_d, src1_d, ne00, ne01, nb01, stream, ne01 > 10);
```

### fattn.cu (f16 redirect)
```cpp
if (K->type == GGML_TYPE_TBQP3_3 && Q->ne[0] == 64) {
    if (side buffer active) {
        K->type = F16; K->data = side_ptr;
        K->ne[1] = side_n;
        K->nb = {2, 1024, 128, ...};
        dispatch f16-TBQ3_2 kernel;
        restore K; return;
    }
}
```

### fattn-vec.cuh
- side buffer 수제 코드 전부 제거
- 원본 TBQP3_3 attention 로직 유지 (fallback용)

---

## 중요 교훈

1. **src1 인덱스**: set-rows의 src1은 KV 캐시 위치 매핑. side buffer도 동일 매핑 필수.
2. **찍어서 비교**: 추측보다 실측. f16 값과 side buffer 값을 직접 비교해서 근본 원인 발견.
3. **ne[1] 패딩**: f16 KV 캐시는 256 패딩, TBQP3_3은 191(미패딩). 커널 분기 차이 유발.
4. **레지스터 레이아웃**: Q_side_reg 추가 시 Q_reg-Q_i32 사이 레이아웃 변경 → 모든 커널 영향.
5. **빌드 캐시**: .cuh 파일 수정 시 반드시 touch 또는 .o 삭제. stale build = false positive.
6. **f16 vec_dot OOB**: D=64에서 cpy_ne=4로 경계 밖 읽기. 실사용에서는 문제없지만 레이아웃 변경 시 다른 가비지 패턴 → 다른 결과.

---

## 미래 작업

### TBQP3_3 오리지널 개선
- 3비트 양자화 자체의 수학 정밀도 향상
- sharpening 파라미터 재튜닝 (side buffer 기준선 활용)

### 3+1비트 분할 양자화 (AmesianX 아이디어)
- VRAM: 3비트 (TBQP3_3 하위 3비트)
- System RAM: 1비트 (cudaHostAlloc zero-copy)
- TBQ4 = 35/35 검증됨 → 3+1 = 4비트급 품질 기대
- PCIe 전송: 200토큰 기준 0.8MB/step = 0.025ms (체감 불가)

### LCP 캐싱 지원
- --no-cache-prompt 없이 멀티턴 지원
- 재활용 구간 K를 TBQP3_3에서 역양자화 → side buffer 복사
- 또는 side buffer를 요청 간 유지 (force_reset 조건 변경)

---

## 환경
- DGX Spark GB10, 128GB 통합 메모리
- GPT OSS 120B (MXFP4), head_dim=64, 8 KV heads, 36 layers (18 SWA + 18 non-SWA)
- llama.cpp b8756-36ac3843b + TurboQuant
