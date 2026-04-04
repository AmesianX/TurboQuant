# TurboQuant Session Log

## 2026-04-04: v1.5.0 — Upstream Rebase + Gemma 4 Support

### Upstream Rebase
- llama.cpp upstream (b7ad48ebd)에 완전 리베이스
- Base commit: `27b93cbd1` (main의 fork point)
- 3-way merge (`diff3`)로 모든 TBQ/TBQP 코드 적용
- 충돌 26개 자동 해결: TBQ 관련 → ours, 비-TBQ → upstream
- `main-old` 브랜치에 기존 코드 백업
- 앞으로 `git fetch upstream && git merge upstream/master`로 동기화

### Gemma 4 지원 (3개 버그 수정)
1. **SWA 캐시 타입 재매핑** (`llama-kv-cache-iswa.cpp`)
   - Gemma 4: global head_dim=512, SWA head_dim=256 혼합
   - SWA head_dim 감지 → TBQ 서브타입 자동 재매핑

2. **가변 GQA attn_rot_k 활성화** (`llama-kv-cache.cpp`)
   - Gemma 4: 레이어마다 head_count_kv 다름 (16/4)
   - TBQ 타입일 때 variable GQA 체크 우회

3. **D=512 vec 커널 2-pass Q WHT** (`fattn-vec.cuh`)
   - 256-block × 2 루프로 512차원 처리
   - QJL WHT도 동일 2-pass

### D=512 TBQ vec dispatch (`fattn.cu`)
- `FATTN_VEC_CASE(512, ...)` + template instances 추가
- `get_best_fattn_kernel`: D≤512 TBQ → VEC, GQA 제한 완화

### 벤치마크
| 모델 | 캐시 | 압축 | 수학 | 파울리 |
|------|------|------|------|--------|
| Gemma 4 31B Dense | tbqp3/tbq3 | 1.8x | 49.2% (f16: 64.6%) | PASS |
| Gemma 4 26B MoE | tbqp3/tbq3 | 5.2x | 46.2% (f16: 56.9%) | PASS |
| Qwen3-14B | tbqp3/tbq3 | 2.5x | 33.8% = f16 33.8% | PASS |
| GLM-4.7-Flash | tbqp3/tbq3 | 3.5x | — | PASS |
| Qwen3.5-27B | tbqp3/tbq3 | — | — | PASS |

### 이슈 대응
- #11 V-cache 버그: f16=TBQ 증명
- #12 Metal: CUDA 전용 설명
- #9, #6 Gemma 4: 완료, 닫음

### 남은 작업
- Gemma 4 수학 정확도 개선 (tbq4, D=512 최적화)
- MMA 텐서코어 D=512 이식
- upstream 정기 동기화
