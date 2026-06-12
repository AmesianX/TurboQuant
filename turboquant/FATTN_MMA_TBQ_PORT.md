# fattn-mma TBQ port — design & progress

Goal (user policy): MMA is the default FA kernel everywhere, vec = fallback only. Port TBQ
dequant into the MMA tile loader (per-tile global→smem dequant to f16 — NOT the full-cache
to_fp16 that freezes GB10 and lacks `_1`-family converters anyway).
Baseline frozen @ 910f335ea (audit-validated). Survey agents' full maps: 2026-06-12 session.

## Architecture (from the survey)

**The seam:** `flash_attn_ext_f16_load_tile` (fattn-mma-f16.cuh:363-448) — generic K/V
global→smem loader, 5 call sites (:606, :625, :1033, :1049, :1366). Forcing `nstages=0` for
quantized types collapses to TWO insertion points (:625 K, :1049 V). Everything downstream
(ldmatrix, mma, softmax, combine) only sees the f16 smem tile. cp.async is incompatible with
dequant-on-load (16B raw copy) — quantized always takes the sync `ggml_cuda_memcpy_1` path.

**Dequant primitives (REUSE, do not rewrite):** `dequantize_V_tbq3_1/amxv3_1/tbqv3_1`
(fattn-common.cuh:937-1045): value = `__half2float(blk->d) * c3[3-bit idx]`, per-pair, reads
via 2-byte-aligned 4B `ggml_cuda_memcpy_1` windows (raw byte indexing froze GB10 — comment
:948-950). Same math works for K tiles (amx3_1 needs a d_wht/stride-108 variant).
TBQP3_1 K folds EXACTLY into a tile value: `norm*cent[idx] + d_qjl*(sign? +1 : -1)`
(Direct Sign dots the same WHT'd q — fattn-common.cuh:1052-1110). TBQP `_0`/`_4` cannot fold
(independent QJL projection of Q) — keep those vec/scalar-correction.

**Domain handling:**
- K cache lives in WHT space; Q must be WHT'd: vec does it in-kernel (fattn-vec.cuh:506-566,
  sign table + 5 shuffle + 2 smem stages, result `WHT(S*q)*scale/D`); MMA should do it
  host-side per call into a temp Q buffer (pattern exists: `tbq_q_wht12_cuda` convert.cu:958,
  D=128 forward-WHT kernel :846-960). Fold `scale/D` there; kernel then runs with scale=1? NO —
  keep kernel scale, fold only the WHT normalization exactly as vec does.
- V dequant yields WHT-domain values → VKQ output needs one IWHT+sign-undo+/128 per output row:
  vec does it at fattn-vec.cuh:1352-1393 with fp32 staging; MMA: run `tbq_output_iwht` kernel
  (convert.cu:930, currently unused) on dst after launch_fattn.
- tbq3_1/amx3_1 outliers (post-RoPE domain, in-block side band ol_idx/ol_val): cannot fold into
  WHT tile. Use the existing QJL side-channel pattern (`raw_K_data` kernel param + in-_iter
  rank-2 scalar correction on KQ_C, fattn-mma-f16.cuh:693-760, currently DKQ==576-gated) with
  RAW Q (vec reference: fattn-vec.cuh:899-919). amx3_1 ol_val==0 ⇒ no-op (env-gated at quant).
- AMX3_1 Part B (polar r/φ for TriAttention) is ignored by FA; Part A is bit-compatible with
  tbq3_1 at d_wht offset 0, block stride 108.

**Constraints:** TBQ blocks never cross K rows (blck 128 for `_1` @ D=128 ⇒ 1 block/row ✓).
nbatch_K2*2 elems block-aligned for all relevant configs (D=128: nbatch_K2=64 = 128 elems = 1
block ✓). TBQ row strides not 16B-aligned ⇒ global reads only via ggml_cuda_memcpy_1<4,2>.
`K->ne[1] % FATTN_KQ_STRIDE(256) == 0` precondition stays. OOB rows must zero-fill like :436.

**Dispatch & instances:**
- Add `type_K/type_V` template params (default F16) to `flash_attn_ext_f16` + `_case`; existing
  20 f16 instance files stay byte-identical via the default; new
  `DECL_FATTN_MMA_F16_TBQ_CASE(DKQ,DV,ncols1,ncols2,tK,tV)` + hand-written per-pair instance
  files `fattn-mma-f16-instance-tbq-<k>-<v>.cu` (generator wipes *.cu on rerun — keep TBQ files
  out of its glob; vec TBQ files survive only because nobody reran it).
- Typed path passes `need_f16_K/V=false` into launch_fattn (kills full-cache dequant) and
  byte-stride for quantized K/V (today `stride_K = nb11/sizeof(half2)` :1883).
- Selection: support table `ggml_cuda_fattn_mma_tbq_supported(D,tK,tV)` generated from the same
  macro list as instances (1:1 or link-abort). Replace the blanket VEC return at fattn.cu:991
  with: supported && turing_mma && gqa/ncols2 prereqs && Q->ne[1] above vec threshold → MMA;
  else VEC (= automatic fallback for uncompiled pairs).
- Production pairs that matter (common.cpp tbq_resolve: 128→`_1`, 64→`_2`(K force `_3` QJL),
  256/512→`_0`, 576→`_4`; TBQP V auto-downgrades): D=128 {tbq3_1×tbq3_1, tbq3_1×tbqv3_1,
  tbqp3_1×tbq3_1, tbqp4_1×tbq4_1, amx3_1×amxv3_1, ×f16-V} first (Qwen 0.87× penalty lives
  here), then `_0` (256/512), `_2/_3` (64), `_4` (576 — partial MMA infra already exists:
  tbqp_wht_mode + raw_K QJL hook).

## Milestones
- [x] M0: TBQV3_1 dispatch hole (92923d367) — f16-K×tbqv3-V prefill reached full-cache dequant.
- [ ] M1: type plumbing (defaulted template params, byte strides, nstages=0 force, need_f16=false)
      + `load_tile_dequant` for TBQ3_1-K × TBQV3_1-V @ D=128 + host Q-WHT + output IWHT +
      outlier rank-2 hook. Validate: MMA-vs-VEC forced A/B greedy identity on Qwen3-14B tbq3
      (env GGML_CUDA_FA_FORCE_KERNEL=vec|mma for testing), numerics tolerance fp16-level.
- [ ] M2: remaining D=128 pairs (tbq4_1, tbqp3_1/tbqp4_1 fold, amx3_1×amxv3_1, f16-V mixes).
- [ ] M3: dispatch flip (MMA default for supported combos, vec fallback) + gates:
      Qwen3-14B tbq3 decode 0.87×→~1.0× of f16; K≈2048 freeze scenario; DSV4 greedy/t/s
      unchanged; full test-backend-ops; long-ctx (8K+) quality spot check.
- [ ] M4: `_0` (256/512 → Gemma4/Qwen3.5/DSV4-kv_base), `_2/_3` (64), `_4` MLA quality re-port
      (the fattn.cu:983 experiment block resolution).

## Session log
- 2026-06-12: survey (3 agents), design doc, M0 landed.
- 2026-06-12 S1 (b7b2da012): type_K/type_V template plumbing, defaulted F16, nstages=0 force
  for quantized; zero behavior change (FLASH_ATTN_EXT suite clean).
- 2026-06-12 S2a (4cb9328da): V-side VALIDATED. flash_attn_ext_f16_load_tile_dequant
  (128-block family), byte stride_V, typed _case (need_f16_V=false + standalone 128-pt output
  IWHT on dst), fattn-mma-tbq.cu dispatcher TU (root dir — generator-safe), env gate
  GGML_CUDA_FA_MMA_TBQ. Validation on Qwen3-14B q4_k_m, -ctk f16 -ctv tbq3:
  * short greedy (14-tok prompt, n=32): text IDENTICAL to vec
  * long prompt (550 tok): greedy diverges at a degenerate repetition point (VEC side was the
    incoherent one) — adjudicated by PPL instead
  * wikitext PPL (8×2048): vec 10.6629 ±0.324 vs MMA 10.6302 ±0.322 → 0.31% apart, parity
  * MMA engagement evidence + speed: prefill 900.7 → 1070.7 t/s (+19%); PPL run wall 25.1s →
    17.4s (-31%)
  Decode (ne1<=2) intentionally stays on vec.
- NEXT S2b (K-side): apply Q-WHT to the Q tile at process_tile load (or host-side pre-pass),
  K tiles through the same dequant loader (tbq3_1 d at offset 0 — works as-is), rank-2 outlier
  correction on KQ_C following the DKQ==576 QJL hook pattern with raw-Q global reads. Then
  TBQP3_1 K via the Direct-Sign fold. Then M3 dispatch flip + 0.87×→1.0× gate.
