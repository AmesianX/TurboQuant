# DSV4 native sm120 FP8 dense GEMM — status / resume

Branch: `feat/dsv4-fp8-native-gemm` (off `feat/dsv4-sparse-attn-gather` @ acba31c79)
Gate: `DSV4_FP8_NATIVE=1` (default off = byte-identical cuBLAS dequant->F16). `DSV4_FP8_VERIFY=1`,
`DSV4_FP8_DEBUG_SYNC=1` are diagnostics.

## PROFILE (measured, DSV4_KERNEL_PROF, 2521-tok prefill, graphs off)
Dense FP8 (type-42 f8_e4m3_b128) GEMMs = **~38% of GPU time = #1 floor term**, ~2x the MoE grouped (20%).
Breakdown: output_b 7127ms, q_b 6992ms, MUL_MAT_ID(4096x1024,n=8) 8056ms, shexp 3397ms, q_a 1244, kv 861.

## PHASE 2 (decisive)
Dense F8_E4M3_B128 is is_quantized=true but NOT in mmq/mmvq switch -> ggml_cuda_op_mul_mat_cublas
dequantizes the whole FP8 weight to F16 every forward + cuBLAS F16 GEMM (ggml-cuda.cu:1836). NOT native.

## IMPLEMENTED
`dsv4-fp8-gemm.cu/.cuh`: native CUTLASS sm120 blockwise-scaled FP8 GEMM (flashinfer core).
- A=weight[M=n_out,K] e4m3, ScaleGranularityM=1 (exact per-row E8M0 weight scale, expanded to fp32 once/weight)
- B=activations[N=ntok,K] e4m3, ScaleGranularityN=128 (per-128-tok-block scale, quantized each call)
- D=[n_out,ntok] fp16 -> f32. MN-major, K=128. Persistent buffers, no per-call malloc -> capture-safe.
- Guards: ntok%16==0 & N%16==0 (CUTLASS align), contiguous, 2D-only, sm120/121 -> else fall back to cuBLAS.
- per-weight Gemm+wsp (mutable CUTLASS Params raced when shared); SHARED transient aq/sfb/dout.

## VALIDATION
- Standalone (scratchpad fp8_test.cu) vs CPU ref, same quant: **rel_l2 0.0002** (layout correct).
- In-server VERIFY vs dequant ref: rel_l2 ~0.014-0.022 per projection = the per-128-tok-block ACTIVATION
  fp8 quant error (baseline keeps activations F16 -> lossless). End-to-end quality: "프랑스의 수도는 파리"
  correct, fluent Korean (matches baseline output).

## MEASURED PERF (fair control, IDENTICAL config CTX=65536/PARALLEL=1/graph-off, per-chunk tok/s)
| tokens | baseline(native off) | native FP8 |
|--------|---------------------|------------|
| 13312  | 326.98              | 342.69     |
| 13769  | 321.77              | 336.81     |
| 14277  | 315.64 (complete)   | 329.98 (complete @ CTX=32768) |
=> **native FP8 = +~4.5% prefill**. (Earlier "20%" was vs a different graph-on/PARALLEL=2 config — not fair.)
Modest because dense GEMMs at these shapes are LPDDR-bandwidth-bound (DSV4 is memory-bound), and the
per-GEMM activation-quant + output-convert prep eats much of the tensor-core saving.

## BLOCKER (memory)
Native duplicates the dense FP8 weights as contiguous e4m3 + fp32 scales = **~6GB total (~3GB/rank under
-sm tensor)**. With DSV4's growing prefill arena, a full 14k prefill crosses the WATCH_MIN_GB=4 watchdog
at CTX>=65536 (killed at progress 0.96-1.00). Completes at CTX=32768. To productionize: free the original
ggml FP8 tensor after unpack (needs a load hook + fallback rethink), or unpack-per-layer-and-reuse.

## NEXT
1. Cut activation-quant error: option B (per-token act M=1 scale + per-128-out-block weight = DeepSeek's
   recipe) for lossless-er quality; or keep option A (exact weights) — confirm via perplexity gate.
2. Memory: eliminate the weight duplication so CTX=262144 fits.
3. Decide if +4.5% prefill is worth the complexity given DSV4 is memory-bound (the bigger floor lever may
   be the MoE GEMM tiling, not the dense FP8).
Build: .66 only, rsync all 4 (llama-server, libllama, libggml-base, libggml-cuda) -> md5 match both boxes.
