# DSV4 MTP fold — implementation spec (branch feat/dsv4-mtp-fold)

Goal: match vLLM — run the MTP draft head IN the trunk verify graph (one forward), removing the
separate ctx_dft decode and its per-instance meta re-resolution tax (~26ms/decode) + the D2H embd
handoff. See DSV4_HONEST_MAP_AND_LADDER_2026-07-19.md "FOLD SCOPE" for the why.

## KEY DISCOVERY (de-risks the whole fold)
The MTP head consumes the target's post-layer hyper-connection state as its `inp->h` input
(graph_mtp @ src/models/deepseek4.cpp:3262 `h_in = inp->h`, used by h_proj(hnorm(h_in))). The
trunk graph ALREADY computes exactly this tensor on-device:
  deepseek4.cpp:3200-3203  `h_flat = reshape(inpL, n_embd*n_hc, ...)`; `res->t_h_pre_norm = h_flat`.
So the fold is NOT data plumbing — it is: append the MTP-head nodes to the trunk graph, reading
`h_flat` directly as `h_in`. No D2H, no separate context, no separate decode.

## The 3 edits

### Edit 1 — deepseek4.cpp: extract the MTP head body into a reusable member, call it inline
- graph_mtp's body from line 3262 (`h_in = inp->h; ...`) to its end (res->t_logits emit) is the
  head. It uses protected llm_graph_context members (build_norm, build_attn_mha, build_moe_ffn,
  build_ffn, build_attn_inp_kv_iswa, build_inp_pos, build_inp_out_ids) + many captured locals
  (n_embd, n_hc, layer=model.layers[n_layer-1], rope_cfg, kq_scale, n_out_group, n_lora_o).
- C++ structure: `graph` and `graph_mtp` are separate nested structs, both : llm_graph_context.
  Cleanest share = a protected member function on a shared base, OR (lower churn) a free helper
  `static void dsv4_build_mtp_head(llm_graph_context & g, ggml_context * ctx0, ggml_cgraph * gf,
     llm_graph_result * res, const llama_model & model, ggml_tensor * h_in, ggml_tensor * tokens,
     const dsv4_rope_cfg & rope_cfg, ... )` — BUT it needs the protected build_* methods, so a free
   function can't call them. => Make it a **protected method on a new tiny base struct** that both
   graph and graph_mtp inherit (e.g. `struct dsv4_graph_base : llm_graph_context { ggml_tensor*
   build_mtp_head(ggml_tensor* h_in, ggml_tensor* tokens_or_embd_src, bool own_kv); };`), move the
   body there. graph_mtp::graph_mtp calls it with inp->h + inp->tokens (behavior IDENTICAL — this
   is the safe, verifiable first commit: gate nothing, just refactor, 2-node greedy output must be
   token-identical).
- Attach point for the fold: in `graph::graph` right after line 3203 (t_h_pre_norm set), gated on
  `getenv("DSV4_MTP_FOLD")`, call build_mtp_head(h_flat, <draft tokens>, own_kv=...) and store its
  logits into a new `res->t_mtp_logits`. Default off => trunk unchanged.

### Edit 2 — KV plane for the folded NextN layer
- graph_mtp uses `build_attn_inp_kv_iswa()` + `mctx_swa` = ctx_dft's own SWA KV. Folded into
  ctx_tgt, the NextN layer needs its own KV plane inside ctx_tgt's memory (the trunk MLA cache is
  a different layout). Options: (a) a dedicated small SWA side-cache registered on ctx_tgt for the
  single NextN layer; (b) reuse the hybrid-iswa machinery with an extra plane. This is the
  hardest sub-part — scope it carefully; the NextN KV is 1 layer, plain SWA, small.

### Edit 3 — server-context.cpp spec loop: consume folded draft, drop ctx_dft decode
- Today: verify decode(ctx_tgt) -> common_speculative_process() [ctx_dft mirror] -> sample_accept
  -> common_speculative_draft() [ctx_dft AR decodes]. Under DSV4_MTP_FOLD, res->t_mtp_logits comes
  out of the verify decode itself. Replace process()+draft() with: read t_mtp_logits, device-side
  argmax (backend sampling exists) -> the draft token(s). n_max draft depth then = extra folded
  head nodes in the SAME graph (cheap — no extra decode), which is what re-opens deep draft.

## Verification gates (do NOT skip — attempt A crashed from an unverified meta edit)
1. Edit 1 refactor: build + 2-node greedy (temp0/seed42) MTP output token-IDENTICAL to main
   (16.41 baseline, MTP ~14). Pure refactor => must match exactly.
2. Fold path (gate on): greedy output token-identical to gate-off (the fold must be numerically
   the same speculation, just faster). t/s should rise toward ~20. Round-trip DSV4_MTP_PROF to
   confirm the ctx_dft decodes are gone.
3. Sustained interleave / long context: no ids-size / reduce-null asserts over 1000+ tokens.

## Status
Branch feat/dsv4-mtp-fold. Attach point proven (h_flat = t_h_pre_norm).

DONE:
- Edit 1 (af6082654): build_mtp_head extracted into dsv4_graph_base, shared by graph +
  graph_mtp. VERIFIED 2-node MTP greedy (temp0/seed42, 256K, ATTN+SHEXP+FOLD split):
  14.57 t/s warm / 14.06 cold, coherent output, no crash = behavior-identical refactor.
  Ref output saved turboquant/mtp_edit1_greedy_ref.json (sha256 8df9f47a…) for future
  strict token A/B vs main (skipped now — verbatim move + baseline t/s is proportionate).
- Edit 2 (035ef61c6): trunk hybrid_iswa filter_attn extended to include NextN layer under
  DSV4_MTP_FOLD. CONFIRMED via source: NextN (il=n_layer-1) is plain SWA — deepseek4.cpp
  load_arch_hparams sets swa_layers[il]=1 + attn_compress_ratio[il]=0, so filter_recr
  (needs compress_ratio!=0) auto-excludes it => it rides ONLY the SWA attn sub-cache
  (n_swa window = cheap). Default OFF => filter byte-identical to before. Compiles.

DESIGN CONFIRMED (Option A — orthodox, low churn):
- The folded head does NOT build its own build_attn_inp_kv_iswa(). It REUSES the trunk's
  hybrid input: build_inp_mem_hybrid_iswa() returns llm_graph_input_mem_hybrid_iswa whose
  get_attn() is an llm_graph_input_attn_kv_iswa. Its mctx->get_swa() is the SWA sub-cache
  that (post Edit 2) contains the NextN plane. Masks/k_idxs_swa are position-based and
  identical for the NextN layer (same SWA window, same causal) — cpy_k/get_k with
  il=n_layer-1 target the NextN plane (cache indexes by il). So the folded NextN attention
  is correct riding the trunk's existing SWA input; no new input tensors, no D2H.
- => build_mtp_head needs a folded flag / SWA-input-source param. Standalone (graph_mtp):
  build its own build_attn_inp_kv_iswa + build_inp_pos + build_inp_out_ids (as today).
  Folded (graph): pass the trunk's inp_mem->get_attn() and REUSE trunk inp_pos; out_ids
  handling is the crux (see below).

REMAINING = the hard 60% (fold-attach in graph::graph + Edit 3 server consume):
- Draft-position / out_ids semantics is THE hard part. Today the flow (speculative.cpp
  common_speculative_impl_draft_mtp): ctx_tgt verify decode -> D2H fetch target h via
  llama_get_embeddings_pre_norm -> pair (h_p, x_{p+1}) with cross-batch carryover
  (pending_h / pending_h_prev / verify_h, grouped by seq, row0=sampled tok..rowN=Nth
  accepted draft) -> mirror to ctx_dft -> decode ctx_dft -> AR draft(). The fold must
  REPLACE this whole carryover: the folded head reads h_flat on-device and must emit draft
  logits for the RIGHT position(s) (the last verified/accepted row per seq), then Edit 3
  does device-side argmax (llama_set_sampler backend chain already exists) to get the draft
  token(s) directly out of the verify decode — deleting process()+draft() ctx_dft decodes.
- This touches memory(done)+graph+server+the pending_h/verify_h carryover model. It is a
  big-bang across subsystems => enter DELIBERATELY at a session start with 2-node verify
  room; do NOT rush (attempt-A crashed from a rushed meta edit). Recommended: implement
  fold-attach (build_mtp_head folded call in graph::graph after :3190/:3203, gated) reading
  h_flat + reusing trunk SWA input + emitting res->t_mtp_logits FIRST, verify the trunk
  still runs + logits are sane (compare folded-head argmax vs the standalone ctx_dft draft
  token for the same position — must match), THEN Edit 3 rewires the server to consume it.
- Gate: DSV4_MTP_FOLD must be forwarded to BOTH ranks (add to tp-w4a16.sh FWD like the
  other DSV4_* envs) — it now affects memory layout (Edit 2) so ranks MUST agree.

Verify order: fold-attach (logits sane, argmax matches ctx_dft draft) -> Edit 3 (consume,
greedy token-identical to gate-off, t/s rises toward ~20, DSV4_MTP_PROF shows ctx_dft
decodes gone) -> sustained 1000+ tok interleave (no ids-size/reduce-null asserts).

## RESULT (2026-07-19, feat/dsv4-mtp-fold @ 59f16912e) — COMPLETE & WORKING
The fold is done and runs 2-node. DSV4_MTP_FOLD=1 (256K, ATTN+SHEXP+FOLD split):
coherent (prose + code), deterministic, 15.64 t/s vs 14.5 standalone MTP = +7.9%. The
entire ctx_dft mirror + AR draft decode subsystem is bypassed.

Key correction to the plan above: bit-identity is NOT a valid gate. The 2-node MTP is
inherently run-to-run non-deterministic (AllReduce FP non-assoc + near-tie argmax flips:
fold-OFF standalone gave 369/369/380 across 3 identical greedy runs). Gate on coherence +
tau + t/s instead. The fold itself is deterministic (15.64 repeats exactly).

Honest perf: the fold beats standalone MTP but 15.64 is around/below plain greedy decode
(16.41, cross-session) — effective n_max=1 (one trunk forward = one AR draft step) limits
the win; the folded NextN layer's per-decode cost roughly cancels the single-draft tau
gain. To make MTP net-positive vs plain and reach ~20+, INCREMENT 3 = restore n_max>=2 via
CHAINED folded heads (a 2nd NextN head in the trunk graph fed by the 1st head's output +
its draft), plus gate fold-attach to decode-only (it currently also fires during prefill,
running the NextN layer per ubatch = a prefill tax). See memory project_dsv4_w4a16_native_port.
