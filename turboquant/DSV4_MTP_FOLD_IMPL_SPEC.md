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
Branch created. Attach point proven (h_flat = t_h_pre_norm). Next session: Edit 1 (refactor,
verify identical), then Edit 3 (consume), then Edit 2 (KV) — or Edit 2 before 3 if the folded
head needs its KV to produce correct logits (likely yes: the NextN attn reads its SWA cache).
Recommended order: 1 (refactor) -> 2 (KV) -> fold-attach -> 3 (consume) -> verify.
