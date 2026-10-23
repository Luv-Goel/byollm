# Roadmap — byollm

The engine ships one verified feature at a time. Every item below has a
corresponding test before it lands, and nothing is pushed unless the suite is
green on real model bytes.

## Phase 1 — File format (done ✅)

- [x] GGUF v1/v2/v3 container parser
- [x] All metadata value types (UINT/INT/FLOAT/BOOL/STRING/ARRAY)
- [x] Dequantisers: F32, F16, BF16, Q8_0, Q4_0
- [x] Architecture detector for Llama-style decoders
- [x] Test suite (8 tests)

## Phase 2 — Forward pass (done ✅)

`byollm/model.py`: RoPE rotary embeddings, RMSNorm, causal masked attention,
GQA head expansion, SwiGLU MLP, residual stream, final norm + LM head.
`ForwardModel.forward(tokens)` runs the full stack on real GGUF tensors and
returns logits. Verified on synthetic fixtures and the real 260K / 15M models
(12 tests green).

**Fixes landed in this phase:** GGUF weight layout is `[in, out]` with no
transpose (`x @ W`); causal mask is `triu(..., k=1)` (0 on diagonal+lower,
`-inf` strictly above); sequence is the leading axis in RoPE.

## Phase 3 — Generation (done ✅)

`byollm/model.py`: KV cache (per-layer K/V tensors, decode without recompute),
samplers (greedy / temperature / top-p nucleus), and `ForwardModel.generate()`
for full autoregressive generation. `generate_next()` kept for single-step use.
Verified: cached decode logits match no-cache forward exactly; generate()
extends prompts deterministically on greedy and stochastically on
temperature/top-p (22 tests green).

## Phase 4 — CLI & UX

- [ ] `byollm chat --model model.gguf` interactive REPL
- [ ] `byollm inspect --model model.gguf` print architecture summary
- [ ] Progress bar + tokens/sec reporting

## Phase 5 — Performance

- [ ] NumPy-vectorised kernels (done in Python loops first)
- [ ] Optional CUDA backend (CuPy) for GTX 1650
- [ ] Quantised KV cache (Q8_0 cache)

## Non-goals

- Training / backpropagation. This is inference only.
- Converting other formats. GGUF is the input; if you have another format,
  convert it to GGUF first.

## Release cadence

One small release per feature. Tags look like `v0.2.0-attention`, and each
release has a demo snippet showing real output from a real model.
