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

## Phase 2 — Forward pass (in progress 🚧)

- [ ] RoPE rotary embeddings
- [ ] Multi-head + grouped-query attention
- [ ] RMSNorm
- [ ] SwiGLU / GeGLU MLP
- [ ] Full transformer block forward pass

## Phase 3 — Generation

- [ ] KV cache
- [ ] Samplers: greedy, temperature, top-p, top-k
- [ ] `generate()` loop: tokens → text

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
