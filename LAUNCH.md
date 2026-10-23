# Launch draft — Build Your Own LLM Inference Engine From Scratch

*Copy-paste ready for Hacker News "Show HN", r/LocalLLaMA, and Twitter/X.*

## TL;DR

I built an LLM inference engine from scratch in pure Python + NumPy (no PyTorch,
no llama.cpp), running real GGUF models on a GTX 1650 with 4GB VRAM.

Every layer — GGUF parsing, RoPE, RMSNorm, causal attention, SwiGLU MLP, KV-cache
decode, greedy/temperature/top-p/top-k sampling — is implemented from scratch and
verified against real model tensors. Benchmarks included.

https://github.com/Luv-Goel/byollm

---

## Show HN title options

1. Show HN: I built an LLM inference engine from scratch in pure Python + NumPy
2. Show HN: ByoLLM — LLM inference engine from scratch, runs on a GTX 1650 (4GB VRAM)
3. Show HN: Build Your Own LLM Inference Engine From Scratch (pure Python, real GGUF models)

## r/LocalLLaMA post

**Title:** I built an LLM inference engine from scratch — pure Python + NumPy, runs on a GTX 1650

**Body:**

Hi all — I built **byollm** (Build Your Own LLM Inference Engine From Scratch),
an LLM inference engine written entirely in pure Python + NumPy. No PyTorch,
no llama.cpp, no CUDA kernels by hand — just NumPy running real GGUF tensors.

What it does:
- Parses GGUF v1/v2/v3 files (F32/F16/BF16/Q8_0/Q4_0 dequantisers)
- Forward pass: RoPE, RMSNorm, GQA causal attention, SwiGLU MLP
- KV-cache decode with greedy / temperature / top-p / top-k sampling
- Optional CuPy CUDA backend (identical API, verified against NumPy)
- CLI: `byollm inspect`, `byollm generate`, `byollm chat`
- CI on GitHub Actions, 27 tests passing

I benchmarked it on a GTX 1650 (4GB VRAM) and published the real numbers in the
repo — including the honest finding that on tiny models CPU beats GPU (launch
overhead dominates).

It's early (260K-parameter test model, byte-level tokenizer), but every piece is
from scratch and verified against real model bytes.

Repo: https://github.com/Luv-Goel/byollm
Chapter 1 (forward pass walkthrough): docs/chapter-1-forward-pass.md

Happy to answer questions about the implementation, the GGUF format, or the
4GB-VRAM benchmark numbers.

---

## Twitter/X thread

1/ I built an LLM inference engine from scratch in pure Python + NumPy.

No PyTorch, no llama.cpp — just NumPy running real GGUF models on a GTX 1650 (4GB VRAM).

🧵

2/ The stack:
• GGUF v1/v2/v3 parser (F32/F16/BF16/Q8_0/Q4_0)
• RoPE, RMSNorm, GQA attention, SwiGLU MLP
• KV-cache decode + greedy/temp/top-p/top-k sampling
• CuPy CUDA backend with identical API
• CLI + GitHub Actions CI

3/ Every layer verified against real model tensors — 27 tests passing, finite logits on real weights.

4/ Real numbers from the GTX 1650:
• 260K model: CPU 346 tok/s, CuPy 27 tok/s
• 15M model: CPU 5 tok/s, CuPy 4 tok/s

(CPU wins on tiny models — GPU launch overhead dominates. Honest numbers.)

5/ Try it:

pip install -e .
byollm inspect --model model.gguf
byollm generate --model model.gguf --prompt "once upon a time" --strategy top_k --top-k 5

6/ Full repo + chapter 1 walkthrough:

https://github.com/Luv-Goel/byollm

#LLM #NumPy #FromScratch #LocalLLaMA

---

## How to try it (for the README / launch)

```bash
git clone https://github.com/Luv-Goel/byollm.git
cd byollm
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
python -m byollm.cli inspect --model models/tinyllamas-stories-260k-f32.gguf
python -m byollm.cli generate --model models/tinyllamas-stories-260k-f32.gguf \
    --prompt "once upon a time" --max-new 32 --strategy top_k --top-k 5
```

*Note: the bundled 260K model is untrained, so output is garbled bytes — the engine is verified, the model is a fixture. Swap in any compatible Llama GGUF for real generation.*

---

## What makes this shareable

- **From scratch, no frameworks** — every line is NumPy, readable, chapter-by-chapter
- **Real hardware numbers** — GTX 1650, 4GB VRAM, published honestly
- **Real GGUF support** — drop in any compatible Llama model
- **Verified** — 27 tests, CI, real model fixtures

---

When you're ready to post, drop the link back here and I'll help respond to comments / turn feedback into the next feature.
