"""byollm — Build Your Own LLM Inference Engine From Scratch.

A from-scratch, dependency-light LLM inference engine that reads real GGUF
model files and runs the forward pass in pure Python + NumPy.

No PyTorch. No llama.cpp bindings. Just the format, the math, and the code
that connects them — written so you can read every line and understand what
it does.

Current state
-------------
- GGUF v1/v2/v3 container parser (spec-accurate, lazy tensor loading)
- Architecture detector for Llama-style decoders (reads hyperparameters from
  file metadata, not hard-coded constants)
- Dequantisers: F32, F16, BF16, Q8_0, Q4_0
- Full test suite against synthetic GGUF files + a real 260K-parameter model

Roadmap
-------
1. Attention + MLP forward pass (this month)
2. KV cache + sampling (greedy / temperature / top-p)
3. CLI: `byollm chat --model models/tiny.gguf`
4. Optimised kernels (NumPy vectorised first, then optional CUDA/Metal)
5. Speculative decoding + quantised KV cache

Why build this?
---------------
Because the best way to understand LLMs is to build one that actually runs
real models. Every chapter of this repo is a verified, tested step you can
reproduce on a laptop — including a GTX 1650 with 4 GB VRAM.
"""

__version__ = "0.1.0"
