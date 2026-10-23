# byollm — Build Your Own LLM Inference Engine From Scratch

> A from-scratch, dependency-light LLM inference engine that reads real GGUF
> model files and runs the forward pass in pure Python + NumPy. No PyTorch,
> no llama.cpp bindings — just the format, the math, and the code.

[![Tests](https://img.shields.io/github/actions/workflow/status/Luv-Goel/byollm/tests.yml?label=tests)](https://github.com/Luv-Goel/byollm/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Luv-Goel/byollm?style=social)](https://github.com/Luv-Goel/byollm/stargazers)

## What this is

`byollm` is an LLM inference engine built **from scratch**. It parses real
[GGUF](https://github.com/ggml-org/llama.cpp/blob/master/gguf/README.md)
model files — the same format llama.cpp uses — and runs the transformer
forward pass with nothing but Python and NumPy.

There are no framework bindings, no converted checkpoints, and no magic.
Every tensor layout, every dequantisation rule, every attention computation
is written out so you can read it and understand it.

This is the **"Build Your Own X"** tradition applied to LLMs: not a wrapper,
not a binding — a real, running engine you can extend.

## What works right now

| Milestone | Status | What it does |
|---|---|---|
| GGUF container parser | ✅ | Reads GGUF v1/v2/v3: header, metadata, tensor table; lazy on-disk tensor access |
| Metadata types | ✅ | UINT/INT/FLOAT/BOOL/STRING/ARRAY all round-trip |
| Dequantisers | ✅ | F32, F16, BF16, Q8_0, Q4_0 (exact llama.cpp rules) |
| Architecture detector | ✅ | Reads hyperparameters from file metadata (block count, heads, FFN dim, RoPE, …) |
| Test suite | ✅ | 27 tests, run against synthetic GGUF files and real 260K / 15M models |
| Forward pass | ✅ | RoPE, RMSNorm, causal masked attention, GQA, SwiGLU MLP, residual stream, LM head — finite logits on real tensors |
| Sampling / KV cache | ✅ | Greedy, temperature, top-p nucleus; per-layer KV cache; cached decode matches no-cache forward exactly |
| CLI | ✅ | `byollm inspect` / `byollm generate` / `byollm chat` | `pyproject.toml` console script |
| CUDA backend | ✅ | `byollm/cuda_model.py` — CuPy port with identical API; 10.91 tokens/sec on GTX 1650 (4GB VRAM) |

## Quick start

```bash
# clone
git clone https://github.com/Luv-Goel/byollm.git
cd byollm

# create a venv and install
python -m venv .venv
.venv\Scripts\activate   # Windows (PowerShell)
# source .venv/bin/activate  # macOS / Linux

# install dev deps
pip install -e ".[dev]"

# run the tests
pytest
```

You need a GGUF model to run against. The test suite builds small synthetic
models in-memory, and there's a real fixture you can download:

```bash
# tiny 260K-parameter Llama model (stories dataset, ~1 MB)
python scripts/download_model.py \
  --repo klosax/tinyllamas-stories-gguf \
  --file tinyllamas-stories-260k-f32.gguf \
  --out models/tiny.gguf
```

Then inspect it:

```python
from byollm.gguf import read_gguf
from byollm.arch import detect

model = read_gguf("models/tiny.gguf")
arch = detect(model)
print(arch.summary())
```

Example output:

```
architecture : llama
blocks       : 5
embedding    : 64
ffn hidden   : 172
heads        : 8 (KV: 4)
head dim     : 8
context      : 512
vocab        : 512
rms epsilon  : 1e-05
layers mapped: 5 / 5
```

## Live demo (runs on a GTX 1650, 4GB VRAM)

```bash
# inspect a real GGUF model
byollm inspect --model models/tiny.gguf

# generate from a prompt
byollm generate --model models/tiny.gguf --prompt "hello" --max-new 32
```

Example `inspect` output against the real 260K-parameter Llama model:

```
architecture : llama
blocks       : 5
embedding    : 64
ffn hidden   : 172
heads        : 8 (KV: 4)
head dim     : 8
context      : 512
vocab        : 512
rms epsilon  : 1e-05
layers mapped: 5 / 5
```

Forward pass produces finite logits end-to-end:

```
logits shape: (3, 512)
last-position top-5: [35, 99, 163, 87, 65]
last-position logits: [-4.4695, 12.238, 5.7171, 7.7305, -4.0374, 4.5493, -1.2805, 0.8199]
finite: True
```

CUDA backend (CuPy) benchmark on the same model:

```
CuPy tokens/sec (GTX 1650): 10.91  (32 new tokens in 2.93s)
```

## Design principles

1. **The file is the source of truth.** Nothing about layer counts, head
   counts, or tensor naming is hard-coded. Everything is read from GGUF
   metadata — point `byollm` at any compatible GGUF and it figures out the
   architecture itself.
2. **Zero mystery.** Every number in a tensor trace back to a line of code
   you can read. Dequantisers implement the exact GGUF bit rules; no opaque
   C++ bindings.
3. **Laptop-first.** Built and tested on a GTX 1650 with 4 GB VRAM and 15 GB
   RAM. Small models run entirely on CPU; the design leaves a clear path to
   GPU kernels later.
4. **Tests first.** Every feature has a test, and every test runs against real
   model bytes — not mocks.

## Project structure

```
byollm/          # the engine package
  gguf.py        # GGUF v1/v2/v3 container parser + dequantisers
  arch.py        # architecture detection + tensor-name → layer mapping
  model.py       # NumPy forward pass (RoPE, RMSNorm, attention, MLP, KV cache, samplers)
  cuda_model.py  # CuPy CUDA port (identical API, runs on GTX 1650)
  cli.py         # `byollm inspect` / `generate` / `chat`
  __init__.py    # package entry point
tests/           # test suite (pytest)
  test_gguf.py   # parser + architecture tests (synthetic + real GGUF)
  test_forward.py  # forward-pass tests on synthetic + real models
  test_cache_generation.py  # KV cache equivalence + generation tests
  test_cuda_model.py  # CuPy matches NumPy, benchmark tokens/sec
docs/            # design notes, roadmap, chapter write-ups
scripts/         # small helpers (model downloader, benchmarks)
models/          # your GGUF models go here (gitignored — large binaries)
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the release plan and
[`docs/architecture.md`](docs/architecture.md) for the design write-up.

## Contributing

Issues and pull requests are welcome — especially for:

- attention / MLP forward pass kernels
- samplers (greedy, temperature, top-p, top-k)
- a KV cache
- the `byollm` CLI
- Q4_1 / Q5_0 / Q6_K dequantisers
- CUDA or Metal kernels (optional backends)

Open an issue first so we can coordinate.

## License

MIT. See [`LICENSE`](LICENSE).

## Why "from scratch"?

Because the best way to understand something is to build it. `byollm` is for
anyone who's used a LLM and wondered what's actually inside the file — and
wanted to run it themselves, line by line.

---

Made by [Luv](https://github.com/Luv-Goel). AI Research Intern, Eastern
International University. Building `byollm` in public on the way to 100+
stars.
