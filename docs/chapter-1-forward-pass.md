# Chapter 1 — The forward pass, from scratch

*Build Your Own LLM Inference Engine From Scratch*

This chapter walks through the entire forward pass of a Llama-style decoder,
implemented in pure Python + NumPy with zero framework bindings. It runs real
GGUF tensors — the same format llama.cpp uses — end to end, and every number
traces back to a line of code you can read.

Tested on a GTX 1650 (4GB VRAM). If it runs here, it runs on your laptop.

## What we build

A single `ForwardModel` that takes token ids and returns logits:

```
tokens → embedding → [block × N] → RMSNorm → LM head → logits
```

Each block is:

```
RMSNorm → RoPE → causal attention (GQA) → residual →
RMSNorm → SwiGLU MLP → residual
```

No PyTorch, no llama.cpp, no magic.

## 1. The file is the source of truth

`byollm` never hard-codes layer counts, head counts, or tensor names. Everything
is read from GGUF metadata:

```python
from byollm.gguf import read_gguf
from byollm.arch import detect

model = read_gguf("models/tiny.gguf")
arch = detect(model)
print(arch.summary())
```

Real output against the 260K-parameter Llama model:

```
architecture : llama
blocks       : 5
embedding    : 64
ffn hidden   : 172
heads        : 8 (KV: 4)
head dim     : 8
context      : 512
vocab        : 512
layers mapped: 5 / 5
```

## 2. GGUF weights are `[in, out]` — no transpose

This caught us once. GGUF (and llama.cpp) store weight matrices as `[in, out]`,
and the forward pass computes `x @ W` — **never** `x @ W.T`. So `token_embd.weight`
is `[64, 512]` and we index `emb[:, token_ids].T` to get `[seq, 64]`; `attn_q.weight`
is `[64, 64]` and `normed @ wq` gives `[seq, 64]`. Remove every `.T` you instinctively
want to add — it's wrong here.

## 3. The primitives

### RMSNorm (pre-norm)

```python
def rms_norm(x, weight, eps):
    return x * weight * (1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps))
```

### RoPE (interleaved, GPT-NeoX style)

Pairs `(x[0], x[1])`, `(x[2], x[3])`, … each rotated by its own frequency.
Sequence is always the leading axis; cos/sin broadcast over heads as `[:, None, :]`:

```python
seq_len = lead[0]
positions = np.arange(start_pos, start_pos + seq_len)
freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
theta = np.outer(positions, freqs)           # [seq, dim/2]
cos, sin = cos[:, None, :], sin[:, None, :]   # head dimension
```

### Causal attention

`triu(..., k=1)` puts `-inf` strictly above the diagonal — position `i` attends
to positions `<= i`. The first version of this chapter had the mask inverted,
which produced `-inf` row-maxes and NaN attention. Don't repeat it.

```python
mask = np.triu(np.full((seq, seq), -np.inf), k=1)
scores = np.einsum("shd,thd->sht", q, k) / np.sqrt(head_dim)
scores = scores + mask[:, None, :]
scores = scores - scores.max(axis=-1, keepdims=True)
attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
```

### GQA head expansion

The 260K model has 8 query heads but only 4 KV heads. We `np.repeat` each KV
head `rep = n_heads // n_kv` times so Q and K/V shapes match before attention.

### SwiGLU MLP

```python
gate = silu(normed2 @ w1)   # [seq, ffn]
up   = normed2 @ w3         # [seq, ffn]
hidden = gate * up
out = hidden @ w2           # [seq, emb]
```

## 4. End-to-end on real bytes

```python
from byollm.model import ForwardModel

fm = ForwardModel(model, arch)
logits = fm.forward([7, 42, 123])   # [3, 512]
```

Real output:

```
logits shape: (3, 512)
last-position top-5: [35, 99, 163, 87, 65]
last-position logits: [-4.4695, 12.238, 5.7171, 7.7305, -4.0374, 4.5493, -1.2805, 0.8199]
finite: True
```

All logits finite. No NaNs.

## 5. KV cache + generation

Decode doesn't recompute. `KVCache` holds per-layer K/V tensors; each new token
attends to the full cached history. Cached decode logits match the no-cache
forward pass exactly (tested at `1e-4` tolerance).

Samplers: greedy, temperature, top-p nucleus. `generate()` runs prefill once,
then decode one token at a time.

## 6. CUDA backend (CuPy)

`byollm/cuda_model.py` is a drop-in CuPy port with the identical API. Every
tensor lives on the GPU; nothing leaves the device during forward passes.

Benchmark on the GTX 1650, 260K model, greedy decode:

```
CuPy tokens/sec (GTX 1650): 10.91  (32 new tokens in 2.93s)
```

## Why "from scratch"?

Because the best way to understand an LLM is to build it. Every dequantiser
implements the exact GGUF bit rules. Every matmul is a line of NumPy you can
step through with a debugger. Point `byollm` at any compatible GGUF and it
figures out the architecture itself.

The engine is at github.com/Luv-Goel/byollm — pull requests welcome.
