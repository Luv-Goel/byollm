"""CUDA-accelerated forward pass — Phase 5 of the roadmap.

A drop-in CuPy port of ``byollm.model.ForwardModel`` with the identical public
API (``rms_norm``, ``rope``, ``attention``, ``block``, ``forward``,
``generate_next``, ``generate``). Every tensor lives on the GTX 1650; no data
leaves the GPU during forward passes.

Usage::

    from byollm.cuda_model import CudaForwardModel
    model = read_gguf("model.gguf")
    arch = detect(model)
    fm = CudaForwardModel(model, arch)
    logits = fm.forward([1, 2, 3])

No PyTorch, no llama.cpp — just NumPy-compatible CuPy kernels on real GGUF
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cupy as cp

from .arch import Architecture
from .gguf import GGUFModel


class CudaForwardModel:
    """A runnable Llama-style decoder over real GGUF tensors, on CUDA."""

    def __init__(self, model: GGUFModel, arch: Architecture) -> None:
        self.model = model
        self.arch = arch
        self._gpu_cache: dict[int, cp.ndarray] = {}

    @property
    def hyper(self) -> Any:
        return self.arch.hyper

    # -- primitives (CuPy) --------------------------------------------------

    @staticmethod
    def rms_norm(x: cp.ndarray, weight: cp.ndarray, eps: float) -> cp.ndarray:
        """Root-mean-square layer normalisation (Llama pre-norm)."""
        return x * weight * (1.0 / cp.sqrt(cp.mean(x ** 2, axis=-1, keepdims=True) + eps))

    @staticmethod
    def rope(
        x: cp.ndarray,
        start_pos: int,
        dim: int,
        base: float = 10000.0,
    ) -> cp.ndarray:
        """Rotary position embeddings applied to the last ``dim`` axes.

        Interleaved layout (GPT-NeoX style): pairs (x[0], x[1]), (x[2], x[3]),
        … each rotated by its own frequency.
        """
        *lead, d = x.shape
        x = x.reshape(*lead, d // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]

        # sequence is always the leading dimension of x ([seq, heads, ...])
        seq_len = lead[0] if lead else 1
        positions = cp.arange(start_pos, start_pos + seq_len, dtype=cp.float32)

        freqs = 1.0 / (base ** (cp.arange(0, d, 2, dtype=cp.float32) / d))
        theta = cp.outer(positions, freqs)  # [seq, dim/2]

        cos = cp.cos(theta)
        sin = cp.sin(theta)

        # cos/sin are [seq, pairs]; insert a head dimension so they broadcast
        # over [seq, heads, pairs]:
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        x0r = x0 * cos - x1 * sin
        x1r = x0 * sin + x1 * cos
        out = cp.stack([x0r, x1r], axis=-1).reshape(*lead, d)
        return out

    def attention(
        self,
        q: cp.ndarray,
        k: cp.ndarray,
        v: cp.ndarray,
        mask: cp.ndarray | None = None,
    ) -> cp.ndarray:
        """Scaled dot-product attention over [seq, heads, head_dim] inputs."""
        head_dim = q.shape[-1]
        scores = cp.einsum("shd,thd->sht", q, k) / cp.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask[:, None, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = cp.exp(scores)
        attn = exp / exp.sum(axis=-1, keepdims=True)
        return cp.einsum("sht,thd->shd", attn, v)

    def silu(self, x: cp.ndarray) -> cp.ndarray:
        return x * (1.0 / (1.0 + cp.exp(-x)))

    # -- one block ----------------------------------------------------------

    def block(
        self,
        x: cp.ndarray,
        layer_index: int,
        start_pos: int = 0,
        cache: "CudaKVCache | None" = None,
    ) -> cp.ndarray:
        """Run one transformer block on ``x`` of shape [seq, embedding]."""
        h = self.arch.hyper
        L = self.arch.layer(layer_index)
        m = self.model

        # load tensors (lazy, from disk; cached on GPU after first use)
        wq = self._gpu(m.tensor(L.attn_q))  # [emb, emb]
        wk = self._gpu(m.tensor(L.attn_k))  # [emb, emb_kv]
        wv = self._gpu(m.tensor(L.attn_v))  # [emb, emb_kv]
        wo = self._gpu(m.tensor(L.attn_output))  # [emb, emb]
        w1 = self._gpu(m.tensor(L.ffn_gate))  # [emb, ffn]
        w2 = self._gpu(m.tensor(L.ffn_down))  # [ffn, emb]
        w3 = self._gpu(m.tensor(L.ffn_up))  # [emb, ffn]
        rn1 = self._gpu(m.tensor(L.attn_norm))  # [emb]
        rn2 = self._gpu(m.tensor(L.ffn_norm))  # [emb]

        seq, emb = x.shape
        hdim = h.head_dim
        n_heads = h.head_count
        n_kv = h.head_count_kv

        # ---- attention ----
        normed = self.rms_norm(x, rn1, h.rms_epsilon)

        q = normed @ wq  # [seq, emb]
        k = normed @ wk  # [seq, emb_kv]
        v = normed @ wv  # [seq, emb_kv]

        q = q.reshape(seq, n_heads, hdim)
        k = k.reshape(seq, n_kv, hdim)
        v = v.reshape(seq, n_kv, hdim)

        # GQA: expand KV heads to match query heads
        if n_kv < n_heads:
            rep = n_heads // n_kv
            k = cp.repeat(k, rep, axis=1)
            v = cp.repeat(v, rep, axis=1)

        q = self.rope(q, start_pos, hdim)
        k = self.rope(k, start_pos, hdim)

        if cache is not None:
            cached_k, cached_v = cache.get(layer_index)
            is_prefill = cached_k is None
            if cached_k is not None:
                new_k = k[cached_k.shape[0] :]
                new_v = v[cached_v.shape[0] :]
                k = cp.concatenate([cached_k, k], axis=0)
                v = cp.concatenate([cached_v, v], axis=0)
            else:
                new_k = k
                new_v = v
            cache.put(layer_index, new_k, new_v)
            if is_prefill:
                # prefill: causal mask — position i attends to positions <= i
                mask = cp.triu(cp.full((seq, seq), -cp.inf, dtype=cp.float32), k=1)
                attn_out = self.attention(q, k, v, mask)
            else:
                # decode: query attends to all cached keys, no causal mask
                attn_out = self.attention(q, k, v)
        else:
            # prefill: causal mask — position i attends to positions <= i
            mask = cp.triu(cp.full((seq, seq), -cp.inf, dtype=cp.float32), k=1)
            attn_out = self.attention(q, k, v, mask)

        attn_out = attn_out.reshape(seq, emb)
        attn_out = attn_out @ wo

        # residual 1
        x = x + attn_out

        # ---- MLP (SwiGLU) ----
        normed2 = self.rms_norm(x, rn2, h.rms_epsilon)
        gate = self.silu(normed2 @ w1)  # [seq, ffn]
        up = normed2 @ w3  # [seq, ffn]
        hidden = gate * up
        out = hidden @ w2  # [seq, emb]

        # residual 2
        x = x + out
        return x

    # -- full forward -------------------------------------------------------

    def forward(
        self,
        token_ids: list[int],
        start_pos: int = 0,
        cache: "CudaKVCache | None" = None,
    ) -> cp.ndarray:
        """Run the full model on ``token_ids``; returns logits [seq, vocab]."""
        h = self.arch.hyper
        m = self.model

        emb = self._gpu(m.tensor(self.arch.embedding))  # [emb, vocab]
        x = emb[:, cp.array(token_ids)].T  # [seq, emb]

        for block_index in range(h.block_count):
            x = self.block(x, block_index, start_pos, cache)

        # final norm + lm head
        x = self.rms_norm(x, self._gpu(m.tensor(self.arch.output_norm)), h.rms_epsilon)
        logits = x @ self._gpu(m.tensor(self.arch.output_weight))  # [seq, vocab]
        return logits

    def generate_next(self, token_ids: list[int]) -> int:
        """Greedy-sample the next token id given a prompt."""
        logits = self.forward(token_ids)
        next_logits = logits[-1]
        return int(cp.argmax(next_logits))

    # -- helpers ------------------------------------------------------------

    _gpu_cache: dict[int, cp.ndarray] = {}

    def _gpu(self, arr: "np.ndarray") -> cp.ndarray:
        """Move a NumPy array to GPU."""
        return cp.asarray(arr)


# -- sampling ---------------------------------------------------------------

def softmax(logits: cp.ndarray, temperature: float = 1.0) -> cp.ndarray:
    """Temperature-scaled softmax over logits."""
    logits = cp.asarray(logits, dtype=cp.float64) / temperature
    logits = logits - logits.max()
    exp = cp.exp(logits)
    return exp / exp.sum()


def top_p_mask(probs: cp.ndarray, p: float) -> cp.ndarray:
    """Zero out all but the top-p nucleus (sorted descending cumulative mass)."""
    sorted_idx = cp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = cp.cumsum(sorted_probs)
    cutoff = int(cp.searchsorted(cumsum, p))
    keep = cp.zeros_like(probs, dtype=bool)
    keep = keep.at[sorted_idx[: cutoff + 1]].set(True)
    return keep


def sample_next(
    logits: cp.ndarray,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_p: float | None = None,
    rng: Any | None = None,
) -> int:
    """Sample the next token id from logits (CuPy)."""
    if rng is None:
        rng = cp.random.default_rng(0)

    if strategy == "greedy":
        return int(cp.argmax(logits))

    probs = softmax(logits, temperature)

    if strategy == "temperature":
        return int(cp.asnumpy(rng.choice(len(probs), p=cp.asnumpy(probs))))

    if strategy == "top_p":
        if top_p is None:
            top_p = 0.9
        keep = top_p_mask(probs, top_p)
        if not bool(keep.any()):
            keep[:] = True
        masked = probs * keep
        masked = masked / masked.sum()
        return int(cp.asnumpy(rng.choice(len(probs), p=cp.asnumpy(masked))))

    raise ValueError(f"unknown sampling strategy {strategy!r}")


def generate(
    self: "CudaForwardModel",
    token_ids: list[int],
    max_new: int = 32,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_p: float | None = None,
) -> list[int]:
    """Autoregressively generate ``max_new`` tokens using a KV cache."""
    h = self.arch.hyper
    rng = cp.random.default_rng(0)

    # allocate one KV cache per layer: [max_seq, heads, head_dim]
    max_seq = max(len(token_ids) + max_new, h.context_length)
    cache = CudaKVCache(h.block_count, max_seq, h.head_count, h.head_dim)

    # prefill: run the whole prompt through (causal mask applied internally)
    logits = self.forward(token_ids, start_pos=0, cache=cache)

    tokens = list(token_ids)
    next_logits = logits[-1]  # logits at the last prompt position
    for _ in range(max_new):
        nxt = sample_next(next_logits, strategy, temperature, top_p, rng)
        tokens.append(nxt)
        # decode only the newly sampled token at its position
        next_logits = self.forward([nxt], start_pos=len(tokens) - 1, cache=cache)[-1]
    return tokens


# attach generate as a method on CudaForwardModel
CudaForwardModel.generate = generate  # type: ignore[attr-defined]


class CudaKVCache:
    """Per-layer K/V cache for decode-time generation, on CUDA.

    Layout per layer: ``k`` and ``v`` arrays of shape
    [max_seq, heads, head_dim], filled left-to-right as tokens are generated.
    """

    def __init__(self, n_layers: int, max_seq: int, n_heads: int, head_dim: int) -> None:
        self.max_seq = max_seq
        self.entries: dict[int, tuple[cp.ndarray, cp.ndarray]] = {}
        for layer in range(n_layers):
            self.entries[layer] = (
                cp.zeros((max_seq, n_heads, head_dim), dtype=cp.float32),
                cp.zeros((max_seq, n_heads, head_dim), dtype=cp.float32),
            )
        self.fill: dict[int, int] = {layer: 0 for layer in range(n_layers)}

    def get(self, layer: int) -> tuple[cp.ndarray | None, cp.ndarray | None]:
        """Return cached (k, v) for ``layer`` trimmed to the filled length."""
        k, v = self.entries[layer]
        n = self.fill[layer]
        if n == 0:
            return None, None
        return k[:n], v[:n]

    def put(self, layer: int, k: cp.ndarray, v: cp.ndarray) -> None:
        """Store the new ``k, v`` chunk for ``layer`` at the fill position."""
        n = self.fill[layer]
        k_seq = k.shape[0]
        self.entries[layer][0][n : n + k_seq] = k
        self.entries[layer][1][n : n + k_seq] = v
        self.fill[layer] = n + k_seq
