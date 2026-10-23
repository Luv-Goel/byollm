"""Forward pass — Phase 2 of the roadmap.

A transformer block, built from scratch in NumPy, that runs the real tensors
from a parsed GGUF file. No PyTorch, no llama.cpp — just the math:

    RoPE -> RMSNorm -> attention -> SwiGLU MLP -> LM head

Every operation is a plain NumPy function you can read and step through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .arch import Architecture
from .gguf import GGUFModel


@dataclass
class ForwardModel:
    """A runnable Llama-style decoder over real GGUF tensors."""

    model: GGUFModel
    arch: Architecture

    @property
    def hyper(self) -> Any:
        return self.arch.hyper

    # -- primitives ----------------------------------------------------------

    @staticmethod
    def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        """Root-mean-square layer normalisation (Llama pre-norm)."""
        return x * weight * (1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps))

    @staticmethod
    def rope(
        x: np.ndarray,
        start_pos: int,
        dim: int,
        base: float = 10000.0,
    ) -> np.ndarray:
        """Rotary position embeddings applied to the last ``dim`` axes.

        Interleaved layout (GPT-NeoX style): pairs (x[0], x[1]), (x[2], x[3]),
        … each rotated by its own frequency.
        """
        *lead, d = x.shape
        x = x.reshape(*lead, d // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]

        # sequence is always the leading dimension of x ([seq, heads, ...])
        seq_len = lead[0] if lead else 1
        positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)

        freqs = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
        theta = np.outer(positions, freqs)  # [seq, dim/2]

        cos = np.cos(theta)
        sin = np.sin(theta)

        # cos/sin are [seq, pairs]; insert a head dimension so they broadcast
        # over [seq, heads, pairs]:
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        x0r = x0 * cos - x1 * sin
        x1r = x0 * sin + x1 * cos
        out = np.stack([x0r, x1r], axis=-1).reshape(*lead, d)
        return out

    def attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scaled dot-product attention over [seq, heads, head_dim] inputs."""
        head_dim = q.shape[-1]
        scores = np.einsum("shd,thd->sht", q, k) / np.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask[:, None, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        attn = exp / exp.sum(axis=-1, keepdims=True)
        return np.einsum("sht,thd->shd", attn, v)

    def silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    # -- one block ----------------------------------------------------------

    def block(self, x: np.ndarray, layer_index: int, start_pos: int = 0,
              cache: "KVCache | None" = None) -> np.ndarray:
        """Run one transformer block on ``x`` of shape [seq, embedding].

        If ``cache`` is provided, K/V are appended to the per-layer cache and
        attention runs over the full cached history (decode step, no causal
        mask). Otherwise a causal mask is applied (prefill).
        """
        h = self.arch.hyper
        L = self.arch.layer(layer_index)
        m = self.model

        # load tensors (lazy, from disk)
        wq = m.tensor(L.attn_q)  # [emb, emb]
        wk = m.tensor(L.attn_k)  # [emb, emb_kv]
        wv = m.tensor(L.attn_v)  # [emb, emb_kv]
        wo = m.tensor(L.attn_output)  # [emb, emb]
        w1 = m.tensor(L.ffn_gate)  # [emb, ffn]
        w2 = m.tensor(L.ffn_down)  # [ffn, emb]
        w3 = m.tensor(L.ffn_up)  # [emb, ffn]
        rn1 = m.tensor(L.attn_norm)  # [emb]
        rn2 = m.tensor(L.ffn_norm)  # [emb]

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
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)

        q = self.rope(q, start_pos, hdim)
        k = self.rope(k, start_pos, hdim)

        if cache is not None:
            cached_k, cached_v = cache.get(layer_index)
            is_prefill = cached_k is None
            if cached_k is not None:
                new_k = k[cached_k.shape[0]:]
                new_v = v[cached_v.shape[0]:]
                k = np.concatenate([cached_k, k], axis=0)
                v = np.concatenate([cached_v, v], axis=0)
            else:
                new_k = k
                new_v = v
            cache.put(layer_index, new_k, new_v)
            if is_prefill:
                # prefill: causal mask — position i attends to positions <= i
                mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
                attn_out = self.attention(q, k, v, mask)
            else:
                # decode: query attends to all cached keys, no causal mask
                attn_out = self.attention(q, k, v)
        else:
            # prefill: causal mask — position i attends to positions <= i
            mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
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

    # -- full forward --------------------------------------------------------

    def forward(self, token_ids: list[int], start_pos: int = 0,
                cache: "KVCache | None" = None) -> np.ndarray:
        """Run the full model on ``token_ids``; returns logits [seq, vocab].

        If ``cache`` is provided, only the last position's logits are returned
        (decode step) and the K/V cache is populated across layers.
        """
        h = self.arch.hyper
        m = self.model

        emb = m.tensor(self.arch.embedding)  # [emb, vocab]
        x = emb[:, np.array(token_ids)].T  # [seq, emb]

        for block_index in range(h.block_count):
            x = self.block(x, block_index, start_pos, cache)

        # final norm + lm head
        x = self.rms_norm(x, m.tensor(self.arch.output_norm), h.rms_epsilon)
        logits = x @ m.tensor(self.arch.output_weight)  # [seq, vocab]
        return logits

    def generate_next(self, token_ids: list[int]) -> int:
        """Greedy-sample the next token id given a prompt."""
        logits = self.forward(token_ids)
        next_logits = logits[-1]
        return int(np.argmax(next_logits))


# -- sampling ---------------------------------------------------------------

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax over logits."""
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def top_p_mask(probs: np.ndarray, p: float) -> np.ndarray:
    """Zero out all but the top-p nucleus (sorted descending cumulative mass)."""
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p)
    keep = np.zeros_like(probs, dtype=bool)
    keep[sorted_idx[: cutoff + 1]] = True
    return keep


def top_k_mask(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the top-``k`` most probable tokens."""
    k = min(k, len(probs))
    keep = np.zeros_like(probs, dtype=bool)
    top_idx = np.argpartition(probs, -k)[-k:]
    keep[top_idx] = True
    return keep


def sample_next(logits: np.ndarray, strategy: str = "greedy",
                temperature: float = 1.0, top_p: float | None = None,
                top_k: int | None = None,
                rng: np.random.Generator | None = None) -> int:
    """Sample the next token id from logits.

    Strategies: ``greedy`` (argmax), ``temperature``, ``top_p`` (nucleus),
    ``top_k``.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if strategy == "greedy":
        return int(np.argmax(logits))

    probs = softmax(logits, temperature)

    if strategy == "temperature":
        return int(rng.choice(len(probs), p=probs))

    if strategy == "top_p":
        if top_p is None:
            top_p = 0.9
        keep = top_p_mask(probs, top_p)
        if not keep.any():
            keep[:] = True
        masked = probs * keep
        masked = masked / masked.sum()
        return int(rng.choice(len(probs), p=masked))

    if strategy == "top_k":
        if top_k is None:
            top_k = 10
        keep = top_k_mask(probs, top_k)
        masked = probs * keep
        masked = masked / masked.sum()
        return int(rng.choice(len(probs), p=masked))

    raise ValueError(f"unknown sampling strategy {strategy!r}")


def generate(self: "ForwardModel", token_ids: list[int], max_new: int = 32,
             strategy: str = "greedy", temperature: float = 1.0,
             top_p: float | None = None, top_k: int | None = None) -> list[int]:
    """Autoregressively generate ``max_new`` tokens using a KV cache.

    Returns the full token list (prompt + generated).
    """
    h = self.arch.hyper
    m = self.model
    rng = np.random.default_rng(0)

    # allocate one KV cache per layer: [max_seq, heads, head_dim]
    max_seq = max(len(token_ids) + max_new, h.context_length)
    cache = KVCache(h.block_count, max_seq, h.head_count, h.head_dim)

    # prefill: run the whole prompt through with no cache mask
    logits = self.forward(token_ids, start_pos=0, cache=cache)

    tokens = list(token_ids)
    next_logits = logits[-1]  # logits at the last prompt position
    for _ in range(max_new):
        nxt = sample_next(next_logits, strategy, temperature, top_p, top_k, rng)
        tokens.append(nxt)
        # decode only the newly sampled token at its position
        next_logits = self.forward([nxt],
                                   start_pos=len(tokens) - 1,
                                   cache=cache)[-1]
    return tokens


class KVCache:
    """Per-layer K/V cache for decode-time generation.

    Layout per layer: ``k`` and ``v`` arrays of shape
    [max_seq, heads, head_dim], filled left-to-right as tokens are generated.
    """

    def __init__(self, n_layers: int, max_seq: int, n_heads: int,
                 head_dim: int) -> None:
        self.max_seq = max_seq
        self.entries: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for layer in range(n_layers):
            self.entries[layer] = (
                np.zeros((max_seq, n_heads, head_dim), dtype=np.float32),
                np.zeros((max_seq, n_heads, head_dim), dtype=np.float32),
            )
        self.fill: dict[int, int] = {layer: 0 for layer in range(n_layers)}

    def get(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return cached (k, v) for ``layer`` trimmed to the filled length."""
        k, v = self.entries[layer]
        n = self.fill[layer]
        if n == 0:
            return None, None
        return k[:n], v[:n]

    def put(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Store the new ``k, v`` chunk for ``layer`` at the fill position.

        During prefill the chunk covers the whole prompt; during decode it's
        a single new token. ``block()`` already slices out only the part
        beyond what's cached.
        """
        n = self.fill[layer]
        k_seq, v_seq = k.shape[0], v.shape[0]
        self.entries[layer][0][n:n + k_seq] = k
        self.entries[layer][1][n:n + v_seq] = v
        self.fill[layer] = n + k_seq


# attach generate as a method on ForwardModel
ForwardModel.generate = generate  # type: ignore[attr-defined]
