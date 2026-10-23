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

    def block(self, x: np.ndarray, layer_index: int, start_pos: int = 0) -> np.ndarray:
        """Run one transformer block on ``x`` of shape [seq, embedding]."""
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

        # causal mask: position i can attend to positions <= i.
        # -inf on strictly-upper triangle (disallowed), 0 on diagonal+lower.
        mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)

        attn_out = self.attention(q, k, v, mask)  # [seq, heads, hdim]
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

    def forward(self, token_ids: list[int], start_pos: int = 0) -> np.ndarray:
        """Run the full model on ``token_ids``; returns logits [seq, vocab]."""
        h = self.arch.hyper
        m = self.model

        emb = m.tensor(self.arch.embedding)  # [emb, vocab]
        x = emb[:, np.array(token_ids)].T  # [seq, emb]

        for block_index in range(h.block_count):
            x = self.block(x, block_index, start_pos)

        # final norm + lm head
        x = self.rms_norm(x, m.tensor(self.arch.output_norm), h.rms_epsilon)
        logits = x @ m.tensor(self.arch.output_weight)  # [seq, vocab]
        return logits

    def generate_next(self, token_ids: list[int]) -> int:
        """Greedy-sample the next token id given a prompt."""
        logits = self.forward(token_ids)
        next_logits = logits[-1]
        return int(np.argmax(next_logits))
