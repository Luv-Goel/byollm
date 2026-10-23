"""Forward-pass tests — Phase 2 of the roadmap.

Uses the synthetic GGUF fixtures from test_gguf plus the real model when
available, so the forward pass is verified on actual tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byollm.arch import detect
from byollm.gguf import GGUFDtype, read_gguf
from byollm.model import ForwardModel
from tests.test_gguf import build_gguf_v1, llama_metadata

REAL_MODELS = [
    p
    for p in Path("models").glob("*.gguf")
    if p.is_file()
]


def _synthetic_model() -> tuple[ForwardModel, int]:
    """Build a tiny 1-block model with random weights; return (model, vocab)."""
    vocab = 16
    emb = 4       # matches llama_metadata(): embedding_length=4
    ffn = 8       # matches llama_metadata(): feed_forward_length=8
    ctx = 16      # matches llama_metadata(): context_length=16
    tensors: dict[str, tuple[list[int], bytes, GGUFDtype]] = {}
    rng = np.random.default_rng(0)

    def w(name: str, shape: tuple[int, ...]) -> None:
        tensors[name] = (list(shape), rng.standard_normal(shape).astype(np.float32).tobytes(), GGUFDtype.F32)

    w("token_embd.weight", (emb, vocab))
    w("output_norm.weight", (emb,))
    w("output.weight", (emb, vocab))   # GGUF [in, out] = [emb, vocab]
    for block in range(2):   # llama_metadata() hardcodes block_count=2
        for n, shape in {
            "attn_q": (emb, emb),
            "attn_k": (emb, emb),
            "attn_v": (emb, emb),
            "attn_output": (emb, emb),
            # GGUF / llama.cpp store weights as [in, out]:
            "ffn_gate": (emb, ffn),   # [emb, ffn]
            "ffn_down": (ffn, emb),   # [ffn, emb]
            "ffn_up": (emb, ffn),     # [emb, ffn]
            "attn_norm": (emb,),
            "ffn_norm": (emb,),
        }.items():
            w(f"blk.{block}.{n}.weight", shape)

    meta = llama_metadata(vocab_size=vocab)
    data = build_gguf_v1(meta, tensors)
    path = Path("/tmp/synthetic_forward.gguf")
    path.write_bytes(data)

    model = read_gguf(path)
    arch = detect(model)
    return ForwardModel(model, arch), vocab


def test_forward_output_shape() -> None:
    fm, vocab = _synthetic_model()
    tokens = [1, 2, 3]
    logits = fm.forward(tokens)
    assert logits.shape == (3, vocab)
    assert np.all(np.isfinite(logits))


def test_forward_deterministic() -> None:
    fm, _ = _synthetic_model()
    a = fm.forward([1, 2, 3])
    b = fm.forward([1, 2, 3])
    assert np.allclose(a, b)


def test_generate_next_returns_id() -> None:
    fm, vocab = _synthetic_model()
    nxt = fm.generate_next([1, 2, 3])
    assert 0 <= nxt < vocab


@pytest.mark.skipif(not REAL_MODELS, reason="no real GGUF fixtures downloaded")
def test_forward_on_real_model() -> None:
    path = REAL_MODELS[0]
    model = read_gguf(path)
    arch = detect(model)
    assert len(arch.layers) == arch.hyper.block_count, f"only {len(arch.layers)}/{arch.hyper.block_count} layers mapped"
    fm = ForwardModel(model, arch)
    logits = fm.forward([0, 1, 2])
    assert logits.shape == (3, arch.hyper.vocab_size)
    assert np.all(np.isfinite(logits))
