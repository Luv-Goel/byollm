"""Phase 3 — Generation tests (KV cache + samplers).

Compares cached decode against the no-cache forward pass to prove the cache
produces identical logits, then checks generate() produces plausible output
on a synthetic model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byollm.arch import detect
from byollm.gguf import GGUFDtype, read_gguf
from byollm.model import KVCache, ForwardModel, generate, sample_next, softmax, top_p_mask
from tests.test_gguf import build_gguf_v1, llama_metadata


def _synthetic_model() -> ForwardModel:
    vocab = 16
    emb = 4
    ffn = 8
    tensors: dict[str, tuple[list[int], bytes, GGUFDtype]] = {}
    rng = np.random.default_rng(1)

    def w(name: str, shape: tuple[int, ...]) -> None:
        tensors[name] = (list(shape), rng.standard_normal(shape).astype(np.float32).tobytes(), GGUFDtype.F32)

    w("token_embd.weight", (emb, vocab))
    w("output_norm.weight", (emb,))
    w("output.weight", (emb, vocab))
    for block in range(2):
        for n, shape in {
            "attn_q": (emb, emb),
            "attn_k": (emb, emb),
            "attn_v": (emb, emb),
            "attn_output": (emb, emb),
            "ffn_gate": (emb, ffn),
            "ffn_down": (ffn, emb),
            "ffn_up": (emb, ffn),
            "attn_norm": (emb,),
            "ffn_norm": (emb,),
        }.items():
            w(f"blk.{block}.{n}.weight", shape)

    meta = llama_metadata(vocab_size=vocab)
    data = build_gguf_v1(meta, tensors)
    path = Path("/tmp/synthetic_gen.gguf")
    path.write_bytes(data)
    model = read_gguf(path)
    arch = detect(model)
    return ForwardModel(model, arch)


def test_kv_cache_matches_no_cache() -> None:
    """Decode step with cache must equal full forward on the same prefix."""
    fm = _synthetic_model()
    tokens = [1, 2, 3, 4]

    # no-cache forward over the whole prompt
    full = fm.forward(tokens)
    last_full = full[-1]

    # prefill with cache (all but last token), then decode the last token
    cache = KVCache(fm.arch.hyper.block_count, 64, fm.arch.hyper.head_count, fm.arch.hyper.head_dim)
    prefill = fm.forward(tokens[:-1], start_pos=0, cache=cache)
    decode = fm.forward([tokens[-1]], start_pos=len(tokens) - 1, cache=cache)[-1]

    # cache decode must equal full forward at the same (last) position
    assert np.allclose(decode, last_full, atol=1e-5)
    # prefill last position must equal full forward at position -2
    assert np.allclose(prefill[-1], full[-2], atol=1e-5)


def test_generate_extends_prompt() -> None:
    fm = _synthetic_model()
    out = fm.generate([1, 2], max_new=5, strategy="greedy")
    assert len(out) == 7
    assert out[:2] == [1, 2]
    assert all(0 <= t < 16 for t in out)


def test_generate_deterministic() -> None:
    fm = _synthetic_model()
    a = fm.generate([1, 2], max_new=5, strategy="greedy")
    b = fm.generate([1, 2], max_new=5, strategy="greedy")
    assert a == b


def test_sample_next_greedy_is_argmax() -> None:
    logits = np.array([0.1, 5.0, 3.0, 2.0])
    assert sample_next(logits, "greedy") == 1


def test_sample_next_temperature_is_probabilistic() -> None:
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(42)
    draws = [sample_next(logits, "temperature", rng=rng) for _ in range(200)]
    assert set(draws) == {0, 1, 2, 3}


def test_softmax_sums_to_one() -> None:
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert probs[2] > probs[1] > probs[0]


def test_top_p_keeps_at_least_one() -> None:
    probs = np.array([0.05, 0.05, 0.05, 0.85])
    keep = top_p_mask(probs, 0.9)
    assert keep.any()
    assert keep[-1]


@pytest.mark.parametrize("strategy", ["greedy", "temperature", "top_p"])
def test_generate_all_strategies(strategy: str) -> None:
    fm = _synthetic_model()
    kwargs: dict = {"strategy": strategy, "max_new": 4}
    if strategy == "temperature":
        kwargs["temperature"] = 0.8
    if strategy == "top_p":
        kwargs["top_p"] = 0.9
    out = fm.generate([1, 2], **kwargs)
    assert len(out) == 6
    assert all(0 <= t < 16 for t in out)
