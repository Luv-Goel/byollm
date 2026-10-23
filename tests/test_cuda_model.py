"""Phase 5 — CuPy CUDA forward pass tests.

Verifies CudaForwardModel matches ForwardModel (NumPy) on real GGUF tensors
within float32 tolerance, then benchmark tokens/sec on the real 260k model.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import cupy as cp  # noqa: E402

def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cupy(), reason="CuPy not installed — CUDA tests skipped"
)


from byollm.arch import detect  # noqa: E402
from byollm.cuda_model import CudaForwardModel, CudaKVCache  # noqa: E402
from byollm.gguf import read_gguf  # noqa: E402
from byollm.model import ForwardModel  # noqa: E402

MODEL_PATH = os.environ.get(
    "BYOLLM_MODEL", "models/tinyllamas-stories-260k-f32.gguf"
)


def _load_pair():
    model = read_gguf(MODEL_PATH)
    arch = detect(model)
    return model, arch


def test_cuda_matches_numpy_forward() -> None:
    """CuPy forward logits must match NumPy forward logits (float32 tol)."""
    model, arch = _load_pair()
    numpy_fm = ForwardModel(model, arch)
    cuda_fm = CudaForwardModel(model, arch)

    tokens = [7, 42, 123, 5, 99]
    np_logits = numpy_fm.forward(tokens)
    cp_logits = cuda_fm.forward(tokens)
    np.testing.assert_allclose(np_logits, cp.asnumpy(cp_logits), rtol=1e-4, atol=1e-4)


def test_cuda_matches_numpy_with_cache() -> None:
    """CuPy cached decode must match NumPy cached decode."""
    model, arch = _load_pair()
    numpy_fm = ForwardModel(model, arch)
    cuda_fm = CudaForwardModel(model, arch)

    tokens = [7, 42, 123, 5, 99]

    # NumPy cached
    from byollm.model import KVCache

    np_cache = KVCache(arch.hyper.block_count, 64, arch.hyper.head_count, arch.hyper.head_dim)
    np_prefill = numpy_fm.forward(tokens[:-1], start_pos=0, cache=np_cache)
    np_decode = numpy_fm.forward([tokens[-1]], start_pos=len(tokens) - 1, cache=np_cache)[-1]

    # CuPy cached
    cp_cache = CudaKVCache(arch.hyper.block_count, 64, arch.hyper.head_count, arch.hyper.head_dim)
    cp_prefill = cuda_fm.forward(tokens[:-1], start_pos=0, cache=cp_cache)
    cp_decode = cuda_fm.forward([tokens[-1]], start_pos=len(tokens) - 1, cache=cp_cache)[-1]

    np.testing.assert_allclose(np_prefill[-1], cp.asnumpy(cp_prefill[-1]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np_decode, cp.asnumpy(cp_decode), rtol=1e-4, atol=1e-4)


def test_cuda_generate_extends_prompt() -> None:
    """CuPy generate() must return prompt + max_new tokens."""
    model, arch = _load_pair()
    cuda_fm = CudaForwardModel(model, arch)
    prompt = [7, 42, 123]
    out = cuda_fm.generate(prompt, max_new=8, strategy="greedy")
    assert out[: len(prompt)] == prompt
    assert len(out) == len(prompt) + 8


def test_cuda_generate_deterministic_greedy() -> None:
    """Greedy generation must be deterministic."""
    model, arch = _load_pair()
    cuda_fm = CudaForwardModel(model, arch)
    prompt = [7, 42, 123]
    out1 = cuda_fm.generate(prompt, max_new=8, strategy="greedy")
    out2 = cuda_fm.generate(prompt, max_new=8, strategy="greedy")
    assert out1 == out2


def test_cuda_tokens_per_second() -> None:
    """Benchmark tokens/sec on the real model; print for the record."""
    import cupy as cp

    model, arch = _load_pair()
    cuda_fm = CudaForwardModel(model, arch)

    prompt = list(range(32))
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    out = cuda_fm.generate(prompt, max_new=32, strategy="greedy")
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = len(out) - len(prompt)
    tps = new_tokens / elapsed
    print(f"\nCuPy tokens/sec (GTX 1650): {tps:.2f}  "
          f"({new_tokens} new tokens in {elapsed:.3f}s)")
    assert tps > 0
