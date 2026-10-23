"""4GB VRAM benchmark harness — Phase 6 of the roadmap.

Benchmarks tokens/sec across model sizes on the GTX 1650 (4GB VRAM), using
both the NumPy (CPU) and CuPy (CUDA) backends. Every number is produced by
running real GGUF tensors end-to-end — no mocks, no framework bindings.

Outputs a markdown table and a JSON record suitable for committing to the repo
and quoting in README / chapter write-ups.

Usage:

    python scripts/benchmark_vram.py \
        --models models/tiny.gguf \
        --max-new 32 \
        --out bench/results.json

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# allow running standalone: python scripts/benchmark_vram.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
import numpy as np

from byollm.arch import detect
from byollm.cuda_model import CudaForwardModel
from byollm.gguf import read_gguf
from byollm.model import ForwardModel


def bench_backend(
    make_model,
    tokens: list[int],
    max_new: int,
    strategy: str = "greedy",
    repeats: int = 3,
) -> float:
    """Return mean tokens/sec over ``repeats`` runs."""
    fm = make_model()
    # warmup
    fm.generate(tokens, max_new=4, strategy=strategy)
    cp.cuda.Stream.null.synchronize()

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fm.generate(tokens, max_new=max_new, strategy=strategy)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        new_tokens = len(out) - len(tokens)
        timings.append(new_tokens / elapsed)
    return float(np.mean(timings))


def model_params(arch) -> int:
    """Rough parameter count from hyperparameters."""
    h = arch.hyper
    emb = h.embedding_length
    ffn = h.feed_forward_length
    blocks = h.block_count
    # embedding + output head
    total = emb * h.vocab_size * 2
    # per block: q/k/v/o + ffn gate/up/down + norms
    per_block = emb * (emb * 3 + emb) + emb * (ffn * 2 + ffn) + emb * 4
    total += blocks * per_block
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="4GB VRAM benchmark")
    parser.add_argument("--models", nargs="+", required=True,
                        help="GGUF model files to benchmark")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--out", default="bench/results.json")
    args = parser.parse_args()

    results = []
    rows = []

    for model_path in args.models:
        path = Path(model_path)
        if not path.exists():
            print(f"skip (missing): {path}")
            continue

        model = read_gguf(str(path))
        arch = detect(model)
        params = model_params(arch)
        tokens = [7, 42, 123, 5, 99]

        def make_numpy():
            return ForwardModel(model, arch)

        def make_cupy():
            return CudaForwardModel(model, arch)

        cpu_tps = bench_backend(make_numpy, tokens, args.max_new)
        gpu_tps = bench_backend(make_cupy, tokens, args.max_new)

        entry = {
            "model": path.name,
            "params": params,
            "blocks": arch.hyper.block_count,
            "embedding": arch.hyper.embedding_length,
            "heads": arch.hyper.head_count,
            "kv_heads": arch.hyper.head_count_kv,
            "cpu_tokens_per_sec": round(cpu_tps, 2),
            "gpu_tokens_per_sec": round(gpu_tps, 2),
            "speedup": round(gpu_tps / cpu_tps, 2),
        }
        results.append(entry)
        rows.append(
            f"| {path.name} | {params:,} | {arch.hyper.block_count} | "
            f"{arch.hyper.embedding_length} | {cpu_tps:.1f} | {gpu_tps:.1f} | "
            f"{entry['speedup']:.1f}x |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("| model | params | blocks | emb | CPU tok/s | GPU tok/s | speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(r)
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
