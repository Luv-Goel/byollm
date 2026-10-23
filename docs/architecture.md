# byollm architecture

## Layer 1 — Container (`byollm/gguf.py`)

Parses the GGUF file format: a header (magic, version, tensor count,
metadata count), a metadata key-value section, a tensor info table
(name, dims, dtype, offset), then raw tensor data.

- Version-aware: GGUF v1 stores counts and string lengths as `uint32`; v2
  and v3 widened them to `uint64`. The reader adapts automatically.
- Lazy tensors: `model.tensor(name)` reads + dequantises on demand — no
  whole-file load into RAM.
- Dequantisers implement the exact GGUF bit rules for F32, F16, BF16, Q8_0,
  Q4_0. Unknown dtypes raise loudly instead of guessing.

## Layer 2 — Architecture (`byollm/arch.py`)

Given a parsed `GGUFModel`, this module figures out what kind of model it is
and maps its flat tensor table onto transformer layers.

Everything is read from file metadata — `llama.block_count`,
`llama.attention.head_count`, `llama.feed_forward_length`, etc. Nothing is
hard-coded. The only convention encoded is the tensor naming scheme
(`blk.N.attn_q.weight`, …), which matches llama.cpp / llama-2.c.

Returns an `Architecture` object with hyperparameters, the embedding tensor,
the output norm/weight, and a `LayerTensors` dataclass per block.

## Layer 3 — Forward pass (planned)

RoPE → attention → RMSNorm → MLP → output head. One block at a time, pure
NumPy, against real tensors from real GGUF files.

## Design rules

1. **File is source of truth.** Architecture comes from metadata, not
   constants.
2. **No mystery numbers.** Every dequantisation constant traces to the GGUF
   spec.
3. **Tests against real bytes.** Synthetic GGUF fixtures plus a real
   260K-parameter model fixture.
4. **Laptop-first.** Runs on CPU; GPU kernels are an optional later layer.
