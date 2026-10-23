"""Tests for the GGUF parser and architecture detector.

Tests build small valid GGUF v1 files in-memory (no downloaded binaries
required), then also run against the real ``tinyllamas-stories-260k`` fixture
when it's present. The whole point of this project is reading *actual* model
files, so both paths matter.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from byollm.arch import detect  # noqa: E402
from byollm.gguf import (  # noqa: E402
    GGUFDtype,
    GGUFError,
    GGUFModel,
    GGUFValueType,
    dequantize,
    read_gguf,
)

REAL_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tinyllamas-stories-260k-f32.gguf"


# -- helpers: build a minimal valid GGUF v1 file -----------------------------


def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _enc_value(v: Any) -> bytes:
    """Encode one metadata value *with* its leading type code (top-level KV)."""
    return _enc_value_typed(v)


def _enc_value_typed(v: Any) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", GGUFValueType.BOOL) + struct.pack("?", v)
    if isinstance(v, int):
        if -2**31 <= v < 2**31:
            return struct.pack("<I", GGUFValueType.INT32) + struct.pack("<i", v)
        return struct.pack("<I", GGUFValueType.INT64) + struct.pack("<q", v)
    if isinstance(v, float):
        return struct.pack("<I", GGUFValueType.FLOAT32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", GGUFValueType.STRING) + _enc_str(v)
    if isinstance(v, bytes):
        return struct.pack("<I", GGUFValueType.STRING) + _enc_str(v.decode("latin-1"))
    if isinstance(v, list):
        if not v:
            raise ValueError("empty lists not supported in helper")
        elem = v[0]
        if isinstance(elem, str):
            et = GGUFValueType.STRING
        elif isinstance(elem, float):
            et = GGUFValueType.FLOAT32
        elif isinstance(elem, int):
            et = GGUFValueType.INT32
        else:
            raise ValueError(f"unsupported array element type: {type(elem)}")
        parts = [struct.pack("<I", GGUFValueType.ARRAY),
                 struct.pack("<I", et),
                 struct.pack("<I", len(v))]
        for item in v:
            # inside an array, elements are raw typed blobs: no leading code
            if isinstance(item, str):
                parts.append(_enc_str(item))
            else:
                parts.append(_enc_raw(item))
        return b"".join(parts)
    raise ValueError(f"unsupported metadata value type: {type(v)}")


def _enc_raw(v: Any) -> bytes:
    """Encode one array element *without* a leading type code."""
    if isinstance(v, str):
        return _enc_str(v)
    if isinstance(v, bool):
        return struct.pack("?", v)
    if isinstance(v, int):
        if -2**31 <= v < 2**31:
            return struct.pack("<i", v)
        return struct.pack("<q", v)
    if isinstance(v, float):
        return struct.pack("<f", v)
    raise ValueError(f"unsupported array element: {type(v)}")


def build_gguf_v1(
    metadata: dict[str, Any],
    tensors: dict[str, tuple[list[int], bytes, GGUFDtype]],
    alignment: int = 32,
) -> bytes:
    """Serialise a GGUF v1 file from pure Python structures.

    Layout: magic + version + counts, then metadata KV pairs, then tensor
    infos (name, dims, dtype, offset), then padding to ``alignment``, then
    tensor data blocks each padded to ``alignment``.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 1)  # version
    out += struct.pack("<II", len(tensors), len(metadata))  # v1: uint32 counts

    # metadata KV pairs
    for k, v in metadata.items():
        out += _enc_str(k)
        out += _enc_value(v)

    # tensor infos — record where each 8-byte offset placeholder lives
    offset_placeholder_positions: list[int] = []
    for name, (shape, data, dtype) in tensors.items():
        out += _enc_str(name)
        out += struct.pack("<I", len(shape))  # v1: uint32 dim count
        for dim in shape:
            out += struct.pack("<I", dim)  # v1: uint32 dims
        out += struct.pack("<I", int(dtype))
        offset_placeholder_positions.append(len(out))
        out += b"\x00" * 8  # offset placeholder

    header_end = len(out)
    data_start = header_end + (-header_end % alignment)

    # compute offsets: each tensor data block is padded to ``alignment``.
    # Offsets are relative to the start of the tensor-data section
    # (GGUF / llama.cpp convention), not absolute file positions.
    cur = 0
    for pos, (_, (shape, data, dtype)) in zip(
        offset_placeholder_positions, tensors.items()
    ):
        out[pos : pos + 8] = struct.pack("<Q", cur)
        cur += (-cur % alignment) + len(data)

    # padding then tensor data
    out += b"\x00" * (data_start - len(out))
    cur = data_start
    for _, (shape, data, dtype) in tensors.items():
        pad = (-cur % alignment)
        out += b"\x00" * pad
        cur += pad
        out += data
        cur += len(data)
    return bytes(out)


# -- llama metadata for a tiny 2-block model ---------------------------------


def llama_metadata(vocab_size: int = 8) -> dict[str, Any]:
    return {
        "general.architecture": "llama",
        "general.name": "test-model",
        "general.alignment": 32,
        "llama.context_length": 16,
        "llama.embedding_length": 4,
        "llama.block_count": 2,
        "llama.feed_forward_length": 8,
        "llama.rope.dimension_count": 4,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 2,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(vocab_size)],
        "tokenizer.ggml.token_type": [1] * vocab_size,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 0,
    }


def f32(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def make_tiny_model() -> bytes:
    vocab = 8
    emb = 4
    tensors: dict[str, tuple[list[int], bytes, GGUFDtype]] = {
        "token_embd.weight": ([vocab, emb], f32([float(i + j * 0.1) for i in range(emb) for j in range(vocab)]), GGUFDtype.F32),
        "output_norm.weight": ([emb], f32([1.0] * emb), GGUFDtype.F32),
        "output.weight": ([vocab, emb], f32([0.25] * (vocab * emb)), GGUFDtype.F32),
    }
    for b in range(2):
        tensors.update({
            f"blk.{b}.attn_q.weight": ([emb, emb], f32([0.1] * (emb * emb)), GGUFDtype.F32),
            f"blk.{b}.attn_k.weight": ([emb, emb], f32([0.1] * (emb * emb)), GGUFDtype.F32),
            f"blk.{b}.attn_v.weight": ([emb, emb], f32([0.1] * (emb * emb)), GGUFDtype.F32),
            f"blk.{b}.attn_output.weight": ([emb, emb], f32([0.1] * (emb * emb)), GGUFDtype.F32),
            f"blk.{b}.ffn_gate.weight": ([emb, 8], f32([0.1] * (emb * 8)), GGUFDtype.F32),
            f"blk.{b}.ffn_down.weight": ([8, emb], f32([0.1] * (8 * emb)), GGUFDtype.F32),
            f"blk.{b}.ffn_up.weight": ([emb, 8], f32([0.1] * (emb * 8)), GGUFDtype.F32),
            f"blk.{b}.attn_norm.weight": ([emb], f32([1.0] * emb), GGUFDtype.F32),
            f"blk.{b}.ffn_norm.weight": ([emb], f32([1.0] * emb), GGUFDtype.F32),
        })
    return build_gguf_v1(llama_metadata(vocab), tensors)


# -- tests -------------------------------------------------------------------


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.gguf"
    p.write_bytes(make_tiny_model())
    return p


def test_read_gguf_round_trip(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    assert model.version == 1
    assert model.metadata["general.architecture"] == "llama"
    assert model.metadata["llama.block_count"] == 2
    assert model.metadata["llama.context_length"] == 16
    assert len(model.tensors) == 3 + 9 * 2


def test_metadata_types_round_trip(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    meta = model.metadata
    assert meta["llama.context_length"] == 16
    assert isinstance(meta["llama.attention.layer_norm_rms_epsilon"], float)
    assert meta["tokenizer.ggml.bos_token_id"] == 1
    assert meta["tokenizer.ggml.eos_token_id"] == 2
    assert meta["tokenizer.ggml.padding_token_id"] == 0
    tokens = meta["tokenizer.ggml.tokens"]
    assert isinstance(tokens, list) and len(tokens) == 8


def test_read_tensor_data(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    emb = model.tensor("token_embd.weight")
    assert emb.shape == (8, 4)
    assert emb.dtype == np.float32
    assert emb.size == 32


def test_detect_architecture(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    arch = detect(model)
    assert arch.kind == "llama"
    h = arch.hyper
    assert h.block_count == 2
    assert h.embedding_length == 4
    assert h.head_count == 2
    assert h.head_dim == 2
    assert h.vocab_size == 8


def test_layer_mapping(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    arch = detect(model)
    assert set(arch.layers) == {0, 1}
    l0 = arch.layer(0)
    assert l0.attn_q == "blk.0.attn_q.weight"
    assert l0.ffn_down == "blk.0.ffn_down.weight"
    assert arch.embedding == "token_embd.weight"


def test_missing_tensor_raises(tiny_model: Path) -> None:
    model = read_gguf(tiny_model)
    arch = detect(model)
    try:
        arch.layer(42)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_bad_magic_raises(tmp_path: Path) -> None:
    p = tmp_path / "not.gguf"
    p.write_bytes(b"NOTGGUF" + b"\x00" * 20)
    try:
        read_gguf(p)
        raise AssertionError("expected GGUFError")
    except GGUFError:
        pass


def test_dequantize_q8_0() -> None:
    bs = GGUFDtype.Q8_0.block_size  # 32 elements per block
    # GGUF Q8_0 layout per block: [f16 scale][int8 quants x 32]
    blocks = []
    for scale in (0.5, -0.25):
        blocks.append(np.float16(scale).tobytes())
        blocks.append((np.arange(bs, dtype=np.int8) - 64).tobytes())
    raw = b"".join(blocks)
    out = dequantize(raw, GGUFDtype.Q8_0, [bs * 2])
    expected_q = np.arange(bs, dtype=np.float32) - 64.0
    expected = np.concatenate([expected_q * 0.5, expected_q * (-0.25)])
    np.testing.assert_allclose(out, expected)


@pytest.mark.skipif(not REAL_FIXTURE.exists(), reason="real GGUF fixture not downloaded")
def test_real_gguf_file() -> None:
    model = read_gguf(REAL_FIXTURE)
    assert model.version == 1
    assert len(model.tensors) == 48
    assert model.metadata["general.architecture"] == "llama"
    arch = detect(model)
    assert arch.hyper.block_count == 5
    assert arch.hyper.embedding_length == 64
    emb = model.tensor("token_embd.weight")
    assert emb.shape == (64, 512)
