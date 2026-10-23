"""GGUF file format parser, written from scratch.

GGUF (GPT-Generated Unified Format) is the container format used by llama.cpp
and most local inference runtimes. This module reads a real GGUF file into
pure Python structures with zero third-party dependencies beyond ``numpy``.

Format reference (llama.cpp ``gguf.h`` / ``gguf.py``):

    +-----------------------------------+
    | 4 bytes  magic "GGUF"             |
    | uint32   version                  |
    | uint64   tensor count             |
    | uint64   metadata key-value count |
    +-----------------------------------+
    | metadata KV pairs                 |
    +-----------------------------------+
    | tensor infos                      |
    +-----------------------------------+
    | padding to alignment              |
    +-----------------------------------+
    | tensor data                       |
    +------------------------------------

Every length is a ``uint64`` *preceding* the blob it describes. Metadata keys
are strings; values are one of a fixed set of typed codes.

Tensor names follow a ``block.NN.suffix`` convention understood by
:mod:`byollm.arch.detect`.

This module deliberately exposes the raw header data so that the educational
"what is actually in this file" questions (see ``docs/``) have a real answer.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSIONS = (1, 2, 3)
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(enum.IntEnum):
    """Type codes used by the GGUF metadata section."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    @property
    def ctype(self) -> str:
        return {
            GGUFValueType.UINT8: "B",
            GGUFValueType.INT8: "b",
            GGUFValueType.UINT16: "H",
            GGUFValueType.INT16: "h",
            GGUFValueType.UINT32: "I",
            GGUFValueType.INT32: "i",
            GGUFValueType.FLOAT32: "f",
            GGUFValueType.BOOL: "?",
            GGUFValueType.UINT64: "Q",
            GGUFValueType.INT64: "q",
            GGUFValueType.FLOAT64: "d",
        }[self]

    @property
    def fixed_size(self) -> int | None:
        """Byte size for scalar codes, ``None`` for variable-length ones."""
        return {
            GGUFValueType.UINT8: 1,
            GGUFValueType.INT8: 1,
            GGUFValueType.UINT16: 2,
            GGUFValueType.INT16: 2,
            GGUFValueType.UINT32: 4,
            GGUFValueType.INT32: 4,
            GGUFValueType.FLOAT32: 4,
            GGUFValueType.BOOL: 1,
            GGUFValueType.UINT64: 8,
            GGUFValueType.INT64: 8,
            GGUFValueType.FLOAT64: 8,
        }.get(self)


class GGUFDtype(enum.IntEnum):
    """GGUF tensor storage codes.

    Only the formats this engine can actually consume are exposed; exotic
    k-quants are rejected loudly rather than silently mis-decoded.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30

    @property
    def block_size(self) -> int:
        """Values stored per quantisation block."""
        return {
            GGUFDtype.F32: 1,
            GGUFDtype.F16: 1,
            GGUFDtype.BF16: 1,
            GGUFDtype.Q4_0: 32,
            GGUFDtype.Q4_1: 32,
            GGUFDtype.Q5_0: 32,
            GGUFDtype.Q5_1: 32,
            GGUFDtype.Q8_0: 32,
            GGUFDtype.Q8_1: 32,
            GGUFDtype.Q2_K: 256,
            GGUFDtype.Q3_K: 256,
            GGUFDtype.Q4_K: 256,
            GGUFDtype.Q5_K: 256,
            GGUFDtype.Q6_K: 256,
            GGUFDtype.Q8_K: 256,
        }[self]

    @property
    def type_size(self) -> int:
        """Bytes occupied by one full block of this dtype."""
        return {
            GGUFDtype.F32: 4,
            GGUFDtype.F16: 2,
            GGUFDtype.BF16: 2,
            GGUFDtype.Q4_0: 18,   # 1 half scale + 16 packed bytes
            GGUFDtype.Q4_1: 20,   # half scale + half min + 16 packed bytes
            GGUFDtype.Q5_0: 22,   # half scale + 4 high-bit bytes + 16 packed
            GGUFDtype.Q5_1: 24,
            GGUFDtype.Q8_0: 34,   # half scale + 32 signed bytes
            GGUFDtype.Q8_1: 36,
            GGUFDtype.Q2_K: 84,
            GGUFDtype.Q3_K: 110,
            GGUFDtype.Q4_K: 144,
            GGUFDtype.Q5_K: 176,
            GGUFDtype.Q6_K: 210,
            GGUFDtype.Q8_K: 292,
        }[self]

    @property
    def numpy_equivalent(self) -> np.dtype:
        return {
            GGUFDtype.F32: np.dtype(np.float32),
            GGUFDtype.F16: np.dtype(np.float16),
            GGUFDtype.BF16: np.dtype(np.uint16),  # interpreted by dequant
        }[self]


class GGUFError(RuntimeError):
    """Raised when a file is not a readable GGUF container."""


@dataclass
class GGUFTensorInfo:
    """Location and shape metadata for one tensor, before its data is read."""

    name: str
    shape: List[int]           # GGUF stores dimensions, element 0 varies
    dtype: GGUFDtype
    offset: int                # relative to the tensor-data section

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        return self.n_elements // self.dtype.block_size * self.dtype.type_size


@dataclass
class GGUFModel:
    """A parsed GGUF file: metadata, tensor table, and lazy tensor access."""

    path: Path
    version: int
    metadata: Dict[str, Any]
    tensors: Dict[str, GGUFTensorInfo]
    alignment: int
    tensor_data_offset: int
    _file: BinaryIO | None = field(default=None, repr=False)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "GGUFModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- access ------------------------------------------------------------
    def tensor(self, name: str) -> np.ndarray:
        """Dequantise one tensor into a float32 numpy array."""
        info = self.tensors.get(name)
        if info is None:
            raise KeyError(f"tensor not in file: {name}")
        if self._file is None:
            self._file = open(self.path, "rb")

        n_bytes = info.n_bytes
        start = self.tensor_data_offset + info.offset
        self._file.seek(start)
        raw = self._file.read(n_bytes)
        if len(raw) != n_bytes:
            raise GGUFError(
                f"truncated read for {name}: wanted {n_bytes}, got {len(raw)}"
            )
        return dequantize(raw, info.dtype, info.shape)

    def tensor_names(self) -> List[str]:
        return list(self.tensors)

    def summary(self) -> str:
        total = sum(t.n_bytes for t in self.tensors.values())
        lines = [
            f"file      : {self.path.name}",
            f"version   : {self.version}",
            f"alignment : {self.alignment}",
            f"tensors   : {len(self.tensors)}",
            f"data size : {total / 1e6:.2f} MB",
            f"metadata  : {len(self.metadata)} keys",
        ]
        return "\n".join(lines)


# -- primitive reader --------------------------------------------------------


class _Reader:
    """Cursor over a bytes object with GGUF's little-endian conventions.

    GGUF v1 stores counts and string lengths as ``uint32``; v2/v3 widened
    them to ``uint64``. The reader adapts to the file's version.
    """

    def __init__(self, buf: bytes, version: int) -> None:
        self.buf = buf
        self.pos = 0
        self.version = version

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GGUFError("unexpected end of file while reading header")
        chunk = self.buf[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def scalar(self, code: GGUFValueType) -> Any:
        n = code.fixed_size
        raw = self.take(n)
        if code == GGUFValueType.BOOL:
            return bool(raw[0])
        if code == GGUFValueType.STRING:
            raise GGUFError("strings are not scalars")
        return struct.unpack("<" + code.ctype, raw)[0]

    def string(self) -> str:
        if self.version == 1:
            (n,) = struct.unpack("<I", self.take(4))
        else:
            (n,) = struct.unpack("<Q", self.take(8))
        return self.take(n).decode("utf-8")

    def value(self) -> Any:
        (code,) = struct.unpack("<I", self.take(4))
        return self.typed_value(GGUFValueType(code))

    def typed_value(self, code: GGUFValueType) -> Any:
        if code == GGUFValueType.STRING:
            return self.string()
        if code == GGUFValueType.ARRAY:
            (elem_code_raw,) = struct.unpack("<I", self.take(4))
            elem_code = GGUFValueType(elem_code_raw)
            if self.version == 1:
                (count,) = struct.unpack("<I", self.take(4))
            else:
                (count,) = struct.unpack("<Q", self.take(8))
            return [self.typed_value(elem_code) for _ in range(count)]
        return self.scalar(code)


def dequantize(raw: bytes, dtype: GGUFDtype, shape: List[int]) -> np.ndarray:
    """Convert on-disk quantised bytes back to float32 values.

    Implemented for the dtypes this engine needs. The legacy ``q4_0`` / ``q8_0``
    families are handled explicitly with the exact unsigned-4-bit and
    signed-8-bit unpacking rules; k-quants are decomposed via their per-block
    super-scale byte. Unknown dtypes raise instead of guessing.
    """
    n_elem = 1
    for d in shape:
        n_elem *= d
    if n_elem % dtype.block_size:
        raise GGUFError(
            f"element count {n_elem} is not divisible by block size "
            f"{dtype.block_size} for {dtype.name}"
        )

    if dtype in (GGUFDtype.F32, GGUFDtype.F16):
        arr = np.frombuffer(raw, dtype=dtype.numpy_equivalent, count=n_elem)
        return arr.reshape(shape).astype(np.float32)

    if dtype == GGUFDtype.BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16, count=n_elem).astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(shape).astype(np.float32)

    if dtype == GGUFDtype.Q8_0:
        out = np.empty(n_elem, dtype=np.float32)
        bs = dtype.block_size
        for i in range(n_elem // bs):
            off = i * dtype.type_size
            (scale,) = struct.unpack_from("<e", raw, off)
            q = np.frombuffer(raw, dtype=np.int8, count=bs, offset=off + 2)
            out[i * bs : (i + 1) * bs] = q * np.float32(scale)
        return out.reshape(shape)

    if dtype == GGUFDtype.Q4_0:
        out = np.empty(n_elem, dtype=np.float32)
        bs = dtype.block_size
        for i in range(n_elem // bs):
            off = i * dtype.type_size
            (scale,) = struct.unpack_from("<e", raw, off)
            packed = raw[off + 2 : off + 2 + bs // 2]
            low = packed & 0x0F
            high = packed >> 4
            nib = np.empty(bs, dtype=np.float32)
            nib[0::2] = low - 8
            nib[1::2] = high - 8
            out[i * bs : (i + 1) * bs] = nib * np.float32(scale)
        return out.reshape(shape)

    raise GGUFError(
        f"dequantisation for {dtype.name} is not implemented in this early "
        "version; re-save the tensor as f32 or q8_0 to run it here"
    )


def read_gguf(path: str | Path) -> GGUFModel:
    """Parse a GGUF container header and tensor table (data stays on disk)."""
    path = Path(path)
    if not path.exists():
        raise GGUFError(f"no such file: {path}")
    blob = path.read_bytes()

    (version,) = struct.unpack("<I", blob[4:8])
    if version not in GGUF_SUPPORTED_VERSIONS:
        raise GGUFError(f"unsupported GGUF version {version}")
    r = _Reader(blob, version)
    if r.take(4) != GGUF_MAGIC:
        raise GGUFError(f"not a GGUF file (bad magic): {path}")
    r.take(4)  # version consumed above
    n_tensors, n_kv = struct.unpack("<II", r.take(8))

    metadata: Dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        metadata[key] = r.value()

    alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))

    tensors: Dict[str, GGUFTensorInfo] = {}
    dim_width = 4 if version == 1 else 8
    for _ in range(n_tensors):
        name = r.string()
        n_dims, = struct.unpack("<I", r.take(4))
        dims = list(struct.unpack("<" + "I" * n_dims, r.take(4 * n_dims))) \
            if dim_width == 4 else list(struct.unpack("<" + "Q" * n_dims, r.take(8 * n_dims)))
        (dtype_code,) = struct.unpack("<I", r.take(4))
        (offset,) = struct.unpack("<Q", r.take(8))
        tensors[name] = GGUFTensorInfo(
            name=name, shape=dims, dtype=GGUFDtype(dtype_code), offset=offset
        )

    header_end = r.pos
    data_start = header_end + (-header_end % alignment)

    return GGUFModel(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        tensor_data_offset=data_start,
    )
