"""Architecture detection for GGUF model files.

Given a parsed :class:`byollm.gguf.GGUFModel`, this module figures out what
kind of model it is (Llama-style decoder, in this early version), what its
hyperparameters are, and how its flat tensor table maps onto actual layers.

The design principle is: **the file is the source of truth**. Nothing is
hard-coded about layer counts, head counts, or tensor naming — everything is
read from ``general.*`` and ``llama.*`` metadata. The mapping of tensor names
to layer roles is the one convention we encode (it's the llama.cpp / llama-2.c
naming convention), and it's exposed as plain dataclasses you can read or
override.

This is the layer that turns a bag of arrays into something an inference
engine can actually run forward through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .gguf import GGUFModel


@dataclass
class LlamaArchitecture:
    """Hyperparameters extracted from GGUF metadata."""

    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    rope_dimension_count: int
    head_count: int
    head_count_kv: int
    rms_epsilon: float
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.embedding_length // self.head_count


@dataclass
class LayerTensors:
    """All tensors belonging to one transformer block."""

    attn_q: str
    attn_k: str
    attn_v: str
    attn_output: str
    ffn_gate: str
    ffn_down: str
    ffn_up: str
    attn_norm: str
    ffn_norm: str


@dataclass
class Architecture:
    """Detected architecture and its tensor mapping."""

    kind: str                      # e.g. "llama"
    hyper: LlamaArchitecture
    embedding: str                 # tensor name for token embeddings
    output_norm: str               # tensor name for final norm
    output_weight: str             # tensor name for lm head
    layers: Dict[int, LayerTensors] = field(default_factory=dict)

    def layer(self, index: int) -> LayerTensors:
        if index not in self.layers:
            raise KeyError(f"no layer {index} in this model ({len(self.layers)} present)")
        return self.layers[index]

    def summary(self) -> str:
        h = self.hyper
        lines = [
            f"architecture : {self.kind}",
            f"blocks       : {h.block_count}",
            f"embedding    : {h.embedding_length}",
            f"ffn hidden   : {h.feed_forward_length}",
            f"heads        : {h.head_count} (KV: {h.head_count_kv})",
            f"head dim     : {h.head_dim}",
            f"context      : {h.context_length}",
            f"vocab        : {h.vocab_size}",
            f"rms epsilon  : {h.rms_epsilon}",
            f"layers mapped: {len(self.layers)} / {h.block_count}",
        ]
        return "\n".join(lines)


def _detect_vocab_size(meta: Dict[str, Any], model: GGUFModel) -> int:
    """Best-effort vocab size from metadata; falls back to the embedding table.

    llama.cpp stores ``token_embd.weight`` as ``[embedding_length, vocab_size]``,
    so the second axis is vocab. Some converted files transpose it, so we prefer
    explicit metadata when available.
    """
    if "llama.vocab_size" in meta:
        return int(meta["llama.vocab_size"])
    if "tokenizer.ggml.tokens" in meta:
        return len(meta["tokenizer.ggml.tokens"])
    if "token_embd.weight" in model.tensors:
        shape = model.tensors["token_embd.weight"].shape
        if len(shape) == 2:
            return int(shape[1])
    return 0


def detect(model: GGUFModel) -> Architecture:
    """Read GGUF metadata and map tensor names to layer roles.

    Reads only metadata — no tensor data is touched. Raises if the file is
    not recognisable as a Llama-style decoder (missing required keys).
    """
    meta = model.metadata
    arch_name = meta.get("general.architecture")
    if arch_name != "llama":
        raise ValueError(
            f"unsupported architecture {arch_name!r}; this early version only "
            "handles llama-style decoders"
        )

    required = [
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.feed_forward_length",
        "llama.rope.dimension_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.attention.layer_norm_rms_epsilon",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"missing required GGUF metadata: {missing}")

    hyper = LlamaArchitecture(
        context_length=int(meta["llama.context_length"]),
        embedding_length=int(meta["llama.embedding_length"]),
        block_count=int(meta["llama.block_count"]),
        feed_forward_length=int(meta["llama.feed_forward_length"]),
        rope_dimension_count=int(meta["llama.rope.dimension_count"]),
        head_count=int(meta["llama.attention.head_count"]),
        head_count_kv=int(meta["llama.attention.head_count_kv"]),
        rms_epsilon=float(meta["llama.attention.layer_norm_rms_epsilon"]),
        vocab_size=_detect_vocab_size(meta, model),
    )

    available = set(model.tensor_names())

    def require(name: str) -> str:
        if name not in available:
            raise ValueError(f"model is missing required tensor {name!r}")
        return name

    embedding = require("token_embd.weight")
    output_norm = require("output_norm.weight")
    output_weight = require("output.weight")

    layers: Dict[int, LayerTensors] = {}
    for block in range(hyper.block_count):
        names = [
            f"blk.{block}.attn_q.weight",
            f"blk.{block}.attn_k.weight",
            f"blk.{block}.attn_v.weight",
            f"blk.{block}.attn_output.weight",
            f"blk.{block}.ffn_gate.weight",
            f"blk.{block}.ffn_down.weight",
            f"blk.{block}.ffn_up.weight",
            f"blk.{block}.attn_norm.weight",
            f"blk.{block}.ffn_norm.weight",
        ]
        if all(n in available for n in names):
            layers[block] = LayerTensors(*names)

    return Architecture(
        kind="llama",
        hyper=hyper,
        embedding=embedding,
        output_norm=output_norm,
        output_weight=output_weight,
        layers=layers,
    )
