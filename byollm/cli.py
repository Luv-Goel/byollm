"""Command-line interface for byollm.

Usage:

    python -m byollm.cli inspect --model models/tiny.gguf
    python -m byollm.cli chat --model models/tiny.gguf --max-new 64
    python -m byollm.cli generate --model models/tiny.gguf --prompt "hello" \
        --strategy top_p --max-new 32

No PyTorch, no llama.cpp — just NumPy running real GGUF tensors.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .arch import detect
from .gguf import read_gguf
from .model import ForwardModel, generate


# -- tiny byte-level tokenizer (no external deps) ---------------------------

def encode(text: str) -> list[int]:
    """UTF-8 bytes -> token ids (each byte + 1, 0 = pad/BOS placeholder)."""
    return [b + 1 for b in text.encode("utf-8")]


def decode(ids: Sequence[int]) -> str:
    """Token ids -> UTF-8 text (ids are bytes + 1)."""
    return bytes(i - 1 for i in ids if 0 < i <= 256).decode("utf-8", errors="replace")


# -- commands ---------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    model = read_gguf(args.model)
    arch = detect(model)
    print(arch.summary())
    print()
    print("tensors:", len(model.tensor_names()))
    for name in model.tensor_names()[:8]:
        t = model.tensor(name)
        print(f"  {name:40s} {tuple(t.shape)} {t.dtype}")
    if len(model.tensor_names()) > 8:
        print(f"  ... and {len(model.tensor_names()) - 8} more")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    model = read_gguf(args.model)
    arch = detect(model)
    fm = ForwardModel(model, arch)

    ids = encode(args.prompt)
    out = generate(
        fm,
        ids,
        max_new=args.max_new,
        strategy=args.strategy,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    text = decode(out)
    print(text)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    model = read_gguf(args.model)
    arch = detect(model)
    fm = ForwardModel(model, arch)

    print(f"byollm chat — {arch.hyper.block_count} blocks, "
          f"{arch.hyper.embedding_length}d, vocab {arch.hyper.vocab_size}")
    print("Type your message, Ctrl-D to exit.\n")

    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        if not line.strip():
            continue
        ids = encode(line)
        out = generate(
            fm,
            ids,
            max_new=args.max_new,
            strategy=args.strategy,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # print only the newly generated part
        new = decode(out[len(ids):])
        print(new, end="", flush=True)
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="byollm",
        description="Build Your Own LLM Inference Engine From Scratch — "
                    "pure Python + NumPy, real GGUF models.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="print architecture summary from a GGUF")
    p_ins.add_argument("--model", required=True, help="path to a .gguf file")
    p_ins.set_defaults(func=cmd_inspect)

    p_gen = sub.add_parser("generate", help="one-shot generation from a prompt")
    p_gen.add_argument("--model", required=True, help="path to a .gguf file")
    p_gen.add_argument("--prompt", required=True, help="text prompt")
    p_gen.add_argument("--max-new", type=int, default=32)
    p_gen.add_argument("--strategy", choices=["greedy", "temperature", "top_p"],
                       default="greedy")
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_chat = sub.add_parser("chat", help="interactive chat REPL")
    p_chat.add_argument("--model", required=True, help="path to a .gguf file")
    p_chat.add_argument("--max-new", type=int, default=32)
    p_chat.add_argument("--strategy", choices=["greedy", "temperature", "top_p"],
                        default="greedy")
    p_chat.add_argument("--temperature", type=float, default=1.0)
    p_chat.add_argument("--top-p", type=float, default=None)
    p_chat.set_defaults(func=cmd_chat)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
