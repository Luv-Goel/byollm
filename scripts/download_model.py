import sys
from pathlib import Path

# add repo root to path so `byollm` imports when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a GGUF model from Hugging Face into models/"
    )
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. klosax/tinyllamas-stories-gguf")
    parser.add_argument("--file", required=True, help="filename, e.g. tinyllamas-stories-260k-f32.gguf")
    parser.add_argument(
        "--out", default="models", help="destination directory (default: models/)"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / Path(args.file).name

    if dest.exists():
        print(f"already present: {dest}")
        return 0

    print(f"downloading {args.repo}/{args.file} ...")
    path = hf_hub_download(repo_id=args.repo, filename=args.file, local_dir=str(out))
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
