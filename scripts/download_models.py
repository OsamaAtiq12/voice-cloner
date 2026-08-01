"""Download OmniVoice (and helper ASR) into ./models for local use."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"]
        )
        from huggingface_hub import snapshot_download

    MODELS.mkdir(parents=True, exist_ok=True)

    targets = [
        ("k2-fsa/OmniVoice", MODELS / "OmniVoice"),
        ("openai/whisper-large-v3-turbo", MODELS / "whisper-large-v3-turbo"),
    ]

    for repo_id, local_dir in targets:
        print(f"\n==> Downloading {repo_id}")
        print(f"    into {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"    done: {repo_id}")

    print("\nAll models downloaded.")
    print("Start the app with start.bat (Windows) or:")
    print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 7870")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
