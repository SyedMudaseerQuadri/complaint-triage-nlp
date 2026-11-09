"""Rebuild everything from scratch: data -> baseline -> (optional) DistilBERT -> error analysis.

Usage: python scripts/rebuild_models.py [--skip-transformer]
The transformer step needs a CUDA PyTorch build for reasonable speed (see README).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print(">>", *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("scripts/download_data.py")
    run("-m", "src.prepare_data")
    run("-m", "src.train_baseline")
    if "--skip-transformer" not in sys.argv:
        run("-m", "src.train_transformer")
        run("-m", "src.error_analysis")
