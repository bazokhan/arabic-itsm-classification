"""
Dedicated Kaggle multi-seed runner for the EgyBERT L1+L2+L3 comparison.

This keeps the standard multi_seed_train.py workflow unchanged while providing
an explicit Kaggle-oriented entry point for the new EgyBERT comparison.

It runs the same matched comparison setting used for the single-seed EgyBERT
experiment across the fixed seeds 42, 123, and 456, and saves one checkpoint
directory per seed.

Default output directories:
    models/egybert_l1_l2_l3_seed42
    models/egybert_l1_l2_l3_seed123
    models/egybert_l1_l2_l3_seed456

Usage:
    python scripts/kaggle_multi_seed_egybert.py
    python scripts/kaggle_multi_seed_egybert.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 123, 456]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Kaggle multi-seed EgyBERT comparison"
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--model",
        default="faisalq/EgyBERT",
        help="HuggingFace model ID for the EgyBERT comparison",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, seed: int) -> tuple[list[str], Path]:
    out_dir = Path(args.output_root) / f"egybert_l1_l2_l3_seed{seed}"
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model",
        args.model,
        "--tasks",
        "l1",
        "l2",
        "l3",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(seed),
    ]
    if args.no_fp16:
        cmd.append("--no-fp16")
    return cmd, out_dir


def main() -> int:
    args = parse_args()

    for seed in SEEDS:
        cmd, out_dir = build_command(args, seed)
        print(f"\n=== Seed {seed} ===")
        print("Output:", out_dir)
        print("Command:", " ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Seed {seed} failed with exit code {result.returncode}")
            return result.returncode

    print("\nAll EgyBERT multi-seed runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
