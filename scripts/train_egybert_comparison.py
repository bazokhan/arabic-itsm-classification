"""
Single-run wrapper for the EgyBERT comparison experiment.

This keeps the existing multi-seed workflow unchanged while providing a
dedicated entry point for the one-off L1+L2+L3 comparison requested for
EgyBERT.

Default configuration:
    - model: faisalq/EgyBERT
    - tasks: l1 l2 l3
    - epochs: 10
    - lr: 1e-5
    - batch size: 16
    - seed: 42
    - output: models/egybert_l1_l2_l3_best
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EgyBERT L1+L2+L3 comparison experiment")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/egybert_l1_l2_l3_best")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "faisalq/EgyBERT",
        "--tasks",
        "l1",
        "l2",
        "l3",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--max-length",
        str(args.max_length),
    ]
    if args.no_fp16:
        cmd.append("--no-fp16")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
