"""
Multi-seed training wrapper for reproducibility analysis.

Addresses reviewer Issue #2: single-seed experiments are insufficient to
characterize training variance. This script runs the 4 key configurations
across 3 random seeds and saves per-seed metrics.

Key configurations:
    1. MARBERTv2  L1-only
    2. MARBERTv2  L1+L2+L3
    3. MARBERTv2  all-5-heads
    4. AraBERTv2  L1+L2+L3

Seeds: 42, 123, 456

Usage (local):
    python scripts/multi_seed_train.py --dry-run   # print commands only
    python scripts/multi_seed_train.py --configs marbert_l1 marbert_l1l2l3

Usage (Kaggle — use the dedicated notebook instead):
    See notebooks/kaggle_multi_seed_training.ipynb

Output:
    results/metrics/multi_seed/<config>_seed<N>_metrics.json  (one per run)
    (Aggregate with: python scripts/aggregate_multi_seed.py)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 123, 456]

CONFIGS = {
    "marbert_l1": {
        "model": "UBC-NLP/MARBERTv2",
        "tasks_flag": "--task l1",
        "epochs": 5,
        "output_prefix": "marbert_l1",
    },
    "marbert_l1l2l3": {
        "model": "UBC-NLP/MARBERTv2",
        "tasks_flag": "--tasks l1 l2 l3",
        "epochs": 10,
        "output_prefix": "marbert_l1_l2_l3",
    },
    "marbert_5heads": {
        "model": "UBC-NLP/MARBERTv2",
        "tasks_flag": "--multi-task",
        "epochs": 10,
        "output_prefix": "marbert_multi_task",
    },
    "arabert_l1l2l3": {
        "model": "aubmindlab/bert-base-arabertv02",
        "tasks_flag": "--tasks l1 l2 l3",
        "epochs": 10,
        "output_prefix": "arabert_l1_l2_l3",
    },
}


def build_command(config_name: str, cfg: dict, seed: int, output_root: Path) -> list[str]:
    seed_dir = output_root / f"{cfg['output_prefix']}_seed{seed}"
    cmd = [
        sys.executable, "scripts/train.py",
        *cfg["tasks_flag"].split(),
        "--model", cfg["model"],
        "--epochs", str(cfg["epochs"]),
        "--seed", str(seed),
        "--output-dir", str(seed_dir),
    ]
    return cmd, seed_dir


def extract_metrics_from_run(seed_dir: Path, config_name: str, seed: int) -> dict | None:
    """
    After training, read the saved metrics file and return a structured dict.
    train.py saves metrics under the checkpoint dir as 'test_metrics.json'.
    """
    metrics_file = seed_dir / "test_metrics.json"
    if not metrics_file.exists():
        print(f"  WARNING: {metrics_file} not found — run may have failed.")
        return None

    with open(metrics_file) as f:
        raw = json.load(f)

    return {
        "config": config_name,
        "seed": seed,
        "metrics": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS.keys()),
                        default=list(CONFIGS.keys()),
                        help="Which configurations to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--metrics-dir", default="results/metrics/multi_seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.configs) * len(args.seeds)
    print(f"Multi-seed training: {total} runs ({len(args.configs)} configs × {len(args.seeds)} seeds)")
    print(f"Seeds: {args.seeds}")
    print(f"Configs: {args.configs}")

    run_n = 0
    for config_name in args.configs:
        cfg = CONFIGS[config_name]
        for seed in args.seeds:
            run_n += 1
            cmd, seed_dir = build_command(config_name, cfg, seed, output_root)
            print(f"\n[{run_n}/{total}] {config_name} | seed={seed}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Output:  {seed_dir}")

            if args.dry_run:
                continue

            # Skip if already done
            out_metrics = metrics_dir / f"{config_name}_seed{seed}_metrics.json"
            if out_metrics.exists():
                print(f"  Already exists, skipping: {out_metrics}")
                continue

            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"  ERROR: run failed with return code {result.returncode}")
                continue

            metrics = extract_metrics_from_run(seed_dir, config_name, seed)
            if metrics:
                with open(out_metrics, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"  Saved metrics to {out_metrics}")

    print("\nDone. Run scripts/aggregate_multi_seed.py to compute mean ± std.")


if __name__ == "__main__":
    main()
