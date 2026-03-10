"""
Aggregate multi-seed training results into mean ± std tables.

Run after multi_seed_train.py has completed all runs.

Usage:
    python scripts/aggregate_multi_seed.py
    python scripts/aggregate_multi_seed.py --metrics-dir results/metrics/multi_seed

Output:
    results/metrics/multi_seed_summary.json  — machine-readable summary
    Printed table suitable for copy-pasting into the paper
"""

import argparse
import json
from pathlib import Path

import numpy as np


CONFIGS_ORDER = ["marbert_l1", "marbert_l1l2l3", "marbert_5heads", "arabert_l1l2l3"]
TASK_LEVELS = ["l1", "l2", "l3", "priority", "sentiment"]


def load_all_results(metrics_dir: Path) -> dict[str, list[dict]]:
    """Load all per-seed JSON files and group by config name."""
    grouped: dict[str, list] = {}
    for f in sorted(metrics_dir.glob("*_seed*_metrics.json")):
        with open(f) as fp:
            data = json.load(fp)
        config = data["config"]
        grouped.setdefault(config, []).append(data)
    return grouped


def extract_macro_f1(metrics: dict, task: str) -> float | None:
    """Extract macro-F1 for a given task from a metrics dict."""
    # Support various JSON shapes from train.py output
    if f"test_macro_f1_{task}" in metrics:
        return float(metrics[f"test_macro_f1_{task}"])
    if "test" in metrics and isinstance(metrics["test"], dict):
        sub = metrics["test"]
        if f"macro_f1_{task}" in sub:
            return float(sub[f"macro_f1_{task}"])
        if task in sub:
            t = sub[task]
            if "macro_f1" in t:
                return float(t["macro_f1"])
    # Flat key
    key_options = [f"{task}_macro_f1", f"macro_f1_{task}", task]
    for k in key_options:
        if k in metrics:
            return float(metrics[k])
    return None


def compute_stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "n": len(values),
        "values": [round(v, 4) for v in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="results/metrics/multi_seed")
    parser.add_argument("--output", default="results/metrics/multi_seed_summary.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)

    if not metrics_dir.exists():
        print(f"Directory not found: {metrics_dir}")
        print("Run scripts/multi_seed_train.py first.")
        return

    print(f"Loading results from {metrics_dir}...")
    grouped = load_all_results(metrics_dir)

    if not grouped:
        print("No result files found. Run multi_seed_train.py first.")
        return

    summary = {}
    table_rows = []

    for config in CONFIGS_ORDER:
        if config not in grouped:
            print(f"  WARNING: No results for {config}")
            continue

        runs = grouped[config]
        print(f"\n{config}: {len(runs)} seeds")

        config_stats = {"seeds": [r["seed"] for r in runs], "tasks": {}}

        for task in TASK_LEVELS:
            values = []
            for r in runs:
                v = extract_macro_f1(r["metrics"], task)
                if v is not None:
                    values.append(v)

            if not values:
                continue

            stats = compute_stats(values)
            config_stats["tasks"][task] = stats

            mean_str = f"{stats['mean']:.4f}"
            std_str = f"{stats['std']:.4f}"
            print(f"  {task}: {mean_str} ± {std_str}  (values: {stats['values']})")

            table_rows.append({
                "config": config,
                "task": task,
                "mean": stats["mean"],
                "std": stats["std"],
                "formatted": f"{stats['mean']:.2f} ± {stats['std']:.2f}",
            })

        summary[config] = config_stats

    # Print paper-ready table
    print("\n\n=== Paper-ready table (mean ± std ×100) ===")
    print(f"{'Config':<25} {'L1':>12} {'L2':>12} {'L3':>12} {'Priority':>12} {'Sentiment':>12}")
    print("-" * 85)
    for config in CONFIGS_ORDER:
        if config not in summary:
            continue
        tasks = summary[config]["tasks"]
        row = [config]
        for task in TASK_LEVELS:
            if task in tasks:
                s = tasks[task]
                row.append(f"{s['mean']*100:.2f}±{s['std']*100:.2f}")
            else:
                row.append("—")
        print(f"{row[0]:<25} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12} {row[5]:>12}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"summary": summary, "table_rows": table_rows}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
