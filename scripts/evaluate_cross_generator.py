"""
Cross-generator generalization evaluation.

Loads trained three-task models and evaluates them on the Claude-generated
cross-generator test set.

Compares:
  - Within-generator F1 from results/metrics/multi_seed_summary.json when available
  - Fallback to checkpoint test_metrics.json for single-run checkpoints
  - Cross-generator F1 on Claude-generated tickets

Usage:
    python scripts/evaluate_cross_generator.py \
        --cross-gen-data ../arabic-itsm-dataset/cross_generator_test.csv \
        --label-encoders data/processed/label_encoders.pkl

Outputs:
    results/metrics/cross_generator_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


MODELS = [
    {
        "name": "MARBERTv2 L1+L2+L3",
        "checkpoint": "models/marbert_l1_l2_l3_best",
        "summary_key": "marbert_l1l2l3",
        "tasks": ["l1", "l2", "l3"],
    },
    {
        "name": "AraBERTv2 L1+L2+L3",
        "checkpoint": "models/arabert_l1_l2_l3_best",
        "summary_key": "arabert_l1l2l3",
        "tasks": ["l1", "l2", "l3"],
    },
    {
        "name": "EgyBERT L1+L2+L3",
        "checkpoint": "models/egybert_l1_l2_l3_best",
        "summary_key": "egybert_l1l2l3",
        "tasks": ["l1", "l2", "l3"],
    },
    {
        "name": "ByT5 L1+L2+L3",
        "checkpoint": "models/byt5_l1_l2_l3_best",
        "summary_key": None,
        "tasks": ["l1", "l2", "l3"],
    },
]


def run_inference(
    model: MarBERTClassifier,
    df: pd.DataFrame,
    tokenizer,
    normalizer: ArabicTextNormalizer,
    tasks: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, list[int]]:
    model.eval()
    model.to(device)

    texts = df["text"].fillna("") if "text" in df.columns else (
        df["title_ar"].fillna("") + " " + df["description_ar"].fillna("")
    )
    texts = [normalizer(text) for text in texts.tolist()]

    preds = {task: [] for task in tasks}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        for task in tasks:
            batch_preds = torch.argmax(outputs[f"logits_{task}"], dim=-1).cpu().tolist()
            preds[task].extend(batch_preds)
    return preds


def encode_labels(df: pd.DataFrame, label_encoders: dict, tasks: list[str]) -> dict[str, dict[str, list[int]]]:
    label_map = {
        "l1": "category_level_1",
        "l2": "category_level_2",
        "l3": "category_level_3",
    }
    encoded: dict[str, dict[str, list[int]]] = {}
    for task in tasks:
        col = label_map.get(task, task)
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in cross-generator data")
            continue

        encoder: LabelEncoder = label_encoders[task]
        valid_mask = df[col].isin(encoder.classes_)
        invalid_count = int((~valid_mask).sum())
        if invalid_count > 0:
            print(f"  WARNING: {invalid_count} tickets have unseen {task} labels, dropping them")
        filtered = df[valid_mask].copy()
        encoded[task] = {
            "true": encoder.transform(filtered[col].tolist()).tolist(),
            "df_indices": filtered.index.tolist(),
        }
    return encoded


def load_within_gen_f1(summary_path: Path, checkpoint_path: Path, summary_key: str | None, tasks: list[str]) -> tuple[dict[str, float], str | None]:
    if summary_key and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f).get("summary", {})

        task_summary = summary.get(summary_key, {}).get("tasks", {})
        within_gen = {}
        for task in tasks:
            stats = task_summary.get(task)
            if stats and "mean" in stats:
                within_gen[task] = round(float(stats["mean"]), 6)
        if within_gen:
            return within_gen, "multi_seed_summary_mean"

    test_metrics_path = checkpoint_path / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return (
            {
                task: round(float(metrics[f"{task}_macro_f1"]), 6)
                for task in tasks
                if f"{task}_macro_f1" in metrics
            },
            "checkpoint_test_metrics",
        )

    return {}, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-gen-data",
        default="../arabic-itsm-dataset/cross_generator_test.csv",
    )
    parser.add_argument("--label-encoders", default="data/processed/label_encoders.pkl")
    parser.add_argument(
        "--within-gen-summary",
        default="results/metrics/multi_seed_summary.json",
    )
    parser.add_argument("--output", default="results/metrics/cross_generator_results.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading cross-generator test data...")
    cross_gen_path = (repo_root / args.cross_gen_data).resolve()
    if not cross_gen_path.exists():
        raise FileNotFoundError(
            f"{cross_gen_path} not found. Run arabic-itsm-dataset/scripts/generate_cross_generator_test.py first."
        )
    cross_gen_df = pd.read_csv(cross_gen_path)
    print(f"  Loaded {len(cross_gen_df)} tickets from {cross_gen_path}")

    with open(repo_root / args.label_encoders, "rb") as f:
        label_encoders = pickle.load(f)

    summary_path = repo_root / args.within_gen_summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = ArabicTextNormalizer()

    all_results = []

    for cfg in MODELS:
        print(f"\n=== {cfg['name']} ===")
        checkpoint = repo_root / cfg["checkpoint"]
        if not checkpoint.exists():
            print(f"  Checkpoint not found: {checkpoint}, skipping.")
            all_results.append(
                {
                    "model": cfg["name"],
                    "checkpoint": str(checkpoint),
                    "status": "skipped_missing_checkpoint",
                }
            )
            continue

        tasks = cfg["tasks"]
        num_classes = {task: len(label_encoders[task].classes_) for task in tasks}
        model = MarBERTClassifier(str(checkpoint), num_classes)

        heads_path = checkpoint / "heads.pt"
        if not heads_path.exists():
            print("  heads.pt not found, skipping.")
            all_results.append(
                {
                    "model": cfg["name"],
                    "checkpoint": str(checkpoint),
                    "status": "skipped_missing_heads",
                }
            )
            continue
        model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)

        encoded = encode_labels(cross_gen_df, label_encoders, tasks)
        if not encoded:
            print("  No valid labels to evaluate, skipping.")
            continue

        valid_indices = set(range(len(cross_gen_df)))
        for task_encoded in encoded.values():
            valid_indices &= set(task_encoded["df_indices"])
        valid_indices = sorted(valid_indices)

        eval_df = cross_gen_df.iloc[valid_indices].reset_index(drop=True)
        print(f"  Evaluating on {len(eval_df)} tickets (after filtering unseen labels)")

        preds = run_inference(
            model,
            eval_df,
            tokenizer,
            normalizer,
            tasks,
            args.max_length,
            device,
            args.batch_size,
        )

        cross_gen_f1 = {}
        label_map = {
            "l1": "category_level_1",
            "l2": "category_level_2",
            "l3": "category_level_3",
        }
        for task in tasks:
            encoder = label_encoders[task]
            true_encoded = encoder.transform(eval_df[label_map[task]].tolist())
            score = f1_score(true_encoded, preds[task], average="macro", zero_division=0)
            cross_gen_f1[task] = round(float(score), 6)
            print(f"  Cross-gen {task}: macro-F1 = {score:.4f}")

        within_gen_f1, within_source = load_within_gen_f1(summary_path, checkpoint, cfg["summary_key"], tasks)

        delta_f1 = {}
        for task in tasks:
            if task in within_gen_f1 and task in cross_gen_f1:
                delta = cross_gen_f1[task] - within_gen_f1[task]
                delta_f1[task] = round(delta, 6)
                print(
                    f"  Delta {task}: {delta:+.4f} "
                    f"(within={within_gen_f1[task]:.4f}, cross={cross_gen_f1[task]:.4f})"
                )

        all_results.append(
            {
                "model": cfg["name"],
                "checkpoint": str(checkpoint),
                "status": "ok",
                "n_cross_gen": len(eval_df),
                "cross_gen_source": "Claude (claude-opus-4-6)",
                "within_gen_source": within_source,
                "cross_gen_macro_f1": cross_gen_f1,
                "within_gen_macro_f1": within_gen_f1,
                "delta_cross_minus_within": delta_f1,
            }
        )

    output = {
        "description": (
            "Cross-generator generalization test: models trained on Gemini-generated data "
            "evaluated on Claude-generated tickets with the same taxonomy. "
            "Delta = cross_gen_F1 - within_gen_F1. Within-generator F1 is loaded "
            "from multi-seed summaries when available, otherwise from checkpoint test_metrics.json."
        ),
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
