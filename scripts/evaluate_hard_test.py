"""
Hard-test evaluation after removing near-duplicate test tickets.

This script reuses saved L1+L2+L3 checkpoints, removes test rows flagged as
near-duplicates of training examples, and reports macro-F1 on both the full
test set and the reduced hard-test subset.

Usage:
    python scripts/evaluate_hard_test.py

Outputs:
    results/metrics/hard_test_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


LABEL_COLUMNS = {
    "l1": "label_l1",
    "l2": "label_l2",
    "l3": "label_l3",
}

MODEL_SPECS = [
    {
        "name": "MARBERTv2 L1+L2+L3",
        "checkpoint": "models/marbert_l1_l2_l3_best",
        "summary_key": "marbert_l1l2l3",
    },
    {
        "name": "AraBERTv2 L1+L2+L3",
        "checkpoint": "models/arabert_l1_l2_l3_best",
        "summary_key": "arabert_l1l2l3",
    },
    {
        "name": "EgyBERT L1+L2+L3",
        "checkpoint": "models/egybert_l1_l2_l3_best",
        "summary_key": "egybert_l1l2l3",
    },
    {
        "name": "ByT5 L1+L2+L3",
        "checkpoint": "models/byt5_l1_l2_l3_best",
        "summary_key": None,
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


def compute_macro_f1(df: pd.DataFrame, preds: dict[str, list[int]]) -> dict[str, float]:
    metrics = {}
    for task, col in LABEL_COLUMNS.items():
        score = f1_score(df[col].tolist(), preds[task], average="macro", zero_division=0)
        metrics[task.upper()] = round(float(score) * 100.0, 2)
    return metrics


def recompute_flagged_test_indices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
) -> list[int]:
    print("  Recomputing full flagged test index set from TF-IDF similarities...")
    train_texts = (train_df["title_ar"].fillna("") + " " + train_df["description_ar"].fillna("")).str.strip()
    test_texts = (test_df["title_ar"].fillna("") + " " + test_df["description_ar"].fillna("")).str.strip()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50_000,
        sublinear_tf=True,
    )
    all_texts = pd.concat([train_texts, test_texts], ignore_index=True)
    vectorizer.fit(all_texts)

    train_matrix = vectorizer.transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    flagged_indices: list[int] = []
    batch_size = 200
    for start in range(0, len(test_texts), batch_size):
        sims = cosine_similarity(test_matrix[start:start + batch_size], train_matrix)
        batch_max = sims.max(axis=1)
        for offset, score in enumerate(batch_max.tolist()):
            if float(score) >= threshold:
                flagged_indices.append(start + offset)
    return sorted(flagged_indices)


def load_flagged_test_indices(report_path: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[int], float, str]:
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    threshold = float(report.get("threshold", 0.9))
    report_indices = sorted({int(pair["test_idx"]) for pair in report.get("flagged_pairs", [])})
    expected_count = int(report.get("n_flagged_above_threshold", len(report_indices)))

    if expected_count > len(report_indices):
        recomputed = recompute_flagged_test_indices(train_df, test_df, threshold)
        if len(recomputed) != expected_count:
            print(
                f"  WARNING: recomputed flagged count {len(recomputed)} does not match report count {expected_count}",
                file=sys.stderr,
            )
        return recomputed, threshold, "recomputed_from_train_test_tfidf"

    return report_indices, threshold, "report_flagged_pairs"


def load_reference_within_gen_metrics(summary_path: Path, checkpoint_dir: Path, summary_key: str | None) -> tuple[dict[str, float] | None, str | None]:
    if summary_key and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f).get("summary", {})
        tasks = summary.get(summary_key, {}).get("tasks", {})
        if tasks:
            return (
                {
                    task.upper(): round(float(stats["mean"]) * 100.0, 2)
                    for task, stats in tasks.items()
                    if task in LABEL_COLUMNS
                },
                "multi_seed_summary_mean",
            )

    test_metrics_path = checkpoint_dir / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return (
            {
                "L1": round(float(metrics["l1_macro_f1"]) * 100.0, 2),
                "L2": round(float(metrics["l2_macro_f1"]) * 100.0, 2),
                "L3": round(float(metrics["l3_macro_f1"]) * 100.0, 2),
            },
            "checkpoint_test_metrics",
        )

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--near-dup-report",
        default="../arabic-itsm-paper/assets/results/near_duplicate_report.json",
    )
    parser.add_argument(
        "--summary-path",
        default="results/metrics/multi_seed_summary.json",
    )
    parser.add_argument("--output", default="results/metrics/hard_test_results.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / args.data_dir
    report_path = (repo_root / args.near_dup_report).resolve()
    summary_path = repo_root / args.summary_path
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    flagged_indices, threshold, flagged_source = load_flagged_test_indices(report_path, train_df, test_df)
    flagged_set = set(flagged_indices)
    hard_df = test_df.drop(index=flagged_indices).reset_index(drop=True)
    print(f"  Test rows: {len(test_df)}")
    print(f"  Removed near-duplicates: {len(flagged_indices)}")
    print(f"  Hard-test rows: {len(hard_df)}")

    tasks = ["l1", "l2", "l3"]
    num_classes = {task: len(label_encoders[task].classes_) for task in tasks}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = ArabicTextNormalizer()

    results = []

    for spec in MODEL_SPECS:
        checkpoint_dir = repo_root / spec["checkpoint"]
        print(f"\n=== {spec['name']} ===")
        if not checkpoint_dir.exists():
            print(f"  Checkpoint missing: {checkpoint_dir}")
            results.append(
                {
                    "model": spec["name"],
                    "checkpoint": str(checkpoint_dir),
                    "status": "skipped_missing_checkpoint",
                }
            )
            continue

        print(f"  Loading model from {checkpoint_dir}...")
        model = MarBERTClassifier(str(checkpoint_dir), num_classes)
        heads_path = checkpoint_dir / "heads.pt"
        if not heads_path.exists():
            results.append(
                {
                    "model": spec["name"],
                    "checkpoint": str(checkpoint_dir),
                    "status": "skipped_missing_heads",
                }
            )
            continue
        model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
        full_preds = run_inference(
            model,
            test_df,
            tokenizer,
            normalizer,
            tasks,
            args.max_length,
            device,
            args.batch_size,
        )
        hard_preds = {
            task: [pred for idx, pred in enumerate(full_preds[task]) if idx not in flagged_set]
            for task in tasks
        }

        original_metrics = compute_macro_f1(test_df, full_preds)
        hard_metrics = compute_macro_f1(hard_df, hard_preds)
        reference_metrics, reference_source = load_reference_within_gen_metrics(
            summary_path, checkpoint_dir, spec["summary_key"]
        )

        entry = {
            "model": spec["name"],
            "checkpoint": str(checkpoint_dir),
            "status": "ok",
            "original_test": original_metrics,
            "hard_test": hard_metrics,
        }
        if reference_metrics:
            entry["reference_within_gen"] = reference_metrics
            entry["reference_within_gen_source"] = reference_source
        results.append(entry)

        print(f"  Original test: {original_metrics}")
        print(f"  Hard test:     {hard_metrics}")

    output = {
        "n_original_test": int(len(test_df)),
        "n_removed": int(len(flagged_indices)),
        "n_hard_test": int(len(hard_df)),
        "near_duplicate_threshold": threshold,
        "flagged_indices_source": flagged_source,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
