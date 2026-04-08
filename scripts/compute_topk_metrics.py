"""
Top-k routing metrics for hierarchical ticket classification.

This script evaluates saved L1+L2+L3 checkpoints, captures logits for L2 and L3,
and measures routing-oriented top-k accuracy with a stricter definition:
- global top-k over all labels at that level
- success counted only when the parent prediction is correct

It also reports unconditional global top-k and path-consistency rate.

Usage:
    python scripts/compute_topk_metrics.py

Outputs:
    results/metrics/topk_routing_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


MODEL_SPECS = [
    {
        "name": "MARBERTv2 L1+L2+L3",
        "checkpoint": "models/marbert_l1_l2_l3_best",
        "consistency_json": "results/metrics/hierarchical_consistency.json",
    },
    {
        "name": "AraBERTv2 L1+L2+L3",
        "checkpoint": "models/arabert_l1_l2_l3_best",
        "consistency_json": "results/metrics/hierarchical_consistency_arabert.json",
    },
    {
        "name": "EgyBERT L1+L2+L3",
        "checkpoint": "models/egybert_l1_l2_l3_best",
        "consistency_json": None,
    },
    {
        "name": "ByT5 L1+L2+L3",
        "checkpoint": "models/byt5_l1_l2_l3_best",
        "consistency_json": None,
    },
]


def run_inference_collect_logits(
    model: MarBERTClassifier,
    df: pd.DataFrame,
    tokenizer,
    normalizer: ArabicTextNormalizer,
    tasks: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    model.eval()
    model.to(device)

    texts = df["text"].fillna("") if "text" in df.columns else (
        df["title_ar"].fillna("") + " " + df["description_ar"].fillna("")
    )
    texts = [normalizer(text) for text in texts.tolist()]

    all_logits = {task: [] for task in tasks}
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
            all_logits[task].append(outputs[f"logits_{task}"].cpu())

    return {task: torch.cat(chunks, dim=0).numpy() for task, chunks in all_logits.items()}


def topk_hit(logits_row: np.ndarray, true_label: int, k: int) -> bool:
    top_indices = np.argsort(logits_row)[::-1][: min(k, logits_row.shape[0])]
    return int(true_label) in set(int(idx) for idx in top_indices)


def compute_topk_metrics(
    logits: np.ndarray,
    true_labels: list[int],
    parent_preds: np.ndarray | None = None,
    true_parents: list[int] | None = None,
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, float | int]:
    total = len(true_labels)
    parent_correct_total = total if parent_preds is None else int(np.sum(parent_preds == np.array(true_parents)))

    result: dict[str, float | int] = {
        "n_eval": total,
        "n_parent_correct": parent_correct_total,
    }

    for k in ks:
        global_hits = sum(topk_hit(row, label, k) for row, label in zip(logits, true_labels))
        if parent_preds is None:
            conditional_hits = global_hits
        else:
            conditional_hits = sum(
                1
                for row, label, parent_pred, parent_true in zip(logits, true_labels, parent_preds, true_parents)
                if int(parent_pred) == int(parent_true) and topk_hit(row, label, k)
            )

        global_rate = global_hits / total if total else 0.0
        conditional_rate = conditional_hits / total if total else 0.0
        conditional_given_parent_rate = (
            conditional_hits / parent_correct_total if parent_correct_total else 0.0
        )

        result[f"global_top_{k}_count"] = int(global_hits)
        result[f"global_top_{k}_rate"] = round(global_rate, 6)
        result[f"global_top_{k}_pct"] = round(global_rate * 100.0, 2)
        result[f"top_{k}_with_correct_parent_count"] = int(conditional_hits)
        result[f"top_{k}_with_correct_parent_rate"] = round(conditional_rate, 6)
        result[f"top_{k}_with_correct_parent_pct"] = round(conditional_rate * 100.0, 2)
        result[f"top_{k}_given_parent_correct_rate"] = round(conditional_given_parent_rate, 6)
        result[f"top_{k}_given_parent_correct_pct"] = round(conditional_given_parent_rate * 100.0, 2)

    return result


def compute_path_consistency(
    preds_l1: np.ndarray,
    preds_l2: np.ndarray,
    preds_l3: np.ndarray,
    true_l1: list[int],
    true_l2: list[int],
) -> dict[str, float | int]:
    n = len(preds_l1)
    l2_parent_correct = int(np.sum(preds_l1 == np.array(true_l1)))
    l3_parent_correct = int(np.sum(preds_l2 == np.array(true_l2)))
    full_parent_correct = int(np.sum((preds_l1 == np.array(true_l1)) & (preds_l2 == np.array(true_l2))))
    return {
        "n_eval": n,
        "l2_parent_accuracy_pct": round((l2_parent_correct / n) * 100.0, 2) if n else 0.0,
        "l3_parent_accuracy_pct": round((l3_parent_correct / n) * 100.0, 2) if n else 0.0,
        "full_parent_chain_accuracy_pct": round((full_parent_correct / n) * 100.0, 2) if n else 0.0,
    }


def compute_taxonomy_path_consistency(
    preds_l1: np.ndarray,
    preds_l2: np.ndarray,
    preds_l3: np.ndarray,
    train_df: pd.DataFrame,
) -> dict[str, float | int]:
    l1_l2_pairs = {(int(r.label_l1), int(r.label_l2)) for r in train_df.itertuples()}
    l2_l3_pairs = {(int(r.label_l2), int(r.label_l3)) for r in train_df.itertuples()}
    n = len(preds_l1)
    consistent = sum(
        1
        for p1, p2, p3 in zip(preds_l1, preds_l2, preds_l3)
        if (int(p1), int(p2)) in l1_l2_pairs and (int(p2), int(p3)) in l2_l3_pairs
    )
    return {
        "consistent_count": int(consistent),
        "inconsistent_count": int(n - consistent),
        "rate": round((consistent / n), 6) if n else 0.0,
        "pct": round((consistent / n) * 100.0, 2) if n else 0.0,
    }


def load_reported_consistency(path: Path | None) -> dict[str, float | int] | None:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    inconsistency = data.get("inconsistency", {})
    full_rate = float(inconsistency.get("full_rate", 0.0))
    n_test = int(data.get("n_test", 0))
    consistent = n_test - int(inconsistency.get("full_count", 0))
    return {
        "consistent_count": consistent,
        "inconsistent_count": int(inconsistency.get("full_count", 0)),
        "rate": round(1.0 - full_rate, 6),
        "pct": round((1.0 - full_rate) * 100.0, 2),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/metrics/topk_routing_results.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / args.data_dir
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

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
        logits = run_inference_collect_logits(
            model,
            test_df,
            tokenizer,
            normalizer,
            tasks,
            args.max_length,
            device,
            args.batch_size,
        )

        preds_l1 = np.argmax(logits["l1"], axis=1)
        preds_l2 = np.argmax(logits["l2"], axis=1)
        preds_l3 = np.argmax(logits["l3"], axis=1)

        l2_topk = compute_topk_metrics(
            logits["l2"],
            test_df["label_l2"].tolist(),
            parent_preds=preds_l1,
            true_parents=test_df["label_l1"].tolist(),
        )
        l3_topk = compute_topk_metrics(
            logits["l3"],
            test_df["label_l3"].tolist(),
            parent_preds=preds_l2,
            true_parents=test_df["label_l2"].tolist(),
        )

        taxonomy_consistency = compute_taxonomy_path_consistency(preds_l1, preds_l2, preds_l3, train_df)
        routing_parent_accuracy = compute_path_consistency(
            preds_l1,
            preds_l2,
            preds_l3,
            test_df["label_l1"].tolist(),
            test_df["label_l2"].tolist(),
        )
        reported_consistency = load_reported_consistency(
            repo_root / spec["consistency_json"] if spec["consistency_json"] else None
        )

        entry = {
            "model": spec["name"],
            "checkpoint": str(checkpoint_dir),
            "status": "ok",
            "n_test": int(len(test_df)),
            "topk_routing_accuracy": {
                "l2_global_ranking": l2_topk,
                "l3_global_ranking": l3_topk,
            },
            "routing_parent_accuracy": routing_parent_accuracy,
            "taxonomy_path_consistency": taxonomy_consistency,
        }
        if reported_consistency is not None:
            entry["reported_existing_path_consistency"] = reported_consistency
        results.append(entry)

        print(
            "  L2 top-3 with correct parent: {:.2f}% | L3 top-3 with correct parent: {:.2f}%".format(
                l2_topk["top_3_with_correct_parent_pct"],
                l3_topk["top_3_with_correct_parent_pct"],
            )
        )
        print(f"  Taxonomy path consistency: {taxonomy_consistency['pct']:.2f}%")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
