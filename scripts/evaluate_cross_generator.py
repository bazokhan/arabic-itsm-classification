"""
Cross-generator generalization evaluation.

Loads the trained MARBERTv2 and AraBERTv2 three-task models and evaluates
them on the Claude-generated cross-generator test set.

Compares:
  - Within-generator F1 (from test_metrics.json in each checkpoint)
  - Cross-generator F1 (Claude-generated tickets)

Usage:
    python scripts/evaluate_cross_generator.py \
        --cross-gen-data ../arabic-itsm-dataset/cross_generator_test.csv \
        --label-encoders data/processed/label_encoders.pkl

Outputs:
    results/metrics/cross_generator_results.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
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
        "model_name": "UBC-NLP/MARBERTv2",
        "tasks": ["l1", "l2", "l3"],
    },
    {
        "name": "AraBERTv2 L1+L2+L3",
        "checkpoint": "models/arabert_l1_l2_l3_best",
        "model_name": "aubmindlab/bert-base-arabertv02",
        "tasks": ["l1", "l2", "l3"],
    },
]


def run_inference(
    model: MarBERTClassifier,
    df: pd.DataFrame,
    tokenizer,
    normalizer,
    tasks: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, list[int]]:
    model.eval()
    texts = df["text"].fillna("") if "text" in df.columns else (
        df["title_ar"].fillna("") + " " + df["description_ar"].fillna("")
    )
    texts = [normalizer(t) for t in texts.tolist()]

    preds = {t: [] for t in tasks}
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(ids, mask)
        for t in tasks:
            p = torch.argmax(out[f"logits_{t}"], dim=-1).cpu().tolist()
            preds[t].extend(p)
    return preds


def encode_labels(df: pd.DataFrame, label_encoders: dict, tasks: list[str]) -> dict:
    """Encode string category columns using the training LabelEncoders."""
    label_map = {"l1": "category_level_1", "l2": "category_level_2", "l3": "category_level_3"}
    encoded = {}
    for task in tasks:
        col = label_map.get(task, task)
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in cross-generator data")
            continue
        le: LabelEncoder = label_encoders[task]
        # Handle unseen labels gracefully
        valid_mask = df[col].isin(le.classes_)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"  WARNING: {n_invalid} tickets have unseen {task} labels, dropping them")
        filtered = df[valid_mask].copy()
        encoded[task] = {
            "true": le.transform(filtered[col].tolist()).tolist(),
            "df_indices": filtered.index.tolist(),
        }
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-gen-data",
                        default="../arabic-itsm-dataset/cross_generator_test.csv")
    parser.add_argument("--label-encoders", default="data/processed/label_encoders.pkl")
    parser.add_argument("--output", default="results/metrics/cross_generator_results.json")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModel

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading cross-generator test data...")
    cg_path = Path(args.cross_gen_data)
    if not cg_path.exists():
        print(f"ERROR: {cg_path} not found. Run arabic-itsm-dataset/scripts/generate_cross_generator_test.py first.")
        return
    cg_df = pd.read_csv(cg_path)
    print(f"  Loaded {len(cg_df)} tickets from {cg_path}")

    with open(args.label_encoders, "rb") as f:
        label_encoders = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = ArabicTextNormalizer()

    all_results = []

    for cfg in MODELS:
        print(f"\n=== {cfg['name']} ===")
        checkpoint = Path(cfg["checkpoint"])
        if not checkpoint.exists():
            print(f"  Checkpoint not found: {checkpoint}, skipping.")
            continue

        tasks = cfg["tasks"]
        num_classes = {t: len(label_encoders[t].classes_) for t in tasks}

        model = MarBERTClassifier(cfg["model_name"], num_classes)
        heads_path = checkpoint / "heads.pt"
        if not heads_path.exists():
            print(f"  heads.pt not found, skipping.")
            continue
        model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
        model.encoder = AutoModel.from_pretrained(str(checkpoint))
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

        # Encode labels
        encoded = encode_labels(cg_df, label_encoders, tasks)
        if not encoded:
            print("  No valid labels to evaluate, skipping.")
            continue

        # Use only rows where all tasks have valid labels
        valid_indices = set(range(len(cg_df)))
        for task_enc in encoded.values():
            valid_indices &= set(task_enc["df_indices"])
        valid_indices = sorted(valid_indices)

        eval_df = cg_df.iloc[valid_indices].reset_index(drop=True)
        print(f"  Evaluating on {len(eval_df)} tickets (after filtering unseen labels)")

        # Run inference
        preds = run_inference(model, eval_df, tokenizer, normalizer,
                              tasks, args.max_length, device)

        # Compute cross-gen F1
        cross_gen_f1 = {}
        for task in tasks:
            label_map = {"l1": "category_level_1", "l2": "category_level_2", "l3": "category_level_3"}
            col = label_map[task]
            le = label_encoders[task]
            true_enc = le.transform(eval_df[col].tolist())
            mf1 = f1_score(true_enc, preds[task], average="macro", zero_division=0)
            cross_gen_f1[task] = round(float(mf1), 6)
            print(f"  Cross-gen {task}: macro-F1 = {mf1:.4f}")

        # Load within-gen F1 from test_metrics.json
        within_gen_f1 = {}
        test_metrics_path = checkpoint / "test_metrics.json"
        if test_metrics_path.exists():
            with open(test_metrics_path) as f:
                within_raw = json.load(f)
            for task in tasks:
                key = f"{task}_macro_f1"
                if key in within_raw:
                    within_gen_f1[task] = round(float(within_raw[key]), 6)
        else:
            print(f"  WARNING: {test_metrics_path} not found — within-gen F1 not available")

        # Compute delta
        delta_f1 = {}
        for task in tasks:
            if task in within_gen_f1 and task in cross_gen_f1:
                delta_f1[task] = round(cross_gen_f1[task] - within_gen_f1[task], 6)
                print(f"  Delta {task}: {delta_f1[task]:+.4f} (within={within_gen_f1[task]:.4f}, cross={cross_gen_f1[task]:.4f})")

        all_results.append({
            "model": cfg["name"],
            "checkpoint": str(checkpoint),
            "n_cross_gen": len(eval_df),
            "cross_gen_source": "Claude (claude-opus-4-6)",
            "within_gen_source": "Gemini (gemini-3-flash)",
            "cross_gen_macro_f1": cross_gen_f1,
            "within_gen_macro_f1": within_gen_f1,
            "delta_cross_minus_within": delta_f1,
        })

    if not all_results:
        print("\nNo models evaluated. Check checkpoint paths.")
        return

    output = {
        "description": (
            "Cross-generator generalization test: models trained on Gemini-generated data "
            "evaluated on Claude-generated tickets with the same taxonomy. "
            "Delta = cross_gen_F1 - within_gen_F1."
        ),
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
