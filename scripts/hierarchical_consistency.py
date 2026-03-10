"""
Hierarchical prediction consistency analysis.

Addresses reviewer Issue #3: the multi-head architecture predicts L1, L2, L3
independently — it can produce taxonomically inconsistent label combinations
(e.g., an L2 prediction that is not a valid child of the predicted L1).

This script:
1. Builds the valid L1→L2→L3 taxonomy from the training data.
2. Runs inference on the test set using the saved three-task model.
3. Measures hierarchical inconsistency rates at L1→L2 and L2→L3 transitions.
4. Applies post-hoc constraint correction (replace inconsistent predictions with
   the most common valid child class) and reports corrected macro-F1.

Usage:
    python scripts/hierarchical_consistency.py
    python scripts/hierarchical_consistency.py --model-dir models/arabert_l1_l2_l3_best \
        --model-name aubmindlab/bert-base-arabertv02 --output results/metrics/hierarchical_consistency_arabert.json

Outputs:
    results/metrics/hierarchical_consistency.json  (default)
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


def build_taxonomy(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build valid parent→children mappings from training data.

    Returns:
        l1_to_l2: dict mapping each L1 label (int) → set of valid L2 labels (int)
        l2_to_l3: dict mapping each L2 label (int) → set of valid L3 labels (int)
    """
    l1_to_l2: dict[int, set] = defaultdict(set)
    l2_to_l3: dict[int, set] = defaultdict(set)

    for _, row in train_df.iterrows():
        l1_to_l2[int(row["label_l1"])].add(int(row["label_l2"]))
        l2_to_l3[int(row["label_l2"])].add(int(row["label_l3"]))

    return dict(l1_to_l2), dict(l2_to_l3)


def run_inference(
    model: MarBERTClassifier,
    test_df: pd.DataFrame,
    tokenizer,
    normalizer: ArabicTextNormalizer,
    tasks: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, list[int]]:
    """Run batched inference on the test set. Returns dict of task→list of predicted labels."""
    from transformers import PreTrainedTokenizerBase

    model.eval()
    model.to(device)

    texts = (test_df["text"].fillna("") if "text" in test_df.columns
             else (test_df["title_ar"].fillna("") + " " + test_df["description_ar"].fillna("")))
    texts = [normalizer(t) for t in texts.tolist()]

    all_preds: dict[str, list] = {t: [] for t in tasks}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(ids, mask)

        for t in tasks:
            preds = torch.argmax(out[f"logits_{t}"], dim=-1).cpu().tolist()
            all_preds[t].extend(preds)

    return all_preds


def apply_constraint_correction(
    preds_l1: list[int],
    preds_l2: list[int],
    preds_l3: list[int],
    l1_to_l2: dict,
    l2_to_l3: dict,
) -> tuple[list[int], list[int]]:
    """
    Post-hoc constraint correction: if predicted L2 is not a valid child of
    predicted L1, replace it with the most frequent valid L2 child.
    Same for L3 → L2. Returns corrected (l2_preds, l3_preds).
    """
    # Precompute most common child per parent (for tie-breaking correction)
    l1_default_l2 = {l1: sorted(children)[0] for l1, children in l1_to_l2.items()}
    l2_default_l3 = {l2: sorted(children)[0] for l2, children in l2_to_l3.items()}

    corrected_l2, corrected_l3 = [], []

    for p_l1, p_l2, p_l3 in zip(preds_l1, preds_l2, preds_l3):
        # Correct L2
        valid_l2 = l1_to_l2.get(p_l1, set())
        c_l2 = p_l2 if p_l2 in valid_l2 else l1_default_l2.get(p_l1, p_l2)
        corrected_l2.append(c_l2)

        # Correct L3 using corrected L2
        valid_l3 = l2_to_l3.get(c_l2, set())
        c_l3 = p_l3 if p_l3 in valid_l3 else l2_default_l3.get(c_l2, p_l3)
        corrected_l3.append(c_l3)

    return corrected_l2, corrected_l3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/marbert_l1_l2_l3_best")
    parser.add_argument("--model-name", default="UBC-NLP/MARBERTv2")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/metrics/hierarchical_consistency.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    print("Building taxonomy from training data...")
    l1_to_l2, l2_to_l3 = build_taxonomy(train_df)
    print(f"  L1 classes: {len(l1_to_l2)} | L2 classes: {sum(len(v) for v in l1_to_l2.values())} valid L1→L2 pairs")
    print(f"  L2 classes: {len(l2_to_l3)} | L3 classes: {sum(len(v) for v in l2_to_l3.values())} valid L2→L3 pairs")

    tasks = ["l1", "l2", "l3"]
    num_classes = {t: len(label_encoders[t].classes_) for t in tasks}

    print(f"Loading model from {model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarBERTClassifier(args.model_name, num_classes)

    heads_path = model_dir / "heads.pt"
    if heads_path.exists():
        model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
        from transformers import AutoModel
        model.encoder = AutoModel.from_pretrained(str(model_dir))
    else:
        raise FileNotFoundError(f"heads.pt not found in {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    normalizer = ArabicTextNormalizer()

    print("Running inference on test set...")
    preds = run_inference(model, test_df, tokenizer, normalizer,
                          tasks, args.max_length, device, args.batch_size)

    preds_l1 = preds["l1"]
    preds_l2 = preds["l2"]
    preds_l3 = preds["l3"]
    true_l1 = test_df["label_l1"].tolist()
    true_l2 = test_df["label_l2"].tolist()
    true_l3 = test_df["label_l3"].tolist()
    n = len(preds_l1)

    # --- Compute inconsistency rates ---
    inconsistent_l1_l2 = sum(
        1 for p1, p2 in zip(preds_l1, preds_l2)
        if p2 not in l1_to_l2.get(p1, set())
    )
    inconsistent_l2_l3 = sum(
        1 for p2, p3 in zip(preds_l2, preds_l3)
        if p3 not in l2_to_l3.get(p2, set())
    )
    inconsistent_full = sum(
        1 for p1, p2, p3 in zip(preds_l1, preds_l2, preds_l3)
        if (p2 not in l1_to_l2.get(p1, set())) or (p3 not in l2_to_l3.get(p2, set()))
    )

    incons_l1_l2_rate = inconsistent_l1_l2 / n
    incons_l2_l3_rate = inconsistent_l2_l3 / n
    incons_full_rate = inconsistent_full / n

    print(f"\nHierarchical inconsistency rates (n={n}):")
    print(f"  L1→L2: {inconsistent_l1_l2}/{n} = {incons_l1_l2_rate:.4f} ({incons_l1_l2_rate*100:.2f}%)")
    print(f"  L2→L3: {inconsistent_l2_l3}/{n} = {incons_l2_l3_rate:.4f} ({incons_l2_l3_rate*100:.2f}%)")
    print(f"  Full:  {inconsistent_full}/{n} = {incons_full_rate:.4f} ({incons_full_rate*100:.2f}%)")

    # --- Uncorrected F1 ---
    f1_l1_raw = f1_score(true_l1, preds_l1, average="macro", zero_division=0)
    f1_l2_raw = f1_score(true_l2, preds_l2, average="macro", zero_division=0)
    f1_l3_raw = f1_score(true_l3, preds_l3, average="macro", zero_division=0)

    print(f"\nUncorrected macro-F1: L1={f1_l1_raw:.4f} | L2={f1_l2_raw:.4f} | L3={f1_l3_raw:.4f}")

    # --- Post-hoc constraint correction ---
    corrected_l2, corrected_l3 = apply_constraint_correction(
        preds_l1, preds_l2, preds_l3, l1_to_l2, l2_to_l3
    )

    f1_l2_corrected = f1_score(true_l2, corrected_l2, average="macro", zero_division=0)
    f1_l3_corrected = f1_score(true_l3, corrected_l3, average="macro", zero_division=0)
    delta_l2 = f1_l2_corrected - f1_l2_raw
    delta_l3 = f1_l3_corrected - f1_l3_raw

    print(f"Corrected macro-F1:   L2={f1_l2_corrected:.4f} (Δ{delta_l2:+.4f}) | L3={f1_l3_corrected:.4f} (Δ{delta_l3:+.4f})")

    # Verify correction removed inconsistencies
    post_incons_l1_l2 = sum(
        1 for p1, c2 in zip(preds_l1, corrected_l2)
        if c2 not in l1_to_l2.get(p1, set())
    )
    post_incons_l2_l3 = sum(
        1 for c2, c3 in zip(corrected_l2, corrected_l3)
        if c3 not in l2_to_l3.get(c2, set())
    )

    result = {
        "model_dir": str(model_dir),
        "model_name": args.model_name,
        "n_test": n,
        "inconsistency": {
            "l1_l2_count": inconsistent_l1_l2,
            "l1_l2_rate": round(incons_l1_l2_rate, 6),
            "l1_l2_pct": round(incons_l1_l2_rate * 100, 2),
            "l2_l3_count": inconsistent_l2_l3,
            "l2_l3_rate": round(incons_l2_l3_rate, 6),
            "l2_l3_pct": round(incons_l2_l3_rate * 100, 2),
            "full_count": inconsistent_full,
            "full_rate": round(incons_full_rate, 6),
            "full_pct": round(incons_full_rate * 100, 2),
        },
        "macro_f1_uncorrected": {
            "l1": round(f1_l1_raw, 6),
            "l2": round(f1_l2_raw, 6),
            "l3": round(f1_l3_raw, 6),
        },
        "macro_f1_corrected": {
            "l2": round(f1_l2_corrected, 6),
            "l3": round(f1_l3_corrected, 6),
            "delta_l2": round(delta_l2, 6),
            "delta_l3": round(delta_l3, 6),
        },
        "post_correction_inconsistencies": {
            "l1_l2_remaining": post_incons_l1_l2,
            "l2_l3_remaining": post_incons_l2_l3,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
