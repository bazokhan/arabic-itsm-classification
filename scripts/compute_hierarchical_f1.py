"""
Compute hierarchical precision, recall, and F1 (hP, hR, hF) for L1+L2+L3 models.

Standard hierarchical evaluation formula (Kiritchenko et al., 2008):
  For each test example:
    - Expand the predicted L3 label to a full taxonomy path using the training taxonomy:
        predicted_path = {implied_L1, implied_L2, pred_L3}
        true_path      = {true_L1,    true_L2,    true_L3}
    - hP_i = |predicted_path ∩ true_path| / |predicted_path|
      hR_i = |predicted_path ∩ true_path| / |true_path|
  Then average hP and hR across all examples to get overall hP, hR, hF.

  For a complete 3-level taxonomy both paths always have 3 nodes, so
  hP = hR = hF = (number of path levels that are correct) / 3.

Usage:
    python scripts/compute_hierarchical_f1.py --model-dir models/marbert_l1_l2_l3_best
    python scripts/compute_hierarchical_f1.py --model-dir models/arabert_l1_l2_l3_best
    python scripts/compute_hierarchical_f1.py --model-dir models/egybert_l1_l2_l3_best
    python scripts/compute_hierarchical_f1.py --model-dir models/byt5_l1_l2_l3_best --max-length 256

Outputs:
    results/metrics/hierarchical_f1_<model_dir_name>.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


def build_reverse_taxonomy(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build reverse parent maps from training data.

    Returns:
        l3_to_l2: dict mapping each L3 label (int) → its L2 parent (int)
        l2_to_l1: dict mapping each L2 label (int) → its L1 parent (int)
    """
    l3_to_l2: dict[int, int] = {}
    l2_to_l1: dict[int, int] = {}

    for _, row in train_df.iterrows():
        l3_to_l2[int(row["label_l3"])] = int(row["label_l2"])
        l2_to_l1[int(row["label_l2"])] = int(row["label_l1"])

    return l3_to_l2, l2_to_l1


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
    """Batched inference on the test set. Returns dict of task → list of predicted labels."""
    model.eval()
    model.to(device)

    texts = (
        test_df["text"].fillna("")
        if "text" in test_df.columns
        else (test_df["title_ar"].fillna("") + " " + test_df["description_ar"].fillna(""))
    )
    texts = [normalizer(t) for t in texts.tolist()]

    all_preds: dict[str, list] = {t: [] for t in tasks}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
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


def hierarchical_f1_score(
    preds_l3: list[int],
    true_l3: list[int],
    l3_to_l2: dict[int, int],
    l2_to_l1: dict[int, int],
) -> dict[str, float]:
    """
    Compute hierarchical precision, recall, and F1.

    Each label is represented as a (level, int) tuple to prevent collisions
    across levels (e.g. L1-class-3 vs L2-class-3).
    """
    hP_list: list[float] = []
    hR_list: list[float] = []

    for p3, t3 in zip(preds_l3, true_l3):
        # True path
        t2 = l3_to_l2[t3]
        t1 = l2_to_l1[t2]
        true_path = {("l1", t1), ("l2", t2), ("l3", t3)}

        # Predicted path expanded from L3 prediction through taxonomy
        p2 = l3_to_l2.get(p3)
        p1 = l2_to_l1.get(p2) if p2 is not None else None
        pred_path = {("l3", p3)}
        if p2 is not None:
            pred_path.add(("l2", p2))
        if p1 is not None:
            pred_path.add(("l1", p1))

        intersection = true_path & pred_path
        hP = len(intersection) / len(pred_path)
        hR = len(intersection) / len(true_path)
        hP_list.append(hP)
        hR_list.append(hR)

    hP_mean = float(np.mean(hP_list))
    hR_mean = float(np.mean(hR_list))
    denom = hP_mean + hR_mean
    hF = 2.0 * hP_mean * hR_mean / denom if denom > 0 else 0.0

    return {
        "hP": round(hP_mean, 6),
        "hR": round(hR_mean, 6),
        "hF": round(hF, 6),
        "hP_pct": round(hP_mean * 100, 2),
        "hR_pct": round(hR_mean * 100, 2),
        "hF_pct": round(hF * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Path to the model directory (contains heads.pt and encoder files)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/metrics",
                        help="Directory where the output JSON will be written")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max input length in tokens/bytes (use 256 for ByT5)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hierarchical_f1_{model_dir.name}.json"

    # Detect encoder family from saved config (for logging only)
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    is_t5 = cfg.get("model_type", "").lower() in ("t5", "mt5")
    print(f"Encoder family: {'T5 (mean pool)' if is_t5 else 'BERT (CLS)'}")

    # Pass the local model_dir path as model_name.
    # MarBERTClassifier._is_t5_model checks for "t5" in the string — byt5_* dirs match correctly.
    # AutoModel/T5EncoderModel.from_pretrained() accept local paths directly.
    model_name_stub = str(model_dir)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    tasks = ["l1", "l2", "l3"]
    num_classes = {t: len(label_encoders[t].classes_) for t in tasks}
    print(f"Classes: {num_classes}")

    print("Building reverse taxonomy...")
    l3_to_l2, l2_to_l1 = build_reverse_taxonomy(train_df)
    print(f"  L3->L2 mappings: {len(l3_to_l2)} | L2->L1 mappings: {len(l2_to_l1)}")

    print(f"Loading model from {model_dir} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Constructor loads the encoder from model_dir; then load the task heads separately.
    model = MarBERTClassifier(model_name_stub, num_classes)
    heads_path = model_dir / "heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"heads.pt not found in {model_dir}")
    model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    normalizer = ArabicTextNormalizer()

    print(f"Running inference (max_length={args.max_length}, batch_size={args.batch_size})...")
    preds = run_inference(
        model, test_df, tokenizer, normalizer,
        tasks, args.max_length, device, args.batch_size,
    )

    true_l3 = test_df["label_l3"].tolist()
    preds_l3 = preds["l3"]

    print("Computing hierarchical F1...")
    scores = hierarchical_f1_score(preds_l3, true_l3, l3_to_l2, l2_to_l1)

    # Also record per-level flat macro-F1 for cross-check
    from sklearn.metrics import f1_score
    flat_f1 = {
        t: round(f1_score(test_df[f"label_{t}"].tolist(), preds[t], average="macro", zero_division=0) * 100, 2)
        for t in tasks
    }

    result = {
        "model_dir": str(model_dir),
        "n_test": len(preds_l3),
        "max_length": args.max_length,
        "hierarchical_f1": scores,
        "flat_macro_f1_pct": flat_f1,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults:")
    print(f"  hP = {scores['hP_pct']:.2f}%  hR = {scores['hR_pct']:.2f}%  hF = {scores['hF_pct']:.2f}%")
    print(f"  Flat macro-F1: L1={flat_f1['l1']}%  L2={flat_f1['l2']}%  L3={flat_f1['l3']}%")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
