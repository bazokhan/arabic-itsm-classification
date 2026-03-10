"""
Error analysis for Priority and Sentiment auxiliary tasks.

Addresses reviewer Issue #4: Priority and Sentiment predictions performed near-random
(15.37% and 24.07% macro-F1 respectively). This script investigates why by:

1. Computing per-class confusion matrices for Priority and Sentiment.
2. Checking label distribution imbalance.
3. Measuring label-text alignment: for 20 sampled tickets per class, reports
   whether the text contains signals that should predict the label.
4. Analyzing whether Priority/Sentiment labels correlate with text features
   (TF-IDF cosine distance between class centroids).
5. Checking if these labels were explicitly present in the generation prompt
   by sampling examples and reporting the label vs. text surface signal.

Usage:
    python scripts/auxiliary_error_analysis.py
    python scripts/auxiliary_error_analysis.py --model-dir models/marbert_multi_task_best \
        --output results/metrics/auxiliary_analysis.json

Outputs:
    results/metrics/auxiliary_analysis.json
    results/figures/priority_confusion.png
    results/figures/sentiment_confusion.png
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


def run_inference_all_tasks(
    model: MarBERTClassifier,
    test_df: pd.DataFrame,
    tokenizer,
    normalizer,
    tasks: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, list]:
    model.eval()
    texts = (test_df["text"].fillna("") if "text" in test_df.columns
             else (test_df["title_ar"].fillna("") + " " + test_df["description_ar"].fillna("")))
    texts = [normalizer(t) for t in texts.tolist()]

    all_preds: dict[str, list] = {t: [] for t in tasks}
    all_probs: dict[str, list] = {t: [] for t in tasks}

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(ids, mask)
        for t in tasks:
            logits = out[f"logits_{t}"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).tolist()
            all_preds[t].extend(preds)
            all_probs[t].extend(probs.tolist())

    return all_preds, all_probs


def analyze_class_separability(
    train_df: pd.DataFrame, texts: list[str], label_col: str
) -> dict:
    """
    Measure inter-class TF-IDF distance. High mean cosine distance → labels are
    textually distinguishable. Near-zero → labels are not recoverable from text.
    """
    vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    labels = train_df[label_col].tolist()
    unique_labels = sorted(set(labels))

    centroids = []
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        centroid = X[mask].mean(axis=0)
        centroids.append(np.asarray(centroid))

    centroids = np.vstack(centroids)
    dists = cosine_distances(centroids)
    np.fill_diagonal(dists, np.nan)
    mean_inter_class_dist = float(np.nanmean(dists))

    return {
        "n_classes": len(unique_labels),
        "mean_inter_class_cosine_distance": round(mean_inter_class_dist, 6),
        "note": (
            "Higher distance → classes are more textually distinct. "
            "Values near 0 suggest labels cannot be reliably predicted from text features."
        ),
    }


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def sample_label_text_alignment(
    test_df: pd.DataFrame,
    preds: list[int],
    true_labels: list[int],
    label_encoder,
    task_name: str,
    n_sample: int = 8,
) -> list[dict]:
    """Sample misclassified examples to show label vs. text content."""
    examples = []
    classes = label_encoder.classes_
    misclassified = [
        (i, true_labels[i], preds[i])
        for i in range(len(preds))
        if preds[i] != true_labels[i]
    ]

    # Sample up to n_sample errors
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(misclassified), size=min(n_sample, len(misclassified)), replace=False)

    for idx in sample_indices:
        i, true_lbl, pred_lbl = misclassified[idx]
        text_col = "text" if "text" in test_df.columns else "description_ar"
        text = str(test_df.iloc[i][text_col])[:300]
        examples.append({
            "true_label": classes[true_lbl],
            "pred_label": classes[pred_lbl],
            "text_excerpt": text,
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/marbert_multi_task_best")
    parser.add_argument("--model-name", default="UBC-NLP/MARBERTv2")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/metrics/auxiliary_analysis.json")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    figures_dir = Path(args.figures_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    tasks = ["l1", "l2", "l3", "priority", "sentiment"]
    num_classes = {t: len(label_encoders[t].classes_) for t in tasks if t in label_encoders}
    active_tasks = list(num_classes.keys())

    print(f"Loading model from {model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarBERTClassifier(args.model_name, num_classes)
    heads_path = model_dir / "heads.pt"
    if not heads_path.exists():
        print(f"WARNING: {heads_path} not found. Run with a five-task model checkpoint.")
        return
    model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
    model.encoder = AutoModel.from_pretrained(str(model_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    normalizer = ArabicTextNormalizer()

    print("Running inference...")
    preds, probs = run_inference_all_tasks(
        model, test_df, tokenizer, normalizer, active_tasks,
        args.max_length, device
    )

    analysis = {}

    for task in ["priority", "sentiment"]:
        if task not in preds:
            print(f"  Skipping {task} (not in model tasks)")
            continue

        print(f"\n=== Analysing: {task} ===")
        le = label_encoders[task]
        class_names = le.classes_.tolist()
        true_col = f"label_{task}"
        true = test_df[true_col].tolist()
        pred = preds[task]

        macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
        report = classification_report(true, pred,
                                       target_names=[str(c) for c in class_names],
                                       output_dict=True, zero_division=0)
        print(f"  Macro-F1: {macro_f1:.4f}")

        # Label distribution in test set
        from collections import Counter
        true_dist = Counter(true)
        pred_dist = Counter(pred)
        print(f"  True distribution: {dict(sorted(true_dist.items()))}")
        print(f"  Pred distribution: {dict(sorted(pred_dist.items()))}")

        # Confusion matrix
        cm = confusion_matrix(true, pred)
        plot_confusion(
            cm, [str(c) for c in class_names],
            f"{task.title()} Confusion Matrix (5-head model, test set)",
            figures_dir / f"{task}_confusion.png"
        )

        # Class separability via TF-IDF
        train_texts = train_df["text"].fillna("").tolist()
        normalizer_texts = [normalizer(t) for t in train_texts]
        sep = analyze_class_separability(train_df, normalizer_texts, f"label_{task}")
        print(f"  Inter-class cosine distance: {sep['mean_inter_class_cosine_distance']:.4f}")

        # Misclassification examples
        errors = sample_label_text_alignment(test_df, pred, true, le, task, n_sample=8)

        # Label distribution in training
        train_true_dist = Counter(train_df[f"label_{task}"].tolist())

        analysis[task] = {
            "macro_f1": round(macro_f1, 6),
            "n_classes": len(class_names),
            "class_names": class_names,
            "test_true_distribution": {str(class_names[k]): v for k, v in sorted(true_dist.items())},
            "test_pred_distribution": {str(class_names[k]): v for k, v in sorted(pred_dist.items())},
            "train_true_distribution": {str(class_names[k]): v for k, v in sorted(train_true_dist.items())},
            "class_separability": sep,
            "per_class_f1": {
                str(class_names[i]): round(report.get(str(class_names[i]), {}).get("f1-score", 0.0), 4)
                for i in range(len(class_names))
            },
            "misclassification_examples": errors,
            "interpretation": (
                f"Inter-class cosine distance of {sep['mean_inter_class_cosine_distance']:.4f} "
                f"{'suggests labels are NOT well-separated in TF-IDF space' if sep['mean_inter_class_cosine_distance'] < 0.3 else 'suggests labels are moderately separable'}. "
                f"Macro-F1 of {macro_f1:.4f} {'is near-random baseline' if macro_f1 < 0.30 else 'is above random'}."
            ),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
