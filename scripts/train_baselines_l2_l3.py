"""
Extended baseline evaluation: TF-IDF + LinearSVC for L2 and L3 classification.

Addresses reviewer Issue #5: the original baseline comparison only covered L1.
This script adds independent flat baselines and a hierarchical pipeline baseline
for L2 (16 classes) and L3 (48 classes), providing a fair comparison against
the transformer multi-task models at all three levels.

Usage:
    python scripts/train_baselines_l2_l3.py
    python scripts/train_baselines_l2_l3.py --data-dir data/processed --output results/metrics

Outputs:
    results/metrics/baseline_l2_l3_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arabic_itsm.data.preprocessing import ArabicTextNormalizer


def normalize_texts(texts: list[str], normalizer: ArabicTextNormalizer) -> list[str]:
    return [normalizer(t) for t in texts]


def build_tfidf_svc_pipeline() -> Pipeline:
    """TF-IDF (word + char n-grams) + LinearSVC — same as L1 baseline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=80_000, sublinear_tf=True
        )),
        ("clf", LinearSVC(max_iter=2000, C=1.0))
    ])


def evaluate(model, X_test: list, y_test: np.ndarray) -> dict:
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    infer_ms_per_sample = elapsed_ms / len(X_test) if X_test else 0.0

    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_test, preds)

    return {
        "macro_f1": round(float(macro_f1), 6),
        "accuracy": round(float(acc), 6),
        "infer_ms_per_sample": round(infer_ms_per_sample, 4),
        "n_classes": len(np.unique(y_test)),
        "n_test": len(y_test),
    }


def train_flat_baseline(
    train_texts, train_labels,
    val_texts, val_labels,
    test_texts, test_labels,
    level_name: str,
) -> dict:
    """Fit and evaluate a flat TF-IDF+LinearSVC baseline for one level."""
    print(f"\n--- Flat baseline: {level_name} ---")
    pipe = build_tfidf_svc_pipeline()

    t0 = time.time()
    pipe.fit(train_texts, train_labels)
    train_time = round(time.time() - t0, 2)

    val_metrics = evaluate(pipe, val_texts, val_labels)
    test_metrics = evaluate(pipe, test_texts, test_labels)

    print(f"  Val macro-F1:  {val_metrics['macro_f1']:.4f}")
    print(f"  Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Train time: {train_time}s")

    return {
        "model": f"TF-IDF+LinearSVC ({level_name})",
        "level": level_name,
        "type": "flat",
        "train_time_s": train_time,
        "val": val_metrics,
        "test": test_metrics,
    }


def train_hierarchical_pipeline(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    train_texts: list, val_texts: list, test_texts: list,
    level_name: str,
    parent_level: str,
) -> dict:
    """
    Hierarchical pipeline: train one LinearSVC per parent class, then
    restrict prediction to classes that are valid children of the predicted parent.

    For L2: parent = L1 (true label used at train; predicted L1 at test time)
    For L3: parent = L2 (true label used at train; predicted L2 at test time)
    """
    print(f"\n--- Hierarchical pipeline: {level_name} (conditioned on {parent_level}) ---")
    target_col = f"category_level_{level_name[-1]}"
    parent_col = f"category_level_{parent_level[-1]}"

    # Map parent class → child classes from training data
    parent_to_children: dict[str, list[str]] = {}
    for parent, group in train_df.groupby(parent_col):
        parent_to_children[parent] = sorted(group[target_col].unique().tolist())

    # Train one classifier per parent class
    parent_classifiers: dict[str, Pipeline] = {}
    for parent, children in parent_to_children.items():
        mask = train_df[parent_col] == parent
        sub_texts = [t for t, m in zip(train_texts, mask) if m]
        sub_labels = train_df.loc[mask, target_col].tolist()

        if len(set(sub_labels)) < 2:
            continue  # skip if only one child class

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                max_features=30_000, sublinear_tf=True
            )),
            ("clf", LinearSVC(max_iter=2000, C=1.0))
        ])
        pipe.fit(sub_texts, sub_labels)
        parent_classifiers[parent] = pipe

    def predict_hierarchical(texts, parent_true_labels):
        """Use true parent labels for parent routing (oracle upper-bound for hier. pipeline)."""
        predictions = []
        for text, parent in zip(texts, parent_true_labels):
            if parent in parent_classifiers:
                pred = parent_classifiers[parent].predict([text])[0]
            else:
                # Fall back: use all children from training data
                fallback_children = parent_to_children.get(parent, [])
                pred = fallback_children[0] if fallback_children else "Unknown"
            predictions.append(pred)
        return predictions

    # Oracle evaluation (true parent labels) — upper bound of the hierarchical approach
    t0 = time.perf_counter()
    val_preds = predict_hierarchical(val_texts, val_df[parent_col].tolist())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    test_preds = predict_hierarchical(test_texts, test_df[parent_col].tolist())

    val_f1 = f1_score(val_df[target_col], val_preds, average="macro", zero_division=0)
    test_f1 = f1_score(test_df[target_col], test_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_df[target_col], val_preds)
    test_acc = accuracy_score(test_df[target_col], test_preds)
    infer_ms = elapsed_ms / len(val_texts) if val_texts else 0.0

    print(f"  Val macro-F1 (oracle parent):  {val_f1:.4f}")
    print(f"  Test macro-F1 (oracle parent): {test_f1:.4f}")

    return {
        "model": f"Hierarchical TF-IDF+LinearSVC ({level_name}|oracle-parent)",
        "level": level_name,
        "type": "hierarchical_oracle_parent",
        "note": "Parent labels taken from ground truth — represents oracle upper bound of hierarchical pipeline.",
        "train_time_s": None,
        "val": {
            "macro_f1": round(float(val_f1), 6),
            "accuracy": round(float(val_acc), 6),
            "infer_ms_per_sample": round(infer_ms, 4),
            "n_classes": len(set(val_df[target_col].tolist())),
            "n_test": len(val_texts),
        },
        "test": {
            "macro_f1": round(float(test_f1), 6),
            "accuracy": round(float(test_acc), 6),
            "infer_ms_per_sample": round(infer_ms, 4),
            "n_classes": len(set(test_df[target_col].tolist())),
            "n_test": len(test_texts),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/metrics/baseline_l2_l3_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    normalizer = ArabicTextNormalizer()

    print("Normalizing text...")
    train_texts = normalize_texts(train_df["text"].fillna("").tolist(), normalizer)
    val_texts = normalize_texts(val_df["text"].fillna("").tolist(), normalizer)
    test_texts = normalize_texts(test_df["text"].fillna("").tolist(), normalizer)

    results = []

    # --- L2 flat baseline ---
    results.append(train_flat_baseline(
        train_texts, train_df["category_level_2"].tolist(),
        val_texts, val_df["category_level_2"].tolist(),
        test_texts, test_df["category_level_2"].tolist(),
        level_name="l2",
    ))

    # --- L3 flat baseline ---
    results.append(train_flat_baseline(
        train_texts, train_df["category_level_3"].tolist(),
        val_texts, val_df["category_level_3"].tolist(),
        test_texts, test_df["category_level_3"].tolist(),
        level_name="l3",
    ))

    # --- L2 hierarchical pipeline (conditioned on L1 oracle) ---
    results.append(train_hierarchical_pipeline(
        train_df, val_df, test_df,
        train_texts, val_texts, test_texts,
        level_name="l2", parent_level="l1",
    ))

    # --- L3 hierarchical pipeline (conditioned on L2 oracle) ---
    results.append(train_hierarchical_pipeline(
        train_df, val_df, test_df,
        train_texts, val_texts, test_texts,
        level_name="l3", parent_level="l2",
    ))

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['model']}: Test macro-F1 = {r['test']['macro_f1']:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
