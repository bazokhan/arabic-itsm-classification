"""
Evaluation metrics for ITSM classification.

All metrics use macro-averaging to treat each class equally regardless of
support size — appropriate for our moderately imbalanced dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list[str] = None,
) -> dict:
    """
    Compute accuracy, macro-F1, macro-precision, macro-recall.

    Parameters
    ----------
    y_true : array-like of int or str
    y_pred : array-like of int or str
    labels : list of class names (optional, for display)

    Returns
    -------
    dict with keys: accuracy, macro_f1, macro_precision, macro_recall
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def classification_report_df(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list[str] = None,
) -> pd.DataFrame:
    """
    Return sklearn classification report as a tidy DataFrame.

    Useful for displaying per-class results in notebooks and saving to CSV.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    df.index.name = "class"
    return df.round(4)
