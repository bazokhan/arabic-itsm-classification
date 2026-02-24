"""
PyTorch Dataset and data loading utilities for Arabic ITSM classification.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Optional


class ITSMDataset(Dataset):
    """
    PyTorch Dataset for Arabic ITSM ticket classification.

    Handles tokenization and label encoding for multi-task classification:
    L1 category, L2 category, and priority prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: title_ar, description_ar, category_level_1,
        category_level_2, priority.
    tokenizer : PreTrainedTokenizerBase
        MarBERT (or any HF) tokenizer.
    normalizer : callable, optional
        Arabic text normalizer applied before tokenization.
    label_encoders : dict[str, LabelEncoder]
        Pre-fit label encoders for each task. If None, fitted from df.
    max_length : int
        Maximum token sequence length (default 128).
    tasks : list[str]
        Which tasks to include labels for. Subset of ['l1', 'l2', 'priority'].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        normalizer=None,
        label_encoders: Optional[dict] = None,
        max_length: int = 128,
        tasks: list[str] = ("l1",),
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.max_length = max_length
        self.tasks = list(tasks)

        # Build text: "[TITLE] [SEP] [DESCRIPTION]"
        titles = df["title_ar"].fillna("").tolist()
        descs = df["description_ar"].fillna("").tolist()

        if self.normalizer is not None:
            titles = [self.normalizer(t) for t in titles]
            descs = [self.normalizer(d) for d in descs]

        self.texts = [f"{t} {d}".strip() for t, d in zip(titles, descs)]

        # Label encoders
        self._task_columns = {
            "l1": "category_level_1",
            "l2": "category_level_2",
            "l3": "category_level_3",
            "priority": "priority",
            "sentiment": "sentiment",
        }

        self.label_encoders = label_encoders or {}
        self.labels: dict[str, list] = {}

        for task in self.tasks:
            col = self._task_columns[task]
            values = df[col].astype(str).tolist()
            if task not in self.label_encoders:
                le = LabelEncoder()
                le.fit(values)
                self.label_encoders[task] = le
            self.labels[task] = self.label_encoders[task].transform(values).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        for task in self.tasks:
            item[f"label_{task}"] = torch.tensor(self.labels[task][idx], dtype=torch.long)

        return item

    @property
    def num_classes(self) -> dict[str, int]:
        return {task: len(self.label_encoders[task].classes_) for task in self.tasks}


def load_splits(
    csv_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    stratify_col: str = "category_level_1",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split the dataset into train/val/test with stratification.

    Returns
    -------
    train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)

    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[stratify_col],
        random_state=random_seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df[stratify_col],
        random_state=random_seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
