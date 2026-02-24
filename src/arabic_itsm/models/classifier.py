"""
MarBERTv2-based multi-task classifier for Arabic ITSM tickets.

Architecture: shared MarBERT encoder + independent linear classification heads.
Supports L1, L2, and priority prediction simultaneously.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class MarBERTClassifier(nn.Module):
    """
    Shared-encoder multi-task classifier built on MarBERTv2.

    The [CLS] token representation from MarBERT is passed through a dropout
    layer and then into independent linear heads — one per classification task.
    Losses are averaged across active tasks during training.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: "UBC-NLP/MARBERTv2".
    num_classes : dict[str, int]
        Number of output classes per task, e.g. {"l1": 6, "l2": 14, "priority": 5}.
    dropout : float
        Dropout rate applied to the [CLS] representation before each head.
    """

    def __init__(
        self,
        model_name: str = "UBC-NLP/MARBERTv2",
        num_classes: dict[str, int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = {"l1": 6}

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, n_cls),
                )
                for task, n_cls in num_classes.items()
            }
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **labels,  # label_l1, label_l2, label_priority, ...
    ) -> dict:
        """
        Parameters
        ----------
        input_ids, attention_mask, token_type_ids : torch.Tensor
            Standard BERT inputs.
        **labels : torch.Tensor
            keyword args named `label_<task>` for each active task.

        Returns
        -------
        dict with keys:
            loss (optional) — total loss summed over active tasks
            logits_<task> — raw logits per task
        """
        encoder_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        result = {}
        total_loss = None

        for task, head in self.heads.items():
            logits = head(cls)
            result[f"logits_{task}"] = logits

            label_key = f"label_{task}"
            if label_key in labels and labels[label_key] is not None:
                task_loss = self.loss_fn(logits, labels[label_key])
                total_loss = task_loss if total_loss is None else total_loss + task_loss

        if total_loss is not None:
            result["loss"] = total_loss / len(self.heads)

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Returns predicted class indices per task (argmax of logits)."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, token_type_ids)
        return {
            task: torch.argmax(out[f"logits_{task}"], dim=-1)
            for task in self.heads
        }
