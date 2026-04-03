"""
Multi-task classifier for Arabic ITSM tickets.

Architecture: shared encoder + independent linear classification heads.
Supports BERT-family encoders (CLS pooling) and T5-family encoders (mean pooling).
Supports L1, L2, L3, Priority, and Sentiment prediction simultaneously.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel, PreTrainedModel


def _is_t5_model(model_name: str) -> bool:
    """Return True if the model name indicates a T5-family encoder (e.g. ByT5, mT5)."""
    return bool(re.search(r't5', str(model_name), re.IGNORECASE))


class MarBERTClassifier(nn.Module):
    """
    Shared-encoder multi-task classifier.

    For BERT-family models (MARBERTv2, AraBERTv2, etc.) the [CLS] token is used.
    For T5-family models (ByT5, mT5, etc.) mean pooling over encoder outputs is used,
    and T5EncoderModel is loaded instead of the full seq2seq T5Model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: "UBC-NLP/MARBERTv2".
    num_classes : dict[str, int]
        Number of output classes per task, e.g. {"l1": 6, "l2": 14, "priority": 5}.
    dropout : float
        Dropout rate applied to the pooled representation before each head.
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

        self._use_mean_pool = _is_t5_model(model_name)
        if self._use_mean_pool:
            self.encoder: PreTrainedModel = T5EncoderModel.from_pretrained(model_name)
        else:
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
        if not self._use_mean_pool and token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        if self._use_mean_pool:
            # Mean pooling over all non-padding token positions (for T5-family)
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            cls = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token (for BERT-family)

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
