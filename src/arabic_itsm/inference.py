"""
Production-grade Inference Engine for Arabic ITSM classification.

This module provides a single, high-level interface to handle raw Arabic ticket
inputs and return structured, multi-task predictions. It abstracts away
normalization, tokenization, and multi-head tensor management.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.models.classifier import MarBERTClassifier


class ITSMInferenceEngine:
    """
    Unified Inference Engine for Arabic ITSM Models.

    Supports dynamic loading of models with varying numbers of classification
    heads (e.g., L1-only, L1+L2, or full Multi-Task).

    Attributes:
        device: The torch device (cuda/cpu).
        tasks: List of tasks supported by the loaded model.
        model_name: HuggingFace model identifier.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        label_encoders_path: Union[str, Path],
        model_name: str = "UBC-NLP/MARBERTv2",
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        """
        Initializes the engine and loads model weights.

        Args:
            checkpoint_dir: Path to the directory containing 'heads.pt' and config files.
            label_encoders_path: Path to the 'label_encoders.pkl' file.
            model_name: The base transformer model string.
            max_length: Maximum sequence length for the tokenizer.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.model_name = model_name

        # 1. Load Label Encoders to determine tasks and class names
        with open(label_encoders_path, "rb") as f:
            self.label_encoders = pickle.load(f)

        # 2. Initialize Normalizer and Tokenizer
        self.normalizer = ArabicTextNormalizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 3. Detect active tasks from the heads.pt state dict
        heads_state = torch.load(self.checkpoint_dir / "heads.pt", map_location="cpu")
        self.tasks = self._detect_tasks(heads_state)
        
        # 4. Initialize and load model
        num_classes = {t: len(self.label_encoders[t].classes_) for t in self.tasks}
        self.model = MarBERTClassifier(model_name, num_classes=num_classes)
        
        # Load weights: heads from heads.pt, encoder from checkpoint_dir
        self.model.heads.load_state_dict(heads_state)
        self.model.encoder = AutoModel.from_pretrained(str(self.checkpoint_dir))
        
        self.model.to(self.device)
        self.model.eval()

    def _detect_tasks(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Infers the task names (l1, l2, priority, etc.) from head keys."""
        # Head keys are usually "l1.1.weight", "l2.1.weight", etc.
        unique_tasks = set()
        for key in state_dict.keys():
            task_name = key.split(".")[0]
            unique_tasks.add(task_name)
        return sorted(list(unique_tasks))

    def predict(self, title: str, description: str) -> Dict[str, Any]:
        """
        Runs the full inference pipeline on a single ticket.

        Args:
            title: Raw Arabic ticket title.
            description: Raw Arabic ticket description.

        Returns:
            Dictionary containing predicted labels and confidence scores for each task.
        """
        # 1. Preprocessing
        clean_title = self.normalizer(title)
        clean_desc = self.normalizer(description)
        combined_text = f"{clean_title} {clean_desc}".strip()

        # 2. Tokenization
        inputs = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

        # 4. Post-processing (Probabilities and Label Mapping)
        results = {"input_text": combined_text, "predictions": {}}
        
        for task in self.tasks:
            logits = outputs[f"logits_{task}"]
            probs = F.softmax(logits, dim=-1).squeeze(0)
            
            conf, idx = torch.max(probs, dim=-1)
            label_str = self.label_encoders[task].inverse_transform([idx.item()])[0]
            
            results["predictions"][task] = {
                "label": label_str,
                "confidence": round(conf.item(), 4)
            }

        return results

    def predict_batch(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Utility for batch inference (e.g., CSV processing)."""
        return [self.predict(t["title"], t["description"]) for t in tickets]


if __name__ == "__main__":
    # Quick sanity check logic
    # usage: python -m src.arabic_itsm.inference
    # (Update paths to match your local environment for testing)
    pass
