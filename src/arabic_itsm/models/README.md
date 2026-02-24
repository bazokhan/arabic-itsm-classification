# Model Architectures

This directory contains the PyTorch model definitions for the Arabic ITSM classification project.

## Why are these separated from the Notebooks?

To adhere to the **DRY (Don't Repeat Yourself)** principle and ensure academic rigor, the model architecture is defined here as a reusable Python package rather than being hardcoded inside Jupyter Notebooks. 

This separation provides several benefits:
1. **Maintainability**: Changes to the model architecture (like adding dropout or changing the loss function) only need to be made once here, and they automatically apply to all training and evaluation notebooks.
2. **Readability**: Notebooks stay focused on data analysis, experiments, and results rather than being cluttered with boilerplate PyTorch code.
3. **Reproducibility**: It ensures that the exact same model logic is used across the L1, L2, L3, and Multi-Task experiments.

## Primary Architecture: `MarBERTClassifier` (`classifier.py`)

The `MarBERTClassifier` is a **Shared-Encoder Multi-Task** architecture.

### Key Components:
- **Shared Encoder**: Uses `UBC-NLP/MARBERTv2` as the base transformer to extract high-quality features from Egyptian Arabic text.
- **Independent Heads**: Dynamically creates linear classification heads for different tasks. This allows the model to predict **Category Level 1, Level 2, Level 3, Priority, and Sentiment** simultaneously.
- **Joint Loss**: During training, the model calculates the loss for all active tasks and averages them. This "Joint Learning" approach helps the model understand hierarchical relationships (e.g., how an L2 sub-category relates to its L1 parent).

## Usage in Notebooks

To use the model in any notebook, simply import it:

```python
from arabic_itsm.models.classifier import MarBERTClassifier

# Initialize for L1 and L2
model = MarBERTClassifier(
    model_name="UBC-NLP/MARBERTv2", 
    num_classes={"l1": 6, "l2": 16}
)
```
