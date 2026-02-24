# Data Processing & Datasets

This directory contains the logic for preparing Arabic ITSM ticket data for training and evaluation.

## Key Modules

### 1. `preprocessing.py`
Contains the `ArabicTextNormalizer` class. This is the "brain" of the text cleaning pipeline.
- **Normalization**: Standardizes Arabic characters (e.g., Alif, Ya, Ta-Marbuta).
- **Noise Removal**: Strips diacritics (Harakat) and non-Arabic punctuation.
- **Consistency**: Ensures that both the training data and future real-time API inputs are cleaned identically.

### 2. `dataset.py`
Contains the `ITSMDataset` (PyTorch Dataset).
- **Tokenization**: Handles the MarBERT WordPiece tokenization logic.
- **Multi-Task Labels**: Dynamically prepares labels for L1, L2, L3, Priority, and Sentiment.
- **Structure**: Encapsulates the logic of combining Ticket Title and Description into a single model input.

## Usage in Notebooks

```python
from arabic_itsm.data.preprocessing import ArabicTextNormalizer
from arabic_itsm.data.dataset import ITSMDataset

normalizer = ArabicTextNormalizer()
train_ds = ITSMDataset(train_df, tokenizer, normalizer, tasks=['l1', 'l2'])
```
