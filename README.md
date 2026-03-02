# Arabic ITSM Ticket Classification
### Fine-Tuning MarBERTv2 for Egyptian Arabic IT Support Ticket Routing

> **Faculty of Computers and Artificial Intelligence — Cairo University**
> Professional Master's in Cloud Computing Networks — February 2026
> **Author**: Mohamed A. Elbaz | **Supervisor**: Dr. Eman E. Sanad

---

## Overview

This repository contains the model training pipeline for a cloud-based Arabic ITSM ticket classification system. The goal is to automatically route Arabic-language IT support tickets (written in Egyptian colloquial Arabic) to the correct service category, priority level, and team — replacing manual triage in enterprise helpdesk workflows.

**Key properties:**
- **Language**: Egyptian Arabic (عامية مصرية) with English code-mixing (e.g., VPN، Outlook، MFA)
- **Model**: [MarBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) — the strongest publicly available encoder for Egyptian dialect NLP
- **Tasks**: Multi-class classification for L1 category (6 classes), L2 sub-category (16 classes), and priority (1–5)
- **Dataset**: [arabic-itsm-dataset](https://github.com/bazokhan/arabic-itsm-dataset) — 10,000 synthetic Egyptian Arabic ITSM tickets with a 3-level taxonomy

---

## Dataset

The training data lives in a companion repository:

| Property | Value |
|----------|-------|
| **Repo** | [`bazokhan/arabic-itsm-dataset`](https://github.com/bazokhan/arabic-itsm-dataset) |
| **HuggingFace** | [`albaz2000/arabic-itsm-dataset`](https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset) |
| **Size** | 10,000 tickets (CSV + JSONL) |
| **Taxonomy** | 6 L1 → 16 L2 → 48 L3 categories |
| **Labels** | category_level_1/2/3, priority (1–5), sentiment |
| **Dialect** | Egyptian Arabic with English technical terms |

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/bazokhan/arabic-itsm-dataset/master/dataset_clean.csv")
```

---

## Model Architecture

```
Input: Arabic ticket (title_ar + description_ar)
    ↓  ArabicTextNormalizer (diacritics removal, alef normalization)
    ↓  MarBERTv2 Tokenizer (WordPiece, max_length=128)
    ↓
MarBERTv2 Encoder  (163M parameters, UBC-NLP/MARBERTv2)
    ↓  [CLS] representation (768-dim)
    ├── Dropout(0.1) → Linear(768→6)   → L1 category     (6 classes)
    ├── Dropout(0.1) → Linear(768→16)  → L2 sub-category  (16 classes)
    ├── Dropout(0.1) → Linear(768→48)  → L3 sub-sub-cat   (48 classes)
    ├── Dropout(0.1) → Linear(768→5)   → Priority         (5 classes: 1–5)
    └── Dropout(0.1) → Linear(768→4)   → Sentiment        (4 classes)
```

**Production mode**: a single forward pass populates all 5 heads from one `marbert_multi_task_best/` checkpoint.
**Demo mode**: milestone checkpoints (`marbert_l1_best/`, `marbert_l2_best/`, etc.) can be shown individually.

**Why MarBERTv2?** See [`docs/model_recommendation.md`](docs/model_recommendation.md) and [`docs/decisions/ADR-001-model-selection.md`](docs/decisions/ADR-001-model-selection.md) for the full rationale. In brief: it outperforms CAMeLBERT and AraBERTv2 on Egyptian colloquial classification tasks (64–85% F1), has a low compute footprint (same size as BERT-base), and is available directly from HuggingFace.

---

## Repository Structure

```
arabic-itsm-classification/
├── CLAUDE.md                        # Project memory & experiment log (AI-assisted)
├── README.md
├── requirements.txt
│
├── configs/
│   ├── model_config.yaml            # Model architecture & training hyperparameters
│   └── data_config.yaml             # Dataset paths, split ratios, preprocessing flags
│
├── notebooks/                       # Ordered demo notebooks — run in sequence
│   ├── 01_data_inspection.ipynb     # EDA: class distribution, text stats, visualizations
│   ├── 02_data_preparation.ipynb    # Normalization, splits, label encoding, tokenization
│   ├── 03_baseline_models.ipynb     # TF-IDF + LR / SVM / NB baselines
│   ├── 04_marbert_finetuning.ipynb  # MarBERTv2 fine-tuning with MLflow tracking
│   └── 05_evaluation_results.ipynb  # Final evaluation, comparison, error analysis
│
├── src/arabic_itsm/                 # Python package
│   ├── data/
│   │   ├── preprocessing.py         # ArabicTextNormalizer
│   │   └── dataset.py               # ITSMDataset (PyTorch) + load_splits()
│   ├── models/
│   │   └── classifier.py            # MarBERTClassifier (multi-head)
│   └── utils/
│       └── metrics.py               # compute_classification_metrics(), classification_report_df()
│
├── scripts/
│   └── train.py                     # CLI training script (wraps Notebook 04 logic)
│
├── docs/
│   ├── abstract.pdf                 # University project proposal
│   ├── model_recommendation.md      # Model selection rationale
│   └── decisions/
│       └── ADR-001-model-selection.md
│
├── models/                          # Saved checkpoints (gitignored — large files)
│   └── marbert_l1_best/             # Best L1 checkpoint (after training)
│
└── results/
    ├── figures/                     # Generated plots from notebooks
    └── metrics/                     # JSON/CSV evaluation results
```

---

## Notebooks

Run notebooks in order for a full walkthrough from raw data to trained model:

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| [`01_data_inspection`](notebooks/01_data_inspection.ipynb) | Exploratory analysis of the dataset | Class distribution, text length, cross-tab plots |
| [`02_data_preparation`](notebooks/02_data_preparation.ipynb) | Preprocessing, splits, tokenization analysis | `data/processed/` train/val/test splits |
| [`03_baseline_models`](notebooks/03_baseline_models.ipynb) | Classical ML baselines (TF-IDF) | `results/metrics/baseline_results.csv` |
| [`04_marbert_finetuning`](notebooks/04_marbert_finetuning.ipynb) | Full MarBERTv2 fine-tuning | `models/marbert_l1_best/` checkpoint |
| [`05_evaluation_results`](notebooks/05_evaluation_results.ipynb) | Final evaluation + model comparison | Confusion matrices, comparison table, error analysis |

---

## Quickstart

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

The dataset repo must be a sibling directory:
```
D:/AI/
├── arabic-itsm-dataset/     ← dataset repo
└── arabic-itsm-classification/  ← this repo
```

Or load from HuggingFace — update `DATA_CSV` at the top of each notebook.

### Run Notebooks

```bash
jupyter notebook notebooks/
```

Open and run notebooks 01 → 02 → 03 → 04 → 05 in order.

### Train via CLI

After running Notebook 02 to generate processed splits:

```bash
# Single-task: L1 only (local GPU, fast)
python scripts/train.py --task l1 --epochs 5 --lr 2e-5

# Joint subset: L1 + L2 + L3 (recommended: Kaggle T4)
python scripts/train.py --tasks l1 l2 l3 --epochs 10 --lr 1e-5 --batch-size 16

# Full multi-task: all 5 tasks (recommended: Kaggle T4)
python scripts/train.py --multi-task --epochs 10 --lr 1e-5 --batch-size 16

# CPU / limited VRAM
python scripts/train.py --task l1 --batch-size 8 --no-fp16
```

---

## Experiment Tracking

Training runs are tracked with [MLflow](https://mlflow.org/):

```bash
# Start the MLflow UI
mlflow ui

# Open http://localhost:5000
```

All runs are logged to `mlruns/` (gitignored). Key metrics per epoch:
- `train_loss`, `val_loss`
- `val_macro_f1`, `val_accuracy`, `val_macro_precision`, `val_macro_recall`

---

## Results

### L1 Classification (6 classes)

| Model | Test Accuracy | Test Macro-F1 | Infer (ms/sample) | Checkpoint |
|-------|:------------:|:-------------:|:-----------------:|------------|
| Naive Bayes (TF-IDF) | 85.55% | 85.26% | 0.04 | — |
| Logistic Regression (TF-IDF) | 87.79% | 87.48% | 0.24 | — |
| LinearSVC (TF-IDF) | 88.70% | 88.40% | 0.23 | — |
| **MarBERTv2 (fine-tuned)** | **89.04%** | **89.10%** | **9.20** | `marbert_l1_best/` |

### L1 + L2 Joint (6 + 16 classes)

| Task | Val Macro-F1 | Checkpoint |
|------|:------------:|------------|
| L1 | 89.31% | `marbert_l2_best/` |
| L2 | 86.57% | `marbert_l2_best/` |

### L3 (48 classes — Kaggle T4)

| Training Mode | Val Macro-F1 | Checkpoint |
|---|:---:|---|
| L3-only single-task | 79.24% | `marbert_l3only_kaggle/` |
| L1+L2+L3 joint | TBD | `marbert_l1_l2_l3_best/` *(pending)* |

### Full Multi-Task (all 5 tasks)

| Checkpoint | Avg Macro-F1 | Status |
|---|:---:|---|
| `marbert_multi_task_best/` | TBD | Pending Kaggle run |

---

## Training Environment Split

Due to local GPU VRAM constraints (RTX 3050 Laptop, 4 GB), training is split:

| Task Scope | Environment | Rationale |
|---|---|---|
| L1 only (`--task l1`) | Local GPU | Fast, fits in 4 GB VRAM |
| L1+L2 joint (`--tasks l1 l2`) | Local GPU | Still fits, ~3.5 GB peak |
| L1+L2+L3 joint (`--tasks l1 l2 l3`) | Kaggle T4 | More heads → more memory; T4 has 15 GB |
| Full multi-task (`--multi-task`) | Kaggle T4 | 5 heads require reliable GPU budget |

**Kaggle notebooks** (in `notebooks/`):
- `kaggle_train_l3_arabic_itsm_classification` — original L3-only run (EXP-004)
- `kaggle_train_l1l2l3_arabic_itsm_classification` — joint L1+L2+L3 (Notebook 08)
- `kaggle_train_multitask_arabic_itsm_classification` — full 5-task (Notebook 09)

---

## Model Registry

| Checkpoint | Tasks | Heads | Environment | Status |
|---|---|---|---|---|
| `marbert_l1_best/` | L1 | 1 | Local GPU (RTX 3050) | Done |
| `marbert_l2_best/` | L1+L2 | 2 | Local GPU (RTX 3050) | Done |
| `marbert_l3only_kaggle/` | L3 only | 1 | Kaggle GPU (T4) | Done — val F1=0.79 |
| `marbert_l1_l2_l3_best/` | L1+L2+L3 | 3 | Kaggle GPU (T4) | Pending |
| `marbert_multi_task_best/` | All 5 | 5 | Kaggle GPU (T4) | Pending |

---

## Related Work & Context

This project sits at the intersection of two research areas:

1. **Arabic NLP**: Transformer-based models for Arabic have rapidly advanced (AraBERT → CAMeLBERT → MarBERT → MarBERTv2), with Egyptian dialect classification remaining challenging due to orthographic variation and code-mixing.

2. **ITSM Automation**: Automated ticket classification reduces mean-time-to-route (MTTR) and enables SLA-aware prioritization. Most prior work targets English tickets; Arabic ITSM resources are scarce.

This project contributes:
- A **first public synthetic Arabic ITSM dataset** with a formal 3-level taxonomy (see companion repo)
- A **fine-tuned MarBERTv2 checkpoint** for Egyptian Arabic ITSM classification
- A **reproducible evaluation pipeline** comparing transformer vs. classical approaches

---

## Citation

If you use this code or the companion dataset in your research:

```bibtex
@misc{elbaz2026arabic,
  title   = {Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models},
  author  = {Elbaz, Mohamed A.},
  year    = {2026},
  note    = {Professional Master's Project, Faculty of Computers and Artificial Intelligence, Cairo University. Supervised by Dr. Eman E. Sanad.},
}
```

---

## License

MIT

---

*Faculty of Computers and Artificial Intelligence — Information Technology Department — Cairo University — Egypt*
