# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Kaggle L3-Only Run

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models
**Author**: Mohamed A. Elbaz
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University
**Date**: February 24, 2026
**Environment**: Kaggle GPU (Tesla T4 × 2, 15 GB VRAM each)

---

## Executive Summary

This run trained a **single-task L3-only** MarBERTv2 model on Kaggle, reaching a best
validation macro-F1 of **0.7924** at epoch 6. It was run before joint-training notebooks
were created, as a stepping-stone to validate Kaggle environment setup and L3 feasibility.

The checkpoint is retained as `marbert_l3only_kaggle/` to distinguish it from the
subsequent joint `marbert_l1_l2_l3_best/` checkpoint. It should **not** be used for
production — it has no L1/L2 heads and L3 performance is expected to improve with joint
training (shared encoder benefits).

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Script | `scripts/train.py --task l3` |
| Tasks | L3 only (single-task) |
| Epochs | 10 |
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Device | Kaggle Tesla T4 GPU, FP16 enabled |
| Base Model | `UBC-NLP/MARBERTv2` |
| Data | `data/processed/` (9,549 deduplicated tickets) |

---

## Training Results

| Epoch | Val Macro-F1 (L3) | Checkpoint Saved |
|-------|:-----------------:|:----------------:|
| 1 | 0.6346 | Yes |
| 2 | 0.7392 | Yes |
| 3 | 0.7692 | Yes |
| 4 | 0.7879 | Yes |
| 5 | 0.7913 | Yes |
| 6 | **0.7924** | Yes (best) |
| 7 | 0.7892 | No |
| 8 | 0.7874 | No |
| 9 | 0.7889 | No |
| 10 | 0.7905 | No |

Best checkpoint saved at **epoch 6**: `models/marbert_l3only_kaggle/`

---

## Analysis

### L3 Performance Context

L3 classification is substantially harder than L1/L2:
- 48 classes vs 6 (L1) or 16 (L2)
- Mean samples per class ≈ 199 (7,000 train ÷ 48 classes × ~1.37 imbalance)
- Some L3 classes have very few training examples

A macro-F1 of 0.79 for 48-class fine-grained classification is a reasonable result.
It represents strong signal that MarBERT can learn the L3 taxonomy from this dataset.

### Limitation: Single-Task Training

This run trained L3 in isolation (`--task l3`). This means:
1. The model encoder only optimized for L3 — no shared signal from L1/L2
2. The `heads.pt` file contains only `l3.1.weight` and `l3.1.bias`
3. The checkpoint cannot be used for L1 or L2 inference

Joint training (`--tasks l1 l2 l3`) is expected to improve L3 F1 by 1–3 pp through
shared representation learning, as observed in the L1→L1+L2 transition (Run 003).

### Convergence Pattern

The model peaked at epoch 6 and showed mild degradation in epochs 7–10, then began
recovering. This suggests slight overfitting at epoch 7–8. The plateau F1 ≈ 0.79
indicates the model is near saturation for single-task L3 with 10K tickets at this LR.

---

## Checkpoint Details

| Artifact | Location |
|----------|----------|
| Classification repo | `models/marbert_l3only_kaggle/` |
| Server | `arabic-itsm-server/models/marbert_l3only_kaggle/` |
| heads.pt keys | `l3.1.weight`, `l3.1.bias` |
| Encoder | MarBERTv2 fine-tuned for L3 only |

**Note**: This checkpoint is intentionally preserved for the academic demo to show the
L3 milestone. For production use, wait for `marbert_l1_l2_l3_best/` (Notebook 08) which
trains L1+L2+L3 jointly.

---

## Next Steps

- Run `kaggle_train_l1l2l3_arabic_itsm_classification` (Notebook 08) to get joint L1+L2+L3 model
- Compare L3 F1: single-task (0.7924) vs joint (expected improvement)
- Run `kaggle_train_multitask_arabic_itsm_classification` (Notebook 09) for full 5-task model
