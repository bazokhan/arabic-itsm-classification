# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Notebook 08 (Joint L1+L2+L3, Kaggle GPU)

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models
**Author**: Mohamed Adel Ebrahim Elbaz
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University
**Date**: TBD (pending Kaggle run)
**Environment**: Kaggle GPU (Tesla T4)

---

> **STATUS: PENDING** — This document is a stub to be filled after running
> `notebooks/kaggle_train_l1l2l3_arabic_itsm_classification` on Kaggle.

---

## Run Configuration (Planned)

| Parameter | Value |
|-----------|-------|
| Script | `scripts/train.py --tasks l1 l2 l3` |
| Tasks | L1 + L2 + L3 (joint, 3 heads) |
| Epochs | 10 |
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Device | Kaggle Tesla T4 GPU, FP16 enabled |
| Base Model | `UBC-NLP/MARBERTv2` |
| Output | `models/marbert_l1_l2_l3_best/` |

---

## Results Placeholder

| Task | Val Macro-F1 | Notes |
|------|:------------:|-------|
| L1   | TBD | Expected: ~0.89+ (similar to local) |
| L2   | TBD | Expected: ~0.87+ (similar to local) |
| L3   | TBD | Expected: >0.7924 (vs L3-only baseline) |
| Average | TBD | |

---

## Analysis

*(Fill in after training run completes)*

Key questions to answer:
1. Does joint L1+L2+L3 training improve L3 F1 vs single-task (0.7924)?
2. Is L1 performance maintained (≥0.8910) when trained with 2 additional heads?
3. Is L2 performance maintained (≥0.8657) in the joint setup?
4. What is the convergence pattern across 10 epochs?

---

## Checkpoint

| Artifact | Location |
|----------|----------|
| Classification repo | `models/marbert_l1_l2_l3_best/` |
| Server | `arabic-itsm-server/models/marbert_l1_l2_l3_best/` |
| heads.pt keys | `l1.1.weight`, `l1.1.bias`, `l2.1.weight`, `l2.1.bias`, `l3.1.weight`, `l3.1.bias` |
