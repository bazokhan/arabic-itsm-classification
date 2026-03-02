# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Notebook 09 (Full Multi-Task, Kaggle GPU)

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models
**Author**: Mohamed A. Elbaz
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University
**Date**: TBD (pending Kaggle run)
**Environment**: Kaggle GPU (Tesla T4)

---

> **STATUS: PENDING** — This document is a stub to be filled after running
> `notebooks/kaggle_train_multitask_arabic_itsm_classification` on Kaggle.

---

## Run Configuration (Planned)

| Parameter | Value |
|-----------|-------|
| Script | `scripts/train.py --multi-task` |
| Tasks | L1 + L2 + L3 + Priority + Sentiment (5 heads) |
| Epochs | 10 |
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Device | Kaggle Tesla T4 GPU, FP16 enabled |
| Base Model | `UBC-NLP/MARBERTv2` |
| Output | `models/marbert_multi_task_best/` |

---

## Results Placeholder

| Task | Val Macro-F1 | Notes |
|------|:------------:|-------|
| L1   | TBD | |
| L2   | TBD | |
| L3   | TBD | |
| Priority | TBD | |
| Sentiment | TBD | |
| Average | TBD | Production metric |

---

## Analysis

*(Fill in after training run completes)*

Key questions to answer:
1. Does adding Priority + Sentiment heads hurt or help L1/L2/L3 performance?
2. What is the average macro-F1 across all 5 tasks?
3. Is this the right production checkpoint vs the simpler `marbert_l1_l2_l3_best`?
4. Inference latency comparison: 5-head model vs separate L12 + L3 checkpoints

---

## Checkpoint

| Artifact | Location |
|----------|----------|
| Classification repo | `models/marbert_multi_task_best/` |
| Server | `arabic-itsm-server/models/marbert_multi_task_best/` |
| heads.pt keys | 10 keys: l1/l2/l3/priority/sentiment × weight+bias |

**This is the production model** — single forward pass for all 5 classification tasks.
