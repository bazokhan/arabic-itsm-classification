# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Run 002 (Post-Fix GPU Execution)

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models  
**Author**: Mohamed A. Elbaz  
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University  
**Date**: February 24, 2026

---

## Executive Summary

Run 002 is the corrected full-pipeline execution after fixing the environment/training failure documented in Run 001. The key difference is that MarBERTv2 now trained on **CUDA with FP16 enabled**, converged normally, and produced valid test-set metrics.

The best classical baseline (LinearSVC) achieved **0.8840 macro-F1** on the deduplicated split. MarBERTv2 reached **0.8910 macro-F1** and **0.8904 accuracy**, yielding a measurable but modest lift over the strongest TF-IDF baseline.

This run transitions the project from infrastructure-debug phase to research-quality evaluation phase. The remaining major work is now: L2/L3 expansion, significance testing, calibration, and multi-seed robustness.

---

## Scope of This Report

This report analyzes updated outputs from:
- Notebook 01: data inspection
- Notebook 02: data preparation
- Notebook 03: baseline models
- Notebook 04: MarBERT fine-tuning
- Notebook 05: final evaluation

Artifacts referenced include:
- `results/metrics/baseline_results.csv`
- `results/metrics/03_baseline_per_class_lin.csv`
- `results/metrics/05_marbert_final_metrics.json`
- `results/metrics/05_marbert_per_class.csv`
- `results/metrics/05_final_comparison.csv`
- `results/figures/04_training_curves.png`
- `results/figures/05_marbert_confusion_matrix.png`
- `results/figures/05_marbert_confusion_normalized.png`

---

## Notebook 01 — Data Inspection

### 1.1 Dataset Integrity

- Shape: **10,000 × 18**
- Missing values: **none**
- Duplicate `ticket_id`: **0**
- Exact duplicate text pairs `(title_ar, description_ar)`: **451**

**Interpretation**: the same structural findings from Run 001 are confirmed. There is no indication of data corruption, but duplicate text pairs remain in source data and must be removed before splitting.

### 1.2 Label Space Reality Check

- L1 classes: 6
- L2 summary count range: min 565 / max 763 / mean 625
- L3 summary count range: min 160 / max 275 / mean 208

**Interpretation**: empirical class counts remain consistent with the corrected taxonomy understanding (L2=16, L3=48). Any old references to L2=14 or L3=31 are legacy documentation artifacts, not current data behavior.

### 1.3 Balance and Risk

- L1 imbalance ratio: **1.65x**

**Interpretation**: mild imbalance. Macro-F1 remains appropriate as primary score; no urgent need for balancing heuristics at L1.

---

## Notebook 02 — Data Preparation

### 2.1 Deduplication and Split

- After deduplication: **9,549 tickets** (451 removed)
- Train: **6,684**
- Validation: **1,432**
- Test: **1,433**

**Interpretation**: leakage risk from exact duplicates is controlled in preprocessing, and split sizes are stable/reproducible.

### 2.2 Label Encoders

- L1: **6 classes**
- L2: **16 classes**
- L3: **48 classes**
- Priority: **5 classes**
- Sentiment: **4 classes**

### 2.3 Tokenization Budget

- Mean token length: **43.6**
- Median token length: **39**
- Max token length: **107**
- Coverage at max_length=128: **100%**

**Interpretation**: no truncation for current data. The 128-token setting is conservative and safe.

---

## Notebook 03 — Classical Baselines

### 3.1 Updated Baseline Table (Dedup Split)

| Model | Val Macro-F1 | Test Macro-F1 | Test Acc | Infer (ms/sample) |
|---|---:|---:|---:|---:|
| LR (word+char TF-IDF) | 0.8852 | 0.8748 | 0.8779 | 0.24 |
| **LinearSVC (word+char TF-IDF)** | 0.8817 | **0.8840** | **0.8870** | 0.23 |
| Naive Bayes (word TF-IDF) | 0.8628 | 0.8526 | 0.8555 | 0.04 |

Best baseline remains LinearSVC.

### 3.2 Baseline Error Structure (LinearSVC)

Per-class F1:
- Access 0.8596
- Hardware 0.8665
- Network 0.9589
- Security 0.8440
- Service 0.8619
- Software 0.9129

**Interpretation**:
- Network/Software remain easiest due to strong lexical signatures.
- Security remains relatively harder, likely due to overlap with Access/Service language.

---

## Notebook 04 — MarBERTv2 Fine-Tuning (Corrected Run)

### 4.1 Environment Status

- `Device: cuda`
- `FP16: True`
- `pin_memory: True`

This is the critical fix versus Run 001 (CPU-only failure).

### 4.2 Training Dynamics

| Epoch | Train Loss | Val Loss | Val Macro-F1 | Val Acc |
|---|---:|---:|---:|---:|
| 1 | 1.0015 | 0.4026 | 0.8721 | 0.8743 |
| 2 | 0.3377 | 0.3043 | **0.8938** | **0.8925** |
| 3 | 0.2586 | 0.2787 | 0.8894 | 0.8883 |
| 4 | 0.2224 | 0.2959 | 0.8866 | 0.8883 |

**Interpretation**:
- Healthy rapid convergence by epoch 2.
- Slight val oscillation after best epoch suggests normal early-overfit behavior; checkpointing on best validation score is correct.
- No evidence of class collapse or non-learning.

---

## Notebook 05 — Final Evaluation

### 5.1 MarBERTv2 Final Metrics (Test)

- Accuracy: **0.8904**
- Macro-F1: **0.8910**
- Macro-Precision: **0.9031**
- Macro-Recall: **0.8835**
- Inference latency: **9.20 ms/sample**
- Error rate: **157 / 1433 = 10.96%**

### 5.2 Per-Class Metrics (MarBERT)

- Access: P 0.8165 / R 0.9021 / F1 0.8571
- Hardware: P 0.8209 / R 0.9170 / F1 0.8663
- Network: P 0.9800 / R 0.9280 / F1 0.9533
- Security: P 0.9648 / R 0.8059 / F1 0.8782
- Service: P 0.9286 / R 0.8571 / F1 0.8914
- Software: P 0.9080 / R 0.8910 / F1 0.8994

### 5.3 Comparison to Best Baseline

| Model | Test Acc | Test Macro-F1 | Infer (ms/sample) |
|---|---:|---:|---:|
| **MarBERTv2 (fine-tuned)** | **0.8904** | **0.8910** | 9.20 |
| LinearSVC (word+char TF-IDF) | 0.8870 | 0.8840 | 0.23 |

Delta (MarBERT - LinearSVC):
- Accuracy: **+0.0035** (+0.35 pp)
- Macro-F1: **+0.0070** (+0.70 pp)

**Interpretation**:
- MarBERT improves quality, but not by a very large margin at L1.
- Latency cost is substantial (roughly 40x vs best baseline).
- For academic reporting, this is a valid improvement; for production, throughput/latency tradeoff should be justified.

### 5.4 Error Breakdown

Errors by true class:
- Security: 33
- Software: 29
- Access: 28
- Service: 26
- Hardware: 22
- Network: 19

**Interpretation**: Security remains the hardest class by recall, despite very high precision, indicating under-prediction risk for true security cases.

---

## Run 001 vs Run 002

| Metric | Run 001 | Run 002 |
|---|---:|---:|
| Device | CPU | CUDA+FP16 |
| Best Val Macro-F1 | 0.1316 | 0.8938 |
| Test Macro-F1 (MarBERT) | 0.1236 | 0.8910 |
| Test Accuracy (MarBERT) | 0.1633 | 0.8904 |
| Error pattern | severe mode collapse | normal distribution |

**Conclusion**: Run 002 fully validates that Run 001 failure was environmental, not conceptual.

---

## Decisions Added in Run 002

### D-002-01 — Model Version Preservation for Academic Traceability

**Decision**: Keep all significant trained checkpoints (including failed/experimental runs) instead of overwriting.

**Rationale**:
- Reproducibility and audit trail for thesis defense.
- Supports error-analysis narrative (Run 001 vs Run 002).
- Enables rollback and ablation comparisons.

**Operational policy**:
- Save versioned checkpoints (e.g., `marbert_l1_exp002_failed`, `marbert_l1_exp003_best`).
- Deploy only one tagged production model, but retain archived versions with metrics and config metadata.

### D-002-02 — Expand Pipeline to L2/L3 Before Final Thesis Freeze

**Decision**: Extend same pipeline style to L2 and L3, not as separate ad-hoc experiments.

**Rationale**: L1-only evidence is insufficient for full taxonomy claims.

### D-002-03 — Statistical Significance as Mandatory Final Reporting Layer

**Decision**: Add significance tests (bootstrap CI + McNemar) before final conclusions about superiority.

**Rationale**: The observed +0.70 pp macro-F1 gain should be statistically verified.

---

## Limitations Still Open

1. Single-seed reporting: no variance yet.
2. L2/L3 not yet trained/evaluated in final published format.
3. No formal calibration metrics (ECE/Brier) yet.
4. Latency/cost tradeoff not yet tied to deployment load model.

---

## Next Steps (Integrated with Existing Pipeline)

1. Execute Notebook 06 (L2/L3 extension) with same training/evaluation structure.
2. Execute Notebook 07 (significance testing) on prediction-level outputs.
3. Add model registry metadata (config hash, seed, split signature, metrics file path).
4. Promote one production candidate model while preserving all academic artifacts.

---

## Final Position

Run 002 is valid for thesis-quality L1 reporting and replaces Run 001 as the authoritative transformer result for L1.

Run 001 remains important as a documented failure case and methodological lesson, not as a performance baseline.
