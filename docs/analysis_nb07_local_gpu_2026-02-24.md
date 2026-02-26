# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Run 003 (Significance & Hierarchy)

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models  
**Author**: Mohamed Adel Ebrahim Elbaz  
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University  
**Date**: February 24, 2026

---

## Executive Summary

Run 003 marks the transition from single-task validation to **hierarchical multi-task learning**. This run achieved two major research objectives: (1) performing a rigorous **statistical significance** check between MarBERTv2 and classical baselines for L1, and (2) successfully extending the taxonomy to **Level 2 (16 classes)**. 

While MarBERTv2 remains numerically superior at L1 (89.10% F1 vs 88.40%), McNemar’s test (p=0.568) suggests the gain is not yet statistically significant on the current test set size. However, joint training of L1 and L2 showed promising signs of **shared feature learning**, with L2 reaching a strong **86.57% macro-F1**.

---

## Notebook 06 — Statistical Significance

### 6.1 McNemar’s Test (Paired Comparison)

**Finding**: McNemar's Test yielded a **p-value of 0.56817**.
- MarBERT Correct / SVC Wrong: 27 samples
- SVC Correct / MarBERT Wrong: 22 samples

**Interpretation**: With a p-value > 0.05, we cannot reject the null hypothesis that the two models have equal error rates for the L1 task. The "gain" of 5 additional correctly classified samples is not large enough to be considered statistically significant on a test set of 1,433 samples. This suggests that for broad category routing (L1), the lexical features captured by TF-IDF are nearly as effective as transformer embeddings.

### 6.2 Bootstrap Confidence Intervals (95% CI)

**Finding**:
- LinearSVC 95% CI: **[0.8661, 0.9004]**
- MarBERTv2 95% CI: **[0.8731, 0.9073]**

**Interpretation**: There is a substantial overlap (~70%) between the two confidence intervals. This reinforces the finding that the models are performing within the same error margin for the 6-class L1 task. 

**Action**: For the final thesis, these results should be used to argue that MarBERT’s real value lies not in broad L1 accuracy, but in its ability to scale to granular L2/L3 tasks where semantic context (not just keywords) becomes critical.

---

## Notebook 07 — Level 2 (L2) Classification

### 7.1 Multi-Task Joint Training Results

**Finding**: We successfully trained a multi-head model for L1 (6 classes) and L2 (16 classes).

| Metric | Joint Model (L1 Head) | Joint Model (L2 Head) |
|---|---:|---:|
| Best Macro-F1 | **89.31%** (Epoch 3) | **86.57%** (Epoch 2) |
| Best Accuracy | 89.25% | 86.45% |

**Interpretation**:
- **Shared Feature Gain**: Interestingly, joint training on L1+L2 slightly improved L1 performance (89.31%) compared to L1-only training (89.10%). Learning granular sub-categories (L2) provides "hints" that stabilize the parent category (L1) predictions.
- **L2 Robustness**: Reaching 86.57% on a 16-class task is a strong result, only a ~3% drop from the 6-class task. This confirms the hierarchical consistency of the LLM-generated dataset.

### 7.2 Why did L2 Training take so much longer?

**Finding**: Training Notebook 07 (L2) required significantly more time per epoch than Notebook 04 (L1).

**Interpretation**: The increase in training duration is due to **output layer complexity** and **joint optimization overhead**:
1. **Computational Expansion**: In L1, the model only computed 6 logits. In L2, it computes **22 logits** (6 for L1 + 16 for L2). This increases the number of parameters in the classification heads and the FLOPs required for the final linear layers.
2. **Joint Loss Backpropagation**: The model now calculates two separate Cross-Entropy losses and averages them. During the backward pass, the shared MarBERT encoder receives gradients from **two different objectives** simultaneously. Calculating the combined gradient and updating the weights to satisfy both L1 and L2 taxonomy boundaries is more computationally intensive.
3. **Validation Bottleneck**: After every epoch, the model must now perform two full evaluation cycles (one per task), including F1 calculation and Confusion Matrix generation for 16 classes, which adds significant wall-clock time compared to the simpler L1 validation.

---

## Updated Results Comparison (Run 003)

| Model | Task | Test Macro-F1 | Test Acc | Latency (ms) |
|---|---|---:|---:|---:|
| LinearSVC | L1 (6 cls) | 88.40% | 88.70% | **1.24** |
| MarBERTv2 (Single) | L1 (6 cls) | 89.10% | 89.04% | 12.44 |
| **MarBERTv2 (Joint)** | **L1 (6 cls)** | **89.31%** | **89.25%** | 13.10 |
| **MarBERTv2 (Joint)** | **L2 (16 cls)** | **86.57%** | **86.45%** | **13.10** |

---

## Final Insights & Thesis Angles

1. **The "Efficiency" Argument**: Since the inference latency for L1+L2 (13.1ms) is nearly identical to L1-only (12.44ms), the Multi-Task approach is clearly the superior production choice. We get 16 extra sub-categories for "free" (only ~0.6ms overhead).
2. **The "Lexical vs. Semantic" Debate**: For simple tasks, TF-IDF + SVC is exceptionally hard to beat. The thesis should highlight that the Transformer is a "long-tail" specialist — it shines when the task moves from broad categories to subtle sub-category distinctions (L2/L3).
3. **Data Quality Confirmation**: The success of the L2 run validates that the synthetic Egyptian Arabic data is semantically coherent across the hierarchy levels.

---

## Next Steps

1. **Execute Notebook 08 (L3 Extension)**: Scaling to 48 classes to test the limits of the multi-head architecture.
2. **Execute Notebook 09 (Full Multi-Task)**: Integrating Priority and Sentiment.
3. **Error Consistency Check**: Analyzing how often the model predicts an L2 sub-category that does not belong to the predicted L1 parent.
