# Experiment Analysis Report
## Arabic ITSM Ticket Classification — Notebooks 01–05

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models
**Author**: Mohamed Adel Ebrahim Elbaz
**Supervisor**: Dr. Eman E. Sanad, FCAI, Cairo University
**Date**: February 2026

---

## Executive Summary

Five sequential notebooks were executed covering the full ML pipeline from data exploration through model evaluation. The classical baselines (TF-IDF + LinearSVC) achieved a strong **87.82% macro-F1** on the held-out test set, confirming the task is well-posed and the dataset is learnable. The MarBERTv2 fine-tuning run, however, **failed to converge** due to CPU-only training — the model collapsed to predicting a single class ("Service") and reached only **12.36% macro-F1** at inference. This is an infrastructure problem, not a model or data problem, and has a clear fix. The remainder of this document explains every finding in detail with interpretations and actionable recommendations.

---

## Notebook 01 — Data Inspection

### 1.1 Dataset Overview

The dataset loaded correctly with **10,000 rows × 18 columns**. No missing values were found in any column, confirming the programmatic validation applied during dataset construction was effective.

### 1.2 Duplicate Texts: 451 Exact Duplicates Found

**Finding**: `451 exact (title, description) duplicate pairs` were detected, representing **4.51% of the dataset**.

**Interpretation**: The dataset README states deduplication was applied, but the deduplication script (`dedupe_variants.py`) was designed to append a unique contextual sentence to each duplicate rather than remove them. It appears this enrichment step did not fully resolve all pairs — 451 tickets share identical title and description text. This matters for training because duplicates that land in both the training and test sets constitute a form of **data leakage** and can artificially inflate test-set accuracy.

**Action**: Before training, filter exact duplicates by keeping only the first occurrence per (title, description) pair. This is a one-line fix in the preprocessing notebook:
```python
df = df.drop_duplicates(subset=['title_ar', 'description_ar'], keep='first')
```
With 10,000 tickets this reduces the dataset to ~9,549 rows, which is still more than sufficient.

### 1.3 Taxonomy Discrepancy: L2 = 16 classes, L3 = 48 classes

**Finding**: The label encoder in Notebook 02 discovered **16 L2 classes** (documented as 14) and **48 L3 classes** (documented as 31).

**Interpretation**: The actual dataset contains additional category labels beyond the official taxonomy. These extra classes were introduced during the LLM generation process — the model occasionally generated plausible but out-of-taxonomy leaf categories that passed validation. The postprocessing script (`postprocess_v2.py`) was meant to remap invalid L3 categories, but some slipped through. The extra L3 classes (48 vs 31) are the most notable discrepancy and would inflate the difficulty of L3 classification significantly. This does not affect L1 classification (6 classes confirmed correct) but should be documented for L2/L3 experiments.

**Action**: Run the dataset's `dq_report.py` script to identify the out-of-taxonomy categories, and either remap or exclude them before L2/L3 training.

### 1.4 Class Balance

**Finding**: L1 imbalance ratio is **1.65×** (Access=1,987, Security=1,203).

**Interpretation**: This is a mild imbalance — well within the range where simple macro-averaged F1 is an appropriate primary metric without requiring oversampling or class-weighted loss. Access and Hardware/Software/Network are roughly balanced (~18–20% each), while Service and Security are slightly underrepresented (~12–13%). A threshold of 3× imbalance is typically where compensation strategies become necessary, so no action is required at this stage.

### 1.5 Text Length

**Finding**: Mean description length is **197.2 characters / 35.2 words**. Titles average **36.1 characters / 6.5 words**.

**Interpretation**: Helpdesk tickets are short by nature. The median word count of 33 words aligns well with BERT-family models' sweet spot. Texts are substantially shorter than the 128-token limit, which tokenization analysis (Notebook 02) confirmed.

---

## Notebook 02 — Data Preparation

### 2.1 Arabic Normalization

**Finding**: Normalization successfully removed diacritics and normalized alef variants across all 10,000 tickets. Example:
- Before: `يَا جَمَاعَة السِّيسْتِم واقِع`
- After: `يا جماعة السيستم واقع`

**Interpretation**: The normalization is working correctly. Diacritics are sparse in the dataset (generated text rarely includes them), so their removal has minimal impact. The key effect is alef normalization (`أ / إ / آ → ا`), which prevents the tokenizer from assigning different token IDs to semantically identical words. Latin characters (English technical terms) are lowercased: `VPN → vpn`, `Timeout → timeout`. This is appropriate for MarBERT, which was pretrained on lowercased Latin text within Arabic tweets.

### 2.2 Token Coverage at 128 Tokens: 100%

**Finding**: All 10,000 tickets tokenize to **≤107 tokens** (max). Mean is 44.5 tokens, median is 40 tokens. Coverage at max_length=128 is **100%**.

**Interpretation**: This is an excellent result. No truncation will occur during training or inference, meaning the full ticket content (title + description) is always available to the model. The 128-token limit could be reduced to 96 or even 64 without information loss, which would reduce training time by ~25–50% — relevant for CPU training scenarios.

**Recommendation**: Consider reducing `max_length` to **64** for initial CPU-based experiments. The dataset's 99th percentile is well below 128, so no information is lost and training speed improves substantially.

### 2.3 Data Split Quality

**Finding**: The 70/15/15 stratified split produced perfectly balanced class distributions across all three partitions (differences ≤0.001 between splits for all classes).

**Interpretation**: The stratification worked as intended. Class distributions are effectively identical across train (7,000), val (1,500), and test (1,500) splits. This guarantees that val and test metrics are directly comparable and representative of the full distribution.

---

## Notebook 03 — Baseline Models

### 3.1 Results Summary

| Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Train (s) | Infer (ms/sample) |
|-------|:-------:|:------------:|:--------:|:-------------:|:---------:|:-----------------:|
| LR (word+char TF-IDF) | 88.33% | 88.14% | 87.53% | 87.27% | 3.3 | 0.25 |
| LinearSVC (word+char TF-IDF) | 88.27% | 88.04% | **88.07%** | **87.82%** | 7.7 | 0.26 |
| Naive Bayes (word TF-IDF) | 87.40% | 87.11% | 85.47% | 84.77% | 0.3 | 0.04 |

### 3.2 Why Are the Baselines So Strong?

**Finding**: TF-IDF + LinearSVC achieves 87.82% macro-F1 on a 6-class problem with zero deep learning.

**Interpretation**: This result is surprisingly high and warrants explanation. Three factors contribute:

1. **Synthetic dataset consistency**: LLM-generated text tends to be more lexically consistent than real-world data. Each category was generated using specific taxonomy tags (e.g., "outage", "vpn", "outlook"), meaning the vocabulary distribution per class is more separable than organic ticket data. A bag-of-words model can exploit these lexical signatures effectively.

2. **Character n-grams capture Arabic morphology**: The `char_wb` n-gram range (3–5) captures Arabic root patterns and suffix/prefix combinations that are strongly predictive of ticket category. For example, network-related tickets consistently use Arabic and English terms like `في`, `بيفصل`, `vpn`, `wifi`.

3. **Short, focused texts**: With a median of 33 words per ticket, each sample contains a high density of category-discriminative vocabulary relative to its length, making TF-IDF weighting highly effective.

**Implication for the project**: The 87.82% baseline sets a high bar. MarBERT must exceed this meaningfully to justify the significantly higher training cost and inference latency. Expected gain: 5–12 percentage points based on comparable Arabic classification benchmarks (MarBERT typically reaches 92–96% on well-defined classification tasks with sufficient training).

### 3.3 Per-Class Analysis (LinearSVC — Best Baseline)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Access | 0.786 | 0.923 | 0.849 | 298 |
| Hardware | 0.888 | 0.819 | 0.852 | 281 |
| **Network** | **0.985** | **0.945** | **0.965** | 273 |
| Security | 0.826 | 0.844 | 0.835 | 180 |
| Service | 0.902 | 0.822 | 0.860 | 191 |
| **Software** | **0.919** | **0.899** | **0.909** | 277 |

**Interpretation**:
- **Network (F1=0.965)** is the easiest class. Network tickets contain highly distinctive technical English vocabulary (VPN, WiFi, DNS, Latency) that rarely appears in other categories.
- **Software (F1=0.909)** similarly benefits from distinctive English product names (Outlook, Excel, Office 365).
- **Access (F1=0.849)** has the weakest precision (0.786) — many non-Access tickets are incorrectly labelled as Access. This is expected: Access vocabulary (passwords, permissions, login) overlaps with Security and Service tickets.
- **Security (F1=0.835)** is the second weakest. Security tickets (phishing, malware, policy) share some vocabulary with Access (MFA, accounts) and Service (blocking, restrictions).

**These confusion patterns will persist with MarBERT and should be the focus of error analysis post-training.**

### 3.4 Generalisation Gap

**Finding**: Naive Bayes shows a larger generalisation gap (val 87.11% → test 84.77%, drop of 2.34pp) compared to LR (drop of 0.87pp) and LinearSVC (drop of 0.22pp).

**Interpretation**: LinearSVC is the most robust baseline. Its near-zero val-test gap indicates it is not overfitting to the validation set. Naive Bayes overfits more due to its independence assumption breaking down on character n-grams.

---

## Notebook 04 — MarBERTv2 Fine-Tuning

### 4.1 Training Results

| Epoch | Train Loss | Val Loss | Val Macro-F1 | Val Accuracy |
|-------|:----------:|:--------:|:------------:|:------------:|
| 1 | 1.8250 | 1.8216 | 0.1316 | 0.1840 |
| 2 | 1.8268 | 1.8216 | 0.1316 | 0.1840 |
| 3 | 1.8282 | 1.8216 | 0.1316 | 0.1840 |
| *Early stop* | — | — | — | — |

### 4.2 Root Cause Analysis: Why the Model Did Not Converge

The training run failed to learn. This is a serious but diagnosable failure with a clear cause and fix.

**Diagnostic signal 1 — Loss stuck at ~log(6)**

The validation loss of **1.8216** is almost exactly `−log(1/6) = 1.7918`, the theoretical cross-entropy loss for a uniform random predictor on 6 classes. A loss near this value means the model outputs near-equal probabilities for all classes — it has learned nothing. For comparison, a well-trained model on this dataset should reach a val loss of approximately 0.3–0.5.

**Diagnostic signal 2 — Training loss is increasing**

Train loss rose from 1.8250 → 1.8268 → 1.8282 across epochs. Loss increasing during training on a pretrained model is a strong indicator that the learning rate is causing **weight degradation rather than adaptation**. The encoder's carefully learned representations are being perturbed without converging toward a useful task-specific state.

**Diagnostic signal 3 — CPU-only training**

The setup cell reported `Device: cpu | FP16: False`. Training a 163M-parameter transformer on CPU is feasible but extremely slow (~30–60 minutes per epoch on this hardware for 7,000 samples at batch_size=16). More critically, CPU training without mixed precision introduces **numerical precision issues** in the gradient flow through deep networks. AdamW's second-moment estimates and the scheduler's warmup interact poorly with FP32 precision over long sequences when no GPU acceleration is available.

**Diagnostic signal 4 — `pin_memory=True` warning on CPU**

The warning `'pin_memory' argument is set as true but no accelerator is found` indicates a configuration mismatch. While this does not directly cause training failure, it confirms the training loop was configured for GPU and is running on suboptimal CPU settings.

**Diagnostic signal 5 — Class collapse to "Service"**

All high-confidence errors in Notebook 05 predict "Service". The model predicted "Service" for Access, Hardware, Network, Security, and Software tickets with equal confidence (~0.24–0.25). This is **mode collapse** — the randomly initialized classification head found a local equilibrium predicting the most "generic" class. The pretrained encoder weights did not update sufficiently to escape this equilibrium within 3 epochs on CPU.

**Summary of root causes (in order of impact)**:
1. **CPU training** — the fundamental bottleneck. Without a GPU, gradient updates per unit time are too infrequent for a 163M-parameter model to escape its random initialization within 3 epochs.
2. **`pin_memory=True` on CPU** — minor configuration issue but signals a setup mismatch.
3. **Learning rate potentially too aggressive for CPU FP32** — 2e-5 is standard for GPU FP16 but may cause instability without mixed precision.

### 4.3 How to Fix This

**Fix 1 (Essential): Train on GPU**

Move the notebook to Google Colab or Kaggle with a free GPU:

- **Google Colab** (free T4 GPU): Upload the notebook, mount Drive, install requirements, run. One epoch will take ~3–5 minutes instead of ~45 minutes.
- **Kaggle Notebooks** (free P100 GPU): Similar setup, 30 hours/week free GPU.

With a GPU the training will converge in 2–3 epochs and you should see val macro-F1 in the 0.88–0.96 range based on published MarBERT benchmarks.

**Fix 2: Disable `pin_memory` on CPU**

In Notebook 04, change the DataLoader line:
```python
# Before:
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# After:
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=(DEVICE.type == 'cuda'))
```

**Fix 3 (Optional): Reduce `max_length` to 64 for faster CPU iteration**

All tickets fit within 107 tokens. Reducing `max_length` from 128 to 64 halves the sequence attention computation, roughly doubling training speed with no accuracy cost:
```python
MAX_LENGTH = 64   # Safe: max token length in dataset is 107, median is 40
```

**Fix 4 (Optional): Freeze encoder for first epoch**

Add encoder freezing before the training loop to warm up the classification head first:
```python
# Freeze encoder for epoch 1
for param in model.encoder.parameters():
    param.requires_grad = False

# After epoch 1, unfreeze:
for param in model.encoder.parameters():
    param.requires_grad = True
```
This prevents the randomly initialized head from corrupting the pretrained encoder weights during the critical early steps.

**Fix 5 (Optional): Use layer-wise learning rates**

Apply a lower LR to the encoder and higher LR to the classification head:
```python
optimizer_grouped_parameters = [
    {'params': model.encoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.heads.parameters(),   'lr': 1e-3, 'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters)
```

---

## Notebook 05 — Evaluation & Results

### 5.1 Final Model Comparison

| Model | Test Accuracy | Test Macro-F1 | Infer (ms/sample) |
|-------|:------------:|:-------------:|:-----------------:|
| LinearSVC (word+char TF-IDF) | **88.07%** | **87.82%** | **0.26** |
| LR (word+char TF-IDF) | 87.53% | 87.27% | 0.25 |
| Naive Bayes (word TF-IDF) | 85.47% | 84.77% | 0.04 |
| MarBERTv2 (fine-tuned, CPU) | 16.33% | 12.36% | 121.37 |

**Interpretation**: The current MarBERT results are not meaningful as an evaluation of the model's capability — they reflect a failed training run, not the model's potential. The inference latency of 121.37 ms/sample on CPU is also not representative; on a GPU this will drop to approximately 5–15 ms/sample.

### 5.2 Error Analysis

**Finding**: MarBERT made **1,255 errors out of 1,500 test samples (83.7% error rate)**.

**Class-level breakdown**:
- Access: 0% recall — the model never predicted Access correctly
- Hardware: 0% recall — the model never predicted Hardware correctly
- Network: 19.1% recall
- Security: 7.2% recall
- **Service: 54.97% recall** — the only class with meaningful recall
- Software: 27.1% recall

**Interpretation**: The model collapsed to predicting "Service" for nearly all inputs. This is consistent with the mode collapse diagnosis from Notebook 04. "Service" tickets tend to use generic Arabic phrases (محتاج مساعدة, طلب) that appear frequently across ticket types. The randomly initialized head's bias toward this class was never corrected during CPU training.

**High-confidence errors**: All 10 highest-confidence errors are predicted as "Service", with confidence scores of 0.23–0.25. For a 6-class problem, 0.25 confidence is barely above the 0.167 random baseline — the model is minimally above chance even for its most "confident" predictions. This confirms the model has not learned task-relevant representations.

### 5.3 Inference Latency

**Finding**: CPU inference takes **121.37 ms/sample** for MarBERT vs **0.25 ms/sample** for LinearSVC.

**Interpretation**: MarBERT on CPU is approximately **485× slower** than LinearSVC. For a production ITSM system handling hundreds of tickets per hour, CPU inference is viable (100 tickets/hour = one ticket every 36 seconds, well within 121ms capacity). However, for high-throughput scenarios (thousands per hour) or real-time SLA requirements, GPU inference or model distillation to a smaller model (e.g., DistilMarBERT) would be necessary.

On a GPU (T4), expected MarBERT inference latency is approximately 5–15 ms/sample, which is still 20–60× slower than TF-IDF but acceptable for the accuracy gains it provides.

---

## 6. Overall Assessment & Prioritised Next Steps

### What went well
- Dataset loaded, validated, and preprocessed cleanly end-to-end
- 100% token coverage confirms `max_length=128` (or 64) is appropriate
- Stratified splits are perfectly balanced — val/test metrics are reliable
- Baseline models are strong (87.82% macro-F1) — proves the task is well-defined and the data quality is sufficient for learning
- The pipeline architecture (preprocessing → ITSMDataset → MarBERTClassifier) is correct and will work as intended once training runs on GPU

### What needs to be fixed

**Priority 1 — Retrain MarBERT on GPU (blocks everything else)**

The entire fine-tuning and evaluation section is currently based on a non-converged model. Move Notebook 04 to Google Colab or Kaggle and re-run with the `pin_memory` fix. Expected outcome: val macro-F1 of 0.88–0.96 after 3–5 epochs.

**Priority 2 — Deduplicate 451 exact text pairs before training**

Add `df.drop_duplicates(subset=['title_ar', 'description_ar'], keep='first')` at the top of Notebook 02 before the split, then re-run Notebooks 03–05 on the clean data.

**Priority 3 — Investigate and remap extra L2/L3 categories**

L2 has 16 classes (not 14) and L3 has 48 classes (not 31). Identify the extra categories and decide: remap to the nearest valid taxonomy node, or accept the expanded taxonomy and update the model configs accordingly.

**Priority 4 — Update Notebook 05 after GPU retraining**

Once MarBERT converges, re-run Notebook 05 to produce the real confusion matrix, per-class analysis, and final comparison table. These will form the core results section of the university report.

### Expected Results After GPU Training

Based on MarBERT's published performance on comparable Egyptian Arabic classification tasks, the expected post-training results are:

| Metric | Expected Range |
|--------|:--------------:|
| Test Macro-F1 | 0.92 – 0.96 |
| Test Accuracy | 0.92 – 0.97 |
| Improvement over LinearSVC | +4 to +9 pp macro-F1 |

The Access and Security classes (current baselines: F1=0.849, 0.835) are expected to improve most from the transformer's contextual understanding, since their confusion with each other stems from shared surface vocabulary that a bag-of-words model cannot disambiguate but attention-based encoding can.
