# Model Selection for Egyptian Arabic ITSM Ticket Classification
## A Comparative Analysis of Transformer-Based Encoders

**Prepared for**: Arabic ITSM Classification Project
**Faculty of Computers and Artificial Intelligence, Cairo University**
**Professional Master's in Cloud Computing Networks — February 2026**

---

## 1. Introduction

The task of automatically classifying Arabic IT support tickets into a structured multi-level taxonomy (category, subcategory, and priority) presents a distinct set of challenges that standard NLP approaches are often not equipped to handle. Egyptian Arabic (EA), the primary dialect of this project's target corpus, is morphologically complex, orthographically inconsistent, and frequently code-mixed with English technical vocabulary — properties that interact adversely with tokenization schemes and pretraining corpora designed for Modern Standard Arabic (MSA) or multilingual text.

Selecting an appropriate pretrained language model is therefore not a minor implementation detail; it is a fundamental methodological decision that directly determines the quality of the learned representations and, consequently, the classification performance. This document provides an evidence-based comparative analysis of the five strongest candidates identified in the recent Arabic NLP literature, evaluates each along criteria specifically relevant to Egyptian colloquial ITSM text, and concludes with a reasoned recommendation and a set of conditions under which alternatives should be preferred.

---

## 2. Background: The Egyptian Arabic NLP Challenge

Egyptian Arabic presents three interrelated challenges that must be factored into model selection:

**Orthographic variation**: EA lacks a standardized spelling system. A single word may appear in dozens of variant forms depending on the writer, platform, and degree of formality. For example, the word *كتير* (many/a lot) is also written *كتيير*, *كتييير*, *كثير* (MSA cognate), or even *kteer* in Arabizi. Standard WordPiece or BPE tokenizers trained on MSA corpora assign high entropy to such variants, producing fragmented and semantically degraded token sequences [1].

**Code-mixing**: Technical communication in Egyptian workplaces routinely intersperses English terms — *VPN*, *Outlook*, *MFA*, *Windows 11*, *Office 365* — within Arabic sentence structures. This Arabic-English code-mixing is a lexical, syntactic, and orthographic challenge simultaneously [2].

**Domain sparsity**: ITSM-specific terminology in Arabic is not well-represented in any publicly available Arabic NLP pretraining corpus. Models must generalize from social or journalistic text to helpdesk vocabulary (e.g., *ريموت أكسيس*, *فايروول*, *دومين*), relying on sub-word and contextual representations to bridge this gap.

---

## 3. Candidate Models

Five transformer encoder models were identified as the strongest candidates based on recent benchmarks in dialectal Arabic classification, sentiment analysis, and dialect identification:

### 3.1 MarBERT and MarBERTv2 (UBC-NLP)

MarBERT [3] and its successor MarBERTv2 are BERT-base-sized bidirectional encoders (12 layers, 768 hidden dimensions, ~163M parameters) pretrained by Abdul-Mageed et al. at the University of British Columbia. The defining characteristic of this model family is the deliberate inclusion of Arabic dialectal text in its pretraining corpus: 128GB of data sourced from Twitter, comprising both MSA and a range of Arabic dialects including Egyptian, Levantine, Gulf, and Moroccan varieties. MarBERTv2 extends the original with additional data and an updated tokenization vocabulary better suited to dialectal spelling variation.

### 3.2 AraBERT (v0.2 and v02-large) (aubmindlab)

AraBERT [4] is a BERT-based encoder pretrained by Antoun et al. at the American University of Beirut on a large MSA corpus (~77GB), including news articles, Wikipedia, and web text. AraBERT has been highly influential and performs strongly across a wide range of Arabic NLP benchmarks. Several variants exist: the base and large configurations of both v0.1 and v0.2, with v0.2 trained on significantly more data and using improved tokenization. All variants are primarily MSA-trained.

### 3.3 CAMeLBERT (CAMeL Lab, NYU Abu Dhabi)

CAMeLBERT [5] is a family of BERT-based models released by Inoue et al. at NYU Abu Dhabi's CAMeL Lab. Unlike MarBERT or AraBERT, CAMeLBERT offers multiple specialized variants pretrained on distinct Arabic genre mixes: MSA-only (`camelbert-msa`), a dialect-aware mix (`camelbert-mix`), and a multi-Arabic genre blend (`camelbert-ca`). The mixed variant was pretrained on MSA, dialectal, and classical Arabic text, offering broader genre coverage. CAMeLBERT achieves state-of-the-art F1 on several Arabic NLP tasks including Classical Arabic NER (97.78% F1) and Arabic poetry meter classification.

### 3.4 ByT5 (Google Research)

ByT5 [6], developed by Xue et al. at Google, represents a fundamentally different approach: rather than using subword tokenization, it operates directly on raw UTF-8 bytes. This token-free architecture entirely sidesteps the tokenization fragmentation problem that plagues dialects with inconsistent spelling. ByT5 is a seq2seq model (encoder-decoder T5 architecture), but its encoder is usable for classification tasks. It achieved state-of-the-art F1 of 74.0% on the QADI dialect identification benchmark [7], demonstrating exceptional robustness to orthographic variation.

### 3.5 XLM-RoBERTa (Meta AI)

XLM-R [8], developed by Conneau et al. at Meta AI, is a multilingual encoder trained on 100 languages using RoBERTa's training procedure on 2.5TB of filtered Common Crawl data. Arabic is a substantial component of this corpus. XLM-R provides reasonable Arabic performance and demonstrates cross-lingual transfer ability, but is not specialized for Arabic dialects.

---

## 4. Multi-Criterion Evaluation

### 4.1 Criterion 1: Dialectal Arabic Handling

This is the most critical criterion for this project. Egyptian colloquial Arabic accounts for the entire target corpus.

**MarBERT / MarBERTv2** is the only model in this shortlist explicitly pretrained on a large-scale dialectal Arabic corpus. Abdul-Mageed et al. [3] demonstrate that MarBERT outperforms multilingual BERT, AraBERT, and other Arabic encoders on a range of dialectal tasks. In the NADI 2023 shared task on Arabic dialect identification, MarBERT-family models consistently rank in the top tier when fine-tuned on dialect-labeled data, with dialect-level F1 reaching 82.87% in specific configurations [9].

**AraBERT** is pretrained exclusively on MSA and performs competitively on MSA classification tasks, but its dialect sensitivity is limited. On dialectal Arabic benchmarks, AraBERT-based models typically lag behind MarBERT by a meaningful margin, particularly on Egyptian and Levantine samples [4].

**CAMeLBERT-mix** incorporates some dialectal data but its primary strength is breadth of genre coverage rather than depth of dialect exposure. It occupies a middle position between the MSA-focused AraBERT and the dialect-first MarBERT.

**ByT5** handles dialect indirectly: by eliminating tokenization, it avoids the failure mode of mapping dialectal spelling variants to sparse or incorrect tokens. This is a complementary advantage rather than the same kind of dialectal pretraining provided by MarBERT.

**XLM-R** treats Arabic as one of 100 languages and has no dialect-specific training. On dialectal Arabic benchmarks, it is generally competitive with AraBERT but underperforms compared to dialect-specialized models [8].

**Winner: MarBERT / MarBERTv2**

---

### 4.2 Criterion 2: Robustness to Orthographic Noise

ITSM helpdesk text is inherently noisy: irregular capitalization, missing diacritics, abbreviated words, inconsistent hamza placement (*أ / إ / ا*), and non-standard spacing. Models must handle this gracefully.

**ByT5** operates on individual bytes, making it inherently invariant to tokenization-level noise. Every character is represented regardless of how unusual its spelling. This is ByT5's strongest advantage [6].

**MarBERT / MarBERTv2** was trained on Twitter data, which is one of the noisiest text genres available: it contains spelling errors, colloquialisms, code-mixing, and orthographic informality at scale. This pretraining exposure provides strong empirical robustness to noisy informal text, as confirmed by its consistent performance on social media-derived benchmarks [3].

**AraBERT** is trained on clean, formal text. It employs an explicit Arabic normalization step (diacritic removal, alef normalization) in its preprocessing pipeline, which helps with some variants but cannot recover from semantic ambiguities introduced by creative informal spelling.

**CAMeLBERT** inherits similar noise sensitivity to AraBERT in its base form, though the mixed variant improves somewhat through exposure to diverse genres.

**XLM-R** shows moderate noise robustness through its large-scale multilingual training, but Egyptian colloquial noise patterns are not a primary optimization target.

**Winner: ByT5 (with MarBERT a strong second)**

---

### 4.3 Criterion 3: Multi-Class Classification Performance

The task requires classification across 6 L1 categories, 14 L2 subcategories, and optionally 31 L3 leaf categories. Performance on multi-class Arabic classification tasks is therefore an important differentiator.

**CAMeLBERT** achieves the strongest published results on structurally complex multi-class Arabic classification tasks. In Classical Arabic NER, it achieves 97.78% F1; in Arabic poetry meter classification (a 16-class problem), it achieves state-of-the-art results [5]. Its broad pretraining genre mix may better generalize to domain-specific multi-class scenarios.

**AraBERT** performs very strongly on news category classification and other multi-class MSA tasks, with published F1 scores often in the 80–92% range on standard benchmarks [4]. It benefits from a large, clean, topically diverse pretraining corpus.

**MarBERT / MarBERTv2** is highly competitive on multi-class classification tasks, particularly when the text style is informal or dialectal. On tasks combining dialect-sensitivity with class discrimination (e.g., dialectal sentiment classification, hate speech detection), it consistently outperforms other encoders [3].

**ByT5** shows strong multi-class performance specifically on dialect identification (QADI: 74% F1) [7] but incurs substantially higher computational cost due to longer byte sequences compared to subword models.

**XLM-R** is competitive on multi-class classification but consistently underperforms dialect-specialized models on Arabic-specific tasks [8].

**Winner: CAMeLBERT (for MSA/structured tasks); MarBERT (for informal/dialectal tasks)**

---

### 4.4 Criterion 4: Fit for the ITSM Domain

Egyptian ITSM helpdesk tickets exhibit three simultaneous properties that jointly define the domain:

1. **Short, informal, colloquial text** — typically 10–60 words per ticket, Egyptian dialect vocabulary
2. **Technical English code-mixing** — product names, error codes, acronyms (*VPN*, *MFA*, *0x8004010F*)
3. **Noisy, unstructured composition** — missing punctuation, spelling variants, fragmented sentences

Matching these properties to model pretraining corpora:

- **MarBERT / MarBERTv2**: Pretrained on Twitter — short, informal, code-mixed, noisy. This is the closest available approximation to helpdesk text in the Arabic NLP pretraining landscape.
- **CAMeLBERT-mix**: Broader genre coverage but no strong Twitter/social media component.
- **AraBERT**: Formal MSA — the opposite of the target domain on informality and orthographic noise.
- **ByT5**: Strong robustness to noise and spelling variation; less domain-matched pretraining data, and higher inference cost.
- **XLM-R**: Domain-neutral; no specific alignment with informal Arabic or technical code-mixing.

**Winner: MarBERT / MarBERTv2**

---

## 5. Summary Comparison Table

| Model | Dialect Fit | Noise Robustness | Multi-Class F1 (Published) | Compute Cost | ITSM Domain Fit | Overall |
|:------|:-----------|:-----------------|:--------------------------|:-------------|:----------------|:--------|
| **MarBERTv2** | **Excellent** | **Strong** | 64–85% (dialectal tasks) | Low (163M) | **Best** | **★★★★★** |
| CAMeLBERT-mix | Good | Moderate | 80–92% (MSA tasks) | Low (163M) | Moderate | ★★★☆☆ |
| AraBERTv2 | Fair | Moderate | 70–85% (MSA tasks) | Low (135M) | Poor | ★★☆☆☆ |
| ByT5-base | Excellent | **Excellent** | ~74% (QADI) | **High (580M)** | Good | ★★★★☆ |
| XLM-R-base | Moderate | Moderate | Varies (multilingual) | Moderate (270M) | Poor | ★★☆☆☆ |

*Compute cost reflects model parameter count and typical inference throughput on ITSM-length inputs (avg. ~35 tokens after normalization).*

---

## 6. Recommendation: MarBERTv2

**Recommended model**: `UBC-NLP/MARBERTv2`
**HuggingFace**: [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2)

### Justification

MarBERTv2 is recommended as the primary encoder for this project based on the convergence of four independent lines of evidence:

1. **Pretraining alignment**: It is the only candidate specifically pretrained on dialectal Arabic text at scale (Twitter corpus), providing the closest available distributional match to Egyptian helpdesk text without requiring domain adaptation from MSA.

2. **Benchmark superiority on dialectal tasks**: MarBERT and MarBERTv2 consistently outperform all other encoders — including AraBERT and CAMeLBERT — on dialect-sensitive classification tasks such as dialect identification (NADI 2023), dialectal sentiment analysis, and hate speech detection [3, 9]. These tasks share the informal, noisy text characteristics of ITSM tickets.

3. **Code-mixing robustness**: Twitter pretraining exposes the model to Arabic-English code-mixing at a scale and naturalness unmatched by news or Wikipedia corpora. This is directly relevant to ITSM text where technical English terms (*VPN*, *Outlook*, *MFA*) appear embedded within Arabic sentences.

4. **Practical efficiency**: MarBERTv2 has the same parameter count as BERT-base (~163M), enabling fine-tuning on consumer-grade GPUs and efficient inference — an important constraint for cloud deployment scenarios where cost per prediction matters.

The model is available as `UBC-NLP/MARBERTv2` on HuggingFace Hub and is compatible with the `AutoModel` / `AutoTokenizer` API, requiring no custom code changes from a standard BERT fine-tuning pipeline.

---

## 7. Conditions for Alternative Choices

The recommendation above is conditioned on the current dataset properties and project constraints. The following circumstances would warrant reconsidering the model choice:

| Condition | Preferred Alternative | Reason |
|-----------|----------------------|--------|
| Dataset augmented with formal/MSA tickets | CAMeLBERT-mix | Broader genre coverage handles MSA–dialect mix better than MarBERT |
| Extreme spelling variation, Arabizi, or mixed scripts | ByT5 | Byte-level processing is robust to arbitrary orthographic noise |
| Very small fine-tuning budget (<500 samples) | AraBERTv2 | Slightly better MSA representations for short fine-tuning when dialects are less dominant |
| Cross-lingual or multi-language ITSM tickets | XLM-R | Trained across 100 languages; useful if non-Arabic tickets are included |
| Ensemble / late fusion for production | MarBERT + CAMeLBERT | Complementary dialect vs. structured text strengths |

---

## 8. Practical Fine-Tuning Notes

Based on the model selection and the properties of the dataset, the following fine-tuning configuration is recommended as a starting point (see `configs/model_config.yaml` for full parameter details):

- **Tokenization**: MarBERT WordPiece, `max_length=128` (covers >95% of dataset tickets)
- **Preprocessing**: Diacritics removal + alef normalization; do *not* apply teh marbuta normalization (hurts dialectal models) or aggressive punctuation removal (MarBERT handles it)
- **Input format**: `[CLS] title [SEP] description [SEP]` or concatenated as a single sequence with a space separator
- **Optimizer**: AdamW with `lr=2e-5`, `weight_decay=0.01`, linear warmup over 6% of total steps
- **Batch size**: Effective batch size of 32 (16 per device × 2 gradient accumulation steps)
- **Early stopping**: Patience of 2 epochs on validation macro-F1 to prevent overfitting on the synthetic dataset

---

## 9. References

[1] Darwish, K., & Magdy, W. (2014). Arabic information retrieval. *Foundations and Trends in Information Retrieval*, 7(4), 239–342.

[2] Soliman, A. B., Eissa, K., & El-Beltagy, S. R. (2017). AraVec: A set of Arabic word embedding models for use in Arabic NLP. *Procedia Computer Science*, 117, 256–265.

[3] Abdul-Mageed, M., Elmadany, A., & Nagoudi, E. M. B. (2021). ARBERT & MARBERT: Deep bidirectional transformers for Arabic. In *Proceedings of ACL-IJCNLP 2021* (pp. 7088–7105).

[4] Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In *Proceedings of the LREC Workshop on Arabic Language Processing* (pp. 9–15).

[5] Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021). The interplay of variant, size, and task type in Arabic pre-trained models. In *Proceedings of EACL 2021* (pp. 92–104).

[6] Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models. *Transactions of the Association for Computational Linguistics*, 10, 291–306.

[7] Abdelali, A., Mubarak, H., Chowdhury, S. A., Darwish, K., & Samih, Y. (2021). QADI: Arabic dialect identification in the wild. In *Proceedings of ACL-IJCNLP 2021*.

[8] Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of ACL 2020* (pp. 8440–8451).

[9] Bouamor, H., et al. (2023). The NADI 2023 shared task on Arabic dialect identification. In *Proceedings of ArabicNLP 2023*.
