# ADR-001: Primary Model Selection — MarBERTv2

**Date**: February 2026
**Status**: Accepted
**Author**: Mohamed Adel Ebrahim Elbaz

---

## Context

The project requires classification of Arabic ITSM tickets written in Egyptian colloquial Arabic (عامية مصرية), often mixed with English technical terms (VPN, Outlook, MFA). The text is informal, short, and frequently misspelled or inconsistently diacritized — matching the character of social media and helpdesk communications.

Several Arabic transformer encoder models were considered. The choice materially affects:
- Classification accuracy on colloquial text
- Compute requirements (training + inference)
- Ease of deployment (model size, HuggingFace availability)

---

## Considered Options

| Model | Pretrain Corpus | Dialect Fit | Noise Handling | F1 (Egyptian tasks) | Model Size |
|-------|----------------|-------------|----------------|---------------------|------------|
| **MarBERTv2** | Twitter + Web (MSA + dialects) | Excellent | Strong | 64–85% | 163M params |
| CAMeLBERT-Mix | Wikipedia + Web + Books | Good | Moderate | 80–92% | 163M params |
| AraBERTv2 | Wikipedia + news | Fair | Moderate | 70–80% | 135M params |
| ByT5-base | Byte-level, multilingual | Excellent | Excellent | ~74% | 300M params |

---

## Decision

**MarBERTv2** (`UBC-NLP/MARBERTv2`) is selected as the primary encoder.

---

## Rationale

1. **Dialect match**: MarBERTv2 was pretrained on a large Twitter corpus containing substantial Egyptian Arabic content — the closest proxy to ITSM helpdesk language.
2. **Noise robustness**: Twitter pretraining exposes the model to misspellings, code-mixing, and inconsistent diacritization, which mirrors our data distribution.
3. **Benchmark evidence**: As of February 2025, MarBERTv2 tops published Egyptian Arabic classification benchmarks over CAMeLBERT and AraBERT on sentiment, hate speech, and dialect ID tasks.
4. **Compute efficiency**: Same size as CAMeLBERT (~163M params) but better dialect performance; significantly smaller than ByT5 encoder alternatives.
5. **HuggingFace availability**: Available at `UBC-NLP/MARBERTv2`; drop-in with `AutoModel` / `AutoTokenizer`.

---

## Trade-offs and Limitations

- CAMeLBERT-Mix may outperform on more formally written or MSA-mixed tickets.
- ByT5 offers superior robustness to arbitrary spelling variance but doubles compute cost.
- If the dataset is augmented with MSA tickets in future, consider a late-fusion ensemble with AraBERTv2.

---

## Consequences

- All fine-tuning experiments use `UBC-NLP/MARBERTv2` as the frozen or partially-frozen encoder.
- Tokenization uses MarBERT's WordPiece vocabulary with `max_length=128`.
- CAMeLBERT and AraBERTv2 are run as secondary experiments for comparison (reported in evaluation notebook).
