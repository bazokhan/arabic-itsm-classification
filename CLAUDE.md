# CLAUDE.md — Project Memory & Tracking
# arabic-itsm-classification

This file is the living memory for Claude Code sessions on this project.
Update it continuously as decisions are made, experiments are run, and milestones are reached.
It will serve as a primary source for the university post-project documentation report.

---

## Project Identity

| Field            | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Title**        | Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models |
| **Author**       | Mohamed Adel Ebrahim Elbaz                                           |
| **Supervisor**   | Dr. Eman E. Sanad, Assistant Professor of IT, FCAI, Cairo University |
| **Degree**       | Professional Master's in Cloud Computing Networks                    |
| **Institution**  | Faculty of Computers and Artificial Intelligence, Cairo University    |
| **Date Started** | February 2026                                                        |
| **Repo**         | arabic-itsm-classification                                           |
| **Dataset Repo** | [bazokhan/arabic-itsm-dataset](https://github.com/bazokhan/arabic-itsm-dataset) |

---

## Dataset Summary

| Property          | Value                                                   |
|-------------------|---------------------------------------------------------|
| Source (GitHub)   | https://github.com/bazokhan/arabic-itsm-dataset         |
| Source (HF)       | https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset |
| Size              | 10,000 tickets                                          |
| Dialect           | Egyptian Arabic (عامية مصرية) with EN code-mixing       |
| Generation        | LLM-generated (Gemini), validated programmatically      |
| L1 categories     | 6 (Access, Network, Hardware, Software, Security, Service) |
| L2 categories     | 14                                                      |
| L3 categories     | 31                                                      |
| Labels            | category_level_1/2/3, priority (1-5), sentiment        |
| Format            | CSV + JSONL                                             |

---

## Architecture Decisions

### ADR-001 — Primary Model: MarBERTv2
- **Date**: February 2026
- **Decision**: Use `UBC-NLP/MARBERTv2` as the primary encoder
- **Rationale**: Best-in-class for Egyptian colloquial Arabic; pretrained on Twitter (dialect-heavy, noisy); validated Feb 2025 benchmarks show 64–85% F1 on Egyptian classification tasks; low compute cost vs ByT5
- **Alternatives considered**: CAMeLBERT (good but MSA-biased), AraBERTv2 (MSA-only), ByT5 (excellent noise handling but high compute)
- **Reference**: `docs/model_recommendation.md`
- **Status**: Accepted

### ADR-002 — Classification Strategy: Multi-Head, Start with L1
- **Date**: February 2026
- **Decision**: Train separate classification heads for L1 (6 classes), L2 (14 classes), and priority (5 classes). Prioritize L1 in initial experiments; cascade to L2/L3 once L1 is stable.
- **Rationale**: Multi-label as flat multiclass is too sparse at L3 (31 classes × 10k = ~322/class). Hierarchical approach follows ITSM routing logic.
- **Status**: Accepted

### ADR-003 — Framework: HuggingFace Transformers + PyTorch
- **Date**: February 2026
- **Decision**: Use `transformers` + `torch` with `Trainer` API for fine-tuning
- **Rationale**: Native support for MarBERTv2; standard in academic NLP work; easy to export to ONNX/TorchScript for deployment
- **Status**: Accepted

### ADR-004 — Data Split: Stratified 70/15/15
- **Date**: February 2026
- **Decision**: Stratified split on L1 label — 70% train, 15% validation, 15% test
- **Rationale**: Maintains class distribution across splits; 1,500 test samples sufficient for reliable metric estimation; validation set sized for hyperparameter search
- **Status**: Accepted

---

## Experiment Log

| ID   | Date | Task | Model | Epochs | LR | Batch | Val F1 (macro) | Test F1 (macro) | Notes |
|------|------|------|-------|--------|-----|-------|----------------|-----------------|-------|
| EXP-001 | — | L1 baseline | TF-IDF + LR | — | — | — | — | — | Pending |
| EXP-002 | — | L1 | MarBERTv2 | — | — | — | — | — | Pending |

---

## Milestones

- [x] Dataset created and published (`arabic-itsm-dataset`)
- [x] Model selection documented (`docs/model_recommendation.md`)
- [x] Repo scaffolded with notebooks, src, configs
- [ ] Notebook 01: Data inspection complete (with outputs)
- [ ] Notebook 02: Data preparation pipeline complete
- [ ] Notebook 03: Baseline models run and results recorded
- [ ] Notebook 04: MarBERTv2 fine-tuning — first run complete
- [ ] Notebook 04: Hyperparameter sweep complete
- [ ] Notebook 05: Final evaluation and results analysis
- [ ] Model exported and saved to `models/`
- [ ] University documentation draft started
- [ ] Web demo prototype (cloud deployment)

---

## Key File Locations

| Artifact                        | Path                                              |
|---------------------------------|---------------------------------------------------|
| Dataset CSV                     | https://raw.githubusercontent.com/bazokhan/arabic-itsm-dataset/master/dataset_clean.csv |
| Dataset JSONL                   | https://raw.githubusercontent.com/bazokhan/arabic-itsm-dataset/master/dataset_clean.jsonl |
| Taxonomy                        | https://raw.githubusercontent.com/bazokhan/arabic-itsm-dataset/master/taxonomy_itsm_v1.json |
| Model config                    | `configs/model_config.yaml`                       |
| Data config                     | `configs/data_config.yaml`                        |
| Best L1 checkpoint              | `models/marbert_l1_best/`                         |
| Results (metrics)               | `results/metrics/`                                |
| Results (figures)               | `results/figures/`                                |

---

## Notes for University Documentation Report

*(Append notes here as the project progresses — these will feed directly into the final report)*

### Chapter 2 — Literature Review Angles
- Arabic NLP challenges: dialect diversity, orthographic variation, code-mixing
- ITSM automation: ticket routing, SLA compliance, priority prediction
- Transformer models for Arabic: AraBERT → CAMeLBERT → MarBERT → MarBERTv2 evolution
- Synthetic data for low-resource languages: LLM-generated datasets for domain adaptation

### Chapter 3 — Methodology Notes
- Dataset: synthetic LLM-generated, 10,000 tickets, Egyptian Arabic, 3-level taxonomy
- Preprocessing: arabert normalization (no diacritics, alif normalization), truncation at 128 tokens
- Model: MarBERTv2 encoder + linear head per task
- Training: AdamW optimizer, linear warmup + decay, early stopping on val macro-F1
- Evaluation: macro-F1, per-class F1, confusion matrix, latency measurement

### Chapter 4 — Results Placeholder
*(Fill in after experiments)*

### Chapter 5 — Deployment Architecture Notes
- Model served via FastAPI
- Cloud: containerized with Docker
- Input: Arabic ticket title + description → JSON response {l1, l2, priority, confidence}

---

## Known Issues & Workarounds

| Issue | Platform | Fix |
|-------|----------|-----|
| `mlflow ui` crashes with `OSError: [WinError 10022]` | Windows | Run `mlflow ui --workers 1` — disables multiprocess socket sharing which fails on Windows |

---

## Session Log

| Date       | Session Summary |
|------------|-----------------|
| 2026-02-24 | Initial repo scaffold: README, CLAUDE.md, folder structure, configs, src package, 5 demo notebooks created |
| 2026-02-24 | Fixed all dataset references to use GitHub/HuggingFace URLs (no local clone required). Rewrote model_recommendation.md as full academic comparative analysis. |
| 2026-02-24 | Diagnosed MLflow WinError 10022 — fixed with `--workers 1` flag |
