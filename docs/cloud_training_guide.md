# Cloud-Based Training Strategy for Granular Classification
## Scaling to L3 and Multi-Task Models via Google Colab & Kaggle

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models  
**Author**: Mohamed Adel Ebrahim Elbaz  
**Status**: Experimental Phase Completion  

---

## 1. Concept: The "Compute Pivot"

As the ITSM taxonomy expands from 6 classes (L1) to 48 classes (L3), the model's output dimensionality and the resulting gradient complexity grow exponentially. Local consumer-grade hardware (e.g., 4GB VRAM) may face "Out of Memory" (OOM) errors or excessive training durations (10+ hours). 

This guide details the **Compute Pivot**—moving the training phase to high-performance cloud environments (Google Colab or Kaggle) while maintaining local control over preprocessing and final deployment.

---

## 2. Platform Selection & Architecture

| Feature | Google Colab (Free/Pro) | Kaggle (Recommended) |
| :--- | :--- | :--- |
| **GPU Architecture** | T4 (16GB) or A100 (40GB) | 2x T4 or P100 (16GB) |
| **Session Limit** | 12 Hours (variable) | 12 Hours (fixed) |
| **Weekly Budget** | Variable | 30 Hours (Free) |
| **Best For** | Pro users with Google Drive sync | Beginners (stable file system) |

### Deployment Loop
1. **Local**: Preprocess data and generate `train.csv`, `val.csv`, and `label_encoders.pkl`.
2. **Cloud**: Upload artifacts, execute `scripts/train.py` via CLI.
3. **Local**: Download `heads.pt` and `config.json` for production inference.

---

## 3. Step-by-Step Execution Guide

### Phase A: Local Preparation
Do not upload the entire raw dataset. Only upload the processed artifacts to minimize bandwidth.
1. Locate your `data/processed/` directory.
2. Create a zip archive:
   ```powershell
   Compress-Archive -Path data/processed/* -DestinationPath processed_data.zip
   ```

### Phase B: Cloud Environment Setup
Open a new notebook on [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/). 

**1. System Initialization**
Run these commands in a notebook cell (prepended with `!`) to clone the repository and install the environment:
```bash
# Clone the repository
!git clone https://github.com/bazokhan/arabic-itsm-classification.git
%cd arabic-itsm-classification

# Install dependencies (ignoring torch to avoid overriding the cloud's optimized CUDA build)
!pip install transformers datasets accelerate evaluate arabert pyarabic statsmodels mlflow tqdm pyyaml
```

**2. Data Upload**
- **Colab**: Click the folder icon on the left sidebar and drag `processed_data.zip` into the pane.
- **Kaggle**: Use the "Add Data" button on the right sidebar to upload the zip as a private dataset.

**3. Artifact Extraction**
```bash
!mkdir -p data/processed
!unzip ../processed_data.zip -d data/processed/
```

### Phase C: Executing the Training Script
Instead of manual cell execution, use the provided `scripts/train.py` for a robust, CLI-driven experience. This script handles MLflow logging and automatic checkpointing.

**To train the Level 3 (48 classes) model:**
```bash
!python scripts/train.py \
    --task l3 \
    --epochs 10 \
    --lr 1e-5 \
    --batch-size 16 \
    --output-dir models/cloud_runs
```

**To train the Final Multi-Task model (L1+L2+L3+Priority+Sentiment):**
```bash
!python scripts/train.py \
    --multi-task \
    --epochs 15 \
    --lr 1e-5 \
    --batch-size 16 \
    --output-dir models/cloud_runs
```

### Phase D: Automated Model Retrieval
Once the "Done" message appears in the logs, the model weights are stored in the cloud's virtual disk.

**On Google Colab:**
```python
from google.colab import files
import os

# Zip the best model for download
!zip -r l3_model.zip models/cloud_runs/marbert_l3_best/
files.download('l3_model.zip')
```

**On Kaggle:**
The files will appear in the `/kaggle/working/models/` directory. Navigate to the "Output" tab on the right sidebar and click "Download All".

---

## 4. Hyperparameter Tuning for Granular Tasks

When moving to Level 3, the following adjustments are recommended to ensure stability:

1. **Learning Rate Decay**: Use `1e-5` (instead of `2e-5`). Lower learning rates prevent the model from "forgetting" the broad L1 features while trying to learn the specific L3 nuances.
2. **Early Stopping**: The `train.py` script monitors `val_macro_f1`. If the score does not improve for 3 epochs, it will terminate. In the cloud, set `--epochs 15` and let the early stopping logic decide the final cutoff.
3. **Weight Decay**: Keep weight decay at `0.01` to regularize the 48 classification heads and prevent overfitting on sparse L3 classes.

---

## 5. Integration with Production
Once you have downloaded the weights (`heads.pt`) and the configuration (`config.json`), place them in your local project:
1. Create a folder: `models/marbert_final/`
2. Move the cloud files into this folder.
3. Update your `inference.py` call:
   ```python
   engine = ITSMInferenceEngine(checkpoint_dir="models/marbert_final", ...)
   ```

This workflow ensures that you use the cloud purely as a "Compute Engine" while keeping your code and inference logic locally managed.
