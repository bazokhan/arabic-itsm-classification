# Cloud-Based Training Strategy for Granular Classification
## Scaling to L3 and Multi-Task Models via Google Colab & Kaggle

**Project**: Cloud-Based ITSM Ticket Classification Platform Using Fine-Tuned Transformer Models  
**Author**: Mohamed A. Elbaz  
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

**1. Hardware Acceleration (Critical)**
By default, cloud notebooks use the CPU. You must manually attach a GPU to avoid 10+ hour training times.

- **Kaggle**: 
  1. Open the right-hand sidebar settings (click the `|>` icon or "Settings").
  2. Locate the **"Accelerator"** dropdown.
  3. Select **"GPU T4 x2"** (standard for multi-tasking) or **"GPU P100"**.
  4. Confirm the session restart.
- **Google Colab**:
  1. Navigate to **Edit** -> **Notebook settings**.
  2. Under "Hardware accelerator", select **"T4 GPU"** (or A100 if on Colab Pro).
  3. Click **Save**.

**2. System Initialization**
Run these commands in a notebook cell (prepended with `!`) to clone the repository and install the environment:
```bash
# Clone the repository
!git clone https://github.com/bazokhan/arabic-itsm-classification.git
%cd arabic-itsm-classification

# Install dependencies
# Note: We skip installing 'torch' to use the cloud's pre-installed, GPU-optimized version.
!pip install transformers datasets accelerate evaluate arabert pyarabic statsmodels mlflow tqdm pyyaml
```

**3. Data Upload**
- **Colab**: Click the folder icon on the left sidebar and drag `processed_data.zip` into the pane.
- **Kaggle**: Use the "Add Data" button on the right sidebar to upload the zip as a private dataset.

**4. Artifact Extraction & Path Mapping**
- **Colab**:
  ```bash
  !mkdir -p data/processed
  !unzip ../processed_data.zip -d data/processed/
  ```
- **Kaggle**: 
  Kaggle unzips datasets into a nested structure. Based on your verification, your exact data path is:
  `/kaggle/input/datasets/mohamedalbaz/processed-data`

  **1. Map the data**:
  Run these exact commands to link the Kaggle input to the project's expected data directory:
  ```bash
  # Clear existing placeholder created by git
  !rm -rf data/processed && mkdir -p data/processed
  
  # Link using the EXACT path confirmed via ls -R
  !cp -rs /kaggle/input/datasets/mohamedalbaz/processed-data/* data/processed/
  ```

  **2. Verify mapping**:
  ```bash
  !ls data/processed
  # Expected output: label_encoders.pkl, test.csv, train.csv, val.csv
  ```

---

## 4. Common Warnings & Troubleshooting

### A. "Device: cpu | FP16: False"
If you see this in the first line of the training output, the script is **not using the GPU**.
- **Fix**: Re-check Phase B, Step 1 (Hardware Acceleration). 
- **Verification**: Run `!nvidia-smi` in a cell. If it returns "command not found" or an error, the GPU is not attached.

### B. "InconsistentVersionWarning: Trying to unpickle LabelEncoder..."
You may see a warning about `scikit-learn` versions (e.g., 1.8.0 vs 1.6.1).
- **Explanation**: This occurs when the `label_encoders.pkl` was created on a local machine with a different version of scikit-learn than the cloud environment.
- **Action**: **Safe to ignore.** For simple `LabelEncoder` objects, this version mismatch does not affect prediction accuracy.

### C. "Warning: You are sending unauthenticated requests to the HF Hub"
- **Action**: **Safe to ignore.** MarBERTv2 is a public model; an API token is not required for download.

---

## 5. Executing the Training Script
Instead of manual cell execution, use the provided `scripts/train.py` for a robust, CLI-driven experience.

**To train the Level 3 (48 classes) model on Kaggle:**
```bash
!python scripts/train.py \
    --task l3 \
    --data-dir data/processed \
    --epochs 10 \
    --lr 1e-5 \
    --batch-size 16
```

**Note on GPU in Cloud**:
If you see `Device: cpu` in the output, ensure you have enabled the GPU accelerator in the notebook settings (Right sidebar -> Accelerator -> GPU T4 x2 or P100).

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
