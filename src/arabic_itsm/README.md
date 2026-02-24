# Inference & Deployment Documentation

This directory contains the production-ready inference engine for the Arabic ITSM classification system. The `inference.py` module is designed to bridge the gap between experimental Jupyter Notebooks and a real-world software system.

## 1. The Production Inference Engine (`inference.py`)

The `ITSMInferenceEngine` class provides a unified interface for all trained models. It is **task-agnostic**, meaning it automatically detects whether you are using an L1-only model, an L1+L2 model, or the full Multi-Task model (L1+L2+L3+Priority+Sentiment).

### Key Features:
- **Seamless Integration**: Handles raw Arabic text normalization, tokenization, and multi-head logic in one call.
- **Dynamic Head Detection**: Automatically identifies active classification tasks by inspecting the checkpoint weights.
- **Structured Output**: Returns JSON-like dictionaries with human-readable labels and confidence scores.

### How to Use (Code Example):

```python
from arabic_itsm.inference import ITSMInferenceEngine

# 1. Initialize the engine with any of your checkpoints
engine = ITSMInferenceEngine(
    checkpoint_dir="models/marbert_l2_best",
    label_encoders_path="data/processed/label_encoders.pkl"
)

# 2. Predict on a new Arabic ticket
result = engine.predict(
    title="مشكلة في الايميل",
    description="مش عارف افتح الاوتلوك والسيستم بيطلع ايرور"
)

# 3. View structured results
print(result['predictions'])
# Output: {'l1': {'label': 'Software', 'conf': 0.98}, 'l2': {'label': 'Office Apps', 'conf': 0.94}}
```

---

## 2. Deployment Architecture

Within the proposed system (as detailed in the abstract), this model functions as the **Ticket Router Component**.

### Deployment Scenarios:

#### A. FastAPI Wrapper (Recommended)
You can wrap this engine in a FastAPI script to create a web API.
- **Input**: `POST /classify` with JSON body `{ "title": "...", "desc": "..." }`.
- **Output**: JSON containing category levels and priority.
- **Benefit**: Can be consumed by any modern frontend (React/Angular) or ITSM tool (ServiceNow/Jira).

#### B. Background Worker (Async)
For high-volume IT departments, place the model behind a message queue (like RabbitMQ or Celery).
- **Process**: Raw ticket lands in DB → Trigger sent to queue → Model classifies → Prediction written back to DB for IT staff.

---

## 3. Deployment Checklist

To move from this repo to a production server (Cloud VM / Docker):
1. **Environment**: Install `requirements.txt`.
2. **GPU vs CPU**: 
   - On **GPU**, inference takes ~10-15ms.
   - On **CPU**, inference takes ~120ms. For a real-time UI, CPU is often sufficient if the ticket volume is under 5 tickets/second.
3. **Artifacts**: Ensure the `models/` and `label_encoders.pkl` files are copied to the server. Use `Path` objects consistently to avoid Windows/Linux slash discrepancies.
