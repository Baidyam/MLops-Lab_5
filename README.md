# ELK-Based ML Monitoring Pipeline

## Project Overview

This project implements an **ML monitoring pipeline** using **Logstash, Elasticsearch, and Kibana (ELK stack)** to track:

* Model training performance (F1, FPR, FNR)
* Real-time data drift detection
* Centralized logging and visualization

Logs are generated from Python pipelines and ingested into Elasticsearch via Logstash for analysis in Kibana.

---

## Changes Made (Dataset Update)

### What was changed

* Replaced previous dataset with **Breast Cancer dataset (from sklearn)**

### Why this dataset

* Fully numeric features (no preprocessing issues)
* Classification problem (compatible with existing pipeline)
* No missing values

---

## Files Modified

### 1. `train_model.py`

**Change: Dataset loading updated**

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
```

---

### 2. `drift_detection.py`

**Change: Data generation logic updated to use same dataset**

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_full, y_full = data.data, data.target

def generate_data(samples=64, drift=False):
    idx = np.random.choice(len(X_full), samples, replace=True)
    X, y = X_full[idx], y_full[idx]
    return X, y
```

---

## No Changes Required

* `logstash.conf` → unchanged (handles logs only)
* `docker-compose.yml` → unchanged
* ELK pipeline → unchanged

---

## Key Insight

Since the new dataset:

* Has **same structure (numeric + classification target)**
* Requires **no additional preprocessing**

Only dataset loading logic needed to be updated.
No changes required in logging or ingestion pipeline.

---

## How to Run

```bash
docker-compose up --build
```

Run pipelines:

```bash
python train_model.py
python drift_detection.py
```

---

## Output

* Logs → Logstash → Elasticsearch
* Visualized in → Kibana (`http://localhost:5601`)

---
