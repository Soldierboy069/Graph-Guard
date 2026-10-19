# GraphGuard: An Explainable Graph Neural Network for Intelligent Intrusion Detection

GraphGuard is an end-to-end intrusion detection system based on temporal graph modeling, GraphSAGE neural networks, feature attribution via Explainable AI (SHAP), and a Threat Propagation Score (TPS) for alert prioritization.

Tested and validated on **CICIDS2017** (`GeneratedLabelledFlows`).

---

## Architecture Overview

```
CICIDS2017 (GeneratedLabelledFlows)
    ↓
Data Preprocessing (Memory-safe streaming, cleaning, scaling, time-aware split)
    ↓
Network Graph Construction (IP → Nodes, Directed Flows → Edges)
    ↓
Adaptive Graph Construction (AGC) (Temporal 5-min windows, EMA edge weights, stale edge pruning)
    ↓
GraphSAGE Node Encoder (2-layer inductive message passing with BatchNorm & L2 norm)
    ↓
Flow/Edge Classifier (MLP: [Source Embedding || Destination Embedding || Traffic Features])
    ↓
Explainable AI (SHAP feature attribution per suspicious flow)
    ↓
Threat Propagation Score (TPS alert prioritization: attack probability, neighbor risk, recency, degree)
    ↓
Interactive Streamlit Dashboard (7-page live operational dashboard)
```

---

## Quickstart

### 1. Environment Setup

```bash
# Uses the dedicated graphguard conda environment
conda activate graphguard
pip install -r requirements.txt
```

### 2. End-to-End Command-Line Pipeline

```bash
# Phase 2: Data Preprocessing (sample mode for development, or full dataset)
python scripts/01_prepare_data.py --sample

# Phase 3: Graph Construction
python scripts/02_build_graph.py

# Phase 4: Adaptive Graph Construction (AGC)
python scripts/03_run_agc.py

# Phase 5: Train GraphGuard GNN
python scripts/04_train_graphguard.py --experiment full_graphguard

# Phase 6: Train Baseline Models (Random Forest & XGBoost)
python scripts/05_train_baselines.py

# Phase 7: Evaluate All Models (Metrics, Plots, Confusion Matrices)
python scripts/06_evaluate.py

# Phase 8: Explainability (SHAP Feature Attribution)
python scripts/07_run_xai.py

# Phase 9: Calculate Threat Propagation Score (TPS)
python scripts/08_calculate_tps.py

# Phase 11: Run Ablation Study
python scripts/09_run_ablation.py
```

### 3. Live Dashboard

Launch the interactive 7-page Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Evaluation Metrics

All models are evaluated on the exact same time-aware split:

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) | FPR |
|---|---|---|---|---|---|---|
| **GraphGuard** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Ablation Study Summary

| Experiment | Description | F1 (Macro) |
|---|---|---|
| `full_graphguard` | Full model with GraphSAGE + AGC + edge features | **1.0000** |
| `no_graph_context` | Tabular features only (zero node embeddings) | **0.4996** |
| `no_agc` | Static graph without temporal adaptation | 1.0000 |
| `no_adaptive_weights` | Uniform edge weights | 1.0000 |

---

## Unit Testing

Run test suite across preprocessing, graph construction, models, and TPS:

```bash
pytest tests/ -v
```

---

## Project Structure

```
graph guard/
├── config.yaml                     # Central configuration
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── data/
│   ├── raw/                        # Original archives
│   ├── processed/                  # Cleaned parquet splits & PyG graph
│   └── samples/                    # Dev samples
├── src/
│   ├── preprocessing/              # Chunked CSV loader, cleaner, encoder, scaler, splitter
│   ├── graph/                      # IP→Node mapping, directed edge builder, graph statistics
│   ├── agc/                        # Adaptive Graph Construction with temporal windowing
│   ├── models/                     # GraphSAGE encoder, edge classifier, trainer
│   ├── baselines/                  # Random Forest and XGBoost classifiers
│   ├── explainability/             # SHAP surrogate explainer
│   ├── threat/                     # Threat Propagation Score (TPS) formula
│   ├── evaluation/                 # Metrics, confusion matrices, training curves
│   └── utils/                      # Logger, seeds, config helpers
├── scripts/                        # 01 through 09 executable workflow
├── results/                        # Metrics, figures, predictions, explanations, graphs
├── dashboard/                      # Multi-page Streamlit application
└── tests/                          # Automated unit test suite
```
