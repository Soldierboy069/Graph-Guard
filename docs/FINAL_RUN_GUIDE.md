# GraphGuard: Final Run & Reproduction Guide

**Purpose:** Comprehensive, End-to-End Command Reference for Reproducing, Testing, and Demonstrating GraphGuard  
**Operating System:** macOS (Apple Silicon / Intel) & Linux  
**Conda Environment:** `graphguard` (Python 3.11)  
**Status:** Frozen & Verified  

---

## 1. Environment Setup & Verification

```bash
# 1. Navigate to project root
cd "/Users/paramesh/Desktop/graph guard"

# 2. Activate Conda environment
conda activate graphguard

# 3. Verify Python version and PyTorch MPS availability
python -c "
import torch
print('Python interpreter:', torch.__file__)
print('PyTorch version:   ', torch.__version__)
print('MPS available:     ', torch.backends.mps.is_available())
"
```

---

## 2. Automated Test Suite

To verify system health and ensure all 15 unit tests pass:

```bash
python -m pytest tests/ -v
```
*Expected Output:* `15 passed, 1 warning in ~1.5 seconds` with zero failures.

---

## 3. End-to-End Pipeline Execution Commands

All scripts are modular, numbered in sequence, and execute against the real Tuesday capture of CICIDS2017:

### Step 1: Preprocessing & Campaign-Window Chronological Split
Cleans raw flows, applies Campaign-Window Chronological Split (70/15/15), purges exact feature duplicates across splits, and normalizes traffic metrics:
```bash
python scripts/01_preprocess.py
```
*Artifacts Generated:*
- `data/processed/train.parquet` (155,544 rows)
- `data/processed/val.parquet` (28,696 rows)
- `data/processed/test.parquet` (29,762 rows)
- `data/processed/scaler.pkl` & `label_encoder.json`
- `data/processed/duplicate_audit.json` & `campaign_provenance.json`

---

### Step 2: Inductive Graph Construction
Constructs $G_{\text{train}}$ from training flows only, prepares $G_{\text{hist}}$ (Train + Val) for historical inference, generates 10 label-free node features, and enforces target-edge exclusion:
```bash
python scripts/02_build_graph.py
```
*Artifacts Generated:*
- `data/processed/graph_data.pt` (PyG inductive Data object: 5,728 nodes, 155,544 train edges, 184,240 hist edges)
- `results/graphs/flow_metadata.parquet` (29,762 test flow metadata records)
- `results/graphs/ip_to_node.json` & `graph_stats.json`

---

### Step 3: Adaptive Graph Construction (AGC)
Executes dynamic temporal windowing (10-minute windows) over historical traffic, applies EMA edge weight decay ($\lambda=0.95$), and prunes stale edges ($w < 0.01$):
```bash
python scripts/03_run_agc.py
```
*Artifacts Generated:*
- `results/graphs/agc_snapshots.json` (29 temporal snapshots)
- `results/graphs/agc_summary.csv`

---

### Step 4: Train GraphGuard (GNN)
Trains the 2-layer GraphSAGE encoder and EdgeClassifier MLP head with weighted cross-entropy loss and early stopping:
```bash
# A. Multiclass Training (BENIGN, FTP-Patator, SSH-Patator)
python scripts/04_train_graphguard.py --experiment full_graphguard

# B. Binary Training (BENIGN vs ATTACK)
python scripts/04_train_graphguard.py --experiment full_graphguard_binary --binary
```
*Artifacts Generated:*
- `models/full_graphguard_best.pt` & `full_graphguard_binary_best.pt`
- `results/predictions/full_graphguard_test_predictions.parquet`
- `results/predictions/full_graphguard_binary_test_predictions.parquet`
- `results/metrics/full_graphguard_training_history.json`

---

### Step 5: Train Baseline Models
Trains Random Forest (200 trees) and XGBoost (200 estimators) independently on raw tabular flow features:
```bash
python scripts/05_train_baselines.py
```
*Artifacts Generated:*
- `models/rf_model_multiclass.pkl` & `rf_model_binary.pkl`
- `models/xgb_model_multiclass.pkl` & `xgb_model_binary.pkl`
- `results/predictions/rf_multiclass_test_predictions.parquet`
- `results/predictions/xgb_multiclass_test_predictions.parquet`
- `results/predictions/rf_binary_test_predictions.parquet`
- `results/predictions/xgb_binary_test_predictions.parquet`

---

### Step 6: Evaluate Models & Generate Comparisons
Computes independent metrics, generates classification reports, and saves confusion matrices:
```bash
python scripts/06_evaluate.py
```
*Artifacts Generated:*
- `results/metrics/model_comparison_multiclass.csv`
- `results/metrics/model_comparison_binary.csv`
- `results/metrics/graphguard_multiclass_metrics.json`
- `results/metrics/graphguard_binary_metrics.json`
- Confusion matrix PNGs in `results/figures/`

---

### Step 7: Explainable AI (XAI / SHAP)
Computes flow-level SHAP attributions on real attack flows using a TreeExplainer surrogate:
```bash
python scripts/07_run_xai.py
```
*Artifacts Generated:*
- `results/explanations/top_20_flow_explanations.json`
- `results/explanations/flow_0_shap.png` through `flow_19_shap.png`

---

### Step 8: Threat Propagation Score (TPS)
Computes normalized Threat Propagation Scores across all 5,728 IP nodes:
```bash
python scripts/08_calculate_tps.py
```
*Artifacts Generated:*
- `results/metrics/tps_scores.csv`
- `results/metrics/tps_summary.json`

---

### Step 9: 5-Component Ablation Study
Evaluates all 5 decoupled architecture variants (`full_graphguard`, `no_agc`, `no_graph_context`, `no_adaptive_weights`, `no_graphsage`):
```bash
python scripts/09_run_ablation.py
```
*Artifacts Generated:*
- `results/metrics/ablation.csv`
- `results/figures/ablation_comparison.png`

---

### Step 10: Generate Presentation Figures (Review-2)
Renders all 10 high-resolution (300 DPI) presentation figures for PowerPoint:
```bash
python scripts/generate_review2_figures.py
```
*Artifacts Generated:*
- `results/review2_figures/01_system_architecture.png`
- `results/review2_figures/02_graph_construction.png`
- `results/review2_figures/03_agc_pipeline.png`
- `results/review2_figures/04_model_architecture.png`
- `results/review2_figures/05_model_comparison.png`
- `results/review2_figures/06_confusion_matrix_graphguard.png`
- `results/review2_figures/07_ablation_comparison.png`
- `results/review2_figures/08_xai_feature_importance.png`
- `results/review2_figures/09_tps_visualization.png`
- `results/review2_figures/10_detection_pipeline.png`

---

## 4. Launching the Interactive Streamlit Dashboard

To launch the live demonstration dashboard:

```bash
streamlit run dashboard/app.py
```
- Open `http://localhost:8501` in your browser.
- Navigate to **"🎓 Review-2 Demo"** to access the guided 9-step faculty demonstration workflow.
