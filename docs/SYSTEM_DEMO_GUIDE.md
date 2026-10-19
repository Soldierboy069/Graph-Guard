# GraphGuard: System Demonstration Guide for Review-2

**Purpose:** Step-by-Step Script and Technical Walkthrough for Faculty Review-2 Live Demonstration  
**Target Application:** Streamlit Interactive Dashboard (`dashboard/app.py`)  
**Status:** Tested & Operational  

---

## 1. Launching the Demonstration Dashboard

Open a terminal in the project directory and execute:

```bash
# 1. Activate conda environment
conda activate graphguard

# 2. Navigate to project root
cd "/Users/paramesh/Desktop/graph guard"

# 3. Launch Streamlit application
streamlit run dashboard/app.py
```

The application will start and automatically open in your default browser at:
`http://localhost:8501`

---

## 2. Live 9-Step Faculty Demonstration Script

Select **"🎓 Review-2 Demo"** at the top of the sidebar navigation. This opens the dedicated 9-step interactive demonstration page.

---

### Step 1: Dataset Overview & Chronological Splitting
- **Action in UI:** Click on the tab **"Step 1: Dataset"**.
- **What is Displayed:**
  - Metric cards: Clean Flows (445,645), Train (155,544), Val (28,696), Test (29,762).
  - Figure 1: End-to-End System Architecture.
- **Examiner Talking Point:**
  > *"Respected panel, GraphGuard is evaluated on the complete Tuesday capture of the CICIDS2017 benchmark. To ensure scientific rigor, we designed a **Campaign-Window Chronological Split** where within each brute-force attack campaign independently, traffic is split into 70% training, 15% validation, and 15% held-out test sets. This ensures temporal causality ($t_{\text{train}} \le t_{\text{val}} \le t_{\text{test}}$) while guaranteeing that the held-out test set contains **710 genuine, verified attack flows** (636 FTP-Patator and 74 SSH-Patator) alongside 29,052 benign flows. Furthermore, 8,204 exact feature duplicate rows across splits were excised to prevent nearest-neighbor memorization."*

---

### Step 2: Network Graph Representation & Inductive Protocol
- **Action in UI:** Click on the tab **"Step 2: Network Graph"**.
- **What is Displayed:**
  - Graph specifications: 5,728 IP nodes, 155,544 Train edges.
  - Figure 2: Graph Construction & Inductive Protocol.
- **Examiner Talking Point:**
  > *"Unlike traditional NIDS that analyze packet flows in total isolation, GraphGuard models network communications as a directed property graph. Nodes represent IP endpoints with 10 label-free topological features—we strictly eradicated `attack_neighbor_ratio` to eliminate label leakage. Flows represent directed edges with 77 continuous CICFlowMeter features. Crucially, we enforce an **inductive target-edge exclusion protocol**: during testing, target flows are NEVER placed into the message-passing graph. Node representations are derived strictly from historical context $G_{\text{hist}}$, and target flows are classified as pure query edges $[\mathbf{h}_u \,\|\, \mathbf{h}_v \,\|\, \mathbf{e}_{uv}]$, guaranteeing zero future topological leakage."*

---

### Steps 3, 4 & 5: Live Suspicious Flow Detection & Confidence
- **Action in UI:** Click on the tab **"Steps 3-5: Flow Inference"**.
- **What is Displayed:**
  - Real Test Attack Flow details (Flow ID `172.16.0.1-192.168.10.50-59382-21-6`, Source IP `172.16.0.1`, Destination IP `192.168.10.50`, Protocol `TCP`, Actual Ground Truth: `FTP-Patator`).
  - GraphGuard prediction card: `FTP-Patator` with **99.96% Confidence**.
  - Figure 4: Detailed Model Architecture & Tensor Dimensions.
- **Examiner Talking Point:**
  > *"Here we demonstrate live inference on a real test attack flow (Index #638). The source IP `172.16.0.1` is initiating an automated brute-force connection to the target server `192.168.10.50` on port 21. GraphGuard aggregates 2-hop neighborhood representations using GraphSAGE ($10 \to 128 \to 64$), concatenates source and destination embeddings with the 77-dimensional flow vector to form a 205-dimensional representation, and passes it through an MLP classifier. GraphGuard correctly predicts `FTP-Patator` with **99.96% confidence**, achieving an exact match with zero false negatives."*

---

### Step 6: Explainable AI (SHAP Root Cause Attribution)
- **Action in UI:** Click on the tab **"Step 6: XAI Explanation"**.
- **What is Displayed:**
  - Table of top feature attribution scores for the flow.
  - SHAP Waterfall plot (`flow_0_shap.png`).
- **Examiner Talking Point:**
  > *"In a real-world Security Operations Center, black-box predictions lead to alert fatigue. GraphGuard incorporates Explainable AI using a TreeExplainer surrogate. For this specific attack flow, the SHAP waterfall chart reveals the exact physical features driving the decision: small `Fwd Packet Length Mean` (0.1009 SHAP) and `Max Packet Length` combined with high `Fwd PSH Flags` and `SYN Flag Count` indicate repetitive, automated authentication handshakes typical of Patator brute-force tools, giving the SOC analyst immediate forensic evidence."*

---

### Step 7: Threat Propagation Score (TPS)
- **Action in UI:** Click on the tab **"Step 7: Threat Prioritization (TPS)"**.
- **What is Displayed:**
  - Node summary metrics: 5,728 nodes, 1 CRITICAL node, 1 MEDIUM node.
  - Figure 9: Top Prioritized IP Nodes & Formula Weights.
- **Examiner Talking Point:**
  > *"To solve the incident triage problem, we developed the Threat Propagation Score (TPS). TPS is an alert-prioritization mechanism combining 5 weighted dimensions: outbound attack probability (35%), suspicious neighbor ratio (25%), recent attack frequency (20%), degree centrality (10%), and graph hub betweenness (10%). Across all 5,728 IP nodes in the network, TPS correctly elevates the attacker machine `172.16.0.1` to **CRITICAL tier (TPS = 1.0000)** and flags the targeted server `52.35.227.152` as **MEDIUM tier (TPS = 0.3368)**, directing analyst focus to the most topologically critical threats."*

---

### Step 8: Comprehensive Model Benchmark Comparison
- **Action in UI:** Click on the tab **"Step 8: Model Comparison"**.
- **What is Displayed:**
  - Multiclass and Binary benchmark tables.
  - Figure 5: Model Comparison Bar Chart.
  - Figure 6: GraphGuard Confusion Matrices.
- **Examiner Talking Point:**
  > *"Comparing GraphGuard against industry-standard baselines (Random Forest and XGBoost) on the 29,762 held-out flows:
  > 1. In Multiclass detection, GraphGuard achieves **0.9787 Macro F1** and **0.9922 Macro Recall**, outperforming both Random Forest (0.9713) and XGBoost (0.9713).
  > 2. In Binary detection, GraphGuard achieves **100% attack recall (710 out of 710 attacks detected)**, whereas Random Forest missed 18 attacks and XGBoost missed 16 attacks.
  > 3. Operationally, GraphGuard achieves a **28x lower False Positive Rate** (0.03% vs 0.85%), virtually eliminating false alarms on legitimate traffic."*

---

### Step 9: Architectural Ablation Study
- **Action in UI:** Click on the tab **"Step 9: Ablation Study"**.
- **What is Displayed:**
  - 5-component ablation results table.
  - Figure 7: Ablation Comparison Bar Chart.
- **Examiner Talking Point:**
  > *"To isolate the empirical contribution of our graph neural network architecture, we trained and evaluated five decoupled variants. When we remove graph topological context (`no_graph_context`, relying solely on isolated tabular flow features), Macro F1 collapses from **0.9787 to 0.7129**—an **absolute drop of 26.58 percentage points, or a 27.16% relative decrease**—and the False Positive Rate surges by **31x** from 0.0003 to 0.0093. This ablation demonstrates the contribution of graph context to GraphGuard's performance under the evaluated test protocol."*

---

## 3. Anticipated Faculty Questions During Demo

| Faculty Question | Recommended Concise Answer |
| :--- | :--- |
| **"Why is your accuracy 0.9992 for all models?"** | *"Because the CICIDS2017 dataset is 97.6% benign traffic; any naive model predicting all benign achieves 97.6% accuracy. That is why we evaluate on **Macro F1** and **Macro Recall**, which weight all classes equally regardless of support size."* |
| **"Why did Random Forest miss 18 attacks while GraphGuard caught all 710?"** | *"Because Random Forest evaluates flows in isolation and cannot see that the source IP has an anomalous zero-in-degree / high-out-degree graph signature communicating with authentication servers. GraphGuard fuses node embeddings with edge features, enabling 100% attack recall."* |
| **"Does TPS predict future attacks?"** | *"No, TPS is strictly a **threat-prioritization mechanism** designed to rank current alerts for SOC analysts based on topological centrality and communication frequency."* |

---

## 4. Verification Check
All 10 figures and live demo files are stored in:
- `results/review2_figures/`
- `results/demo/demo_attack_flow.json`
- `dashboard/app.py`
