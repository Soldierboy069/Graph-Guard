# GraphGuard: Review-2 Implementation Technical Documentation

**Project Title:** GraphGuard: An Explainable Graph Neural Network for Intelligent Intrusion Detection  
**Review Stage:** Academic Final-Year Review-2  
**Status:** Scientifically Audited & Frozen  
**Scientific Validity Verdict:** **VERIFIED**  

---

## 1. Problem Statement
Modern enterprise computer networks face sophisticated, automated intrusion campaigns such as credential brute-forcing, lateral pivoting, and multi-stage reconnaissance. Traditional Network Intrusion Detection Systems (NIDS) primarily rely on either:
1. **Signature-based detection:** Brittle against novel mutation patterns and zero-day variants.
2. **Point-based tabular Machine Learning (e.g., Random Forest, XGBoost):** Classifies each bidirectional network flow in total topological isolation, ignoring spatial relationship contexts, structural neighborhood patterns, and host roles.

GraphGuard formulates network intrusion detection as an **inductive edge classification task over a dynamic, time-aware property graph**, capturing both granular flow traffic metrics and higher-order topological communication patterns.

---

## 2. Motivation
- **Network Traffic is Inherently a Graph:** IP addresses function as communicating vertices, and bidirectional flows represent directed, attributed edges.
- **Topological Significance:** Attackers establish anomalous topological signatures (e.g., high fan-out scanning, asymmetric degree ratios, repeated querying of dedicated authentication servers) that tabular classifiers evaluating flows in isolation fail to correlate.
- **Explainability & Triage Need:** Black-box detection creates alert fatigue in Security Operations Centers (SOC). Operators require forensic feature attribution (XAI) and alert prioritization (TPS) to triage critical threats quickly.

---

## 3. Dataset
- **Benchmark:** Canadian Institute for Cybersecurity **CICIDS2017**.
- **Capture Window:** Complete Tuesday, July 4, 2017 traffic capture (`GeneratedLabelledFlows/Tuesday-WorkingHours.pcap_ISCX.csv` and `MachineLearningCSV/Tuesday-WorkingHours.pcap_ISCX.csv`).
- **Raw Volume:** 445,909 raw flows; 445,645 clean valid flows.
- **Attack Types Present:**
  - `FTP-Patator`: Automated FTP authentication brute-force attacks (Morning campaign).
  - `SSH-Patator`: Automated SSH authentication brute-force attacks (Afternoon campaign).
  - `BENIGN`: Legitimate background enterprise communications.
- **Network 5-Tuple:** Source IP, Source Port, Destination IP, Destination Port, Protocol, and Timestamp.

---

## 4. Preprocessing Pipeline (`src/preprocessing/`)
- **Cleaning (`cleaner.py`):**
  - Strips leading/trailing whitespace from column headers.
  - Replaces `+Infinity`, `-Infinity` with NaN; imputes with column medians.
  - Removes 264 corrupt/empty records.
  - Drops invariant zero-variance feature columns.
- **Normalization (`scaler.py`):**
  - Robust scaling (`StandardScaler` fitted **strictly on training flows only** to prevent distribution leakage).
  - Scaler parameters saved to `data/processed/scaler.pkl` and applied downstream to validation/test sets.
- **Label Encoding (`encoder.py`):**
  - Multiclass mapping: `{"BENIGN": 0, "FTP-Patator": 1, "SSH-Patator": 2}`.
  - Binary mapping: `{"BENIGN": 0, "ATTACK": 1}`.

---

## 5. Graph Representation
- **Graph Type:** Directed attributed multigraph $G = (V, E, \mathbf{X}, \mathbf{E}_{\text{attr}})$.
- **Vertices ($V$):** Distinct IP addresses ($|V| = 5,728$ endpoints).
- **Edges ($E$):** Directed network flows from Source IP $u$ to Destination IP $v$.
- **Edge Direction:** Unambiguously directed from source client/initiator to destination server/target.

---

## 6. Node Construction (`src/graph/node_builder.py`)
To prevent target label leakage, all node features are **100% label-independent** and computed strictly from topology and traffic volume:

| Feature | Mathematical Definition | Purpose |
| :--- | :--- | :--- |
| `in_degree` | Total incoming flows: $\sum_w \mathbb{I}((w, u) \in E)$ | Identifies servers receiving heavy traffic |
| `out_degree` | Total outgoing flows: $\sum_w \mathbb{I}((u, w) \in E)$ | Identifies proactive communication initiators |
| `total_degree` | $\text{in\_degree} + \text{out\_degree}$ | Total node activity level |
| `degree_ratio` | $\frac{\text{in\_degree}}{\text{total\_degree} + 10^{-6}}$ | Structural orientation ($0 \to$ client, $1 \to$ server) |
| `unique_neighbors` | Distinct communicating IP peers: $|\{w \mid (u,w) \in E \lor (w,u) \in E\}|$ | Replaces leaked `attack_neighbor_ratio` |
| `avg_in_bytes` | Mean `Total Length of Fwd Packets` across inbound flows | Payload volume received |
| `avg_out_bytes` | Mean `Total Length of Bwd Packets` across outbound flows | Payload volume transmitted |
| `avg_flow_duration` | Mean `Flow Duration` of attached flows | Session duration profile |
| `is_attacker_ip` | Structural heuristic: $\mathbb{I}(\text{out\_degree} > 50 \land \text{in\_degree} == 0)$ | Scanners/brute-forcers with zero ingress |
| `is_victim_ip` | Structural heuristic: $\mathbb{I}(\text{in\_degree} > 50 \land \text{out\_degree} == 0)$ | Targeted servers with zero egress |

---

## 7. Edge Construction (`src/graph/edge_builder.py`)
Each directed edge represents an individual communication flow and carries a **77-dimensional continuous feature vector** $\mathbf{e}_{uv} \in \mathbb{R}^{77}$ computed by CICFlowMeter:
- Inter-arrival times (Flow IAT Mean, Std, Max, Min).
- Packet length moments (Mean, Std, Max, Min of Forward and Backward packets).
- TCP flag statistics (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR).
- Window sizes and active/idle timing metrics.
- Subflow volume characteristics.

---

## 8. Adaptive Graph Construction (AGC) (`src/agc/adaptive_graph.py`)
Networks evolve dynamically; static graphs accumulate stale connections. AGC maintains temporal relevance:
- **Sliding Temporal Windows:** Traffic partitioned into 10-minute sliding windows ($W_1, W_2, \dots, W_{29}$).
- **Exponential Moving Average (EMA) Edge Decay:**
  $$w_{uv}^{(t)} = w_{uv}^{(t-1)} \cdot \lambda \quad (\lambda = 0.95)$$
- **Active Reinforcement:**
  $$w_{uv}^{(t)} = w_{uv}^{(t-1)} \cdot \lambda + \log(\text{count}_{uv}^{(t)} + 1)$$
- **Dynamic Pruning:** Edges decaying below $w_{\text{min}} = 0.01$ are excised, discarding inactive communication channels.

---

## 9. GraphSAGE Node Encoder (`src/models/graphsage.py`)
GraphGuard computes structural node embeddings using a 2-layer GraphSAGE architecture with mean aggregation:
- **Input:** Node feature matrix $\mathbf{X} \in \mathbb{R}^{5728 \times 10}$, edge adjacency matrix $\mathcal{E}$.
- **Layer 1:**
  $$\mathbf{h}_v^{(1)} = \sigma\left(\mathbf{W}_1 \cdot \left[\mathbf{x}_v \,\|\, \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{x}_u\right]\right) \quad (10 \to 128)$$
  Followed by $\text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(p=0.30)$.
- **Layer 2:**
  $$\mathbf{h}_v^{(2)} = \sigma\left(\mathbf{W}_2 \cdot \left[\mathbf{h}_v^{(1)} \,\|\, \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(1)}\right]\right) \quad (128 \to 64)$$
  Followed by $\text{BatchNorm1d} \to \text{L2-Normalization} \implies \mathbf{h}_v \in \mathbb{R}^{64}$.

---

## 10. Edge Classifier Head (`src/models/edge_classifier.py`)
To classify an individual flow $e_{uv}$:
1. **Representation Fusion:** The source node embedding $\mathbf{h}_u \in \mathbb{R}^{64}$, destination node embedding $\mathbf{h}_v \in \mathbb{R}^{64}$, and raw edge traffic vector $\mathbf{e}_{uv} \in \mathbb{R}^{77}$ are concatenated:
   $$\mathbf{z}_{uv} = [\mathbf{h}_u \,\|\, \mathbf{h}_v \,\|\, \mathbf{e}_{uv}] \in \mathbb{R}^{205}$$
2. **Multi-Layer Perceptron (MLP):**
   $$\mathbf{z}^{(1)} = \text{Dropout}_{0.3}(\text{ReLU}(\text{BatchNorm}(\mathbf{W}_{\text{dense}1} \mathbf{z}_{uv} + \mathbf{b}_1))) \quad (205 \to 256)$$
   $$\mathbf{z}^{(2)} = \text{Dropout}_{0.3}(\text{ReLU}(\text{BatchNorm}(\mathbf{W}_{\text{dense}2} \mathbf{z}^{(1)} + \mathbf{b}_2))) \quad (256 \to 128)$$
   $$\hat{\mathbf{y}}_{uv} = \mathbf{W}_{\text{out}} \mathbf{z}^{(2)} + \mathbf{b}_{\text{out}} \quad (128 \to C)$$
   Where $C=3$ for Multiclass or $C=2$ for Binary.

---

## 11. Baseline Models (`src/baselines/`)
Trained and evaluated on identical splits using raw 77-dimensional flow features:
1. **Random Forest (`random_forest.py`):**
   - 200 unpruned Decision Trees, balanced class weights, Gini impurity.
2. **XGBoost (`xgboost_model.py`):**
   - 200 estimators, max depth 6, learning rate 0.10, subsample 0.80, colsample 0.80, histogram tree method.

---

## 12. Explainability (XAI) (`src/explainability/explainer.py`)
- **Methodology:** SHAP (SHapley Additive exPlanations) using `TreeExplainer` on a fitted surrogate model over edge feature attributions.
- **Outputs:** Flow-level SHAP attributions and waterfall visualizations identifying specific packet parameters driving the attack alert.

---

## 13. Threat Propagation Score (TPS) (`src/threat/tps.py`)
TPS provides an alert-prioritization ranking in $[0, 1]$ for SOC analysts:
$$\text{TPS}(u) = w_1 \cdot \text{AttProb}(u) + w_2 \cdot \text{SuspNeigh}(u) + w_3 \cdot \text{RecFreq}(u) + w_4 \cdot \text{DegCent}(u) + w_5 \cdot \text{HubScore}(u)$$
- Weights: $0.35, 0.25, 0.20, 0.10, 0.10$.
- Tiers: CRITICAL ($\ge 0.75$), HIGH ($\ge 0.50$), MEDIUM ($\ge 0.25$), LOW ($< 0.25$).
- *Clarification:* TPS is an alert-prioritization mechanism; it does not claim future attack certainty.

---

## 14. Evaluation Methodology
- **Campaign-Window Chronological Split:** Chronological 70% Train, 15% Validation, 15% Testing independently within each attack campaign window.
- **Inductive Protocol with Target-Edge Exclusion:**
  - $G_{\text{train}}$: 155,544 Train edges.
  - $G_{\text{hist}}$: 184,240 Train + Val edges.
  - Test edges are **NEVER** in the message-passing graph; evaluated purely as query edges.
- **Exact-Duplicate Feature Purge:** 4,634 validation duplicates and 3,570 test duplicates appearing in training were excised.

---

## 15. Ablation Methodology
Five decoupled variants evaluated to isolate component contributions:
1. `full_graphguard`: AGC + Graph Context + GraphSAGE.
2. `no_agc`: Static topology without temporal windowing.
3. `no_graph_context`: Edge features only (node embeddings zeroed).
4. `no_adaptive_weights`: Static uniform edge weights.
5. `no_graphsage`: Direct node+edge MLP without message passing.

---

## 16. Software Architecture
```
graph guard/
├── config.yaml               # System parameters and hyperparameters
├── data/
│   ├── raw/                  # CICIDS2017 raw captures
│   └── processed/            # Sanitized parquet splits & graph_data.pt
├── models/                   # PyTorch & pickle model checkpoints
├── results/
│   ├── explanations/         # SHAP JSONs and waterfall plots
│   ├── figures/              # Metric figures & curves
│   ├── metrics/              # CSV/JSON evaluation files
│   ├── predictions/          # Parquet prediction arrays
│   ├── review2_figures/      # 10 presentation-ready 300-DPI charts
│   └── demo/                 # Real demo attack flow JSON
├── scripts/                  # 10 numbered pipeline execution scripts
├── src/                      # Modular Python library
│   ├── agc/                  # Adaptive graph constructor
│   ├── baselines/            # RF & XGBoost models
│   ├── evaluation/           # Metrics & confusion matrices
│   ├── explainability/       # SHAP explainer
│   ├── graph/                # Node/edge builders & stats
│   ├── models/               # PyTorch GraphSAGE & EdgeClassifier
│   ├── preprocessing/        # Cleaner, scaler, encoder, splitter
│   ├── threat/               # TPS calculator
│   └── utils/                # Config, logger, seeds
├── dashboard/                # Streamlit interactive application
└── tests/                    # 15 automated pytest unit tests
```

---

## 17. Implementation Status
- **Pipeline Implementation:** Complete across all 12 planned phases.
- **Testing:** 15/15 unit tests passing in 1.54s.
- **Verification Verdict:** **VERIFIED** — Academically sound, leak-free, and frozen for Review-2 presentation.
