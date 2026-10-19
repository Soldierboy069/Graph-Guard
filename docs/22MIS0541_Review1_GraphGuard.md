# GraphGuard: An Explainable Graph Neural Network for Intelligent Intrusion Detection

**Reg.No:** 22MIS0541 – PARAMESHWAR P  
**Guide:** Dr. PRABHU J, Professor Grade 1  
**Review:** REVIEW – 1  

---

## ABSTRACT
GraphGuard is an explainable Graph Neural Network (GNN)-based framework designed to improve intrusion detection in modern enterprise, cloud, and Internet of Things (IoT) networks. Unlike conventional Intrusion Detection Systems (IDS) that primarily analyse network traffic as independent records, GraphGuard models communication as a graph, where network devices are represented as nodes and communication flows as edges. This enables the system to capture both traffic characteristics and relationships between network entities, helping identify coordinated and evolving cyber attacks. The framework introduces an Adaptive Graph Construction (AGC) mechanism to dynamically update network relationships according to changing communication behaviour. Explainable Artificial Intelligence (XAI) techniques are incorporated to provide understandable explanations for detected threats, improving the transparency and trustworthiness of the detection process. A proposed Threat Propagation Score (TPS) estimates the potential severity and spread of detected attacks, supporting effective alert prioritization. GraphGuard will be evaluated using benchmark intrusion detection datasets, including CICIDS2017, and compared with conventional machine-learning and graph-based approaches.

**Keywords:** Graph Neural Networks, Intrusion Detection System, GraphSAGE, Adaptive Graph Construction, Explainable AI, Threat Propagation Score, CICIDS2017

---

## [1] INTRODUCTION
Network intrusion detection is a fundamental security requirement in modern enterprise, cloud, and Internet of Things (IoT) environments. Conventional Intrusion Detection Systems (IDS) predominantly analyse individual network flow records using signature-based rules or machine-learning classifiers, focusing primarily on traffic-level features while treating each flow as an independent observation. This approach misses important relational information about how hosts communicate over time and fails to capture coordinated or multi-stage attack patterns.

Graph Neural Networks (GNNs) have recently emerged as a powerful method for learning from graph-structured data, making them well-suited to network intrusion detection where the relationships between communicating entities carry important security signals. GraphGuard proposes an end-to-end GNN-based intrusion detection framework that models network traffic as a dynamic graph, applies Adaptive Graph Construction (AGC), produces inductive node embeddings using GraphSAGE, and classifies communication flows using an MLP edge classifier. The system further provides SHAP-based explainability and a Threat Propagation Score (TPS) for analyst-friendly alert prioritization.

---

## [2] PROBLEM STATEMENT
Existing network intrusion detection approaches face several important limitations:
1. **Flow independence:** Classical ML and deep-learning IDS treat each network flow as isolated, ignoring relationships between communicating entities and structural network patterns.
2. **Static graph representations:** Early GNN-based IDS construct fixed graphs that do not adapt to changing network communication behaviour or new/disappearing connections.
3. **Limited explainability:** Most GNN-based IDS provide predictions without clear forensic justification, reducing analyst trust and operational usefulness.
4. **Absence of threat propagation awareness:** Existing IDS do not provide explicit scores to estimate the potential severity or spread of a detected threat across connected network entities.

GraphGuard addresses all four limitations through adaptive graph construction, inductive GNN learning, integrated SHAP explainability, and a Threat Propagation Score.

---

## [3] OBJECTIVES
* Design a GNN-based NIDS that models network traffic as a directed communication graph.
* Implement Adaptive Graph Construction (AGC) to dynamically update graph relationships over time.
* Apply GraphSAGE inductive learning to generate topology-aware node embeddings.
* Train an MLP edge classifier using combined graph embeddings and flow traffic features.
* Provide SHAP-based explainability to identify important features, nodes, and relationships per prediction.
* Design and compute a Threat Propagation Score (TPS) for alert prioritization.
* Evaluate against conventional ML and GNN-based IDS baselines on CICIDS2017.
* Conduct component-wise ablation to quantify each component's contribution.

---

## [4] SCOPE OF THE PROJECT
GraphGuard covers: graph-based network intrusion detection, adaptive graph construction over temporal windows, inductive GNN-based classification, SHAP explainability, Threat Propagation Score, and an interactive Streamlit demonstration dashboard. The system operates on CICIDS2017 network-flow data. Future work may include live packet ingestion, real-time streaming, and deployment on real network infrastructure.

---

## [5] PROPOSED SYSTEM
1. **Data Preprocessing:** CICIDS2017 flow records are loaded, cleaned, normalized, and split chronologically.
2. **Network Graph Construction:** Nodes = IP endpoints. Edges = communication flows. Edge features = CICFlowMeter statistics. Node features = label-independent structural/traffic aggregates.
3. **Adaptive Graph Construction (AGC):** Dynamically updates the graph: new nodes/edges are added, active relationships are reinforced, edge weights decay exponentially, stale edges are pruned.
4. **GraphSAGE Encoder:** Two-layer inductive GNN that produces 64-dim node embeddings from graph structure without requiring test nodes during training.
5. **MLP Edge Classifier:** Concatenates source embedding + destination embedding + edge features -> MLP -> BENIGN / FTP-Patator / SSH-Patator classification.
6. **SHAP Explainability (XAI):** Provides forensic attribution for selected predictions by identifying the most important contributing traffic features.
7. **Threat Propagation Score (TPS):** Propagates detection probabilities across the graph to rank IP nodes by suspicion for analyst prioritization.
8. **Evaluation & Ablation:** Benchmarked against RF and XGBoost baselines with 5-component ablation study.

---

## [6] LITERATURE SURVEY
*(Please refer to the full DOCX file for the detailed 15-paper literature survey table).*

### [6.1] FINDINGS IN LITERATURE SURVEY
The literature survey indicates that network intrusion detection has been extensively studied using machine learning, deep learning, GNNs, dynamic graph learning, self-supervised learning, and XAI. The following key gaps were identified:
1. **Limited adaptive graph construction:** AGC based on continuously evolving communication behaviour is not sufficiently integrated into a unified IDS framework.
2. **Insufficient integration of traffic and structural relationships:** A unified approach considering both traffic features and changing structural relationships is needed.
3. **Limited explainability in GNN-based IDS:** Explainability is not consistently integrated with adaptive GNN-based detection.
4. **Lack of threat propagation and alert prioritization:** Existing approaches do not provide an explicit TPS to estimate severity or spread of detected threats.
5. **Challenges in adapting to evolving attacks:** Changing network patterns require frameworks that can adapt their graph representation.
6. **Need for an integrated and interpretable framework:** AGC + relationship-aware detection + explainability + TPS have not been combined into one system.

---

## [7] METHODOLOGY
1. **Literature Review and Baseline Selection:** Review existing GNN, dynamic graph, and explainable IDS approaches. Select RF and XGBoost baselines.
2. **Dataset Acquisition and Preprocessing:** Clean CICIDS2017, handle missing values, normalize, and divide into train/val/test using chronological splits.
3. **Network Graph Construction:** Transform flows into graphs. Nodes = IP endpoints, Edges = communication flows.
4. **Adaptive Graph Construction (AGC):** Update graphs over temporal windows with weight decay and pruning.
5. **GNN-Based Intrusion Detection:** GraphSAGE learns representations from node/edge features and graph structure.
6. **Model Training and Attack Classification:** Train the GNN/MLP models on the training data.
7. **Explainable AI Analysis:** SHAP identifies important features for forensic transparency.
8. **Threat Propagation Score Calculation:** Estimate severity of detected threats to prioritize alerts.
9. **Evaluation and Benchmarking:** Evaluate using Macro F1, recall, FPR, and compare against baselines.
10. **Ablation and Explainability Analysis:** Component-wise ablation (AGC, graph context, GraphSAGE).

---

## [8] SOFTWARE REQUIREMENTS

**Functional Requirements:**
* FR1: Load and preprocess CICIDS2017 dataset.
* FR2: Construct a graph representation of network traffic.
* FR3: Implement Adaptive Graph Construction (AGC).
* FR4: Train a GNN to learn structural relationships and classify network activities.
* FR5: Implement an XAI module to identify important traffic features.
* FR6: Calculate a Threat Propagation Score (TPS) for alert prioritization.
* FR7: Evaluate and benchmark against conventional ML baselines.

**Non-Functional Requirements:**
* NFR1: Accuracy & Reliability
* NFR2: Performance & Efficiency
* NFR3: Scalability
* NFR4: Adaptability
* NFR5: Explainability & Usability
* NFR6: Maintainability & Extensibility
* NFR7: Reproducibility & Security

---

## [9] SYSTEM ARCHITECTURE
CICIDS2017 -> Preprocessing -> Chronological split -> Graph construction -> AGC -> GraphSAGE -> Node embeddings -> [Source + Destination embeddings + Edge features] -> MLP Classifier -> Prediction -> SHAP XAI + TPS -> Dashboard. 

*(See `results/review2_figures/01_system_architecture.png` for diagram).*

---

## [10] UML DIAGRAM
*(See `results/uml/use_case_diagram.png` and `results/uml/class_diagram.png` for full diagrams).*

---

## [11] IMPLEMENTATION

### 11.1 PROJECT PLAN
* Phase 1: Literature review and baseline selection.
* Phase 2: Dataset acquisition (CICIDS2017) and preprocessing pipeline.
* Phase 3: Network graph construction module.
* Phase 4: Adaptive Graph Construction (AGC) implementation.
* Phase 5: GraphSAGE encoder training.
* Phase 6: MLP edge classifier training and evaluation.
* Phase 7: Independent Random Forest and XGBoost baseline training.
* Phase 8: SHAP explainability module integration.
* Phase 9: Threat Propagation Score (TPS) implementation.
* Phase 10: 5-component ablation study.
* Phase 11: Streamlit demonstration dashboard development.
* Phase 12: Documentation and Review preparation.

### 11.2 SAMPLE CODE

**GraphSAGE Encoder (`src/models/graphsage.py`):**
```python
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128,
                 embedding_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [in_channels]+[hidden_dim]*(num_layers-1)+[embedding_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
```

**MLP Edge Classifier (`src/models/edge_classifier.py`):**
```python
class EdgeClassifier(nn.Module):
    # input = concat(h_src, h_dst, edge_features)
    def __init__(self, node_embed_dim, edge_feat_dim,
                 num_classes, hidden_dims=[256,128], dropout=0.3):
        super().__init__()
        in_dim = 2 * node_embed_dim + edge_feat_dim  # 205-dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)
```

### 11.3 TESTING STRATEGIES
* **Unit Tests:** 15/15 pytest unit tests passing — covers preprocessing, graph construction, AGC, model shape validation, prediction pipeline, and dashboard.
* **Data Leakage Audit:** Verified no feature leakage across train/val/test splits. `attack_neighbor_ratio` permanently removed as a label-leaking feature.
* **Model Integrity:** Confirmed GraphSAGE and MLP edge classifier are the GraphGuard components; RF and XGBoost are independent baselines.
* **Ablation Testing:** 5-component ablation (Full, No AGC, No Graph Context, No Adaptive Weights, No GraphSAGE) to quantify each component's contribution.
* **Integration Testing:** End-to-end pipeline from raw CICIDS2017 CSV to dashboard output verified.

### 11.4 SAMPLE SCREENSHOTS (From Review-2)

**Screenshot 1: Dataset Overview**
![Overview](../../results/dashboard_screenshots/ss_01_overview.png)

**Screenshot 2: Intrusion Detection**
![Detection](../../results/dashboard_screenshots/ss_02_intrusion_detection.png)

**Screenshot 3: SHAP Explainability**
![SHAP](../../results/dashboard_screenshots/ss_04_explainability.png)

**Screenshot 4: Threat Propagation Score (TPS)**
![TPS](../../results/dashboard_screenshots/ss_06_threat_propagation.png)

**Screenshot 5: Model Comparison**
![Comparison](../../results/dashboard_screenshots/ss_03_model_comparison.png)

**Screenshot 6: Ablation Study**
![Ablation](../../results/dashboard_screenshots/ss_05_ablation.png)

---

## [12] SUMMARY
GraphGuard is an explainable Graph Neural Network (GNN)-based framework designed to improve intrusion detection in modern networks. Unlike conventional IDS that treat each network flow independently, GraphGuard models communication as a directed graph where network devices are nodes and communication flows are edges. Adaptive Graph Construction (AGC) dynamically updates the graph over temporal windows. GraphSAGE computes inductive node embeddings, which combined with edge traffic features are passed through an MLP edge classifier for intrusion prediction. SHAP-based XAI provides forensic attribution for predictions, and a Threat Propagation Score (TPS) prioritizes suspicious IP nodes for analysts.

GraphGuard is evaluated on CICIDS2017 under a strict campaign-window chronological split with verified attack flows in the held-out test set. Results: Macro F1 97.87% (multiclass), 100% attack recall, Macro FPR 0.04%. Ablation confirms graph context contributes 26.58 percentage points of Macro F1. Future work includes live packet ingestion, real-time streaming, and broader attack families.

---

*(References are detailed in the full Review DOCX file).*
