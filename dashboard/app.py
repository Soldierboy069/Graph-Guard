"""
GraphGuard — Streamlit Dashboard
=================================
Main entry point. Reads all pipeline outputs and renders 7-page dashboard.

Run:
    streamlit run dashboard/app.py

All data shown is read from actual pipeline output files.
No hardcoded metrics, no placeholder charts.
"""
import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="GraphGuard — Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.config import load_config, resolve_path, get_project_root


@st.cache_resource
def get_config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


cfg = get_config()
root = get_project_root(cfg)

# ─── Sidebar Navigation ────────────────────────────────────────────────────
st.sidebar.title("🛡️ GraphGuard")
st.sidebar.caption("Explainable GNN for Intrusion Detection")
st.sidebar.markdown("---")

PAGES = {
    "📊 Overview": "overview",
    "🌐 Network Graph": "network_graph",
    "🔍 Intrusion Detection": "intrusion_detection",
    "💡 Explainability": "explainability",
    "⚠️ Threat Propagation": "threat_propagation",
    "📈 Model Comparison": "model_comparison",
    "🔬 Ablation Study": "ablation",
}

selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
page = PAGES[selection]

st.sidebar.markdown("---")
st.sidebar.caption(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")
st.sidebar.caption(f"Dataset: CICIDS2017")

# ─── Load shared data ──────────────────────────────────────────────────────

@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_parquet(path):
    import pandas as pd
    return pd.read_parquet(path)

@st.cache_data
def load_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def file_exists(path):
    return Path(path).exists()


# ─── Helper: missing data notice ───────────────────────────────────────────
def show_missing(component: str, script: str):
    st.warning(
        f"**{component}** data not found. "
        f"Run `python scripts/{script}` to generate it.",
        icon="⚠️"
    )


if page == "overview":
    st.title("📊 GraphGuard — Project Overview")

    stats_path = root / "results" / "graphs" / "graph_stats.json"
    preproc_path = root / "data" / "processed" / "preprocessing_stats.json"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset")
        if file_exists(preproc_path):
            preproc = load_json(str(preproc_path))
            st.metric("Total Flows (raw)", f"{preproc['total_rows_raw']:,}")
            st.metric("Flows (after cleaning)", f"{preproc['total_rows_after_cleaning']:,}")
            st.metric("Train / Val / Test",
                      f"{preproc['split']['train']:,} / {preproc['split']['val']:,} / {preproc['split']['test']:,}")
            st.metric("Mode", preproc.get("mode", "unknown").upper())
        else:
            show_missing("Preprocessing stats", "01_prepare_data.py")

    with col2:
        st.subheader("Graph")
        if file_exists(stats_path):
            gstats = load_json(str(stats_path))
            st.metric("Unique IP Nodes", f"{gstats['num_nodes']:,}")
            st.metric("Total Edges (Flows)", f"{gstats['num_edges']:,}")
            st.metric("Benign Flows", f"{gstats['benign_edges']:,}")
            st.metric("Attack Flows", f"{gstats['attack_edges']:,}")
        else:
            show_missing("Graph statistics", "02_build_graph.py")

    st.markdown("---")
    st.subheader("Attack Class Distribution")
    if file_exists(stats_path):
        gstats = load_json(str(stats_path))
        import pandas as pd
        import plotly.express as px
        dist = gstats["label_distribution"]
        df_dist = pd.DataFrame(list(dist.items()), columns=["Class", "Count"])
        df_dist = df_dist.sort_values("Count", ascending=False)
        fig = px.bar(df_dist, x="Class", y="Count", color="Class",
                     title="Edge (Flow) Label Distribution",
                     labels={"Count": "Number of Flows"})
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing("Graph statistics", "02_build_graph.py")

    st.markdown("---")
    st.subheader("Pipeline Status")
    pipeline_files = {
        "✅ Preprocessing": root / "data" / "processed" / "train.parquet",
        "✅ Graph Construction": root / "results" / "graphs" / "graph_stats.json",
        "✅ AGC": root / "results" / "graphs" / "agc_summary.csv",
        "✅ GraphGuard Training": root / "models" / "full_graphguard_best.pt",
        "✅ Baselines": root / "models" / "rf_model.pkl",
        "✅ Evaluation": root / "results" / "metrics" / "model_comparison.csv",
        "✅ Explainability": root / "results" / "explanations" / "flow_0_explanation.json",
        "✅ TPS": root / "results" / "metrics" / "tps_scores.csv",
        "✅ Ablation": root / "results" / "metrics" / "ablation.csv",
    }
    for label, fpath in pipeline_files.items():
        icon = "✅" if file_exists(fpath) else "⏳"
        label_clean = label.lstrip("✅").strip()
        st.write(f"{icon} {label_clean}: `{'Ready' if file_exists(fpath) else 'Pending'}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "network_graph":
    st.title("🌐 Network Communication Graph")

    meta_path = root / "results" / "graphs" / "flow_metadata.parquet"
    tps_path = root / "results" / "metrics" / "tps_scores.csv"

    if not file_exists(meta_path):
        show_missing("Flow metadata", "02_build_graph.py")
        st.stop()

    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np

    meta = load_parquet(str(meta_path))
    tps_df = load_csv(str(tps_path)) if file_exists(tps_path) else pd.DataFrame()

    st.sidebar.subheader("Graph Filters")
    show_attack_only = st.sidebar.checkbox("Show attack flows only", value=False)
    max_nodes = st.sidebar.slider("Max nodes to display", 20, 500, 100)
    selected_threat = st.sidebar.multiselect("Filter by threat level",
                                             ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                             default=["CRITICAL", "HIGH"])

    # Subsample for display
    display_meta = meta.copy()
    if show_attack_only and "Label" in display_meta.columns:
        display_meta = display_meta[display_meta["Label"] != "BENIGN"]

    # Get top-TPS nodes
    if len(tps_df) > 0 and selected_threat:
        top_ips = set(tps_df[tps_df["threat_level"].isin(selected_threat)]["ip"].values)
        if top_ips:
            filtered = display_meta[
                display_meta["Source IP"].isin(top_ips) |
                display_meta["Destination IP"].isin(top_ips)
            ]
            if len(filtered) > 0:
                display_meta = filtered

    display_meta = display_meta.head(5000)

    # Build unique nodes
    all_ips = list(set(display_meta["Source IP"].tolist() + display_meta["Destination IP"].tolist()))[:max_nodes]
    ip_pos = {ip: i for i, ip in enumerate(all_ips)}

    # Assign positions using a circle layout
    angles = np.linspace(0, 2 * np.pi, len(all_ips), endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Build edges
    edge_x, edge_y = [], []
    edge_colors = []
    for _, row in display_meta.iterrows():
        src, dst = row.get("Source IP", ""), row.get("Destination IP", "")
        if src in ip_pos and dst in ip_pos:
            si, di = ip_pos[src], ip_pos[dst]
            edge_x += [x_pos[si], x_pos[di], None]
            edge_y += [y_pos[si], y_pos[di], None]

    # Node colors from TPS
    tps_map = {}
    if len(tps_df) > 0:
        tps_map = dict(zip(tps_df["ip"], tps_df["tps"]))
    node_colors = [tps_map.get(ip, 0.0) for ip in all_ips]
    node_sizes = [max(8, 20 * tps_map.get(ip, 0.01)) for ip in all_ips]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#aaa"),
        hoverinfo="none", name="Flows"
    ))
    fig.add_trace(go.Scatter(
        x=x_pos[:len(all_ips)], y=y_pos[:len(all_ips)],
        mode="markers",
        marker=dict(size=node_sizes, color=node_colors, colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="TPS"),
                    cmin=0, cmax=1),
        text=all_ips,
        hovertemplate="<b>%{text}</b><br>TPS: %{marker.color:.3f}<extra></extra>",
        name="IP Nodes"
    ))
    fig.update_layout(
        title=f"Network Graph (showing {len(all_ips)} nodes)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Node detail on selection
    st.subheader("Node Details")
    if len(tps_df) > 0:
        selected_ip = st.selectbox("Select an IP to inspect", tps_df["ip"].head(50).tolist())
        if selected_ip:
            node_row = tps_df[tps_df["ip"] == selected_ip]
            if len(node_row) > 0:
                row = node_row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TPS", f"{row['tps']:.4f}")
                c2.metric("Threat Level", row["threat_level"])
                c3.metric("Out Degree", row["out_degree"])
                c4.metric("In Degree", row["in_degree"])

                c5, c6 = st.columns(2)
                c5.metric("Attack Probability", f"{row['attack_probability']:.4f}")
                c6.metric("Suspicious Neighbor Ratio", f"{row['suspicious_neighbor_ratio']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INTRUSION DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "intrusion_detection":
    st.title("🔍 Intrusion Detection — Flow Inspector")

    pred_path = root / "results" / "predictions" / "full_graphguard_test_predictions.parquet"
    tps_path = root / "results" / "metrics" / "tps_scores.csv"
    enc_path = root / "data" / "processed" / "label_encoder.json"

    if not file_exists(pred_path):
        show_missing("Test predictions", "04_train_graphguard.py")
        st.stop()

    import pandas as pd

    pred_df = load_parquet(str(pred_path))
    tps_df = load_csv(str(tps_path)) if file_exists(tps_path) else pd.DataFrame()
    label_enc = load_json(str(enc_path)) if file_exists(enc_path) else {}
    int_to_label = {v: k for k, v in label_enc.items()}

    pred_df["predicted_label_str"] = pred_df["predicted_label"].map(int_to_label)
    pred_df["true_label_str"] = pred_df["true_label"].map(int_to_label)

    st.subheader("Flow Search")
    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search_ip = st.text_input("Search by IP address (Source or Destination)", "")
    with filter_col:
        filter_pred = st.selectbox("Filter by prediction", ["All"] + sorted(pred_df["predicted_label_str"].dropna().unique().tolist()))

    display_df = pred_df.copy()
    if search_ip:
        mask = pd.Series([False] * len(display_df))
        for col in ["Source IP", "Destination IP"]:
            if col in display_df.columns:
                mask |= display_df[col].astype(str).str.contains(search_ip, na=False)
        display_df = display_df[mask]

    if filter_pred != "All":
        display_df = display_df[display_df["predicted_label_str"] == filter_pred]

    st.write(f"Showing {min(100, len(display_df))} of {len(display_df)} flows")

    # Select columns to show
    show_cols = [c for c in ["Source IP", "Source Port", "Destination IP", "Destination Port",
                              "Protocol", "predicted_label_str", "true_label_str"] if c in display_df.columns]
    st.dataframe(display_df[show_cols].head(100), use_container_width=True)

    # Flow detail
    st.markdown("---")
    st.subheader("Flow Detail")
    idx = st.number_input("Flow index (row number)", 0, len(pred_df) - 1, 0)
    if idx < len(pred_df):
        flow = pred_df.iloc[idx]
        c1, c2, c3 = st.columns(3)
        c1.metric("Source IP", str(flow.get("Source IP", "—")))
        c2.metric("Destination IP", str(flow.get("Destination IP", "—")))
        c3.metric("Protocol", str(flow.get("Protocol", "—")))

        c4, c5, c6 = st.columns(3)
        c4.metric("Prediction", str(flow.get("predicted_label_str", "—")))
        c5.metric("True Label", str(flow.get("true_label_str", "—")))

        src_ip = str(flow.get("Source IP", ""))
        if len(tps_df) > 0 and src_ip in tps_df["ip"].values:
            tps_row = tps_df[tps_df["ip"] == src_ip].iloc[0]
            c6.metric("TPS (Source IP)", f"{tps_row['tps']:.4f}")
        else:
            c6.metric("TPS (Source IP)", "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "explainability":
    st.title("💡 Explainability — Why was this flow flagged?")

    expl_dir = root / "results" / "explanations"
    top_expl_path = expl_dir / "top_20_flow_explanations.json"

    if not file_exists(top_expl_path):
        show_missing("Flow explanations", "07_run_xai.py")
        st.stop()

    import pandas as pd
    import plotly.express as px

    explanations = load_json(str(top_expl_path))
    enc_path = root / "data" / "processed" / "label_encoder.json"
    label_enc = load_json(str(enc_path)) if file_exists(enc_path) else {}
    int_to_label = {v: k for k, v in label_enc.items()}

    st.write(f"Showing explanations for **{len(explanations)}** suspicious flows.")

    flow_labels = [
        f"Flow {i}: Predicted={exp['predicted_label']} | True={int_to_label.get(exp.get('true_label', -1), '?')}"
        for i, exp in enumerate(explanations)
    ]
    selected_flow_idx = st.selectbox("Select a flow to explain", range(len(explanations)), format_func=lambda i: flow_labels[i])
    exp = explanations[selected_flow_idx]

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Class", exp["predicted_label"])
    c2.metric("True Label", int_to_label.get(exp.get("true_label", -1), "—"))

    for meta_key in ["Source IP", "Destination IP", "Protocol", "Flow ID"]:
        if meta_key in exp:
            st.write(f"**{meta_key}:** `{exp[meta_key]}`")

    st.subheader("Top Contributing Features (SHAP)")
    top_feats = exp["top_features"]
    feat_df = pd.DataFrame(top_feats)[["feature", "shap_value", "feature_value"]].head(15)
    feat_df["direction"] = feat_df["shap_value"].apply(lambda v: "Increases Attack Risk" if v > 0 else "Reduces Attack Risk")

    fig = px.bar(
        feat_df, x="shap_value", y="feature", orientation="h",
        color="direction",
        color_discrete_map={"Increases Attack Risk": "#d73027", "Reduces Attack Risk": "#4575b4"},
        title="SHAP Feature Attribution — Impact on Prediction",
        labels={"shap_value": "SHAP Value", "feature": "Traffic Feature"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw feature values"):
        st.dataframe(feat_df, use_container_width=True)

    # Show saved plot if available
    plot_path = expl_dir / f"flow_{selected_flow_idx}_shap.png"
    if file_exists(plot_path):
        st.image(str(plot_path), caption="SHAP Explanation Plot")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THREAT PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "threat_propagation":
    st.title("⚠️ Threat Propagation Score (TPS)")

    tps_path = root / "results" / "metrics" / "tps_scores.csv"
    summary_path = root / "results" / "metrics" / "tps_summary.json"

    if not file_exists(tps_path):
        show_missing("TPS scores", "08_calculate_tps.py")
        st.stop()

    import pandas as pd
    import plotly.express as px

    tps_df = load_csv(str(tps_path))
    summary = load_json(str(summary_path)) if file_exists(summary_path) else {}

    # Summary metrics
    st.subheader("Threat Level Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Critical", summary.get("critical_nodes", 0))
    c2.metric("🟠 High", summary.get("high_nodes", 0))
    c3.metric("🟡 Medium", summary.get("medium_nodes", 0))
    c4.metric("🟢 Low", summary.get("low_nodes", 0))

    st.markdown("---")

    # TPS distribution
    st.subheader("TPS Distribution")
    fig_hist = px.histogram(tps_df, x="tps", color="threat_level",
                             nbins=50, title="TPS Score Distribution",
                             color_discrete_map={"CRITICAL": "red", "HIGH": "orange",
                                                 "MEDIUM": "gold", "LOW": "green"})
    st.plotly_chart(fig_hist, use_container_width=True)

    # Top suspicious nodes
    st.subheader("Top Suspicious Nodes")
    top_n = st.slider("Show top N nodes", 5, 50, 20)
    top_tps = tps_df.head(top_n)

    fig_top = px.bar(top_tps, x="ip", y="tps", color="threat_level",
                     color_discrete_map={"CRITICAL": "red", "HIGH": "orange",
                                         "MEDIUM": "gold", "LOW": "green"},
                     title=f"Top {top_n} Nodes by TPS",
                     labels={"ip": "IP Address", "tps": "Threat Propagation Score"})
    fig_top.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_top, use_container_width=True)

    # Node detail
    st.markdown("---")
    st.subheader("Node Detail")
    selected_ip = st.selectbox("Select IP", tps_df["ip"].head(100).tolist())
    if selected_ip:
        row = tps_df[tps_df["ip"] == selected_ip].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("TPS", f"{row['tps']:.4f}")
        c2.metric("Threat Level", row["threat_level"])
        c3.metric("Attack Probability", f"{row['attack_probability']:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Suspicious Neighbors", f"{row['suspicious_neighbor_ratio']:.4f}")
        c5.metric("Recent Attack Freq", f"{row['recent_attack_frequency']:.4f}")
        c6.metric("Degree Centrality", f"{row['degree_centrality']:.4f}")

    # Formula explanation
    st.markdown("---")
    st.subheader("TPS Formula")
    weights = summary.get("formula_weights", {})
    st.latex(
        r"TPS = w_1 \cdot P_{attack} + w_2 \cdot R_{neighbor} + w_3 \cdot F_{recent} + w_4 \cdot D_{centrality} + w_5 \cdot B_{proxy}"
    )
    st.write("**Component weights:**")
    for k, v in weights.items():
        st.write(f"- `{k}`: **{v}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "model_comparison":
    st.title("📈 Model Comparison")

    comp_path = root / "results" / "metrics" / "model_comparison.csv"
    figures_dir = root / "results" / "figures"

    if not file_exists(comp_path):
        show_missing("Model comparison", "06_evaluate.py")
        st.stop()

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    comp_df = load_csv(str(comp_path))
    st.subheader("Performance Summary")
    st.dataframe(comp_df.set_index("Model"), use_container_width=True)

    # Bar chart comparison
    metrics_to_plot = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision (Macro)", "Recall (Macro)"]
    metrics_to_plot = [m for m in metrics_to_plot if m in comp_df.columns]

    if metrics_to_plot:
        import plotly.graph_objects as go
        fig = go.Figure()
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric, x=comp_df["Model"], y=comp_df[metric],
                text=comp_df[metric].round(4), textposition="auto",
            ))
        fig.update_layout(barmode="group", title="Model Performance Comparison",
                          yaxis=dict(range=[0, 1.05]), height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    for col, (model_name, fname) in zip(
        [cm_col1, cm_col2, cm_col3],
        [("GraphGuard", "graphguard_confusion_matrix.png"),
         ("Random Forest", "rf_confusion_matrix.png"),
         ("XGBoost", "xgb_confusion_matrix.png")]
    ):
        cm_path = figures_dir / fname
        if file_exists(cm_path):
            col.image(str(cm_path), caption=model_name, use_container_width=True)
        else:
            col.write(f"{model_name}: Not available")

    # Training curves
    st.markdown("---")
    st.subheader("GraphGuard Training Curves")
    curve_path = figures_dir / "graphguard_training_curves.png"
    if file_exists(curve_path):
        st.image(str(curve_path), use_container_width=True)
    else:
        st.info("Training curve not available. Run 06_evaluate.py after training.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ablation":
    st.title("🔬 Ablation Study")
    st.caption("Comparing contribution of each GraphGuard component using actual experiment results.")

    ablation_path = root / "results" / "metrics" / "ablation.csv"
    ablation_fig_path = root / "results" / "figures" / "ablation_comparison.png"

    if not file_exists(ablation_path):
        show_missing("Ablation results", "09_run_ablation.py")
        st.stop()

    import pandas as pd
    import plotly.express as px

    ablation_df = load_csv(str(ablation_path))

    st.subheader("Ablation Results Table")
    st.dataframe(ablation_df, use_container_width=True)

    st.subheader("F1 Score Comparison")
    if file_exists(ablation_fig_path):
        st.image(str(ablation_fig_path), use_container_width=True)

    valid = ablation_df.dropna(subset=["f1_macro"])
    if len(valid) > 0:
        fig = px.bar(valid, x="experiment", y="f1_macro", color="experiment",
                     text="f1_macro",
                     title="Ablation Study — F1 Macro by Experiment",
                     labels={"experiment": "Experiment", "f1_macro": "F1 (Macro)"})
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis=dict(range=[0, 1.05]), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Experiment Descriptions")
    cfg_ablation = cfg.get("ablation", {}).get("experiments", [])
    for exp in cfg_ablation:
        st.write(f"**{exp['name']}**: {exp['description']}")
