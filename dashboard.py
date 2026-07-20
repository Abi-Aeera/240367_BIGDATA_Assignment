import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BODS Operator Compliance Dashboard", layout="wide")

DB_PATH = "output/bods_processed.db"


@st.cache_data
def load_table(name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


predictions = load_table("predictions")
metrics = load_table("model_metrics")
clusters = load_table("operator_clusters")

full_metrics = metrics[metrics["variant"] == "full (with sub-status flags)"].set_index("model")
conservative_metrics = metrics[metrics["variant"] == "conservative (without sub-status flags)"].set_index("model")
best_row = full_metrics[full_metrics["is_best"] == 1].iloc[0]
best_model_name = best_row.name

# One row per operator: risk prediction joined to its K-Means segment.
predictions = predictions.merge(
    clusters[["org_key", "cluster_id", "cluster_name"]],
    on="org_key", how="left",
)

st.title("🚌 BODS Operator Compliance Benchmarking Dashboard")
st.markdown(
    "**The classification task:** predict whether an operator will be flagged "
    "**`Requires Attention`** in BODS's compliance report — i.e. is this operator "
    "publishing its timetable, fares, and live vehicle location (AVL) feeds "
    "reliably enough that a Transport Authority doesn't need to step in? "
    "Four classifiers were trained and compared on this exact question; the best "
    f"one (**{best_model_name}**) is used for the operator-level predictions below."
)

tab_models, tab_explorer, tab_clusters = st.tabs(
    ["📊 Model Comparison", "🔍 Operator Explorer", "🧩 Operator Segmentation"]
)

# ============================================================ TAB 1: models
with tab_models:
    st.subheader("Classification performance — all 4 models")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_secs"]
    st.dataframe(
        full_metrics[metric_cols].style.format(
            {c: "{:.3f}" for c in metric_cols}
        ).highlight_max(subset=["accuracy", "precision", "recall", "f1", "roc_auc"], color="#c6f6c6"),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Accuracy / Precision / Recall / F1 / ROC-AUC by model**")
        plot_df = full_metrics.reset_index().melt(
            id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
            var_name="metric", value_name="score",
        )
        fig = px.bar(plot_df, x="model", y="score", color="metric", barmode="group")
        fig.update_yaxes(range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Training time per model**")
        fig2 = px.bar(full_metrics.reset_index(), x="model", y="train_secs",
                       labels={"train_secs": "Training time (s)"})
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Leakage check: with vs. without sub-status flag columns")
    st.caption(
        "The sub-status columns (e.g. 'Timetables Published Status') describe "
        "*why* an operator was already judged compliant, so they risk leaking "
        "the label. This compares F1 with those columns included vs. excluded."
    )
    leak_df = pd.DataFrame({
        "F1 (with flags)": full_metrics["f1"],
        "F1 (without flags)": conservative_metrics["f1"],
    })
    leak_df["Drop"] = leak_df["F1 (with flags)"] - leak_df["F1 (without flags)"]
    st.dataframe(leak_df.style.format("{:.3f}"), use_container_width=True)

# ============================================================ TAB 2: explorer
with tab_explorer:
    st.caption(
        f"Predictions below are from the winning classifier — **{best_model_name}** "
        f"(Accuracy {best_row['accuracy']:.3f}, F1 {best_row['f1']:.3f}, "
        f"ROC-AUC {best_row['roc_auc']:.3f})."
    )

    st.sidebar.header("Filters")
    regions = ["All"] + sorted(predictions["Traveline Region"].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Traveline Region", regions)

    reg_statuses = ["All"] + sorted(predictions["Registration Status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Registration Status", reg_statuses)

    cluster_options = ["All"] + sorted(clusters["cluster_name"].dropna().unique().tolist())
    selected_cluster = st.sidebar.selectbox("Operator Segment", cluster_options)

    filtered = predictions.copy()
    if selected_region != "All":
        filtered = filtered[filtered["Traveline Region"] == selected_region]
    if selected_status != "All":
        filtered = filtered[filtered["Registration Status"] == selected_status]
    if selected_cluster != "All":
        filtered = filtered[filtered["cluster_name"] == selected_cluster]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Services shown", f"{len(filtered):,}")
    col2.metric("Predicted 'Requires Attention'", f"{(filtered['prediction'] == 1.0).sum():,}")
    col3.metric("Actual 'Requires Attention'", f"{(filtered['label'] == 1.0).sum():,}")
    match_rate = (filtered["prediction"] == filtered["label"]).mean() if len(filtered) else 0
    col4.metric("Match rate (this view)", f"{match_rate:.1%}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted risk by region")
        region_risk = (
            filtered.groupby("Traveline Region")["risk_probability"]
            .mean().sort_values(ascending=False).reset_index()
        )
        fig1 = px.bar(region_risk, x="Traveline Region", y="risk_probability",
                       labels={"risk_probability": "Avg. predicted risk"})
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.subheader("Risk probability distribution")
        fig2 = px.histogram(filtered, x="risk_probability", nbins=30,
                             color="Requires Attention")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Highest-risk operators")
    operator_risk = (
        filtered.groupby(["Organisation Name", "cluster_name"])
        .agg(
            services=("org_key", "count"),
            avg_risk=("risk_probability", "mean"),
            flagged_count=("prediction", lambda s: (s == 1.0).sum()),
        )
        .sort_values("avg_risk", ascending=False)
        .reset_index()
    )
    st.dataframe(operator_risk, use_container_width=True, height=400)

    st.divider()
    st.subheader("Service-level detail")
    st.dataframe(
        filtered[["Organisation Name", "cluster_name", "Traveline Region", "Registration Status",
                  "Requires Attention", "prediction", "risk_probability"]]
        .sort_values("risk_probability", ascending=False),
        use_container_width=True, height=400,
    )

# ============================================================ TAB 3: clusters
with tab_clusters:
    st.markdown(
        "**The clustering task:** a separate, *unsupervised* K-Means step that groups "
        "operators by scale and publishing behaviour — service volume, published ratio, "
        "AVL match score, and route count — rather than by predicted compliance risk. "
        "It complements the classification model above by showing whether risk is spread "
        "evenly across operator types or concentrated in a particular kind of operator."
    )

    seg_left, seg_right = st.columns(2)
    with seg_left:
        st.subheader("Segment sizes")
        cluster_counts = clusters["cluster_name"].value_counts().reset_index()
        cluster_counts.columns = ["cluster_name", "operators"]
        fig3 = px.bar(cluster_counts, x="cluster_name", y="operators",
                       labels={"cluster_name": "Segment", "operators": "Operators"})
        st.plotly_chart(fig3, use_container_width=True)

    with seg_right:
        st.subheader("Segment profile")
        fig4 = px.scatter(
            clusters, x="op_published_ratio", y="op_avg_avl_score",
            color="cluster_name", size="op_total_rows",
            hover_data=["Operator", "op_distinct_lines"],
            labels={"op_published_ratio": "Published ratio", "op_avg_avl_score": "Avg. AVL match score"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Does compliance risk concentrate in a particular segment?")
    risk_by_cluster = (
        predictions.groupby("cluster_name")
        .agg(
            operators=("org_key", "nunique"),
            avg_risk=("risk_probability", "mean"),
            flagged_rate=("label", "mean"),
        )
        .sort_values("avg_risk", ascending=False)
        .reset_index()
    )
    st.dataframe(
        risk_by_cluster.style.format({"avg_risk": "{:.3f}", "flagged_rate": "{:.1%}"}),
        use_container_width=True,
    )

    st.divider()
    st.subheader("All operators by segment")
    st.dataframe(
        clusters[["Operator", "cluster_name", "op_total_rows", "op_published_ratio",
                  "op_avg_avl_score", "op_distinct_lines"]]
        .sort_values(["cluster_name", "op_total_rows"], ascending=[True, False]),
        use_container_width=True, height=400,
    )
