# streamlit_model_gallery.py
# Run with: streamlit run streamlit_model_gallery.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Enterprise Model Gallery", layout="wide")

# -----------------------------
# Sample / Load Data
# -----------------------------
@st.cache_data

def load_data():
    try:
        df = pd.read_csv("model_registry_v2_enterprise_schema.csv")
    except:
        # fallback demo dataset
        df = pd.DataFrame({
            "model_name": ["Credit Risk PD", "Demand Forecasting", "Fraud Detection"],
            "domain": ["Banking", "Retail", "Payments"],
            "model_stage": ["prod", "canary", "shadow"],
            "owner_team": ["Risk AI", "Supply Chain DS", "Trust & Safety"],
            "last_retrained_date": ["2025-11-01", "2025-09-12", "2025-12-20"],
            "sla_tier": ["Tier-1", "Tier-2", "Tier-1"],
            "monitoring_status": ["Healthy", "Drift detected", "Healthy"],
            "approval_status": ["Approved", "Pending", "Approved"],
            "inference_endpoint_id": ["ep_112", "ep_231", "ep_443"],
            "feature_store_dependency": ["customer_fs_v2", "demand_fs_v1", "fraud_graph_fs"]
        })
    return df


df = load_data()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("🔎 Discover Models")

domain_filter = st.sidebar.multiselect("Domain", options=df["domain"].unique(), default=df["domain"].unique())
stage_filter = st.sidebar.multiselect("Lifecycle Stage", options=df["model_stage"].unique(), default=df["model_stage"].unique())
sla_filter = st.sidebar.multiselect("SLA Tier", options=df["sla_tier"].unique(), default=df["sla_tier"].unique())

filtered_df = df[
    (df["domain"].isin(domain_filter)) &
    (df["model_stage"].isin(stage_filter)) &
    (df["sla_tier"].isin(sla_filter))
]

# -----------------------------
# Header
# -----------------------------
st.title("🤖 Enterprise Model Gallery")
st.caption("Discover, evaluate, and reuse production-grade ML models across the organization")

# -----------------------------
# Metrics row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Models", len(df))
col2.metric("Production Models", len(df[df.model_stage == "prod"]))
col3.metric("Healthy Monitoring", len(df[df.monitoring_status == "Healthy"]))
col4.metric("Domains Covered", df.domain.nunique())

st.divider()

# -----------------------------
# Model explorer
# -----------------------------
st.subheader("📦 Model Explorer")

selected_model = st.selectbox("Select a model to view deep insights", filtered_df["model_name"])
model_row = filtered_df[filtered_df.model_name == selected_model].iloc[0]

colA, colB = st.columns([2, 1])

with colA:
    st.markdown(f"### {model_row['model_name']}")
    st.write("**Domain:**", model_row['domain'])
    st.write("**Owner Team:**", model_row['owner_team'])
    st.write("**Lifecycle Stage:**", model_row['model_stage'])
    st.write("**Monitoring Status:**", model_row['monitoring_status'])
    st.write("**Approval Status:**", model_row['approval_status'])

with colB:
    st.info(f"**SLA Tier:** {model_row['sla_tier']}")
    st.success(f"**Endpoint:** {model_row['inference_endpoint_id']}")
    st.warning(f"**Feature Store:** {model_row['feature_store_dependency']}")

st.divider()

# -----------------------------
# Data scientist wow section
# -----------------------------
st.subheader("🧠 Why this model matters")

colX, colY = st.columns(2)

with colX:
    st.markdown("#### 📈 Performance & Reliability")
    perf = np.random.uniform(0.78, 0.96)
    drift = np.random.uniform(0.01, 0.2)
    st.metric("Validation AUC", round(perf, 3))
    st.metric("Data Drift Score", round(drift, 3))

with colY:
    st.markdown("#### 🔁 Reuse Signals")
    reuse_count = np.random.randint(4, 32)
    pipelines = np.random.randint(2, 12)
    st.metric("Downstream Consumers", reuse_count)
    st.metric("Pipelines Using Model", pipelines)

st.divider()

# -----------------------------
# Model comparison table
# -----------------------------
st.subheader("📊 Compare Models")
st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# Feedback loop
# -----------------------------
st.subheader("💬 Data Scientist Feedback")
feedback = st.text_area("What would make you reuse this model?")

if st.button("Submit Feedback"):
    st.success("Feedback captured — this helps improve model discovery & reuse!")

# -----------------------------
# Footer
# -----------------------------
st.caption("Enterprise AI Platform • Model Registry • Governance • Monitoring • Reuse-first culture")
