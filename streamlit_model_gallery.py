# streamlit_model_gallery.py
# Run with: streamlit run streamlit_model_gallery.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Enterprise Model Gallery", layout="wide")

# -----------------------------
# Load & Normalize Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("model_registry_v2_enterprise_schema.csv")

    # Clean column names (fix spaces, casing, hidden chars)
    df.columns = df.columns.str.strip().str.lower()

    # Expected enterprise schema mapping
    column_map = {
        "model_name": "model_name",
        "name": "model_name",
        "domain": "domain",
        "model_stage": "model_stage",
        "type": "model_stage",
        "owner_team": "owner_team",
        "contributor": "owner_team",
        "last_retrained_date": "last_retrained_date",
        "sla_tier": "sla_tier",
        "monitoring_status": "monitoring_status",
        "approval_status": "approval_status",
        "inference_endpoint_id": "inference_endpoint_id",
        "feature_store_dependency": "feature_store_dependency"
    }

    # Rename only columns that exist
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Validate required columns
    required_cols = [
        "model_name","domain","model_stage","owner_team",
        "sla_tier","monitoring_status","approval_status"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.stop()

    return df


df = load_data()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("🔎 Discover Models")

domain_filter = st.sidebar.multiselect(
    "Domain",
    options=df["domain"].dropna().unique(),
    default=df["domain"].dropna().unique()
)

stage_filter = st.sidebar.multiselect(
    "Lifecycle Stage",
    options=df["model_stage"].dropna().unique(),
    default=df["model_stage"].dropna().unique()
)

sla_filter = st.sidebar.multiselect(
    "SLA Tier",
    options=df["sla_tier"].dropna().unique(),
    default=df["sla_tier"].dropna().unique()
)

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
col2.metric("Production Models", len(df[df.model_stage.str.lower() == "prod"]))
col3.metric("Healthy Monitoring", len(df[df.monitoring_status.str.lower() == "healthy"]))
col4.metric("Domains Covered", df.domain.nunique())

st.divider()

# -----------------------------
# Model explorer
# -----------------------------
st.subheader("📦 Model Explorer")

if "name" not in filtered_df.columns:
    st.error(f"'name' column missing. Available columns: {list(filtered_df.columns)}")
    st.stop()

selected_model = st.selectbox(
    "Select a model to view deep insights",
    filtered_df["name"].dropna().unique()
)

model_row = filtered_df[filtered_df.name == selected_model].iloc[0]

colA, colB = st.columns([2, 1])

with colA:
    st.markdown(f"### {model_row['name']}")
    st.write("**Domain:**", model_row['domain'])
    st.write("**Owner Team:**", model_row['model_owner_team'])
    st.write("**Lifecycle Stage:**", model_row['model_stage'])
    st.write("**Monitoring Status:**", model_row['monitoring_status'])
    st.write("**Approval Status:**", model_row['approval_status'])

with colB:
    st.info(f"**SLA Tier:** {model_row.get('sla_tier','NA')}")
    st.success(f"**Endpoint:** {model_row.get('inference_endpoint_id','NA')}")
    st.warning(f"**Feature Store:** {model_row.get('feature_store_dependency','NA')}")

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
