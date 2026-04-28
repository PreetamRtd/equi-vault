import streamlit as st
import pandas as pd
import requests
import json
import os
from components import plot_utility_metrics, plot_bias_metrics, plot_vulnerability, display_recommendation_card, display_vulnerability_metrics

# --- Page Configuration ---
st.set_page_config(page_title="Equi-Vault", layout="wide", page_icon="🛡️")

# --- Optional CSS Loader ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "assets", "custom_style.css")
if os.path.exists(css_path):
    load_css(css_path)

# --- NEW: Auto-Schema Scanner ---
def auto_map_schema(columns):
    """Scans column names and makes intelligent guesses for the schema mapping."""
    target_keywords = ['income', 'result', 'status', 'class', 'label', 'target', 'disease']
    protected_keywords = ['gender', 'sex', 'race', 'ethnicity']
    qi_keywords = ['age', 'date', 'room', 'zip', 'education', 'marital', 'country', 'relationship', 'workclass']
    sa_keywords = ['condition', 'billing', 'amount', 'medication', 'capital', 'salary', 'disease']

    mapped = {"target": None, "protected": None, "qis": [], "sas": []}

    # 1. Find Target
    for col in columns:
        if any(kw in col.lower() for kw in target_keywords):
            mapped["target"] = col
            break
            
    # 2. Find Protected Attribute
    for col in columns:
        if any(kw in col.lower() for kw in protected_keywords):
            mapped["protected"] = col
            break

    # 3. Find QIs and SAs
    for col in columns:
        c_low = col.lower()
        if col == mapped["target"] or col == mapped["protected"]:
            continue
            
        if any(kw in c_low for kw in qi_keywords):
            mapped["qis"].append(col)
        elif any(kw in c_low for kw in sa_keywords):
            mapped["sas"].append(col)

    # Fallbacks if the scanner couldn't find a match
    if not mapped["target"]: mapped["target"] = columns[-1] # Usually the last column is the target
    if not mapped["protected"]: mapped["protected"] = columns[0]

    return mapped

# --- Main App ---
st.title("🛡️ Equi-Vault: Privacy & Fairness Auditing Sandbox")
st.markdown("Upload your sensitive tabular data to mathematically evaluate the tradeoff between Data Privacy, ML Utility, and Algorithmic Bias.")

# --- File Upload & Domain Selection ---
st.sidebar.header("1. Upload & Context")
uploaded_file = st.sidebar.file_uploader("Upload Sensitive Dataset (CSV)", type="csv")
domain = st.sidebar.selectbox("Select Data Domain", ["Healthcare", "Finance", "Criminal Justice"])

if uploaded_file is not None:
    # Read data to populate dropdowns
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()
    
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(3))
    
    st.markdown("---")
    st.write("### ⚙️ Schema Mapping")
    st.markdown("Tag your columns so the engine knows how to process the data. *Equi-Vault has auto-mapped these based on your column names.*")
    
    # --- Execute Auto-Scanner ---
    auto_mapped = auto_map_schema(columns)
    
    # Find the numeric index of the target and protected columns for the dropdowns
    target_idx = columns.index(auto_mapped["target"]) if auto_mapped["target"] in columns else 0
    protected_idx = columns.index(auto_mapped["protected"]) if auto_mapped["protected"] in columns else 0
    
    # --- UI Layout for Mapping ---
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox("🎯 Target Variable (What the ML predicts)", columns, index=target_idx)
        protected_col = st.selectbox("⚖️ Protected Attribute (Audit for bias against this)", columns, index=protected_idx)
        
    with col2:
        qi_cols = st.multiselect("🔍 Quasi-Identifiers (To be Generalized/Blurred)", columns, default=auto_mapped["qis"])
        sa_cols = st.multiselect("🔒 Sensitive Attributes (To be mathematically protected)", columns, default=auto_mapped["sas"])

    st.markdown("---")
    
    # --- Trigger Execution ---
    if st.button("🚀 Run Complete Privacy & Bias Audit", type="primary"):
        if not target_col or not protected_col or not qi_cols or not sa_cols:
            st.error("Please map at least one column for Target, Protected, QIs, and SAs before running.")
        else:
            with st.spinner("Executing Mathematical Transformations, Mock Attacks, and Model Training... This may take a minute."):
                
                # Format data for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                data = {
                    "domain": domain,
                    "target_col": target_col,
                    "protected_col": protected_col,
                    "qi_cols": ",".join(qi_cols),
                    "sa_cols": ",".join(sa_cols)
                }
                
                # Call the local FastAPI server
                try:
                    response = requests.post("http://127.0.0.1:8000/audit/", files=files, data=data)
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.success("Audit Complete!")
                        
                        # --- Dashboard Rendering ---
                        display_recommendation_card(results.get("recommendation", ""))
                        
                        # Explicit Metric Cards
                        st.markdown("---")
                        display_vulnerability_metrics(results["vulnerability_analysis"])
                        
                        # Donut Chart
                        total_rows = results["dataset_info"]["total_rows"]
                        plot_vulnerability(results["vulnerability_analysis"], total_rows)
                        
                        st.markdown("---")
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            plot_utility_metrics(results["ml_audit"])
                        with r_col2:
                            plot_bias_metrics(results["ml_audit"])
                            
                        # --- Download & Iterative Recommendation ---
                        st.markdown("---")
                        st.write("### 🛠️ Phase 2: Next Steps & Export")
                        
                        action_col1, action_col2 = st.columns(2)
                        
                        with action_col1:
                            st.markdown("#### Export Optimized Data")
                            st.markdown("Download the dataset modified by the recommended architecture.")
                            st.download_button(
                                label="📥 Download Anonymized Dataset (CSV)",
                                data=results.get("downloadable_csv", ""),
                                file_name=f"anonymized_{uploaded_file.name}",
                                mime="text/csv",
                                type="primary"
                            )
                            
                        with action_col2:
                            st.markdown("#### Continuous Privacy Optimization")
                            if results["vulnerability_analysis"]["homogeneity_attack"]["exposed_records"] > 0:
                                st.warning("⚠️ **Vulnerability Detected:** Your k-anonymized data still has exposed records due to homogeneity.")
                                st.markdown(f"**Recommendation:** Stack techniques. Apply **Differential Privacy** to the `{sa_cols[0] if sa_cols else 'Sensitive'}` column of your newly downloaded dataset to break the homogeneity without sacrificing demographic utility.")
                            else:
                                st.info("✅ **Data is Secure:** No tabular linkage vulnerabilities detected. To further prevent Machine Learning Membership Inference attacks, consider applying a light layer of Differential Privacy (ε=2.0).")
                            
                            if st.button("Apply Hybrid Strategy (Simulate)"):
                                st.success("Hybrid Strategy Applied! (In a full deployment, this would push the downloaded dataset back through the pipeline for a Phase 2 audit).")

                        with st.expander("View Raw JSON Output"):
                            st.json(results)
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")