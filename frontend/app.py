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
                        
                        winner_name = results.get("winner_name", "k-Anonymity")
                        
                        # --- 🏆 BOLD WINNER DISPLAY ---
                        st.markdown(f"""
                        <div style="background-color:#0e1117; padding:20px; border-radius:10px; border: 2px solid #00ff00;">
                            <h2 style="color:#00ff00; text-align:center;">🏆 Optimal Architecture: {winner_name}</h2>
                            <p style="font-size:18px; text-align:center;">{results['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- 📊 CLEAN DATA TABLE (NO MORE JSON) ---
                        st.markdown("### 📈 Comprehensive Metrics Table")
                        st.markdown("Compare the exact Utility and Fairness tradeoffs across all evaluated architectures.")
                        
                        # Convert the JSON dictionary into a Pandas DataFrame for a beautiful table
                        audit_df = pd.DataFrame(results["ml_audit"]).T
                        audit_df = audit_df.rename(columns={"F1_Score": "Utility (F1-Score)", "Bias_Score": "Bias (Lower is Better)"})
                        
                        # Display the table, highlighting the best scores
                        st.dataframe(
                            audit_df.style.highlight_max(subset=['Utility (F1-Score)'], color='rgba(0,255,0,0.2)')
                                        .highlight_min(subset=['Bias (Lower is Better)'], color='rgba(0,0,255,0.2)'),
                            use_container_width=True
                        )
                        
                        st.markdown("---")
                        display_vulnerability_metrics(results["vulnerability_analysis"])
                        total_rows = results["dataset_info"]["total_rows"]
                        plot_vulnerability(results["vulnerability_analysis"], total_rows)
                        
                        st.markdown("---")
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            plot_utility_metrics(results["ml_audit"])
                        with r_col2:
                            plot_bias_metrics(results["ml_audit"])
                            
                        # --- 🛠️ DYNAMIC PHASE 2 (HYBRID STACKING) ---
                        st.markdown("---")
                        st.write("### 🛠️ Phase 2: Next Steps & Hybrid Export")
                        
                        action_col1, action_col2 = st.columns(2)
                        
                        with action_col1:
                            st.markdown(f"#### Export {winner_name} Data")
                            st.markdown(f"Download the securely modified dataset using the winning **{winner_name}** architecture.")
                            st.download_button(
                                label=f"📥 Download {winner_name} Dataset (CSV)",
                                data=results.get("downloadable_csv", ""),
                                file_name=f"equi_vault_{winner_name.lower().replace(' ', '_')}.csv",
                                mime="text/csv",
                                type="primary"
                            )
                            
                        with action_col2:
                            st.markdown("#### Continuous Privacy Optimization")
                            
                            # Dynamic Logic: Recommend the missing puzzle piece
                            if "Privacy" in winner_name:
                                st.info(f"✅ **SAs Protected:** {winner_name} mathematically secured your Sensitive Attributes.")
                                st.markdown(f"**Recommendation:** Stack techniques! Your newly downloaded data is ready for Phase 2. Apply **k-Anonymity** to the Quasi-Identifiers to prevent hacker Linkage Attacks.")
                            else:
                                st.info(f"✅ **QIs Protected:** {winner_name} successfully blurred your Quasi-Identifiers into groups.")
                                st.markdown(f"**Recommendation:** Stack techniques! Your newly downloaded data is ready for Phase 2. Apply **Differential Privacy** to the numerical Sensitive Attributes to prevent Attribute Disclosure and smooth out ML bias.")
                            
                            if st.button("Apply Complementary Technique (Simulate)"):
                                st.success("Hybrid Data Generated! (In a full deployment, Equi-Vault runs this in a continuous CI/CD loop).")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {str(e)}")