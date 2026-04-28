import streamlit as st
import pandas as pd
import requests
import os
from components import plot_utility_metrics, plot_bias_metrics, plot_vulnerability, display_vulnerability_metrics

st.set_page_config(page_title="Equi-Vault", layout="wide", page_icon="🛡️")

if "phase1_results" not in st.session_state:
    st.session_state.phase1_results = None
if "phase2_results" not in st.session_state:
    st.session_state.phase2_results = None

def auto_map_schema(columns):
    target_keywords = ['income', 'result', 'status', 'class', 'label', 'target', 'disease']
    protected_keywords = ['gender', 'sex', 'race', 'ethnicity']
    qi_keywords = ['age', 'date', 'room', 'zip', 'education', 'marital', 'country', 'relationship', 'workclass']
    sa_keywords = ['condition', 'billing', 'amount', 'medication', 'capital', 'salary', 'disease']

    mapped = {"target": None, "protected": None, "qis": [], "sas": []}
    for col in columns:
        if any(kw in col.lower() for kw in target_keywords): mapped["target"] = col; break
    for col in columns:
        if any(kw in col.lower() for kw in protected_keywords): mapped["protected"] = col; break
    for col in columns:
        c_low = col.lower()
        if col in [mapped["target"], mapped["protected"]]: continue
        if any(kw in c_low for kw in qi_keywords): mapped["qis"].append(col)
        elif any(kw in c_low for kw in sa_keywords): mapped["sas"].append(col)

    if not mapped["target"]: mapped["target"] = columns[-1]
    if not mapped["protected"]: mapped["protected"] = columns[0]
    return mapped

st.title("🛡️ Equi-Vault: Privacy & Fairness Auditing Sandbox")

st.sidebar.header("1. Upload & Context")
uploaded_file = st.sidebar.file_uploader("Upload Sensitive Dataset (CSV)", type="csv")
# Restored Domain Dropdown
domain = st.sidebar.selectbox("Select Data Domain", ["Healthcare", "Finance", "Criminal Justice", "Custom"])

BASE_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") # Production URL can be set via Streamlit Secrets

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()
    auto_mapped = auto_map_schema(columns)
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("🎯 Target Variable", columns, index=columns.index(auto_mapped["target"]) if auto_mapped["target"] in columns else 0)
        protected_col = st.selectbox("⚖️ Protected Attribute", columns, index=columns.index(auto_mapped["protected"]) if auto_mapped["protected"] in columns else 0)
    with col2:
        qi_cols = st.multiselect("🔍 Quasi-Identifiers", columns, default=auto_mapped["qis"])
        sa_cols = st.multiselect("🔒 Sensitive Attributes", columns, default=auto_mapped["sas"])

    st.markdown("---")
    
    if st.button("🚀 Run Phase 1: Complete Pipeline Audit", type="primary"):
        with st.spinner("Executing Architectures & Model Training..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            data = {"domain": domain, "target_col": target_col, "protected_col": protected_col, "qi_cols": ",".join(qi_cols), "sa_cols": ",".join(sa_cols)}
            
            response = requests.post(f"{BASE_API_URL}/audit/", files=files, data=data)
            if response.status_code == 200:
                st.session_state.phase1_results = response.json()
                st.session_state.phase2_results = None 
                st.success("Phase 1 Complete!")
            else:
                st.error(f"Backend Error (Status {response.status_code}): {response.text}")

    # ==========================================
    # RENDER PHASE 1 UI
    # ==========================================
    if st.session_state.phase1_results:
        results = st.session_state.phase1_results
        winner_name = results.get("winner_name", "Optimized")
        
        st.markdown(f"""
        <div style="background-color:#0e1117; padding:20px; border-radius:10px; border: 2px solid #00ff00;">
            <h2 style="color:#00ff00; text-align:center;">🏆 Phase 1 Winner: {winner_name}</h2>
            <p style="font-size:18px; text-align:center;">{results['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        # 1. THE DYNAMIC THREAT LANDSCAPE
        st.write("### 🚨 The Threat Landscape & Attack Analysis")
        st.markdown(f"Equi-Vault actively evaluates the data against the mathematical vulnerabilities specific to each architecture. Since **{winner_name}** won Phase 1, here is how it holds up against the primary attack vectors:")
        
        atk_col1, atk_col2 = st.columns(2)
        with atk_col1:
            st.markdown("#### 1. Linkage Attacks (Homogeneity)")
            st.info("**Target:** Datasets using raw data or basic k-Anonymity.\n\n**Analysis:** Hackers isolate a 'blurred' group (e.g., Males aged 30-40) and notice everyone in that group has the same Sensitive Attribute. *l-Diversity specifically patches this attack.*")
        with atk_col2:
            st.markdown("#### 2. Similarity & Skewness Attacks")
            st.info("**Target:** Datasets using k-Anonymity and l-Diversity.\n\n**Analysis:** Hackers observe that while a group has diverse conditions, they are all conceptually similar (e.g., all are gastric diseases), leaking semantic identity. *t-Closeness specifically patches this.*")
        
        st.markdown("#### 3. Membership Inference & Reconstruction")
        st.info("**Target:** Any deterministic masking algorithm (k-Anon, l-Div, t-Close).\n\n**Analysis:** A rogue AI model attempts to reverse-engineer records to prove if a specific person was in the training set. *Differential Privacy patches this by injecting Laplacian Noise.*")
        
        st.markdown("---")
        st.write(f"### 🛡️ Vulnerability Audit for {winner_name}")
        display_vulnerability_metrics(results["vulnerability_analysis"])
        
        st.markdown("---")
        st.write("### 🧠 ML Utility & Fairness Tradeoffs")
        r_col1, r_col2 = st.columns(2)
        with r_col1: plot_utility_metrics(results["ml_audit"])
        with r_col2: plot_bias_metrics(results["ml_audit"])

        # ==========================================
        # PHASE 2: ITERATIVE STACKING
        # ==========================================
        st.markdown("---")
        st.write("### 🛠️ Phase 2: Iterative Stacking Optimization")
        st.markdown(f"If the graphs above show that **{winner_name}** still has exposed vulnerabilities or high algorithmic bias, you can stack a complementary architecture. Equi-Vault will apply it, train a new model, and analyze the delta.")
        
        p2_col1, p2_col2 = st.columns(2)
        with p2_col1:
            phase2_tech = st.selectbox("Select Architecture to Stack:", ["Differential Privacy", "k-Anonymity", "l-Diversity", "t-Closeness"])
            if st.button("Stack & Train Phase 2 Model", type="secondary"):
                with st.spinner(f"Injecting {phase2_tech} on top of {winner_name}..."):
                    files = {"file": ("phase1_winner.csv", results["downloadable_csv"].encode('utf-8'), "text/csv")}
                    data = {"target_col": target_col, "protected_col": protected_col, "qi_cols": ",".join(qi_cols), "sa_cols": ",".join(sa_cols), "technique": phase2_tech}
                    
                    p2_resp = requests.post(f"{BASE_API_URL}/audit_phase2/", files=files, data=data)
                    if p2_resp.status_code == 200:
                        st.session_state.phase2_results = p2_resp.json()

        # Render Phase 2 Dynamic Results
        if st.session_state.phase2_results:
            p2 = st.session_state.phase2_results
            st.markdown("---")
            st.write(f"### 📊 Phase 2 Results: {winner_name} + {p2['technique']}")
            
            # Phase 2 Recommendation Logic
            base_f1 = p2["phase1_baseline"]["F1_Score"]
            new_f1 = p2["phase2_results"]["F1_Score"]
            base_bias = p2["phase1_baseline"]["Bias_Score"]
            new_bias = p2["phase2_results"]["Bias_Score"]
            vuln_exposed = p2["vulnerability_analysis"]["homogeneity_attack"]["exposed_records"]
            
            # Dynamically color the Phase 2 box based on success or failure
            if new_f1 > (base_f1 * 0.8) and vuln_exposed == 0:
                box_color = "#00ff00" # Green
                title = f"🏆 Phase 2 Success: Optimal Stack Achieved"
                p2_reco = f"Stacking {p2['technique']} on top of {winner_name} successfully patched vulnerabilities while keeping ML accuracy within the enterprise threshold."
            elif vuln_exposed > 0:
                box_color = "#ffcc00" # Yellow
                title = f"⚠️ Phase 2 Warning: Vulnerabilities Remain"
                p2_reco = f"The stacked data still has {vuln_exposed} exposed records. Try stacking t-Closeness or Differential Privacy instead."
            else:
                box_color = "#ff4b4b" # Red
                title = f"🚨 Phase 2 Alert: Utility Destroyed"
                p2_reco = f"While the data is now highly secure, {p2['technique']} destroyed ML utility. The AI model is likely unusable."
                
            # Render the styled box
            st.markdown(f"""
            <div style="background-color:#0e1117; padding:20px; border-radius:10px; border: 2px solid {box_color}; margin-bottom: 20px;">
                <h2 style="color:{box_color}; text-align:center;">{title}</h2>
                <p style="font-size:18px; text-align:center;">{p2_reco}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Phase 2 Vulnerability Donut
            display_vulnerability_metrics(p2["vulnerability_analysis"])
            plot_vulnerability(p2["vulnerability_analysis"], p2["dataset_info"]["total_rows"])
            
            # Phase 2 ML Charts
            p2_r1, p2_r2 = st.columns(2)
            with p2_r1: plot_utility_metrics(p2["ml_audit"])
            with p2_r2: plot_bias_metrics(p2["ml_audit"])
            
            st.download_button(
                label=f"📥 Download Final Secure Dataset ({winner_name} + {p2['technique']})",
                data=p2["downloadable_csv"],
                file_name=f"equi_vault_stacked_{phase2_tech.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                type="primary"
            )