import streamlit as st  # type: ignore[import]
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_utility_metrics(ml_audit):
    """Renders a comparative bar chart for F1 Scores (Utility)."""
    # Extract data from the JSON response
    techniques = list(ml_audit.keys())
    f1_scores = [v.get("F1_Score", 0) for v in ml_audit.values()]
    
    df_plot = pd.DataFrame({
        "Technique": techniques,
        "F1 Score": f1_scores
    })
    
    fig = px.bar(
        df_plot, 
        x="Technique", 
        y="F1 Score", 
        color="Technique",
        title="Model Utility Preservation (F1 Score)",
        text_auto='.2f',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_bias_metrics(ml_audit):
    """Renders a comparative bar chart for Demographic Parity (Bias)."""
    techniques = list(ml_audit.keys())
    bias_scores = [v.get("Bias_Score", 0) for v in ml_audit.values()]
    
    df_plot = pd.DataFrame({
        "Technique": techniques,
        "Bias (Demographic Parity)": bias_scores
    })
    
    # For bias, lower is better. 0.0 is perfectly fair.
    fig = px.bar(
        df_plot, 
        x="Technique", 
        y="Bias (Demographic Parity)", 
        color="Technique",
        title="Algorithmic Fairness (Lower is Better)",
        text_auto='.3f',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_vulnerability(vulnerability_data, total_rows):
    """Renders a donut chart showing the percentage of exposed rows."""
    homogeneity = vulnerability_data.get("homogeneity_attack", {})
    skewness = vulnerability_data.get("skewness_attack", {})
    
    exposed = homogeneity.get("exposed_records", 0)
    skewed = skewness.get("highly_skewed_records", 0)
    secure = total_rows - exposed - skewed
    if secure < 0: secure = 0
    
    labels = ['100% Exposed (Homogeneity)', 'Highly Skewed (>90% Risk)', 'Statistically Secure']
    values = [exposed, skewed, secure]
    colors = ['#EF553B', '#FFA15A', '#00CC96'] # Red, Orange, Green
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
    fig.update_layout(title_text="Re-identification Vulnerability (k-Anonymized Baseline)")
    st.plotly_chart(fig, use_container_width=True)

def display_recommendation_card(recommendation_text):
    """Renders a stylized highlight box for the final AI recommendation."""
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h3 style="margin-top:0;">🛡️ AI Architecture Recommendation</h3>
            <p style="font-size: 18px;">{recommendation_text}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
def display_vulnerability_metrics(vulnerability_data):
    homogeneity = vulnerability_data.get("homogeneity_attack", {})
    st.write("### 🚨 Vulnerability Exposure Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Equivalence Classes", value=homogeneity.get("total_classes", 0))
    with col2:
        st.metric(label="Vulnerable Classes (100% Homogeneous)", value=homogeneity.get("vulnerable_classes", 0))
    with col3:
        st.metric(label="Total Exposed Records", value=homogeneity.get("exposed_records", 0), delta="High Risk" if homogeneity.get("exposed_records", 0) > 0 else "Secure", delta_color="inverse")