import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time

# --- Configuration & Setup ---
st.set_page_config(page_title="NIDS Telemetry Dashboard", layout="wide", page_icon="🛡️")

# --- Custom CSS for Enterprise UI ---
st.markdown("""
<style>
    /* Styling for Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        border-left: 5px solid #E63946; /* Red cyber accent */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Make headers look sharper */
    h1, h2, h3 {
        color: #F1FAEE;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Adjust top padding to remove empty space */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

LOGS_DIR = "/app/logs"
RESIDUALS_FILE = os.path.join(LOGS_DIR, "residual_metrics.csv")
ALERTS_FILE = os.path.join(LOGS_DIR, "nids_final_alerts.log")
JSONL_FILE = os.path.join(LOGS_DIR, "master_traffic_records.jsonl")

def load_csv_data(filepath):
    """Safely loads CSV data if the file exists and is not empty."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            return pd.read_csv(filepath)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def load_alerts(filepath):
    """Reads the raw text log for critical alerts, ignoring empty lines."""
    alerts = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Filter empty lines and fetch the last 10
            alerts = [line.strip() for line in lines if line.strip() != ""][-10:]
    return alerts

# --- Dashboard Layout ---
st.title("🛡️ Autonomous NIDS Telemetry Dashboard")
st.markdown("Real-time monitoring of Kinematic Residuals, Stateful Protocols, and Neural Network Confidence.")

# Auto-refresh logic (refreshes every 5 seconds)
st_autorefresh = st.empty()
time.sleep(5) 

# --- Data Loading ---
df_residuals = load_csv_data(RESIDUALS_FILE)
alerts_data = load_alerts(ALERTS_FILE)

# --- Top Level Metrics ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Active Monitoring Engine", value="Online", delta="Operational")
with col2:
    alert_count = len(alerts_data)
    st.metric(label="Recent Critical Alerts", value=alert_count, delta_color="inverse")
with col3:
    # Safely get the latest velocity if available
    latest_vel = df_residuals['velocity'].iloc[-1] if not df_residuals.empty and 'velocity' in df_residuals else 0.0
    st.metric(label="Current Kinematic Velocity", value=f"{latest_vel:.4f}")

st.markdown("---")

# --- Middle Section: Kinematic Analysis (Paper Logic) ---
st.subheader("📈 Real-Time Kinematic Error Analysis")

if not df_residuals.empty and 'timestamp' in df_residuals:
    # Assuming the CSV has columns: timestamp, residual, velocity, acceleration
    
    fig_kinematics = go.Figure()
    
    if 'velocity' in df_residuals:
        fig_kinematics.add_trace(go.Scatter(x=df_residuals['timestamp'], y=df_residuals['velocity'], 
                                            mode='lines', name='Velocity (v)', line=dict(color='#457B9D', width=2)))
    
    if 'acceleration' in df_residuals:
        fig_kinematics.add_trace(go.Scatter(x=df_residuals['timestamp'], y=df_residuals['acceleration'], 
                                            mode='lines', name='Acceleration (a)', line=dict(color='#E63946', width=2)))
    
    # Threshold lines based on paper
    fig_kinematics.add_hline(y=0.65, line_dash="dash", line_color="#E9C46A", annotation_text="Velocity Threshold")
    fig_kinematics.add_hline(y=0.80, line_dash="dash", line_color="#E63946", annotation_text="Acceleration Threshold")
    
    fig_kinematics.update_layout(
        title="Residual Velocity & Acceleration Tracker",
        xaxis_title="Timeline", 
        yaxis_title="Kinematic Values",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig_kinematics, use_container_width=True)
else:
    st.info("Awaiting residual metric data from Volumetric Consumer... (Generate some network traffic to see the graphs!)")

st.markdown("---")

# --- Bottom Section: AI Confidence & Live Logs ---
col_ai, col_logs = st.columns([1, 1])

with col_ai:
    st.subheader("🧠 Neural Network Monitor")
    st.markdown("Tracking anomaly confidence levels from the Adaptive MLP.")
    
    # Placeholder for a gauge chart (mocked value here, can be tied to JSONL later)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 0.15, # Replace with actual dynamic mean probability from JSONL
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Threat Confidence", 'font': {'size': 20, 'color': "white"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "#333",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps' : [
                {'range': [0, 0.5], 'color': "#2A9D8F"},  # Muted green
                {'range': [0.5, 0.85], 'color': "#E9C46A"}, # Muted orange
                {'range': [0.85, 1.0], 'color': "#E63946"}], # Muted red
            'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 0.95}
        }
    ))
    fig_gauge.update_layout(
        height=300, 
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Courier New"}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_logs:
    st.subheader("Final NIDS Alerts")
    st.markdown("Recent events triggered by the Stateful or AI engines:")
    
    if alerts_data:
        for alert in alerts_data:
            st.error(alert, icon="⚠️")
    else:
        st.success("No active threats detected. Network is stable.", icon="✅")

# Trigger a rerun to keep the dashboard live
st.rerun()