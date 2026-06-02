import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# --- Configuration & Setup ---
st.set_page_config(page_title="NIDS Telemetry Dashboard", layout="wide", page_icon="🛡️")

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
    """Reads the raw text log for critical alerts."""
    alerts = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Fetch the last 10 alerts
            alerts = [line.strip() for line in lines[-10:]]
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
                                            mode='lines', name='Velocity (v)', line=dict(color='royalblue')))
    
    if 'acceleration' in df_residuals:
        fig_kinematics.add_trace(go.Scatter(x=df_residuals['timestamp'], y=df_residuals['acceleration'], 
                                            mode='lines', name='Acceleration (a)', line=dict(color='firebrick')))
    
    # Threshold lines based on paper
    fig_kinematics.add_hline(y=0.65, line_dash="dash", line_color="orange", annotation_text="Velocity Threshold")
    fig_kinematics.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="Acceleration Threshold")
    
    fig_kinematics.update_layout(title="Residual Velocity & Acceleration Tracker",
                                 xaxis_title="Timeline", yaxis_title="Kinematic Values",
                                 template="plotly_dark")
    st.plotly_chart(fig_kinematics, use_container_width=True)
else:
    st.info("Awaiting residual metric data from Volumetric Consumer...")

st.markdown("---")

# --- Bottom Section: AI Confidence & Live Logs ---
col_ai, col_logs = st.columns([1, 1])

with col_ai:
    st.subheader("🧠 Neural Network Monitor")
    st.markdown("Tracking anomaly confidence levels from the Adaptive MLP.")
    
    # Placeholder for a gauge chart (mocked value here, can be tied to JSONL later)
    # This demonstrates how the XAI (Explainable AI) meter will look
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 0.15, # Replace with actual dynamic mean probability from JSONL
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Threat Confidence"},
        gauge = {
            'axis': {'range': [None, 1]},
            'steps' : [
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 0.85], 'color': "orange"},
                {'range': [0.85, 1.0], 'color': "red"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.95}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_logs:
    st.subheader("🚨 Final NIDS Alerts")
    st.markdown("Recent events triggered by the Stateful or AI engines:")
    
    if alerts_data:
        for alert in alerts_data:
            st.error(alert)
    else:
        st.success("No active threats detected. Network is stable.")

# Trigger a rerun to keep the dashboard live
st.rerun()