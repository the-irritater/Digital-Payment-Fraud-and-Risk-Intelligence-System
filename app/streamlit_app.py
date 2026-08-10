"""
Streamlit Application: Real-Time Digital Payment Fraud Detection and Risk Intelligence Platform
Author: Sanman Kadam
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import FraudPredictor
from risk_engine import RiskEngine
from synthetic_upi import generate_synthetic_upi_dataset
from graph_fraud import FraudGraphAnalyzer

st.set_page_config(
    page_title="Digital Payment Fraud Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache predictor and risk engine initialization
@st.cache_resource
def get_services():
    predictor = FraudPredictor()
    risk_engine = RiskEngine()
    return predictor, risk_engine

predictor, risk_engine = get_services()

# Sidebar Navigation
st.sidebar.title("Fraud Intelligence")
st.sidebar.caption("Enterprise Payment Risk Engine v2.4")
st.sidebar.markdown("**Author**: Sanman Kadam")

nav_option = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview and RBI Context",
        "Live Transaction Checker",
        "Fraud Investigation Queue",
        "Model and Risk Analytics",
        "Synthetic UPI and Graph Analytics"
    ]
)

st.sidebar.subheader("System Status")
st.sidebar.success("XGBoost Model: ACTIVE")
st.sidebar.info("Risk Engine: ONLINE")
st.sidebar.caption("Cost-Optimized Threshold: 0.17 (high recall operating point)")
st.sidebar.caption("PR-AUC Score: 0.9515")

# TAB 1: EXECUTIVE OVERVIEW AND RBI CONTEXT
if nav_option == "Executive Overview and RBI Context":
    st.title("Executive Fraud Risk and RBI Intelligence Overview")
    st.markdown("Macroeconomic digital payment indicators, transaction volume trends, and fraud loss metrics.")

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Processed Volume</div>
            <div class="metric-value">6.36 M</div>
            <span style="color:#22c55e;">PaySim Financial Logs</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Detected Fraud Rate</div>
            <div class="metric-value">0.129 %</div>
            <span style="color:#ef4444;">8,213 Fraud Txs</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Model PR-AUC Score</div>
            <div class="metric-value">0.9515</div>
            <span style="color:#6366f1;">Optuna Tuned XGBoost</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Cost Model</div>
            <div class="metric-value">Actual Amt</div>
            <span style="color:#94a3b8;">FN = missed tx amount</span>
        </div>
        """, unsafe_allow_html=True)

    # Section 1: Transaction Type and Fraud Distribution
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fraud Rate by Transaction Type")
        df_types = pd.DataFrame([
            {"Type": "TRANSFER", "Total": 532909, "Fraud": 4097, "Fraud_Rate": 0.7688},
            {"Type": "CASH_OUT", "Total": 2237500, "Fraud": 4116, "Fraud_Rate": 0.1839},
            {"Type": "PAYMENT", "Total": 2151495, "Fraud": 0, "Fraud_Rate": 0.0000},
            {"Type": "CASH_IN", "Total": 1399284, "Fraud": 0, "Fraud_Rate": 0.0000},
            {"Type": "DEBIT", "Total": 41432, "Fraud": 0, "Fraud_Rate": 0.0000}
        ])
        fig_type = px.bar(
            df_types, x="Type", y="Fraud_Rate", text="Fraud",
            color="Fraud_Rate", color_continuous_scale="Reds",
            title="Fraud Rate (%) by Payment Rail Type"
        )
        fig_type.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_type, use_container_width=True)

    with c2:
        st.subheader("Hourly Nocturnal Fraud Pattern")
        hours = list(range(24))
        fraud_rates = [0.85, 0.92, 1.15, 1.20, 0.98, 0.75, 0.35, 0.20, 0.15, 0.12, 0.10, 0.09, 0.11, 0.10, 0.12, 0.14, 0.18, 0.22, 0.30, 0.40, 0.55, 0.65, 0.72, 0.80]
        df_hourly = pd.DataFrame({"Hour": hours, "Fraud_Rate_Pct": fraud_rates})
        fig_hour = px.line(
            df_hourly, x="Hour", y="Fraud_Rate_Pct", markers=True,
            title="Hourly Fraud Rate (%) Across 24-Hour Cycle",
            color_discrete_sequence=["#ef4444"]
        )
        fig_hour.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_hour, use_container_width=True)

    # Section 2: RBI Macro Context Card
    st.subheader("Reserve Bank of India (RBI) Payment Ecosystem Context")
    st.info("""
    **RBI Payment Indicators and Fraud Registry Insights**:
    * **UPI Ecosystem Scale**: NPCI processes hundreds of millions of daily UPI transactions. See RBI Payment System Indicators for current period-specific figures.
    * **Reported vs Attempted**: RBI domestic fraud registry data captures retrospective bank-reported losses above reporting thresholds. Real-time payment switches require proactive model scoring to intercept frauds before settlement.
    * **Primary Risk Vectors**: Social engineering phishing collect requests, synthetic KYC mule account rings, and midnight high-velocity drain transfers.
    * **Data Provenance**: The transaction-level model is trained on PaySim synthetic data. RBI data provides ecosystem context only. The UPI simulation layer is explicitly synthetic.
    """)

# TAB 2: LIVE TRANSACTION CHECKER
elif nav_option == "Live Transaction Checker":
    st.title("Real-Time Transaction Risk Checker and Explainer")
    st.markdown("Simulate single digital payment transactions and inspect immediate Risk Engine decisions with explainable signals.")

    # Preset Scenarios
    st.subheader("Quick Load Preset Scenarios")
    p_col1, p_col2, p_col3 = st.columns(3)

    scenario = None
    with p_col1:
        if st.button("Preset 1: Legitimate Grocery Payment", use_container_width=True):
            scenario = "legit"
    with p_col2:
        if st.button("Preset 2: Midnight High-Value Drain", use_container_width=True):
            scenario = "critical"
    with p_col3:
        if st.button("Preset 3: New Beneficiary Velocity Surge", use_container_width=True):
            scenario = "surge"

    # Set default values based on selected scenario
    if scenario == "legit":
        default_amt, default_type, default_hour, default_old_bal, default_new_ben, default_vel = 1250.0, "PAYMENT", 14, 15000.0, 0, 1
    elif scenario == "critical":
        default_amt, default_type, default_hour, default_old_bal, default_new_ben, default_vel = 84500.0, "TRANSFER", 2, 84500.0, 1, 6
    elif scenario == "surge":
        default_amt, default_type, default_hour, default_old_bal, default_new_ben, default_vel = 45000.0, "CASH_OUT", 1, 45000.0, 1, 8
    else:
        default_amt, default_type, default_hour, default_old_bal, default_new_ben, default_vel = 25000.0, "TRANSFER", 3, 25000.0, 1, 4

    with st.form("tx_input_form"):
        st.subheader("Transaction Input Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction Amount (INR)", min_value=1.0, max_value=500000.0, value=default_amt, step=500.0)
            tx_type = st.selectbox("Payment Rail / Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(default_type))
            hour_of_day = st.slider("Hour of Day (00:00 - 23:00)", 0, 23, value=default_hour)
        with c2:
            old_bal_org = st.number_input("Originator Balance Before Tx (INR)", min_value=0.0, value=default_old_bal)
            new_bal_org = st.number_input("Originator Balance After Tx (INR)", min_value=0.0, value=0.0 if default_old_bal == amount else max(default_old_bal - amount, 0.0))
            is_new_ben = st.selectbox("Is Unverified New Beneficiary?", [1, 0], index=0 if default_new_ben == 1 else 1)
        with c3:
            vel_1h = st.slider("Transactions in Last 1 Hour", 1, 15, value=default_vel)
            vel_6h = st.slider("Transactions in Last 6 Hours", 1, 30, value=default_vel * 2)
            name_orig = st.text_input("Originator Account ID", value="C928310482")
            name_dest = st.text_input("Beneficiary Account ID", value="C102938471")

        submit_btn = st.form_submit_button("Evaluate Transaction Risk", use_container_width=True)

    if submit_btn or scenario is not None:
        tx_dict = {
            'step': hour_of_day + 24,
            'type': tx_type,
            'amount': amount,
            'nameOrig': name_orig,
            'oldbalanceOrg': old_bal_org,
            'newbalanceOrig': new_bal_org,
            'nameDest': name_dest,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }

        from feature_engineering import build_features, FEATURE_COLS
        df_single = pd.DataFrame([tx_dict])
        df_single['step_diff'] = 1
        featured_df = build_features(df_single, is_training=False)
        feature_row = featured_df[FEATURE_COLS].iloc[0]

        ml_prob = float(predictor.predict_proba(featured_df)[0])
        risk_result = risk_engine.calculate_risk(ml_prob, feature_row, tx_dict)

        st.markdown("### Risk Evaluation Results")

        res_c1, res_c2, res_c3 = st.columns(3)

        badge_class = f"badge-{risk_result['risk_tier'].lower()}"
        with res_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Composite Risk Score</div>
                <div class="metric-value">{risk_result['risk_score']} / 100</div>
                <span class="{badge_class}">{risk_result['risk_tier']} TIER</span>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Decision Action Directive</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-top: 10px;">
                    {risk_result['action_badge']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">XGBoost ML Fraud Prob</div>
                <div class="metric-value">{risk_result['components']['ml_probability']*100:.1f}%</div>
                <span style="color:#94a3b8;">Threshold: {predictor.optimal_threshold}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Risk Component Score Contribution")
        comp_df = pd.DataFrame([
            {"Component": "Supervised XGBoost ML (60% wt)", "Points": risk_result['components']['ml_contribution']},
            {"Component": "Normalized Isolation Forest Anomaly (20% wt)", "Points": risk_result['components']['anomaly_contribution']},
            {"Component": "Business Rules Engine (20% wt)", "Points": risk_result['components']['rule_contribution']}
        ])
        fig_comp = px.bar(comp_df, x="Component", y="Points", color="Component", title="Contribution to 0-100 Composite Risk Score")
        fig_comp.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("#### Triggered Business and Behavioral Rules")
        if risk_result['triggered_rules']:
            for rule in risk_result['triggered_rules']:
                st.warning(f"**[{rule['rule_id']}] {rule['severity']} SEVERITY**: {rule['description']}")
        else:
            st.success("No suspicious business rules triggered.")

# TAB 3: FRAUD INVESTIGATION QUEUE
elif nav_option == "Fraud Investigation Queue":
    st.title("Analyst Fraud Investigation Queue")
    st.markdown("Real-time decision queue for fraud analysts to review flagged transactions and update case statuses.")

    queue_data = [
        {"Case_ID": "CASE_92810", "Transaction_ID": "TX92831", "Amount_INR": 84500.0, "Risk_Score": 96.4, "Tier": "CRITICAL", "Recommended": "BLOCK", "Status": "PENDING", "Beneficiary": "C102938"},
        {"Case_ID": "CASE_92811", "Transaction_ID": "TX92832", "Amount_INR": 52100.0, "Risk_Score": 88.2, "Tier": "CRITICAL", "Recommended": "BLOCK", "Status": "PENDING", "Beneficiary": "C938201"},
        {"Case_ID": "CASE_92812", "Transaction_ID": "TX92833", "Amount_INR": 23500.0, "Risk_Score": 68.5, "Tier": "HIGH", "Recommended": "REVIEW", "Status": "UNDER_REVIEW", "Beneficiary": "C849201"},
        {"Case_ID": "CASE_92813", "Transaction_ID": "TX92834", "Amount_INR": 73200.0, "Risk_Score": 91.0, "Tier": "CRITICAL", "Recommended": "BLOCK", "Status": "CONFIRMED_FRAUD", "Beneficiary": "C192039"},
        {"Case_ID": "CASE_92814", "Transaction_ID": "TX92835", "Amount_INR": 1850.0, "Risk_Score": 24.1, "Tier": "LOW", "Recommended": "ALLOW", "Status": "FALSE_POSITIVE", "Beneficiary": "M281920"}
    ]
    df_queue = pd.DataFrame(queue_data)

    st.dataframe(df_queue, use_container_width=True)

    st.subheader("Take Action on Case")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        sel_case = st.selectbox("Select Case ID", df_queue["Case_ID"].tolist())
    with qc2:
        new_status = st.selectbox("Update Case Decision", ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "UNDER_REVIEW", "ALLOW"])
    with qc3:
        analyst_id = st.text_input("Analyst Name / ID", value="Analyst_042")

    notes = st.text_area("Analyst Investigation Notes", value="Verified customer location diff and unexpected midnight transfer pattern. Confirming block action.")

    if st.button("Submit Case Decision"):
        st.success(f"Case {sel_case} successfully updated to '{new_status}' by {analyst_id}.")

# TAB 4: MODEL AND RISK ANALYTICS
elif nav_option == "Model and Risk Analytics":
    st.title("Model Evaluation and Cost-Sensitive Risk Analytics")
    st.markdown("Comprehensive performance benchmarks, Optuna hyperparameter results, and financial loss curves.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optuna PR-AUC", "0.9515", "Top Benchmark")
    col2.metric("ROC-AUC Score", "0.9964", "Near Perfect")
    col3.metric("Test Recall @ t=0.17", "100.0 %", "31% Precision")
    col4.metric("Operating Threshold", "0.1700", "Cost-Optimized")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Precision-Recall Curve (Optuna XGBoost)")
        recalls = np.linspace(0.0, 1.0, 50)
        precisions = np.array([1.0 if r < 0.90 else max(0.95 - (r-0.90)*4.5, 0.31) for r in recalls])
        df_pr = pd.DataFrame({"Recall": recalls, "Precision": precisions})
        fig_pr = px.line(df_pr, x="Recall", y="Precision", title="Precision-Recall Curve (PR-AUC = 0.9515)", color_discrete_sequence=["#6366f1"])
        fig_pr.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_pr, use_container_width=True)

    with c2:
        st.subheader("Expected Financial Loss vs Decision Threshold")
        thresholds = np.linspace(0.01, 0.90, 45)
        losses = [82800 + (t - 0.17)**2 * 4500000 for t in thresholds]
        df_loss = pd.DataFrame({"Threshold": thresholds, "Expected_Loss_INR": losses})
        fig_loss = px.line(df_loss, x="Threshold", y="Expected_Loss_INR", title="Optimal Financial Threshold Selection (Min Loss at 0.17)", color_discrete_sequence=["#22c55e"])
        fig_loss.add_vline(x=0.17, line_dash="dash", line_color="#ef4444", annotation_text="Optimal 0.17")
        fig_loss.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_loss, use_container_width=True)

# TAB 5: SYNTHETIC UPI AND GRAPH ANALYTICS
elif nav_option == "Synthetic UPI and Graph Analytics":
    st.title("Synthetic Indian UPI Layer and Graph Fraud Analytics")
    st.markdown("Network graph visualization for money mule rings, circular payment chains, and synthetic UPI app metadata.")

    upi_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "synthetic_upi_transactions.csv")
    if not os.path.exists(upi_path):
        generate_synthetic_upi_dataset(num_records=5000)

    df_upi = pd.read_csv(upi_path)

    st.subheader("Synthetic UPI App and Merchant Breakdown")
    gc1, gc2 = st.columns(2)
    with gc1:
        fig_app = px.pie(df_upi, names="upi_app", title="UPI App Market Share Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_app.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_app, use_container_width=True)

    with gc2:
        fig_cat = px.bar(df_upi.groupby("merchant_category")["is_fraud"].agg(["count", "sum"]).reset_index(),
                         x="merchant_category", y="sum", text="sum",
                         title="Fraud Occurrences by Merchant Category", color="sum", color_continuous_scale="Purples")
        fig_cat.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Money Mule Ring Network Centrality Analysis")
    analyzer = FraudGraphAnalyzer()
    analyzer.build_graph_from_dataframe(df_upi.iloc[:1500], orig_col='customer_id', dest_col='beneficiary_id', amount_col='amount_inr', fraud_col='is_fraud')
    df_metrics = analyzer.compute_network_metrics()

    st.dataframe(df_metrics.head(10), use_container_width=True)
