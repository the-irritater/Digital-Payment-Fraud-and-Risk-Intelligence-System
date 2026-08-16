"""
Streamlit Application: Digital Payment Fraud Detection and Risk Intelligence Platform
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

# Add src and sql to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql")))

from predict import FraudPredictor
from risk_engine import RiskEngine
from synthetic_upi import generate_synthetic_upi_dataset
from graph_fraud import FraudGraphAnalyzer
from db_manager import get_db_manager

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

# Cache predictor, risk engine, and metadata
@st.cache_resource
def get_services():
    predictor = FraudPredictor()
    risk_engine = RiskEngine()
    db_mgr = get_db_manager()
    
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    threshold_csv = os.path.join(os.path.dirname(__file__), "..", "reports", "threshold_cost_table.csv")
    df_thresh = pd.read_csv(threshold_csv) if os.path.exists(threshold_csv) else pd.DataFrame()
    
    # Load dataset summary if available
    summary_path = os.path.join(os.path.dirname(__file__), "..", "reports", "dataset_summary.json")
    dataset_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            dataset_summary = json.load(f)
    
    return predictor, risk_engine, db_mgr, meta, df_thresh, dataset_summary

predictor, risk_engine, db_mgr, meta, df_thresh, dataset_summary = get_services()

# Extract dynamic metadata from single source of truth
opt_thresh = meta.get("optimal_threshold", 0.5)
test_metrics = meta.get("test_metrics", {})
pr_auc_val = test_metrics.get("pr_auc", 0.0)
roc_auc_val = test_metrics.get("roc_auc", 0.0)
recall_val = test_metrics.get("recall", 0.0)
precision_val = test_metrics.get("precision", 0.0)
loss_val = test_metrics.get("total_financial_loss_inr", 0.0)
brier_val = test_metrics.get("brier_score", 0.0)
ece_val = test_metrics.get("expected_calibration_error", 0.0)
review_rate_val = test_metrics.get("review_rate", 0.0)
fraud_capture_val = test_metrics.get("fraud_capture_rate", 0.0)
blocked_value_val = test_metrics.get("blocked_fraud_value_inr", 0.0)

# Sidebar Navigation
st.sidebar.title("Fraud Intelligence")
st.sidebar.caption("Fraud Risk Intelligence Prototype v3.0")
st.sidebar.markdown("**Author**: Sanman Kadam")

nav_option = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Transaction Risk Checker",
        "Fraud Investigation Queue",
        "Model and Risk Analytics",
        "Synthetic UPI and Graph Analytics"
    ]
)

st.sidebar.subheader("System Status")
st.sidebar.success("XGBoost Model: ACTIVE (Calibrated)")
if risk_engine.is_anomaly_active:
    st.sidebar.success("Isolation Forest: ACTIVE (Fitted)")
else:
    st.sidebar.warning("Isolation Forest: INACTIVE (Unfitted baseline)")
st.sidebar.info(f"Risk Engine: ONLINE ({risk_engine.w_ml:.0%} ML / {risk_engine.w_anomaly:.0%} Anomaly / {risk_engine.w_rules:.0%} Rules)")
st.sidebar.caption(f"Locked Threshold: **{opt_thresh:.4f}** (Validation Locked)")
st.sidebar.caption(f"Test PR-AUC Score: **{pr_auc_val:.4f}**")

# TAB 1: EXECUTIVE OVERVIEW
if nav_option == "Executive Overview":
    st.title("Executive Fraud Risk Intelligence Overview")
    st.markdown("Model performance metrics, dataset provenance, and financial loss indicators — all loaded from pipeline artifacts.")

    # Top KPI Row (Dynamically Populated from model_metadata.json)
    col1, col2, col3, col4 = st.columns(4)
    
    # Dataset volume from dataset_summary.json or metadata
    total_records = dataset_summary.get("total_records", meta.get("data_provenance", {}).get("sample_size", "N/A"))
    total_fraud = dataset_summary.get("total_fraud", "N/A")
    
    with col1:
        display_vol = f"{total_records:,}" if isinstance(total_records, int) else str(total_records)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Processed Dataset Volume</div>
            <div class="metric-value">{display_vol}</div>
            <span style="color:#22c55e;">PaySim Financial Simulation</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Fraud Recall</div>
            <div class="metric-value">{recall_val*100:.1f} %</div>
            <span style="color:#ef4444;">@ Locked Threshold {opt_thresh:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test PR-AUC Score</div>
            <div class="metric-value">{pr_auc_val:.4f}</div>
            <span style="color:#6366f1;">Calibrated Optuna XGBoost</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Expected Financial Loss</div>
            <div class="metric-value">INR {loss_val:,.0f}</div>
            <span style="color:#94a3b8;">Actual Missed Fraud Amounts + FP Cost</span>
        </div>
        """, unsafe_allow_html=True)

    # Section 1: Transaction Type and Fraud Distribution (from dataset_summary.json)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fraud Rate by Transaction Type")
        fraud_by_type = dataset_summary.get("fraud_by_type", [])
        if fraud_by_type:
            df_types = pd.DataFrame(fraud_by_type)
            df_types.rename(columns={
                "type": "Type", "total_transactions": "Total", 
                "fraud_count": "Fraud", "fraud_rate_pct": "Fraud_Rate"
            }, inplace=True)
        else:
            st.info("Run `python src/train.py` to generate dataset summary.")
            df_types = pd.DataFrame(columns=["Type", "Total", "Fraud", "Fraud_Rate"])
        
        if not df_types.empty:
            fig_type = px.bar(
                df_types, x="Type", y="Fraud_Rate", text="Fraud",
                color="Fraud_Rate", color_continuous_scale="Reds",
                title="Fraud Rate (%) by Payment Rail Type"
            )
            fig_type.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_type, use_container_width=True)

    with c2:
        st.subheader("Business KPIs at Operating Threshold")
        kpi_data = pd.DataFrame([
            {"KPI": "Fraud Capture Rate", "Value": f"{fraud_capture_val*100:.1f}%"},
            {"KPI": "Review Rate", "Value": f"{review_rate_val*100:.1f}%"},
            {"KPI": "Precision", "Value": f"{precision_val*100:.1f}%"},
            {"KPI": "Blocked Fraud Value (INR)", "Value": f"₹{blocked_value_val:,.0f}"},
            {"KPI": "Expected Financial Loss (INR)", "Value": f"₹{loss_val:,.0f}"},
            {"KPI": "Brier Score (Calibration)", "Value": f"{brier_val:.6f}"},
            {"KPI": "Expected Calibration Error", "Value": f"{ece_val:.6f}"},
        ])
        st.dataframe(kpi_data, use_container_width=True, hide_index=True)

    # Section 2: Data Provenance Card
    st.subheader("Data Provenance and Model Context")
    st.info("""
    **Data and Model Provenance**:
    * **Training Data**: PaySim synthetic mobile-money simulation (Kaggle). This is explicitly synthetic data — not real banking transactions.
    * **Transaction Scope**: Model trained on TRANSFER and CASH_OUT transaction types only, as PaySim fraud labels are concentrated in these categories.
    * **Threshold Selection**: Operating threshold locked on Validation set. Test set used only for final one-time evaluation.
    * **Calibration**: Probabilities calibrated via Platt scaling (sigmoid) on validation set.
    * **Download Dataset**: Obtain `PS_20174392719_1491204439457_log.csv` from [Kaggle PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place in `data/`.
    """)

# TAB 2: TRANSACTION RISK CHECKER
elif nav_option == "Transaction Risk Checker":
    st.title("Interactive Transaction Risk Checker")
    st.markdown("Simulate single digital payment transactions with **stateful customer baselines** and inspect Risk Engine decisions.")

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

    if scenario == "legit":
        default_amt, default_type, default_hour, default_old_bal, default_orig, default_dest = 1250.0, "PAYMENT", 14, 15000.0, "C928310482", "M102938471"
    elif scenario == "critical":
        default_amt, default_type, default_hour, default_old_bal, default_orig, default_dest = 84500.0, "TRANSFER", 2, 84500.0, "C928310482", "C938201941"
    elif scenario == "surge":
        default_amt, default_type, default_hour, default_old_bal, default_orig, default_dest = 45000.0, "CASH_OUT", 1, 45000.0, "C482019482", "C849201942"
    else:
        default_amt, default_type, default_hour, default_old_bal, default_orig, default_dest = 25000.0, "TRANSFER", 3, 25000.0, "C928310482", "C102938471"

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
        with c3:
            name_orig = st.text_input("Originator Account ID", value=default_orig)
            name_dest = st.text_input("Beneficiary Account ID", value=default_dest)

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

        # Stateful Single Prediction (computes genuine historical velocity via CustomerStateStore)
        pred_res = predictor.predict_single(tx_dict, update_state=True)
        ml_prob = pred_res['fraud_probability']
        feature_row = pd.Series(pred_res['raw_features'])

        # Calculate Risk Engine score
        risk_result = risk_engine.calculate_risk(ml_prob, feature_row, tx_dict)

        # Log evaluation to SQLite Database
        db_mgr.log_evaluation(tx_dict, risk_result)

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
                <div class="metric-title">Calibrated Fraud Probability</div>
                <div class="metric-value">{ml_prob*100:.1f}%</div>
                <span style="color:#94a3b8;">Locked Threshold: {predictor.optimal_threshold:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        # Show component weights from engine config
        engine_cfg = risk_result.get("engine_config", {})
        weight_label = f"({engine_cfg.get('w_ml', 0.6)*100:.0f}% ML, {engine_cfg.get('w_anomaly', 0.2)*100:.0f}% Anomaly, {engine_cfg.get('w_rules', 0.2)*100:.0f}% Rules)"
        anomaly_status = "ACTIVE" if risk_result['components'].get('anomaly_model_active', False) else "BASELINE"
        
        st.markdown(f"#### Risk Component Score Contribution {weight_label}")
        comp_df = pd.DataFrame([
            {"Component": f"Calibrated XGBoost ML", "Points": risk_result['components']['ml_contribution']},
            {"Component": f"Isolation Forest Anomaly [{anomaly_status}]", "Points": risk_result['components']['anomaly_contribution']},
            {"Component": "Business Rules Engine", "Points": risk_result['components']['rule_contribution']}
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
    st.markdown("Decision queue connected to the **SQLite Warehouse (`data/processed/fraud_intelligence.db`)**.")

    # Fetch live queue from SQLite DB
    df_queue = db_mgr.get_investigation_queue()
    st.dataframe(df_queue, use_container_width=True)

    st.subheader("Take Action on Case")
    qc1, qc2, qc3 = st.columns(3)
    case_list = df_queue["case_id"].tolist() if not df_queue.empty else ["CASE_92810"]
    with qc1:
        sel_case = st.selectbox("Select Case ID", case_list)
    with qc2:
        new_status = st.selectbox("Update Case Decision", ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "UNDER_REVIEW", "ALLOW"])
    with qc3:
        analyst_id = st.text_input("Analyst Name / ID", value="Analyst_042")

    notes = st.text_area("Analyst Investigation Notes", value="")

    if st.button("Submit Case Decision"):
        db_mgr.update_case_decision(sel_case, new_status, analyst_id, notes)
        st.success(f"Case {sel_case} successfully updated to '{new_status}' in SQLite Database by {analyst_id}.")

# TAB 4: MODEL AND RISK ANALYTICS
elif nav_option == "Model and Risk Analytics":
    st.title("Model Evaluation and Cost-Sensitive Risk Analytics")
    st.markdown("Performance benchmarks, calibration metrics, and **actual validation threshold-cost curves** loaded from pipeline artifacts.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC (Test)", f"{pr_auc_val:.4f}")
    col2.metric("ROC-AUC (Test)", f"{roc_auc_val:.4f}")
    col3.metric("Recall @ Locked Thresh", f"{recall_val*100:.1f}%", f"{precision_val*100:.1f}% Precision")
    col4.metric("Locked Threshold", f"{opt_thresh:.4f}", "Validation Locked")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Precision-Recall Curve (from Threshold Table)")
        if not df_thresh.empty and "recall" in df_thresh.columns and "precision" in df_thresh.columns:
            fig_pr = px.line(
                df_thresh.sort_values("recall"), x="recall", y="precision", 
                title=f"Precision-Recall Curve (PR-AUC = {pr_auc_val:.4f})", 
                color_discrete_sequence=["#6366f1"]
            )
            fig_pr.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning("Threshold cost table not found. Run `python src/train.py` to generate.")

    with c2:
        st.subheader("Expected Financial Loss vs Decision Threshold")
        if not df_thresh.empty and "threshold" in df_thresh.columns and "total_financial_loss_inr" in df_thresh.columns:
            fig_loss = px.line(
                df_thresh, x="threshold", y="total_financial_loss_inr", 
                title=f"Validation Threshold Selection (Min Loss at {opt_thresh:.2f})", 
                color_discrete_sequence=["#22c55e"]
            )
            fig_loss.add_vline(x=opt_thresh, line_dash="dash", line_color="#ef4444", annotation_text=f"Locked {opt_thresh:.2f}")
            fig_loss.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.warning("Threshold cost table not found. Run `python src/train.py` to generate.")

    # Model Comparison Table
    st.subheader("Model Comparison (from Pipeline Artifacts)")
    baseline_comp = meta.get("baseline_comparison", {})
    if baseline_comp:
        comp_rows = []
        for model_name, metrics in baseline_comp.items():
            comp_rows.append({
                "Model": model_name.replace("_", " "),
                "PR-AUC": metrics.get("pr_auc", 0),
                "ROC-AUC": metrics.get("roc_auc", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1": metrics.get("f1_score", 0),
                "Total Loss (INR)": f"₹{metrics.get('total_financial_loss_inr', 0):,.0f}"
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Calibration metrics
    cal_info = meta.get("calibration", {})
    if cal_info:
        st.subheader("Probability Calibration")
        cal_c1, cal_c2, cal_c3 = st.columns(3)
        cal_c1.metric("Calibration Method", cal_info.get("method", "N/A"))
        cal_c2.metric("Brier Score", f"{cal_info.get('brier_score', 0):.6f}")
        cal_c3.metric("Expected Calibration Error", f"{cal_info.get('expected_calibration_error', 0):.6f}")

# TAB 5: SYNTHETIC UPI AND GRAPH ANALYTICS
elif nav_option == "Synthetic UPI and Graph Analytics":
    st.title("Synthetic Indian UPI Layer and Graph Fraud Analytics")
    st.markdown("Network MultiDiGraph visualization for money mule rings, circular payment chains, and synthetic UPI app metadata.")

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

    st.subheader("Money Mule Ring MultiDiGraph Network Centrality Analysis")
    analyzer = FraudGraphAnalyzer()
    analyzer.build_graph_from_dataframe(df_upi.iloc[:1500], orig_col='customer_id', dest_col='beneficiary_id', amount_col='amount_inr', fraud_col='is_fraud')
    df_metrics = analyzer.compute_network_metrics()

    st.dataframe(df_metrics.head(10), use_container_width=True)
