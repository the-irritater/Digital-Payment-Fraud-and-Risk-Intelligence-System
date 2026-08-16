# Digital Payment Fraud and Risk Intelligence Platform

<p align="center">
  <strong>Real Time Fraud Detection. Explainable Risk Intelligence. Cost Sensitive Decisioning.</strong>
</p>

<p align="center">
  An end to end fraud intelligence platform that combines supervised machine learning, anomaly detection, behavioral analytics, explainable AI, graph analytics and rule based decisioning to identify, explain and prioritize suspicious digital payment transactions.
</p>

<p align="center">
  <a href="YOUR_STREAMLIT_URL">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQL-MySQL%20%7C%20SQL%20Server-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="SQL">
  <img src="https://img.shields.io/badge/XGBoost-Optuna-FF6600?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Analytics-4C8BF5?style=for-the-badge" alt="NetworkX">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
</p>

<p align="center">
  <strong>Primary Metric: PR AUC 0.9515</strong>
  <br>
  <strong>Temporal Test Recall: 100%</strong>
  <br>
  <strong>ROC AUC: 0.9964</strong>
</p>

---

## Live Demo

<p align="center">
  <a href="YOUR_STREAMLIT_URL">
    <img src="https://img.shields.io/badge/Open%20Live%20Fraud%20Risk%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Dashboard">
  </a>
</p>

The deployed Streamlit application provides an interactive interface for exploring the fraud intelligence pipeline.

The dashboard is designed around the complete risk decision workflow:

```text
Transaction
     │
     ▼
Fraud Prediction
     │
     ▼
Risk Score
     │
     ▼
Risk Tier
     │
     ├── ML Probability
     ├── Anomaly Signal
     ├── Business Rules
     └── Explainability
             │
             ▼
      Operational Decision
```

### Dashboard Capabilities

| Capability          | Description                              |
| :------------------ | :--------------------------------------- |
| Transaction Scoring | Evaluate individual payment transactions |
| Fraud Probability   | View supervised model probability        |
| Anomaly Detection   | Identify unusual behavioral patterns     |
| Risk Score          | Composite score from 0 to 100            |
| Risk Tier           | LOW, MEDIUM, HIGH and CRITICAL           |
| Decisioning         | ALLOW, REVIEW or BLOCK                   |
| SHAP Explainability | Understand important model drivers       |
| Graph Intelligence  | Explore structural transaction risk      |
| Behavioral Analysis | Identify unusual transaction behavior    |

---

# Project Overview

Digital payment fraud is not simply a binary classification problem.

A practical fraud intelligence platform must answer three questions:

```text
1. Is the transaction suspicious?

2. Why is the transaction suspicious?

3. What action should be taken?
```

This project addresses these questions through a hybrid risk intelligence architecture combining:

```text
Supervised Machine Learning
            +
Unsupervised Anomaly Detection
            +
Behavioral Analytics
            +
Business Rules
            +
Explainable AI
            +
Graph Analytics
            +
Cost Sensitive Decisioning
```

The result is an end to end fraud risk platform designed to move beyond model accuracy and support operational fraud investigation.

---

# Key Results

The final XGBoost model was evaluated using a strict temporal test set.

| Metric                  |     Result    |
| :---------------------- | :-----------: |
| PR AUC                  |   **0.9515**  |
| ROC AUC                 |   **0.9964**  |
| Operating Threshold     |    **0.17**   |
| Precision               |   **0.3100**  |
| Recall                  |   **1.0000**  |
| F1 Score                |   **0.4733**  |
| Observed Fraud Detected | **186 / 186** |

At the selected operating threshold of 0.17 the model detected all observed fraud cases in the temporal test set.

The threshold intentionally prioritizes recall because missing fraudulent transactions can have a significantly higher operational cost than investigating legitimate transactions.

---

# System Architecture

```text
                         TRANSACTION LOG
                                │
                                ▼
                 ┌──────────────────────────┐
                 │     DATA VALIDATION      │
                 │      PREPROCESSING       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    FEATURE ENGINEERING   │
                 │     CAUSAL FEATURES      │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │  XGBoost    │  │  Isolation  │  │  Business   │
      │ Fraud Model │  │   Forest    │  │    Rules    │
      └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │   HYBRID RISK      │
                    │      ENGINE         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     RISK SCORE      │
                    │       0 to 100      │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
            ALLOW           REVIEW            BLOCK
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ SHAP EXPLANATIONS        │
                 │ GRAPH INTELLIGENCE       │
                 │ ANALYST DASHBOARD        │
                 └──────────────────────────┘
```

---

# Fraud Intelligence Pipeline

```text
Raw Transactions
       │
       ▼
Data Validation
       │
       ▼
Temporal Ordering
       │
       ▼
Causal Feature Engineering
       │
       ▼
Model Benchmarking
       │
       ▼
Optuna Hyperparameter Optimization
       │
       ▼
Temporal Model Evaluation
       │
       ▼
Threshold Cost Optimization
       │
       ▼
SHAP Explainability
       │
       ▼
Isolation Forest
       │
       ▼
Behavioral Analytics
       │
       ▼
Graph Risk Analysis
       │
       ▼
Hybrid Risk Engine
       │
       ▼
Streamlit Risk Dashboard
```

---

# Data Provenance

The project deliberately separates transaction level modelling data from Indian payment ecosystem context.

| Data Source                   | Purpose                                               | Nature                            |
| :---------------------------- | :---------------------------------------------------- | :-------------------------------- |
| PaySim                        | Transaction level fraud model training and evaluation | Synthetic mobile money simulation |
| RBI Payment System Indicators | Indian payment ecosystem context                      | Official aggregate statistics     |
| Synthetic UPI Generator       | Indian UPI like behavioral simulation                 | Explicitly synthetic              |

## PaySim Dataset

The transaction level fraud model is trained and evaluated using the PaySim synthetic financial transaction dataset.

The dataset contains approximately 6.3 million simulated transactions and provides transaction level information for fraud classification and behavioral analysis.

PaySim is synthetic and therefore model performance on PaySim should not be interpreted as equivalent to performance on real banking transactions.

## RBI Payment System Indicators

RBI payment system statistics are used to provide macro level context around India's digital payment ecosystem.

RBI aggregate statistics are not used as transaction level labels for the fraud classification model.

## Synthetic UPI Generator

The project includes a synthetic UPI transaction generator designed to simulate Indian payment behavior.

The generated data is explicitly synthetic and is not presented as real banking or customer data.

---

# India Payment Ecosystem Context

India's digital payment ecosystem operates at very high transaction volumes which creates a strong requirement for automated fraud risk assessment.

The project considers representative risk patterns including:

```text
Social Engineering
        │
        ▼
Phishing and Unauthorized Transactions


High Velocity Activity
        │
        ▼
Rapid Transaction Bursts


Mule Accounts
        │
        ▼
Multiple Account Relationships


Behavioral Changes
        │
        ▼
Unusual Amounts, Devices and Beneficiaries
```

RBI payment system statistics are used only as ecosystem context.

---

# Machine Learning Models

Three classification approaches were benchmarked.

```text
Logistic Regression
        │
        ▼
Random Forest
        │
        ▼
XGBoost
        │
        ▼
Optuna Optimization
```

## Model Comparison

| Model                   |     PR AUC | Threshold |  Precision |     Recall |         F1 |
| :---------------------- | ---------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression     |     0.7959 |      0.50 |     0.1840 |     0.7634 |     0.2966 |
| Random Forest           |     0.8660 |      0.50 |     0.4180 |     0.8817 |     0.5670 |
| **XGBoost with Optuna** | **0.9515** |  **0.17** | **0.3100** | **1.0000** | **0.4733** |

XGBoost was selected as the primary supervised fraud model.

---

# Why PR AUC?

Fraud detection is typically an imbalanced classification problem.

Accuracy can therefore be misleading.

The project uses **Precision Recall AUC** as the primary model selection metric because it provides a more informative evaluation of fraud detection performance under class imbalance.

The model was also evaluated using ROC AUC, precision, recall and F1 score.

---

# Temporal Validation

A temporal evaluation strategy was used instead of a conventional random train test split.

```text
Earlier Transactions
        │
        ▼
Training Data
        │
        ▼
Model Training
        │
        ▼
Later Transactions
        │
        ▼
Temporal Test Data
```

This approach better represents the direction of real transaction data and helps reduce the risk of unrealistic evaluation caused by random temporal mixing.

---

# Feature Engineering

The project uses causal feature construction to create transaction level risk indicators.

| Category    | Example Signals                 |
| :---------- | :------------------------------ |
| Transaction | Amount and transaction type     |
| Velocity    | Recent transaction frequency    |
| Behavioral  | Changes in transaction behavior |
| Device      | Device changes and consistency  |
| Beneficiary | Beneficiary churn               |
| Temporal    | Time based transaction patterns |
| Network     | Degree centrality and PageRank  |

The feature engineering pipeline is implemented in:

```text
src/feature_engineering.py
```

---

# Cost Sensitive Threshold Optimization

Fraud detection should not rely on a default classification threshold such as 0.50.

The project evaluates different operating thresholds using an explicit financial loss model.

```text
Financial Loss
=
Actual Amount of Missed Fraud
+
False Positive Investigation Cost
```

The current investigation cost assumption is:

```text
False Positive Investigation Cost = 200
```

The threshold analysis allows the system to evaluate the operational tradeoff between:

```text
Higher Recall
      vs
Higher Precision
      vs
Investigation Cost
      vs
Missed Fraud Loss
```

The complete analysis is available in:

```text
reports/threshold_cost_table.csv
```

---

# Hybrid Risk Engine

The fraud probability is combined with anomaly detection and deterministic business rules.

```text
Risk Score
=
ML Probability × 100 × 0.60
+
Anomaly Score × 100 × 0.20
+
Rule Score × 0.20
```

| Component           |   Scale  | Weight | Purpose                       |
| :------------------ | :------: | :----: | :---------------------------- |
| XGBoost Probability | 0 to 100 |   60%  | Supervised fraud probability  |
| Isolation Forest    | 0 to 100 |   20%  | Unsupervised anomaly signal   |
| Business Rules      | 0 to 100 |   20%  | Deterministic risk indicators |

This architecture allows multiple sources of risk evidence to contribute to the final decision.

---

# Risk Tiers and Decisioning

| Risk Score | Tier     | Action | Operational Response                       |
| :--------: | :------- | :----- | :----------------------------------------- |
|   0 to 30  | LOW      | ALLOW  | Automated approval                         |
|  31 to 60  | MEDIUM   | REVIEW | Step up authentication                     |
|  61 to 80  | HIGH     | REVIEW | Enhanced authentication and analyst review |
|  81 to 100 | CRITICAL | BLOCK  | Block transaction and escalate account     |

The thresholds are configurable and should be calibrated using representative production data before real world deployment.

---

# Explainable AI

A fraud prediction without an explanation can be difficult for analysts to act upon.

The project uses **SHAP** to identify the features contributing to individual model predictions.

```text
Transaction
      │
      ▼
Fraud Probability
      │
      ▼
SHAP Analysis
      │
      ├── Transaction Amount
      ├── Transaction Velocity
      ├── Behavioral Change
      ├── Beneficiary Activity
      └── Other Model Features
              │
              ▼
       Analyst Explanation
```

SHAP explanations provide transparency into model behavior and help analysts understand why a transaction received a particular fraud probability.

---

# Anomaly Detection

The platform uses **Isolation Forest** as an additional unsupervised risk signal.

The objective is to identify transactions with unusual behavioral characteristics that may not be fully represented by supervised fraud labels.

The architecture therefore combines:

```text
Supervised Fraud Signal
          +
Unsupervised Anomaly Signal
          +
Deterministic Business Rules
          │
          ▼
Hybrid Risk Assessment
```

---

# Behavioral Risk Analytics

The platform incorporates behavioral patterns such as:

```text
Transaction Velocity
        │
        ▼
Amount Behavior
        │
        ▼
Device Changes
        │
        ▼
Beneficiary Churn
        │
        ▼
Behavioral Risk
```

These signals are intended to provide additional context around suspicious transaction activity.

---

# Synthetic UPI Simulation

The project includes a synthetic UPI generator with three behavioral personas.

| Persona    | Characteristics                                    |
| :--------- | :------------------------------------------------- |
| Normal     | Stable transaction behavior and lower velocity     |
| Suspicious | Elevated velocity and unusual transaction patterns |
| Mule       | High beneficiary churn and rapid movement of funds |

The synthetic generator is intended for experimentation and risk engineering.

It does not represent real customer behavior or real banking transaction data.

---

# Graph Based Fraud Intelligence

Fraud can involve networks of accounts rather than isolated transactions.

The project uses **NetworkX** to construct transaction relationship graphs.

The graph analysis includes:

| Graph Feature        | Purpose                                 |
| :------------------- | :-------------------------------------- |
| Degree Centrality    | Measures account connectivity           |
| PageRank             | Identifies structurally important nodes |
| Connected Components | Identifies relationship structures      |

The workflow is:

```text
Transactions
      │
      ▼
Account Relationships
      │
      ▼
Transaction Graph
      │
      ▼
Graph Features
      │
      ▼
Structural Risk Indicators
```

Graph features are treated as risk indicators and do not establish that an account or transaction is fraudulent.

---

# SQL Analytics

The project includes a SQL layer for structured transaction and fraud analysis.

```text
sql/
├── schema.sql
└── fraud_analysis.sql
```

The SQL layer provides a foundation for analytical workflows around:

```text
Transaction Analysis
Fraud Patterns
Risk Segmentation
Behavioral Analysis
Account Activity
```

---

# Streamlit Application

The project includes an interactive Streamlit dashboard for analyst oriented fraud investigation.

```text
Transaction Input
       │
       ▼
Model Prediction
       │
       ▼
Hybrid Risk Engine
       │
       ▼
Risk Score
       │
       ├── Fraud Probability
       ├── Anomaly Score
       ├── Rule Score
       └── SHAP Explanation
               │
               ▼
        Decision Recommendation
```

The application is designed to make the underlying analytical pipeline accessible through an interactive interface.

---

# Model Governance

The project includes a model card documenting model provenance, evaluation methodology, intended use and known limitations.

```text
reports/model_card.md
```

## Important Limitations

* PaySim is synthetic
* Synthetic UPI data is generated by the project
* Model performance on PaySim does not guarantee equivalent real world performance
* Risk thresholds require calibration using real operational costs
* Graph features indicate structural risk rather than proof of fraud
* RBI aggregate statistics are not used as transaction level fraud labels
* Production deployment would require extensive validation and governance

---

# Testing

Automated tests are included for important components of the platform.

```text
tests/
├── test_features.py
├── test_risk_engine.py
└── test_prediction.py
```

Run the complete test suite:

```bash
pytest tests/ -v
```

Testing focuses on:

```text
Feature Engineering
Risk Tier Logic
Decision Actions
Prediction Pipeline
```

---

# Repository Structure

```text
digital-payment-fraud-intelligence/
│
├── data/
│   ├── PS_20174392719_1491204439457_log.csv
│   └── processed/
│
├── notebooks/
│   └── Digital_Payment_Fraud_Intelligence_Complete.ipynb
│
├── src/
│   ├── create_merged_notebook.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── risk_engine.py
│   ├── synthetic_upi.py
│   └── graph_fraud.py
│
├── sql/
│   ├── schema.sql
│   └── fraud_analysis.sql
│
├── app/
│   ├── streamlit_app.py
│   └── style.css
│
├── models/
│   ├── xgboost_model.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── model_card.md
│   ├── model_comparison.csv
│   ├── threshold_cost_table.csv
│   └── rbi_india_context.md
│
├── tests/
│   ├── test_features.py
│   ├── test_risk_engine.py
│   └── test_prediction.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/the-irritater/digital-payment-fraud-intelligence.git

cd digital-payment-fraud-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Model Training

Run the training pipeline:

```bash
python src/train.py
```

The training workflow includes:

```text
Data Processing
Feature Engineering
Temporal Split
Model Benchmarking
Optuna Optimization
Model Evaluation
Threshold Cost Analysis
Model Metadata Generation
```

---

# Run Tests

```bash
pytest tests/ -v
```

---

# Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# Complete Project Notebook

The complete analytical workflow is available in:

```text
notebooks/Digital_Payment_Fraud_Intelligence_Complete.ipynb
```

The notebook integrates:

```text
Data Processing
       ↓
Feature Engineering
       ↓
Temporal Validation
       ↓
Model Benchmarking
       ↓
Optuna Optimization
       ↓
Threshold Optimization
       ↓
SHAP Explainability
       ↓
Isolation Forest
       ↓
Synthetic UPI Simulation
       ↓
Graph Analytics
       ↓
Hybrid Risk Engine
```

---

# Technology Stack

| Technology   | Purpose                                |
| :----------- | :------------------------------------- |
| Python       | Core development                       |
| Pandas       | Data processing                        |
| NumPy        | Numerical computing                    |
| Scikit Learn | Machine learning and anomaly detection |
| XGBoost      | Fraud classification                   |
| Optuna       | Hyperparameter optimization            |
| SHAP         | Explainable AI                         |
| NetworkX     | Graph analytics                        |
| SQL          | Analytical processing                  |
| Streamlit    | Interactive dashboard                  |
| Matplotlib   | Visualization                          |
| Pytest       | Automated testing                      |
| Jupyter      | Research and experimentation           |

---

# Future Roadmap

The current platform is designed as a strong research and engineering prototype.

Potential production extensions include:

### Real Time Fraud Scoring

```text
Payment Event
      │
      ▼
Kafka
      │
      ▼
Feature Store
      │
      ▼
Fraud API
      │
      ▼
Risk Engine
      │
      ▼
ALLOW
REVIEW
BLOCK
```

### Planned Enhancements

* FastAPI based model serving
* Kafka based transaction streaming
* Online feature computation
* Feature store integration
* MLflow experiment tracking
* Model drift monitoring
* Data quality monitoring
* Docker based deployment
* Automated model retraining
* Champion challenger modelling
* Real time risk APIs
* Advanced graph embeddings
* Account level risk profiling
* Analyst feedback loops
* Production model monitoring

---

# Why This Project Is Different

Many fraud detection projects focus primarily on model accuracy.

This project focuses on the complete fraud intelligence lifecycle:

```text
                 ┌──────────────────┐
                 │     Detection    │
                 │    XGBoost       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Explanation   │
                 │      SHAP        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Risk Assessment │
                 │ ML + Anomaly +   │
                 │ Rules            │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Decisioning   │
                 │ Allow / Review / │
                 │ Block            │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Analyst Dashboard│
                 │    Streamlit     │
                 └──────────────────┘
```

The project therefore demonstrates practical application of:

```text
Machine Learning
       +
Statistical Thinking
       +
Fraud Analytics
       +
Explainable AI
       +
Anomaly Detection
       +
Graph Intelligence
       +
SQL Analytics
       +
Cost Sensitive Decisioning
       +
Software Testing
       +
Interactive Deployment
```

---

# Responsible Use

This project is an analytical and engineering prototype.

The transaction level fraud model is trained and evaluated on synthetic PaySim data.

The UPI like transaction generator produces synthetic data.

Neither dataset should be interpreted as real banking customer data.

The model should not be deployed for real financial decisions without validation using representative production data, appropriate calibration, monitoring, governance and human oversight.

Risk scores should support fraud investigation and operational decision making rather than independently determining adverse outcomes.

---

# Data Sources

### PaySim

https://www.kaggle.com/datasets/ealaxi/paysim1

### Reserve Bank of India Payment System Indicators

https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41

---

# Author

**Sanman Kadam**

MSc Statistics
Data Analytics and Data Science

This project demonstrates the application of statistical modelling, machine learning, explainable AI, anomaly detection, graph analytics and business decision intelligence to digital payment fraud risk.

---

# Project Summary

```text
DIGITAL PAYMENT FRAUD
          │
          ▼
   DATA INTELLIGENCE
          │
          ▼
   MACHINE LEARNING
          │
          ▼
  ANOMALY DETECTION
          │
          ▼
   GRAPH ANALYTICS
          │
          ▼
 EXPLAINABLE RISK SCORE
          │
          ▼
  BUSINESS DECISION
          │
     ┌────┼────┐
     ▼    ▼    ▼
   ALLOW REVIEW BLOCK
          │
          ▼
    STREAMLIT DASHBOARD
```

<p align="center">
  <strong>Detection. Explanation. Decision.</strong>
</p>

<p align="center">
  Built for fraud intelligence research and practical risk analytics.
</p>
