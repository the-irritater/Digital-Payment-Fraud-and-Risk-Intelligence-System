# Digital Payment Fraud and Risk Intelligence Platform

<p align="center">
  <strong>Real Time Fraud Detection. Explainable Risk Intelligence. Cost Sensitive Decisioning.</strong>
</p>

<p align="center">
  An end to end fraud intelligence platform combining supervised machine learning, anomaly detection, behavioral analytics, explainable AI, graph analytics and rule based decisioning to identify, explain and prioritize suspicious digital payment transactions.
</p>

<p align="center">
  <a href="https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/">
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
  <strong>PR AUC 0.9515</strong>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>ROC AUC 0.9964</strong>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>Recall 100%</strong>
</p>

---

## Live Demo

### Explore the deployed fraud intelligence platform

[Open the Live Streamlit Fraud Risk Intelligence Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/?utm_source=chatgpt.com)

The live application provides an interactive interface for exploring transaction level fraud predictions, hybrid risk scores, anomaly signals, business rules and explainable AI outputs.

### Dashboard Workflow

```text
Transaction
     │
     ▼
Fraud Prediction
     │
     ▼
Hybrid Risk Engine
     │
     ▼
Risk Score
     │
     ├── ML Probability
     ├── Anomaly Score
     ├── Business Rules
     └── SHAP Explanation
             │
             ▼
      Risk Classification
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     ALLOW REVIEW BLOCK
```

---

# Project Overview

Digital payment fraud is not simply a binary classification problem.

A practical fraud intelligence platform needs to answer three questions:

```text
1. Is this transaction suspicious?

2. Why is this transaction suspicious?

3. What action should be taken?
```

This project addresses these questions through a hybrid fraud intelligence architecture combining:

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

The result is an end to end fraud risk intelligence platform designed to move beyond model accuracy and provide actionable risk intelligence.

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

The threshold intentionally prioritizes recall because missing fraudulent transactions can have a substantially higher operational cost than investigating legitimate transactions.

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
                 │    CAUSAL FEATURE        │
                 │       ENGINEERING        │
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
                 │ STREAMLIT DASHBOARD      │
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

| Data Source                   | Purpose                               | Nature                            |
| :---------------------------- | :------------------------------------ | :-------------------------------- |
| PaySim                        | Fraud model training and evaluation   | Synthetic mobile money simulation |
| RBI Payment System Indicators | Indian payment ecosystem context      | Official aggregate statistics     |
| Synthetic UPI Generator       | Indian UPI like behavioral simulation | Explicitly synthetic              |

## PaySim

The transaction level fraud model is trained and evaluated using the PaySim synthetic financial transaction dataset.

The dataset contains approximately 6.3 million simulated transactions and provides transaction level information suitable for fraud classification and behavioral analysis.

PaySim is synthetic. Therefore model performance on PaySim should not be interpreted as equivalent to performance on real banking transactions.

## RBI Payment System Indicators

RBI payment system statistics are used strictly as macro level context for India's digital payment ecosystem.

RBI aggregate statistics are not used as transaction level fraud labels for the machine learning model.

## Synthetic UPI Generator

The project includes a synthetic UPI transaction generator designed to simulate Indian payment behavior.

The generated transactions are explicitly synthetic and are not presented as real banking or customer data.

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

RBI payment system statistics are used only for ecosystem context.

---

# Machine Learning Models

Three supervised classification approaches were benchmarked.

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

# Why PR AUC

Fraud detection is typically an imbalanced classification problem.

Accuracy can therefore provide a misleading view of model performance.

The project uses **Precision Recall AUC** as the primary evaluation metric because it provides a more informative assessment of fraud detection under class imbalance.

The model is also evaluated using:

```text
PR AUC
ROC AUC
Precision
Recall
F1 Score
```

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

This better represents the direction of real transaction data and reduces the risk of unrealistic evaluation caused by randomly mixing historical and future observations.

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

The threshold analysis evaluates the tradeoff between:

```text
Higher Recall
      vs
Higher Precision
      vs
Investigation Cost
      vs
Missed Fraud Loss
```

The complete threshold analysis is available in:

```text
reports/threshold_cost_table.csv
```

---

# Hybrid Risk Engine

The platform combines three risk signals into a composite risk score.

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

This allows multiple sources of risk evidence to contribute to the final decision.

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

Fraud detection requires more than a prediction.

Analysts need to understand why a transaction received a high risk score.

The platform uses **SHAP** to identify features contributing to individual model predictions.

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

SHAP explanations improve transparency and help analysts understand the drivers behind individual predictions.

---

# Anomaly Detection

The platform uses **Isolation Forest** as an additional unsupervised risk signal.

The objective is to identify unusual behavioral patterns that may not be fully represented by supervised fraud labels.

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

The platform incorporates behavioral indicators including:

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

These features provide additional context around suspicious transaction activity.

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

# Streamlit Risk Intelligence Dashboard

The project includes a deployed Streamlit application for interactive fraud risk analysis.

### Live Application

[Digital Payment Fraud and Risk Intelligence Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/?utm_source=chatgpt.com)

The dashboard provides:

| Dashboard Feature    | Purpose                                 |
| :------------------- | :-------------------------------------- |
| Transaction Input    | Evaluate transaction level risk         |
| Fraud Probability    | Display supervised model probability    |
| Anomaly Score        | Display unsupervised risk signal        |
| Rule Score           | Display deterministic rule contribution |
| Composite Risk Score | Combine multiple risk signals           |
| Risk Tier            | Classify transaction risk               |
| Decision             | Recommend ALLOW, REVIEW or BLOCK        |
| SHAP Analysis        | Explain model predictions               |
| Graph Analysis       | Explore structural risk indicators      |

---

# SQL Analytics

The project includes a SQL layer for structured fraud analysis.

```text
sql/
├── schema.sql
└── fraud_analysis.sql
```

The SQL layer supports analytical workflows involving:

```text
Transaction Analysis
Fraud Patterns
Risk Segmentation
Behavioral Analysis
Account Activity
```

---

# Model Governance

A dedicated model card documents model provenance, evaluation methodology, intended use and known limitations.

```text
reports/model_card.md
```

## Important Limitations

* PaySim is synthetic
* Synthetic UPI transactions are generated by the project
* Model performance on PaySim does not guarantee equivalent real world performance
* Risk thresholds require calibration using real operational costs
* Graph features represent structural risk rather than proof of fraud
* RBI aggregate statistics are not used as transaction level fraud labels
* Production deployment requires additional validation and governance

---

# Testing

Automated tests are included for important components.

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

Testing covers:

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

# Run the Streamlit Application Locally

```bash
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

For the deployed application:

[Open Live Streamlit Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/?utm_source=chatgpt.com)

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
| NumPy        | Numerical computation                  |
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

The current platform is designed as a research and engineering prototype.

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

Many fraud detection projects stop at:

```text
Dataset
   ↓
Model
   ↓
Accuracy
```

This project extends the workflow into an operational risk intelligence system:

```text
Data
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Cost Optimization
   ↓
Explainability
   ↓
Anomaly Detection
   ↓
Graph Intelligence
   ↓
Risk Scoring
   ↓
Decision Engine
   ↓
Streamlit Dashboard
   ↓
Testing and Governance
```

The project demonstrates practical application of:

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

[PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1?utm_source=chatgpt.com)

### Reserve Bank of India Payment System Indicators

[RBI Payment System Indicators](https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41&utm_source=chatgpt.com)

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

