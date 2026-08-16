# Digital Payment Fraud and Risk Intelligence Platform

<p align="center">
  <strong>Real Time Fraud Detection. Explainable Risk Intelligence. Cost Sensitive Decisioning.</strong>
</p>

<p align="center">
  An end to end fraud intelligence platform designed to detect suspicious digital payment transactions, explain risk drivers and recommend operational actions using machine learning, anomaly detection, behavioral analytics and graph based risk intelligence.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQL-MySQL%20%7C%20SQL%20Server-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="SQL">
  <img src="https://img.shields.io/badge/XGBoost-Optuna-FF6600?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Analytics-4C8BF5?style=for-the-badge" alt="NetworkX">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

## Project Overview

Digital payment fraud requires more than a binary fraud classifier.

A production oriented fraud detection system must answer three questions:

```text
Is this transaction fraudulent?
             │
             ▼
Why is this transaction suspicious?
             │
             ▼
What action should the payment system take?
```

The **Digital Payment Fraud and Risk Intelligence Platform** addresses these questions through a hybrid fraud intelligence architecture combining supervised machine learning, unsupervised anomaly detection, behavioral risk analytics, deterministic business rules, explainable AI and graph based network analysis.

The platform is designed around a cost sensitive fraud detection philosophy where missing a fraudulent transaction can have substantially greater financial impact than investigating a legitimate transaction.

## Key Highlights

| Capability                  | Implementation                             |
| :-------------------------- | :----------------------------------------- |
| Fraud Classification        | XGBoost                                    |
| Hyperparameter Optimization | Optuna                                     |
| Anomaly Detection           | Isolation Forest                           |
| Explainability              | SHAP                                       |
| Behavioral Analytics        | Velocity and transaction behavior features |
| Graph Intelligence          | NetworkX                                   |
| Decision Engine             | Rule based risk scoring                    |
| Risk Score                  | Composite score from 0 to 100              |
| Decision Actions            | Allow, Review and Block                    |
| Dashboard                   | Streamlit                                  |
| Database Layer              | SQL                                        |
| Testing                     | Pytest                                     |
| Evaluation Strategy         | Temporal validation                        |
| Primary Metric              | PR AUC                                     |
| Cost Optimization           | Threshold based financial loss analysis    |

## Architecture

```text
                         TRANSACTION LOG
                                │
                                ▼
                 ┌──────────────────────────┐
                 │   DATA VALIDATION        │
                 │   & PREPROCESSING        │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   CAUSAL FEATURE         │
                 │   ENGINEERING            │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │  XGBoost    │  │  Isolation  │  │   Business  │
      │ Fraud Model │  │   Forest    │  │    Rules    │
      └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │   HYBRID RISK     │
                    │      ENGINE        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   RISK SCORE       │
                    │      0 to 100      │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
           ALLOW            REVIEW            BLOCK
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ SHAP EXPLANATIONS  │
                    │ GRAPH INTELLIGENCE  │
                    │ ANALYST DASHBOARD   │
                    └────────────────────┘
```

## Data Provenance

The project deliberately separates transaction level model data from Indian payment ecosystem context.

| Data Source                   | Purpose                                | Nature                            |
| :---------------------------- | :------------------------------------- | :-------------------------------- |
| PaySim                        | Fraud model development and evaluation | Synthetic mobile money simulation |
| RBI Payment System Indicators | Indian payment ecosystem context       | Official aggregate statistics     |
| Synthetic UPI Generator       | Indian UPI like behavioral simulation  | Explicitly synthetic              |

### PaySim

The fraud detection models are trained and evaluated using the **PaySim synthetic financial transaction dataset**.

The dataset contains approximately 6.3 million simulated transactions and provides transaction level information suitable for fraud classification and behavioral analysis.

### RBI Context

RBI payment system statistics are used strictly as ecosystem context.

RBI aggregate statistics are not used as transaction level fraud labels or as training data for the PaySim fraud classifier.

### Synthetic UPI Data

The project includes a synthetic UPI transaction generator designed to simulate Indian payment behaviors.

These transactions are explicitly synthetic and are not presented as real banking or customer data.

## India Payment Ecosystem Context

India's digital payment ecosystem operates at extremely high transaction volumes. This creates a requirement for automated fraud intelligence capable of assessing transactions before settlement.

The project considers several representative risk patterns:

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
Multiple Accounts and Beneficiary Relationships

Behavioral Changes
        │
        ▼
Unusual Amounts, Devices and Transaction Patterns
```

RBI payment system statistics are used to provide macro level context around India's digital payment ecosystem.

## Machine Learning Pipeline

The modelling workflow follows a temporal validation strategy to reduce the risk of unrealistic evaluation caused by random train test splitting.

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
Training Period
       │
       ▼
XGBoost + Optuna
       │
       ▼
Temporal Test Period
       │
       ▼
PR AUC Evaluation
       │
       ▼
Threshold Cost Optimization
```

## Feature Engineering

The platform constructs features designed to represent transaction behavior while avoiding future information leakage.

Feature categories include:

| Feature Category | Examples                                             |
| :--------------- | :--------------------------------------------------- |
| Transaction      | Amount, transaction type and balance information     |
| Velocity         | Recent transaction frequency and activity            |
| Behavioral       | Transaction patterns and unusual activity            |
| Device           | Device changes and behavioral consistency            |
| Beneficiary      | Beneficiary churn and relationship changes           |
| Temporal         | Time based transaction behavior                      |
| Network          | Degree centrality, PageRank and connected structures |

## Model Benchmark

The models were evaluated using a strict temporal test set.

| Model                   |     PR AUC |      ROC AUC | Threshold |  Precision |     Recall |         F1 |
| :---------------------- | ---------: | -----------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression     |     0.7959 | Not reported |      0.50 |     0.1840 |     0.7634 |     0.2966 |
| Random Forest           |     0.8660 | Not reported |      0.50 |     0.4180 |     0.8817 |     0.5670 |
| **XGBoost with Optuna** | **0.9515** |   **0.9964** |  **0.17** | **0.3100** | **1.0000** | **0.4733** |

### Why PR AUC?

Fraud detection is typically an imbalanced classification problem.

Accuracy can therefore provide a misleading picture of model quality.

This project prioritizes **Precision Recall AUC** because it evaluates the model's ability to identify fraudulent transactions while accounting for the tradeoff between precision and recall.

## XGBoost Operating Point

The optimized XGBoost model achieved:

```text
PR AUC       0.9515
ROC AUC      0.9964
Threshold    0.17
Recall       100%
Fraud Cases  186 / 186
Precision    31%
```

At the selected operating threshold of **0.17**, the model detected all observed fraud cases in the temporal test set.

The threshold intentionally favors recall because fraud detection systems often place a substantially higher cost on missed fraud than on legitimate transactions being sent for additional review.

The complete threshold analysis is available in:

```text
reports/threshold_cost_table.csv
```

## Financial Loss Model

The project incorporates financial impact into threshold selection.

```text
Financial Loss
=
Actual Amount of Missed Fraud
+
False Positive Investigation Cost
```

The current investigation cost assumption is:

```text
False Positive Cost = 200
```

This converts model evaluation from a purely statistical problem into an operational decision problem.

The objective is not simply to maximize a model metric.

The objective is to identify an operating threshold that balances fraud prevention against investigation costs.

## Hybrid Risk Engine

The platform combines three independent risk signals into a composite score.

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
| Isolation Forest    | 0 to 100 |   20%  | Behavioral anomaly signal     |
| Business Rules      | 0 to 100 |   20%  | Deterministic risk indicators |

This hybrid approach prevents the platform from relying exclusively on a single machine learning model.

## Risk Decision Framework

| Risk Score | Tier     | Action | Operational Response                       |
| :--------: | :------- | :----- | :----------------------------------------- |
|   0 to 30  | LOW      | ALLOW  | Automated approval                         |
|  31 to 60  | MEDIUM   | REVIEW | Step up authentication                     |
|  61 to 80  | HIGH     | REVIEW | Enhanced authentication and analyst review |
|  81 to 100 | CRITICAL | BLOCK  | Block transaction and escalate account     |

The thresholds are configurable and can be calibrated according to business requirements and observed fraud economics.

## Explainable AI

Fraud detection requires more than a prediction.

Analysts need to understand why a transaction received a high risk score.

The platform uses **SHAP based explainability** to identify influential features behind model predictions.

```text
Transaction
     │
     ▼
Fraud Probability
     │
     ▼
SHAP Analysis
     │
     ├── High transaction amount
     ├── Unusual transaction behavior
     ├── High transaction velocity
     ├── Beneficiary changes
     └── Other model features
             │
             ▼
       Analyst Explanation
```

This improves transparency and supports analyst investigation.

## Anomaly Detection

The platform uses **Isolation Forest** as an unsupervised anomaly detection layer.

The purpose is to identify unusual behavioral patterns that may not be fully represented by supervised fraud labels.

This provides a second signal alongside the XGBoost fraud probability.

```text
Supervised Signal
       +
Unsupervised Signal
       +
Business Rules
       │
       ▼
Hybrid Risk Assessment
```

## Synthetic UPI Behavioral Simulation

The project includes a synthetic UPI generator that models three behavioral personas.

| Persona    | Behavioral Profile                                                        |
| :--------- | :------------------------------------------------------------------------ |
| Normal     | Stable amounts, lower velocity and consistent behavior                    |
| Suspicious | Elevated velocity, unusual amounts and behavioral changes                 |
| Mule       | High beneficiary churn, rapid movement of funds and network relationships |

The generator is designed for experimentation and risk engineering.

It does not represent real customer behavior or real banking transactions.

## Graph Based Fraud Intelligence

Fraud can involve relationships between accounts rather than isolated transactions.

The project therefore uses **NetworkX** to construct transaction relationship graphs.

Graph features include:

```text
Degree Centrality
       │
       ├── Account connectivity

PageRank
       │
       ├── Network importance

Connected Components
       │
       └── Relationship structures
```

These features can highlight patterns associated with potential mule networks.

Graph indicators are treated as **risk signals rather than proof of fraud**.

## Streamlit Risk Intelligence Dashboard

The project includes an analyst oriented Streamlit dashboard designed to expose the fraud intelligence pipeline through an interactive interface.

The dashboard can support:

```text
Transaction Input
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
       ├── Model Probability
       ├── Anomaly Signal
       ├── Rule Score
       └── SHAP Explanation
```

## SQL Analytics

The project includes a dedicated SQL layer for structured fraud analysis.

```text
sql/
├── schema.sql
└── fraud_analysis.sql
```

The SQL layer supports analytical workflows around transaction behavior, fraud patterns and risk segmentation.

## Model Governance

A dedicated model card is included to document model provenance, evaluation methodology, intended use and known limitations.

```text
reports/model_card.md
```

The model should be interpreted as a research and engineering prototype rather than a production banking fraud model.

Important limitations include:

* PaySim is synthetic
* UPI transaction data generated by this project is synthetic
* Model performance on PaySim does not imply equivalent performance on real banking data
* Risk thresholds require calibration against real operational costs
* Graph features indicate structural risk and do not establish fraud
* RBI aggregate statistics do not provide transaction level fraud labels for this model

## Testing

The project includes automated tests covering key components of the system.

```text
tests/
├── test_features.py
├── test_risk_engine.py
└── test_prediction.py
```

Run the complete test suite with:

```bash
pytest tests/ -v
```

## Repository Structure

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

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/the-irritater/digital-payment-fraud-intelligence.git

cd digital-payment-fraud-intelligence

pip install -r requirements.txt
```

## Model Training

Run the complete training pipeline:

```bash
python src/train.py
```

The training workflow includes data processing, feature engineering, Optuna optimization, model evaluation and threshold cost analysis.

## Run Tests

Execute the automated test suite:

```bash
pytest tests/ -v
```

## Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Complete Project Notebook

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
Graph Fraud Analytics
       ↓
Risk Intelligence
```

## Technology Stack

| Technology   | Purpose                                |
| :----------- | :------------------------------------- |
| Python       | Core development                       |
| Pandas       | Data processing                        |
| NumPy        | Numerical computation                  |
| Scikit Learn | Machine learning and anomaly detection |
| XGBoost      | Fraud classification                   |
| Optuna       | Hyperparameter optimization            |
| SHAP         | Model explainability                   |
| NetworkX     | Graph analytics                        |
| SQL          | Analytical data processing             |
| Streamlit    | Interactive dashboard                  |
| Matplotlib   | Visualization                          |
| Pytest       | Automated testing                      |
| Jupyter      | Research and experimentation           |

## Why This Project Matters

Traditional fraud classification answers:

> Is this transaction fraudulent?

A risk intelligence platform should answer three questions:

```text
Detection
Is the transaction suspicious?

Explanation
Why is it suspicious?

Decision
What should happen next?
```

This project combines these three layers into a unified fraud intelligence workflow.

The result is a system that moves beyond model accuracy and focuses on **risk prioritization, explainability, operational decisioning and financial impact**.

## Future Enhancements

Potential production oriented extensions include:

```text
Real Time Event Streaming
        │
        ▼
Kafka or Event Driven Architecture
        │
        ▼
Online Feature Store
        │
        ▼
Real Time Fraud Scoring
        │
        ▼
Low Latency Decision API
        │
        ▼
Continuous Monitoring
```

Additional enhancements could include:

* Real banking transaction data for model calibration
* Online feature computation
* Model drift monitoring
* Champion challenger model deployment
* Automated retraining pipelines
* Real time Kafka based transaction ingestion
* REST API deployment
* Advanced graph embeddings
* Account level risk profiling
* Feedback driven analyst learning
* Production grade model monitoring
* Cost sensitive learning using real fraud economics

## Responsible Use

This project is an analytical and engineering prototype.

The PaySim dataset is synthetic and the UPI transaction generator is explicitly synthetic.

The model should not be interpreted as evidence of real world banking fraud performance without validation using representative production data.

Risk scores should support human and operational decision making rather than automatically determining adverse actions without appropriate validation and governance.

## Data Sources

**PaySim**

https://www.kaggle.com/datasets/ealaxi/paysim1

**Reserve Bank of India Payment System Indicators**

https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41

## Author

**Sanman Kadam**

MSc Statistics
Data Analytics and Data Science

This project demonstrates practical application of statistical thinking, machine learning, explainable AI, anomaly detection, graph analytics and business decision intelligence to digital payment fraud risk.

## Project Focus

```text
Machine Learning
       +
Statistical Thinking
       +
Explainable AI
       +
Behavioral Analytics
       +
Graph Intelligence
       +
Cost Sensitive Decisioning
       =
Fraud Risk Intelligence
```

<p align="center">
  <strong>Digital Payment Fraud and Risk Intelligence Platform</strong>
</p>

<p align="center">
  Detection. Explanation. Decision.
</p>
