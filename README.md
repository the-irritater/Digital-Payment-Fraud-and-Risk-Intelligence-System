# Digital Payment Fraud and Risk Intelligence Platform

| Metadata           | Details                                                                                                                  |
| :----------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Author**         | Sanman Kadam                                                                                                             |
| **Project**        | Real Time Digital Payment Fraud Detection and Explainable Risk Intelligence Platform                                     |
| **Dataset**        | PaySim Synthetic Financial Transactions                                                                                  |
| **Context**        | Reserve Bank of India Payment System Indicators                                                                          |
| **Primary Metric** | PR AUC and Average Precision                                                                                             |
| **Live Demo**      | [Streamlit Fraud Risk Intelligence Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/) |

An explainable and cost sensitive fraud risk platform using PaySim synthetic transaction data, synthetic UPI like simulations, XGBoost with Optuna optimization, Isolation Forest anomaly detection, behavioral analytics, graph based risk analysis and rule based decisioning.

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

---

## Live Demo

Explore the deployed fraud intelligence dashboard:

**[Open Digital Payment Fraud and Risk Intelligence Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/)**

The Streamlit application provides an interactive interface for transaction level fraud prediction, hybrid risk scoring, anomaly detection, rule based decisioning and explainable fraud analysis.

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
     ├── Rule Score
     └── SHAP Explanation
             │
             ▼
    ALLOW / REVIEW / BLOCK
```

---

## System Architecture

```text
                              Transaction Log
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │   Hybrid Risk Intelligence   │
                     └──────────────┬───────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
 1. Is it fraudulent?      2. Why is it suspicious?    3. What action to take?
    (ML & Anomaly Score)     (SHAP XAI & Rules)           (ALLOW / REVIEW / BLOCK)
```

The platform is designed around three core fraud intelligence questions:

```text
Detection
Is the transaction fraudulent or suspicious?

Explanation
Why is the transaction suspicious?

Decision
What action should be taken?
```

---

## Data Provenance

| Data Source                                                                                             | Purpose                                               | Nature                                                          |
| :------------------------------------------------------------------------------------------------------ | :---------------------------------------------------- | :-------------------------------------------------------------- |
| **PaySim** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1))                                   | Transaction level fraud model training and evaluation | Synthetic mobile money simulation based on aggregated real logs |
| **RBI Payment System Indicators** ([rbi.org.in](https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41)) | India digital payment ecosystem context               | Official aggregate statistics                                   |
| **Synthetic UPI Generator** (`src/synthetic_upi.py`)                                                    | Realistic Indian UPI attribute simulation             | Explicitly synthetic                                            |

> **Important:** The transaction level fraud model is trained and evaluated on PaySim synthetic mobile money data. RBI data is used for Indian payment system context only. The UPI like dataset generated by this project is explicitly synthetic and is not presented as real banking data.

---

## India Payment Ecosystem Context

RBI publishes aggregate payment system statistics and domestic fraud related information that provide context for India's rapidly expanding digital payment ecosystem.

Key risk considerations include:

| Risk Area                   | Example Pattern                                   |
| :-------------------------- | :------------------------------------------------ |
| Social Engineering          | Phishing and unauthorized transactions            |
| Fraudulent Collect Requests | Manipulated payment authorization                 |
| Mule Accounts               | Coordinated account and beneficiary relationships |
| High Velocity Activity      | Multiple transactions within short periods        |
| Behavioral Changes          | Unusual amounts, devices and beneficiaries        |
| Midnight Activity           | High velocity fund movement during unusual hours  |

Exact payment volumes vary by reporting period. Current ecosystem figures should be referenced directly from RBI Payment System Indicators.

*Source: RBI Payment System Indicators and RBI Annual Reports.*

---

## Model Performance

Models were evaluated using a strict temporal test set where earlier transactions were used for training and later transactions were used for testing.

### Benchmark Comparison

| Model                        |   PR AUC   |   ROC AUC  | Threshold |  Precision |   Recall   |     F1     |
| :--------------------------- | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: |
| Logistic Regression Baseline |   0.7959   |      —     |    0.50   |   0.1840   |   0.7634   |   0.2966   |
| Random Forest                |   0.8660   |      —     |    0.50   |   0.4180   |   0.8817   |   0.5670   |
| **XGBoost Optuna Tuned**     | **0.9515** | **0.9964** |  **0.17** | **0.3100** | **1.0000** | **0.4733** |

### XGBoost Operating Point

At the cost optimized operating threshold of **0.17**, the model detected:

```text
Fraud Cases Detected    186 / 186
Recall                  100%
Precision               31%
PR AUC                  0.9515
ROC AUC                 0.9964
```

The low operating threshold intentionally favors recall over precision.

This is a deliberate fraud detection strategy where missing fraudulent activity can have a significantly higher cost than investigating legitimate transactions.

See `reports/threshold_cost_table.csv` for the complete threshold cost tradeoff analysis.

---

## Financial Loss Model

The project incorporates financial impact into threshold selection.

Financial loss is calculated using the actual transaction amounts of missed fraudulent transactions.

```text
Loss
=
Sum of Actual Amount of Missed Fraud
+
False Positive Count × 200 Investigation Cost
```

The current false positive investigation cost is:

```text
Investigation Cost = 200 per false positive
```

This allows threshold selection to consider business impact rather than relying only on statistical performance metrics.

See `reports/model_card.md` for model provenance, assumptions, limitations and intended use.

---

## Hybrid Risk Engine

The Risk Engine synthesizes three analytical components into a composite **Risk Score from 0 to 100**.

```text
Risk Score
=
(ML Probability × 100 × 0.60)
+
(Anomaly Score × 100 × 0.20)
+
(Rule Score × 0.20)
```

| Component              |       Scale       | Weight | Description                  |
| :--------------------- | :---------------: | :----: | :--------------------------- |
| XGBoost Probability    | 0 to 1 → 0 to 100 |   60%  | Supervised fraud probability |
| Isolation Forest Score | 0 to 1 → 0 to 100 |   20%  | Normalized anomaly score     |
| Business Rules         |      0 to 100     |   20%  | Deterministic rule triggers  |

This hybrid architecture combines statistical prediction, behavioral anomaly detection and deterministic business intelligence.

---

## Risk Tiers and Decision Actions

| Risk Score | Tier     | Action | Description                                                 |
| :--------: | :------- | :----- | :---------------------------------------------------------- |
|   0 to 30  | LOW      | ALLOW  | Automated payment approval                                  |
|  31 to 60  | MEDIUM   | REVIEW | Step up authentication                                      |
|  61 to 80  | HIGH     | REVIEW | Escalated analyst review and enhanced authentication        |
|  81 to 100 | CRITICAL | BLOCK  | Transaction blocked and account escalated for investigation |

The risk thresholds are configurable and should be calibrated against real operational costs before production deployment.

---

## Explainable AI

The platform uses **SHAP** to explain the factors contributing to individual fraud predictions.

```text
Transaction
     │
     ▼
XGBoost Fraud Probability
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

The explainability layer helps transform a model prediction into an interpretable risk signal that can support analyst investigation.

---

## Anomaly Detection

The platform uses **Isolation Forest** as an unsupervised anomaly detection component.

The purpose is to identify unusual behavioral patterns that may not be fully captured by supervised fraud labels.

```text
Supervised Fraud Signal
          +
Unsupervised Anomaly Signal
          +
Business Rules
          │
          ▼
Hybrid Risk Assessment
```

This provides a second analytical perspective beyond the supervised fraud classifier.

---

## Synthetic UPI and Behavioral Simulation

The Synthetic UPI Generator creates transactions from three behavioral personas:

| Persona        | Behavioral Profile                                                    |
| :------------- | :-------------------------------------------------------------------- |
| **Normal**     | Stable transaction amounts, lower velocity and consistent behavior    |
| **Suspicious** | Elevated velocity, unusual amounts and behavioral changes             |
| **Mule**       | High beneficiary churn, rapid fund movement and network relationships |

The generator is designed for experimentation, simulation and risk engineering.

The generated data is explicitly synthetic and does not represent real customer or banking transaction data.

---

## Graph Risk Analysis

Fraud can involve networks of accounts rather than isolated transactions.

The project uses **NetworkX** to compute structural graph features including:

| Graph Feature        | Purpose                                 |
| :------------------- | :-------------------------------------- |
| PageRank             | Identifies structurally important nodes |
| Degree Centrality    | Measures account connectivity           |
| Connected Components | Identifies relationship structures      |

The graph workflow is:

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

These features can help identify patterns associated with potential money mule networks.

They are structural risk indicators and are **not proof of fraud**.

---

## Complete Project Notebook

All major stages of data processing, feature engineering, model training, threshold tuning, SHAP explainability, anomaly detection and graph analytics are integrated into a single comprehensive notebook.

```text
notebooks/
└── Digital_Payment_Fraud_Intelligence_Complete.ipynb
```

The notebook provides the complete analytical workflow from raw transaction data to fraud risk intelligence.

---

## Repository Structure

```text
digital-payment-fraud-intelligence/
├── data/
│   ├── PS_20174392719_1491204439457_log.csv    # Raw PaySim 6.3M rows
│   └── processed/                              # Processed outputs and synthetic UPI
│
├── notebooks/
│   └── Digital_Payment_Fraud_Intelligence_Complete.ipynb
│
├── src/
│   ├── create_merged_notebook.py               # Complete notebook generator
│   ├── data_processing.py                      # Loading, validation and temporal split
│   ├── feature_engineering.py                  # Causal feature construction
│   ├── train.py                                # Training, Optuna and threshold analysis
│   ├── predict.py                              # Inference engine
│   ├── risk_engine.py                          # Hybrid risk scoring
│   ├── synthetic_upi.py                        # Persona based UPI generator
│   └── graph_fraud.py                          # NetworkX graph risk analysis
│
├── sql/
│   ├── schema.sql                              # Database schema
│   └── fraud_analysis.sql                      # Analytical SQL queries
│
├── app/
│   ├── streamlit_app.py                        # Analyst dashboard
│   └── style.css                               # Dashboard styling
│
├── models/
│   ├── xgboost_model.pkl                       # Trained model
│   └── model_metadata.json                     # Hyperparameters and metrics
│
├── reports/
│   ├── model_card.md                           # Model provenance and limitations
│   ├── model_comparison.csv                    # Experiment results
│   ├── threshold_cost_table.csv                # Threshold cost analysis
│   └── rbi_india_context.md                    # RBI ecosystem context
│
├── tests/
│   ├── test_features.py                        # Feature engineering tests
│   ├── test_risk_engine.py                     # Risk tier and action tests
│   └── test_prediction.py                      # Inference pipeline tests
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Model Training

```bash
python src/train.py
```

The training pipeline performs data processing, feature engineering, model benchmarking, Optuna optimization and threshold cost analysis.

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Access the local dashboard at:

```text
http://localhost:8501
```

For the deployed application:

**[Open the Live Streamlit Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/)**

---

## Technology Stack

| Technology       | Purpose                                |
| :--------------- | :------------------------------------- |
| **Python**       | Core development                       |
| **Pandas**       | Data processing                        |
| **NumPy**        | Numerical computation                  |
| **Scikit Learn** | Machine learning and anomaly detection |
| **XGBoost**      | Fraud classification                   |
| **Optuna**       | Hyperparameter optimization            |
| **SHAP**         | Explainable AI                         |
| **NetworkX**     | Graph analytics                        |
| **SQL**          | Data and analytical queries            |
| **Streamlit**    | Interactive fraud dashboard            |
| **Matplotlib**   | Visualization                          |
| **Pytest**       | Automated testing                      |
| **Jupyter**      | Research and experimentation           |

---

## Testing

The project includes automated tests covering key components of the platform.

```text
tests/
├── test_features.py
├── test_risk_engine.py
└── test_prediction.py
```

Run the test suite with:

```bash
pytest tests/ -v
```

The tests cover:

```text
Feature Engineering
Risk Tier Logic
Decision Actions
Prediction Pipeline
```

---

## Model Governance

The project includes a dedicated model card:

```text
reports/model_card.md
```

The model card documents:

```text
Model Provenance
Evaluation Methodology
Cost Assumptions
Intended Use
Known Limitations
Risk Considerations
```

### Important Limitations

* PaySim is synthetic
* The Synthetic UPI Generator produces synthetic transactions
* PaySim performance does not guarantee equivalent performance on real banking data
* Risk thresholds require calibration against real operational costs
* Graph features indicate structural risk and do not establish fraud
* RBI statistics provide ecosystem context rather than transaction level labels
* Production deployment would require representative real world data, monitoring and governance

---

## Responsible Use

This project is an analytical and engineering prototype intended for research, portfolio demonstration and fraud risk experimentation.

The transaction level fraud model is trained and evaluated using synthetic PaySim data.

The UPI like transaction generator is explicitly synthetic.

The platform should not be used to make real financial decisions without extensive validation using representative production data, appropriate calibration, monitoring, governance and human oversight.

Risk scores should support fraud investigation and operational decision making rather than independently determining adverse outcomes.

---

## Future Enhancements

Potential production oriented extensions include:

```text
Real Time Payment Event
          │
          ▼
Kafka Event Streaming
          │
          ▼
Online Feature Store
          │
          ▼
Fraud Scoring API
          │
          ▼
Hybrid Risk Engine
          │
          ▼
ALLOW / REVIEW / BLOCK
```

Potential enhancements include:

* FastAPI based model serving
* Kafka based real time transaction streaming
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

## Why This Project Matters

A basic fraud detection project typically follows:

```text
Dataset
   │
   ▼
Machine Learning Model
   │
   ▼
Accuracy
```

This project extends the workflow into a complete fraud intelligence system:

```text
Transaction Data
       │
       ▼
Feature Engineering
       │
       ▼
Fraud Detection
       │
       ▼
Anomaly Detection
       │
       ▼
Explainable AI
       │
       ▼
Graph Intelligence
       │
       ▼
Cost Sensitive Optimization
       │
       ▼
Hybrid Risk Score
       │
       ▼
Business Decision
       │
       ├── ALLOW
       ├── REVIEW
       └── BLOCK
              │
              ▼
      Streamlit Dashboard
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
Behavioral Analytics
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

## Data Sources

### PaySim

[PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Reserve Bank of India

[RBI Payment System Indicators](https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41)

---

## Author

**Sanman Kadam**

MSc Statistics
Data Analytics and Data Science

This project demonstrates the application of statistical modelling, machine learning, explainable AI, anomaly detection, graph analytics and business decision intelligence to digital payment fraud risk.

---

## Project Summary

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
  <strong>Built for fraud intelligence research and practical risk analytics.</strong>
</p>
