# Digital Payment Fraud and Risk Intelligence Platform

| Metadata           | Details                                                                              |
| :----------------- | :----------------------------------------------------------------------------------- |
| **Author**         | Sanman Kadam                                                                         |
| **Project**        | Real Time Digital Payment Fraud Detection and Explainable Risk Intelligence Platform |
| **Dataset**        | PaySim Synthetic Financial Transactions                                              |
| **Context**        | Reserve Bank of India Payment System Indicators                                      |
| **Primary Metric** | PR AUC and Average Precision                                                         |

An explainable and cost sensitive fraud risk platform using PaySim synthetic transaction data, synthetic UPI like simulations, XGBoost with Optuna optimization, Isolation Forest anomaly detection, behavioral analytics, graph based risk analysis and rule based decisioning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQL-MySQL%20%7C%20SQL%20Server-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="SQL">
  <img src="https://img.shields.io/badge/XGBoost-Optuna-FF6600?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Analytics-4C8BF5?style=for-the-badge" alt="NetworkX">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
</p>

---

## Business Problem

The rapid growth of digital payments has increased the importance of identifying fraudulent transactions quickly and accurately.

Traditional fraud detection systems that rely only on static rules can struggle with evolving transaction behavior. A practical fraud detection platform needs to identify suspicious activity while also explaining the underlying risk and recommending an appropriate operational response.

The core business problem is:

```text
How can digital payment transactions be evaluated in real time
to identify fraud, understand why a transaction is suspicious
and determine whether it should be allowed, reviewed or blocked?
```

The platform addresses this problem by combining machine learning, anomaly detection, behavioral intelligence, graph analysis and deterministic business rules.

---

## Project Objective

The primary objective is to develop an end to end fraud intelligence platform capable of:

| Objective            | Description                                                    |
| :------------------- | :------------------------------------------------------------- |
| Fraud Detection      | Predict the probability that a transaction is fraudulent       |
| Risk Identification  | Detect unusual transaction behavior using anomaly detection    |
| Explainability       | Identify the factors contributing to fraud predictions         |
| Behavioral Analysis  | Capture transaction velocity, device and beneficiary behavior  |
| Network Intelligence | Identify structural patterns across transaction relationships  |
| Cost Optimization    | Select decision thresholds using financial loss considerations |
| Risk Scoring         | Combine multiple signals into a unified score from 0 to 100    |
| Decisioning          | Recommend ALLOW, REVIEW or BLOCK actions                       |
| Analyst Support      | Provide an interactive risk intelligence dashboard             |

---

## Key Business Questions

The platform is designed to answer three operational questions:

```text
1. Is the transaction fraudulent or suspicious?

2. Why is the transaction considered risky?

3. What action should be taken?
```

These questions form the foundation of the system architecture.

---

## Solution Approach

The project uses a hybrid fraud intelligence framework:

```text
Transaction Data
       │
       ▼
Data Validation
       │
       ▼
Feature Engineering
       │
       ▼
┌─────────────────────────────────────────────┐
│                                             │
│  XGBoost      Isolation Forest      Rules   │
│  Fraud ML     Anomaly Detection     Engine  │
│                                             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
               Hybrid Risk Engine
                       │
                       ▼
                 Risk Score
                   0 to 100
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        ALLOW        REVIEW        BLOCK
                       │
                       ▼
             SHAP and Graph Analysis
                       │
                       ▼
                Analyst Dashboard
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
    ML & Anomaly Score        SHAP XAI & Rules           ALLOW / REVIEW / BLOCK
```

---

## Data Provenance

| Data Source                                                                                             | Purpose                                               | Nature                                                          |
| :------------------------------------------------------------------------------------------------------ | :---------------------------------------------------- | :-------------------------------------------------------------- |
| **PaySim** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1))                                   | Transaction level fraud model training and evaluation | Synthetic mobile money simulation based on aggregated real logs |
| **RBI Payment System Indicators** ([rbi.org.in](https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=41)) | Indian digital payment ecosystem context              | Official aggregate statistics                                   |
| **Synthetic UPI Generator** (`src/synthetic_upi.py`)                                                    | Indian UPI like behavioral simulation                 | Explicitly synthetic                                            |

> **Important:** The transaction level fraud model is trained and evaluated on PaySim synthetic mobile money data. RBI data is used only for Indian payment system context. The UPI like dataset generated by this project is explicitly synthetic and is not presented as real banking data.

---

## Data Understanding

### PaySim Dataset

PaySim is used as the primary transaction level dataset for fraud detection.

The dataset contains approximately 6.3 million simulated financial transactions and includes transaction characteristics that support fraud classification and behavioral analysis.

The fraud model is trained and evaluated exclusively on the synthetic PaySim transaction data.

### RBI Context

RBI Payment System Indicators provide macro level context around India's digital payment ecosystem.

RBI aggregate statistics are not used as transaction level labels for model training.

### Synthetic UPI Data

The project includes a synthetic UPI generator to simulate Indian payment behaviors and experiment with risk characteristics.

The generated UPI like transactions are explicitly synthetic.

---

## India Payment Ecosystem Context

The project considers several representative digital payment risk patterns.

| Risk Vector                 | Example Pattern                                   |
| :-------------------------- | :------------------------------------------------ |
| Social Engineering          | Phishing and unauthorized transactions            |
| Fraudulent Collect Requests | Manipulated payment authorization                 |
| Mule Accounts               | Coordinated account and beneficiary relationships |
| High Velocity Activity      | Multiple rapid transactions                       |
| Behavioral Change           | Unusual transaction amounts or devices            |
| Beneficiary Churn           | Rapid changes in recipients                       |
| Unusual Timing              | High activity during unusual hours                |

Exact payment volumes and ecosystem statistics vary over time and should be referenced directly from current RBI publications.

---

## Analytical Methodology

The analytical pipeline follows a structured sequence:

```text
Raw Data
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
Optuna Optimization
   │
   ▼
Temporal Evaluation
   │
   ▼
Threshold Cost Optimization
   │
   ▼
Explainability
   │
   ▼
Risk Intelligence
```

---

## Feature Engineering

The project develops causal transaction features across multiple analytical dimensions.

| Feature Category | Examples                        |
| :--------------- | :------------------------------ |
| Transaction      | Amount and transaction type     |
| Velocity         | Recent transaction frequency    |
| Behavioral       | Changes in transaction behavior |
| Device           | Device changes and consistency  |
| Beneficiary      | Beneficiary churn               |
| Temporal         | Time based activity patterns    |
| Network          | Degree centrality and PageRank  |

The feature engineering pipeline is implemented in:

```text
src/feature_engineering.py
```

---

## Machine Learning Approach

The project benchmarks multiple supervised learning algorithms before selecting the final model.

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

XGBoost was selected as the primary supervised fraud classifier because it provided the strongest performance on the temporal test set.

---

## Model Evaluation

Models were evaluated using a strict temporal test set where earlier observations were used for training and later observations were used for testing.

### Benchmark Comparison

| Model                        |   PR AUC   |   ROC AUC  | Threshold |  Precision |   Recall   |     F1     |
| :--------------------------- | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: |
| Logistic Regression Baseline |   0.7959   |      —     |    0.50   |   0.1840   |   0.7634   |   0.2966   |
| Random Forest                |   0.8660   |      —     |    0.50   |   0.4180   |   0.8817   |   0.5670   |
| **XGBoost Optuna Tuned**     | **0.9515** | **0.9964** |  **0.17** | **0.3100** | **1.0000** | **0.4733** |

---

## Key Model Result

At the selected operating threshold of **0.17**, the XGBoost model achieved:

```text
PR AUC                 0.9515
ROC AUC                0.9964
Precision              0.3100
Recall                 1.0000
F1 Score               0.4733
Observed Fraud Cases   186
Fraud Cases Detected   186
```

The model detected all observed fraud cases in the temporal test set.

The selected threshold deliberately favors recall over precision because the cost of missing fraudulent transactions can be considerably higher than the cost of investigating legitimate transactions.

---

## Why PR AUC

Fraud detection is generally a highly imbalanced classification problem.

Accuracy can therefore provide an incomplete view of performance.

The primary evaluation metric for this project is **PR AUC and Average Precision** because it focuses on the relationship between precision and recall under class imbalance.

Additional evaluation metrics include:

```text
PR AUC
ROC AUC
Precision
Recall
F1 Score
```

---

## Temporal Validation

A temporal validation strategy was used instead of random train test splitting.

```text
Earlier Transactions
        │
        ▼
Training Data
        │
        ▼
Model Development
        │
        ▼
Later Transactions
        │
        ▼
Temporal Test Data
```

This approach better represents the chronological nature of financial transaction data and reduces the risk of unrealistic evaluation caused by mixing future observations into training.

---

## Cost Sensitive Threshold Optimization

A fraud detection model should not automatically use a probability threshold of 0.50.

The project evaluates thresholds using an explicit financial loss function.

```text
Loss
=
Sum of Actual Amount of Missed Fraud
+
False Positive Count × 200
```

The investigation cost assumption is:

```text
False Positive Investigation Cost = 200
```

The threshold optimization process evaluates the tradeoff between:

```text
Recall
Precision
Missed Fraud Loss
False Positive Investigation Cost
```

The complete threshold analysis is available at:

```text
reports/threshold_cost_table.csv
```

---

## Hybrid Risk Engine

The Risk Engine combines three independent analytical signals.

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
| Business Rules         |      0 to 100     |   20%  | Deterministic risk triggers  |

The resulting score provides a unified representation of transaction risk.

---

## Risk Classification

| Risk Score | Tier     | Action | Description                                          |
| :--------: | :------- | :----- | :--------------------------------------------------- |
|   0 to 30  | LOW      | ALLOW  | Automated payment approval                           |
|  31 to 60  | MEDIUM   | REVIEW | Step up authentication                               |
|  61 to 80  | HIGH     | REVIEW | Escalated analyst review and enhanced authentication |
|  81 to 100 | CRITICAL | BLOCK  | Transaction blocked and account escalated            |

The thresholds are configurable and should be calibrated against real operational data before production use.

---

## Explainable AI

The platform uses **SHAP** to explain individual fraud predictions.

```text
Transaction
     │
     ▼
XGBoost Probability
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
       Risk Explanation
```

The goal of the explainability layer is to help analysts understand the drivers behind a model prediction rather than treating the classifier as a black box.

---

## Anomaly Detection

Isolation Forest provides an additional unsupervised risk signal.

The model is designed to detect unusual observations based on behavioral characteristics without relying directly on the fraud label.

```text
Supervised Fraud Detection
          +
Unsupervised Anomaly Detection
          +
Business Rules
          │
          ▼
Hybrid Risk Intelligence
```

---

## Behavioral Analytics

The platform incorporates behavioral characteristics including:

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

These features provide additional context around potentially suspicious activity.

---

## Synthetic UPI Simulation

The Synthetic UPI Generator creates transactions from three behavioral personas.

| Persona        | Characteristics                                                       |
| :------------- | :-------------------------------------------------------------------- |
| **Normal**     | Stable amounts, lower velocity and consistent behavior                |
| **Suspicious** | Elevated velocity, unusual amounts and behavioral changes             |
| **Mule**       | High beneficiary churn, rapid fund movement and network relationships |

The simulator is intended for experimentation and risk engineering.

It is not a representation of real UPI customer behavior.

---

## Graph Based Risk Analysis

Fraudulent activity can involve interconnected accounts rather than isolated transactions.

The project uses **NetworkX** to construct transaction relationship graphs and derive structural features.

| Graph Feature        | Purpose                                 |
| :------------------- | :-------------------------------------- |
| PageRank             | Identifies structurally important nodes |
| Degree Centrality    | Measures account connectivity           |
| Connected Components | Identifies relationship structures      |

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

Graph features are treated as supporting risk indicators and not as proof of fraudulent activity.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard for fraud risk analysis.

The application brings together:

```text
Transaction Input
       │
       ▼
Fraud Prediction
       │
       ▼
Anomaly Detection
       │
       ▼
Rule Evaluation
       │
       ▼
Hybrid Risk Score
       │
       ▼
Risk Tier
       │
       ▼
Decision Recommendation
       │
       ▼
Explainability
```

The application is available at:

[Digital Payment Fraud and Risk Intelligence Dashboard](https://digital-payment-fraud-and-risk-intelligence-system.streamlit.app/)

---

## SQL Analytics

The project includes a SQL analytics layer.

```text
sql/
├── schema.sql
└── fraud_analysis.sql
```

The SQL components support analytical workflows involving:

```text
Transaction Analysis
Fraud Patterns
Risk Segmentation
Behavioral Analysis
Account Activity
```

---

## Complete Project Notebook

The complete analytical workflow is integrated into:

```text
notebooks/
└── Digital_Payment_Fraud_Intelligence_Complete.ipynb
```

The notebook covers:

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
Risk Engine
```

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
│   └── fraud_analysis.sql                      # Analytical queries
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
│   └── rbi_india_context.md                    # RBI macro context
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

---

## Testing

Automated tests are included for key parts of the system.

```text
tests/
├── test_features.py
├── test_risk_engine.py
└── test_prediction.py
```

The test suite covers:

```text
Feature Engineering
Risk Tier Logic
Decision Actions
Prediction Pipeline
```

Run the tests using:

```bash
pytest tests/ -v
```

---

## Model Governance

A dedicated model card is included at:

```text
reports/model_card.md
```

It documents:

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
* UPI like transactions are synthetically generated
* Model performance on PaySim does not guarantee equivalent performance on real banking data
* Risk thresholds require calibration against real operational costs
* Graph features indicate structural risk rather than proving fraud
* RBI statistics provide ecosystem context and are not transaction level fraud labels
* Production use would require representative real world data, monitoring and governance

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
| **Streamlit**    | Interactive dashboard                  |
| **Matplotlib**   | Visualization                          |
| **Pytest**       | Automated testing                      |
| **Jupyter**      | Research and experimentation           |

---

## Business Impact

The platform is designed to translate machine learning outputs into operational fraud decisions.

```text
Fraud Probability
       +
Anomaly Signal
       +
Behavioral Risk
       +
Network Risk
       +
Business Rules
       │
       ▼
Unified Risk Score
       │
       ▼
Operational Action
```

This supports a shift from:

```text
Prediction Only
```

toward:

```text
Prediction
     +
Explanation
     +
Risk Prioritization
     +
Decisioning
```

---

## Project Outcome

The project demonstrates an end to end approach to digital payment fraud analytics that integrates statistical modelling, machine learning and operational risk intelligence.

Key outcomes include:

```text
XGBoost Fraud Detection
        │
        ▼
PR AUC of 0.9515
        │
        ▼
Cost Optimized Threshold
        │
        ▼
100% Recall on Observed Test Fraud
        │
        ▼
Hybrid Risk Score
        │
        ▼
Explainable Decisioning
        │
        ▼
Interactive Streamlit Dashboard
```

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

## Responsible Use

This project is an analytical and engineering prototype intended for research, portfolio demonstration and fraud risk experimentation.

The transaction level fraud model is trained and evaluated using synthetic PaySim data.

The UPI like transaction generator is explicitly synthetic.

The system should not be used for real financial decision making without validation using representative production data, appropriate calibration, monitoring, governance and human oversight.

Risk scores should support investigation and operational decision making rather than independently determining adverse outcomes.

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

This project demonstrates the application of statistical modelling, machine learning, explainable AI, anomaly detection, behavioral analytics, graph analytics and business decision intelligence to digital payment fraud risk.

---

<p align="center">
  <strong>Digital Payment Fraud and Risk Intelligence Platform</strong>
</p>

<p align="center">
  Detection. Explanation. Decision.
</p>
