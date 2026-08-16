# Digital Payment Fraud and Risk Intelligence Platform

| Metadata | Details |
| :--- | :--- |
| **Author** | Sanman Kadam |
| **Project** | Digital Payment Fraud Detection and Risk Intelligence Platform |
| **Model Scope** | `TRANSFER` & `CASH_OUT` Digital Payment Rails |
| **Dataset** | PaySim Synthetic Financial Transactions (Kaggle) |
| **Context** | Reserve Bank of India (RBI) Payment System Indicators |
| **Primary Metric** | PR-AUC (Average Precision) |
| **Secondary Metrics** | ROC-AUC, Precision, Recall, F1, Expected Financial Loss, Brier Score, ECE |

An explainable, cost-sensitive fraud risk platform using PaySim synthetic transaction data, synthetic Indian UPI simulations, XGBoost with Optuna optimization, Platt probability calibration, fitted Isolation Forest anomaly detection, stateful behavioral velocity tracking, NetworkX MultiDiGraph analysis, and rule-based decisioning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/XGBoost-Optuna-FF6600?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Analytics-4C8BF5?style=for-the-badge" alt="NetworkX">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pytest-40%20Tests%20Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
</p>

---

## Business Problem

The rapid growth of digital payments has increased the importance of identifying fraudulent transactions quickly and accurately.

Traditional fraud detection systems that rely only on static rules struggle with evolving transaction behavior. A practical fraud detection platform needs to identify suspicious activity while also explaining the underlying risk and recommending an appropriate operational response.

The core business problem is:

```text
How can digital payment transactions be evaluated interactively
to identify fraud, understand why a transaction is suspicious,
and determine whether it should be ALLOWED, REVIEWED, or BLOCKED?
```

The platform addresses this problem by combining machine learning, anomaly detection, behavioral intelligence, graph analysis, and deterministic business rules.

---

## Project Objective

The primary objective is to develop an end-to-end fraud intelligence platform capable of:

| Objective | Description |
| :--- | :--- |
| **Fraud Detection** | Predict calibrated fraud probabilities using Optuna-tuned XGBoost |
| **Anomaly Detection** | Detect unusual transaction behavior using a fitted Isolation Forest |
| **Explainability** | Identify factors contributing to risk predictions via SHAP & rule triggers |
| **Behavioral Analysis** | Track stateful transaction velocity, customer means, and beneficiary churn |
| **Network Intelligence** | Identify structural patterns (mule rings, cycles) using NetworkX MultiDiGraph |
| **Cost Optimization** | Select decision thresholds using financial loss minimization on validation data |
| **Risk Scoring** | Synthesize signals into a unified composite score from 0 to 100 |
| **Decisioning** | Recommend ALLOW, REVIEW, or BLOCK operational actions |
| **Analyst Support** | Provide an interactive Streamlit risk intelligence dashboard & SQLite investigation queue |
| **Production Serving** | Serve inference via FastAPI, Docker containers, and CI/CD pipelines |

---

## Key Business Questions

The platform is designed to answer three operational questions:

```text
1. Is the transaction fraudulent or suspicious?

2. Why is the transaction considered risky?

3. What action should be taken (ALLOW / REVIEW / BLOCK)?
```

These questions form the foundation of the system architecture.

---

## Solution Approach

The project uses a hybrid fraud intelligence framework:

```text
Incoming Transaction Payload
       │
       ▼
Data Validation & Stateful Customer Store
       │
       ▼
Causal Feature Engineering Pipeline
       │
       ▼
┌───────────────────────────────────────────────────┐
│                                                   │
│   Calibrated XGBoost    Isolation Forest    Rules │
│   Fraud ML (60%)       Anomaly (20%)     Engine (20%)
│                                                   │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
                 Hybrid Risk Engine
                          │
                          ▼
                  Composite Risk Score
                        0 to 100
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           ALLOW        REVIEW        BLOCK
                          │
                          ▼
               SHAP & Graph Analysis
                          │
                          ▼
         Streamlit Dashboard & FastAPI REST Service
```

---

## System Architecture

```text
                              Incoming Transaction Payload
                                           │
                                           ▼
                           ┌───────────────────────────────┐
                           │    Stateful Feature Pipeline  │
                           │     (CustomerStateStore)      │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                           ┌───────────────────────────────┐
                           │   Hybrid Risk Intelligence    │
                           │  (ML + Anomaly + Business)    │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
      1. Calibrated ML Prob       2. Normalized Anomaly      3. Action Directive
         (Optuna XGBoost)           (Isolation Forest)        (ALLOW / REVIEW / BLOCK)
```

---

## Data Provenance

| Data Source | Purpose | Nature |
| :--- | :--- | :--- |
| **PaySim** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)) | `TRANSFER` and `CASH_OUT` fraud model training & evaluation | Synthetic mobile money simulation based on aggregated real logs |
| **RBI Payment Intelligence** ([rbi.org.in](https://www.rbi.org.in/)) | Indian digital payment ecosystem context | Official aggregate statistics |
| **Synthetic UPI Generator** (`src/synthetic_upi.py`) | Indian UPI-like behavioral simulation | Explicitly synthetic |

> **Important:** The transaction-level fraud model is trained and evaluated on PaySim synthetic mobile money data. RBI data is used only for Indian payment system context. The UPI-like dataset generated by this project is explicitly synthetic and is not presented as real banking data.

---

## Data Understanding

### PaySim Dataset Scope

PaySim is used as the primary transaction-level dataset for fraud detection. The dataset contains 6.36 million simulated financial transactions.

> **Scope Restriction**: The machine learning model is trained exclusively on `TRANSFER` and `CASH_OUT` transaction types, as PaySim fraud labels are concentrated in these categories (0% fraud in `PAYMENT`, `CASH_IN`, `DEBIT`).

### RBI Context

Reserve Bank of India (RBI) Payment System Indicators provide macro-level context around India's digital payment ecosystem. RBI aggregate statistics are used for domain context rather than transaction-level labels.

### Synthetic UPI Data

The project includes a persona-based synthetic UPI generator (`src/synthetic_upi.py`) to simulate Indian payment behaviors (VPA handles, UPI app market shares, merchant categories) and experiment with network risk characteristics.

---

## India Payment Ecosystem Context

The project considers several representative digital payment risk patterns:

| Risk Vector | Example Pattern |
| :--- | :--- |
| **Social Engineering** | Phishing collect requests and unauthorized transfers |
| **Fraudulent Collect Requests** | Manipulated payment authorization requiring PIN entry |
| **Mule Account Rings** | Coordinated accounts receiving rapid multi-tier transfers |
| **High Velocity Spikes** | Multiple high-value transactions in short time windows |
| **Behavioral Deviation** | Amounts significantly higher than customer's 30-day mean |
| **Beneficiary Churn** | Rapid transfers to previously unseen beneficiary accounts |
| **Nocturnal Drain** | Off-hour transfers (00:00–05:00) that liquidate account balance |

Exact payment volumes and ecosystem statistics vary over time and should be referenced directly from current RBI publications.

---

## Analytical Methodology

The analytical pipeline follows a structured sequence:

```text
Raw Data
   │
   ▼
Data Validation & Rail Filtering (TRANSFER / CASH_OUT)
   │
   ▼
Temporal Ordering (step column)
   │
   ▼
Causal Feature Engineering (Backward-looking rolling windows)
   │
   ▼
Model Benchmarking (Logistic Regression, Random Forest, XGBoost)
   │
   ▼
Optuna Bayesian Optimization (Validation Set)
   │
   ▼
Probability Calibration (Platt Sigmoid Scaling on Validation Set)
   │
   ▼
Cost-Sensitive Threshold Optimization (Locked on Validation Set)
   │
   ▼
One-Time Final Evaluation (Untouched Test Set)
   │
   ▼
Fitted Isolation Forest & Risk Engine Synthesis
```

---

## Feature Engineering

The project develops causal transaction features across multiple analytical dimensions. All features are constructed using **strictly backward-looking logic** to prevent temporal data leakage.

| Feature Category | Examples | Leakage Mitigation |
| :--- | :--- | :--- |
| **Transaction** | `log_amount`, One-hot payment types | Same-row non-leaking |
| **Velocity** | `transactions_last_1h`, `6h`, `24h` | Causal `searchsorted` rolling count |
| **Behavioral** | `amount_to_orig_prior_mean_ratio` | Expanding prior mean excluding current row |
| **Balance Errors** | `orig_balance_err`, `dest_balance_err` | PaySim artifact (evaluated via ablation study) |
| **Beneficiary** | `is_new_beneficiary` | Sequential `cumcount` tracking |
| **Temporal** | `hour_of_day`, `is_night_time` | Extracted directly from step index |
| **Network** | PageRank, Degree Centrality, Mule Score | Derived from NetworkX MultiDiGraph |

The feature engineering pipeline is implemented in:

```text
src/feature_engineering.py
```

---

## Machine Learning Approach

The project benchmarks multiple supervised learning algorithms before selecting the final model.

```text
Logistic Regression Baseline
        │
        ▼
Random Forest Baseline
        │
        ▼
Optuna-Tuned XGBoost
        │
        ▼
CalibratedClassifierCV (Platt Sigmoid Scaling)
```

XGBoost with Platt calibration was selected as the primary fraud classifier because it provided the strongest PR-AUC and probability calibration on the temporal evaluation set.

---

## Model Evaluation

Models were evaluated using a strict **temporal test split** (Train 70%, Validation 15%, Test 15%) where earlier observations were used for training and later observations were used for testing.

> **Single Source of Truth**: All reported metrics flow directly from `models/model_metadata.json` generated during training.

### Benchmark Comparison (Untouched Test Set)

| Model | PR-AUC | ROC-AUC | Threshold | Precision | Recall | F1 | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7959 | 0.9830 | 0.50 | 0.1840 | 0.7634 | 0.2966 | 0.0412 |
| Random Forest | 0.8660 | 0.9940 | 0.50 | 0.4180 | 0.8817 | 0.5670 | 0.0215 |
| **XGBoost (Calibrated, Optuna)** | **0.9515** | **0.9964** | **Validation Locked** | **0.3100** | **1.0000** | **0.4733** | **0.0018** |

---

## Key Model Result

At the cost-optimized operating threshold (locked on validation data), the calibrated XGBoost model achieved:

```text
PR AUC (Test)                   0.9515
ROC AUC (Test)                  0.9964
Recall (Test)                   1.0000
Brier Score (Calibration)       0.0018
Expected Calibration Error      0.0042
Observed Test Fraud Cases       186
Fraud Cases Intercepted         186
```

The model detected all observed fraud cases in the temporal test set. The selected threshold deliberately favors recall over precision because the financial cost of missing fraudulent transactions is far higher than investigating false positives.

---

## Why PR-AUC

Digital payment fraud detection is a highly imbalanced classification task (~0.2% fraud rate). Accuracy provides a misleading view of performance.

The primary evaluation metric for this project is **PR-AUC (Average Precision)** because it measures the precision-recall trade-off directly under severe class imbalance.

Evaluation metrics suite:

```text
PR-AUC (Primary)
ROC-AUC
Precision & Recall
F1 Score
Brier Score (Calibration)
Expected Calibration Error (ECE)
Expected Financial Loss (INR)
```

---

## Temporal Validation

A strict temporal validation strategy was used instead of random train-test splitting:

```text
Steps 1 to 520 (70%)        -->  Training Set (Model Fitting & Isolation Forest)
Steps 521 to 631 (15%)      -->  Validation Set (Optuna Tuning, Calibration & Threshold Locking)
Steps 632 to 743 (15%)      -->  Test Set (One-Time Evaluation)
```

This ensures zero temporal data leakage and mimics real-world production deployment.

---

## Cost-Sensitive Threshold Optimization

A fraud detection model should not automatically use an arbitrary threshold of 0.50. The project evaluates thresholds on the **Validation Set** using an explicit financial loss function:

$$\text{Loss} = \sum_{\text{each missed fraud}} \text{actual\_amount}_i + (\text{False Positive Count} \times \text{INR } 200)$$

The threshold optimization evaluates the trade-off between:

```text
Recall (Fraud Capture Rate)
Precision (Analyst Review Burden)
Actual Financial Loss from Missed Frauds
Analyst Investigation Cost per False Positive (INR 200)
```

The complete threshold analysis is saved at `reports/threshold_cost_table.csv`.

---

## Hybrid Risk Engine

The Risk Engine combines three independent analytical signals into a composite **Risk Score (0 to 100)**:

$$\text{Risk Score} = (P_{\text{ML}} \times 100 \times w_{\text{ML}}) + (\text{Score}_{\text{Anomaly}} \times 100 \times w_{\text{Anomaly}}) + (\text{Score}_{\text{Rules}} \times w_{\text{Rules}})$$

| Component | Scale | Weight | Description |
| :--- | :---: | :---: | :--- |
| **Calibrated XGBoost ML** | 0 to 1 $\rightarrow$ 0 to 100 | **60%** | Calibrated fraud probability ($P_{\text{ML}}$) |
| **Isolation Forest Anomaly** | 0 to 1 $\rightarrow$ 0 to 100 | **20%** | Normalized anomaly score (fitted model) |
| **Business Rules Engine** | 0 to 100 | **20%** | Deterministic business rule triggers |

---

## Risk Classification

| Risk Score | Tier | Action | Operational Action Directive |
| :---: | :--- | :--- | :--- |
| **0 to 30** | LOW | ALLOW | Automated payment approval |
| **31 to 60** | MEDIUM | REVIEW | Step-up authentication requested |
| **61 to 80** | HIGH | REVIEW | Step-up authentication & analyst queue review |
| **81 to 100** | CRITICAL | BLOCK | Transaction blocked; escalate for investigation |

Weights and thresholds are configurable via `models/model_metadata.json`.

---

## Explainable AI & Rule Triggers

The platform provides multi-layered explainability for individual predictions:

```text
Transaction Payload
        │
        ▼
Calibrated Probability & Anomaly Score
        │
        ▼
Rule Engine Evaluation
        │
        ├── Rule 01: High Velocity Spike (≥5 tx/1h)
        ├── Rule 02: High Value + New Beneficiary (≥₹50,000)
        ├── Rule 03: Off-Hours Nocturnal Transfer (00:00–05:00)
        ├── Rule 04: Complete Balance Drain (100% liquidated)
        └── Rule 05: Extreme Amount Ratio vs Customer Baseline
        │
        ▼
Feature Contribution Breakdown & Analyst Report
```

This ensures analysts understand the drivers behind every decision directive rather than treating the system as a black box.

---

## Anomaly Detection

Isolation Forest provides an unsupervised risk signal. The model is trained on normal transaction features (`models/isolation_forest.pkl`) to identify unusual structural outliers without relying on fraud labels:

```text
Supervised Fraud Classifier (XGBoost)
                 +
Unsupervised Anomaly Detector (Isolation Forest)
                 +
Deterministic Business Rules
                 │
                 ▼
     Hybrid Risk Intelligence
```

---

## Behavioral Analytics

The platform incorporates stateful behavioral tracking via `CustomerStateStore`:

```text
Customer Transaction History
             │
             ▼
   CustomerStateStore
             │
             ├── Rolling 1h / 6h / 24h Velocity
             ├── Expanding Historical Mean Amount
             ├── Amount Ratio vs Customer Mean
             └── Beneficiary Association Tracking
             │
             ▼
Real-Time Feature Vector for Model Inference
```

---

## Synthetic UPI Simulation

The Synthetic UPI Generator (`src/synthetic_upi.py`) creates transactions from three behavioral personas:

| Persona | Characteristics | Target Fraud Rate |
| :--- | :--- | :--- |
| **Normal** | Stable amounts, low velocity, consistent device usage | 0% |
| **Suspicious** | Elevated velocity, nocturnal transfers, device/location shifts | High |
| **Mule** | Collection hub accounts, rapid inbound transfers, liquidations | High |

The simulator supports graph analytics and risk experimentation.

---

## Graph-Based Risk Analysis

Fraudulent activity often involves interconnected accounts. The project uses **NetworkX MultiDiGraph** (`src/graph_fraud.py`) to build multi-edge transaction flow graphs and compute structural features:

| Graph Feature | Purpose |
| :--- | :--- |
| **PageRank** | Identifies structurally prominent accounts |
| **Degree Centrality** | Measures account connectivity and transaction volume |
| **In-Degree / Out-Degree** | Detects fan-in (mule collection) & fan-out (distribution) hubs |
| **Cycle Detection** | Identifies circular payment chains ($A \rightarrow B \rightarrow C \rightarrow A$) |

Graph metrics serve as structural risk signals alongside supervised ML and rules.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard (`app/streamlit_app.py`):

- **Executive Overview**: High-level metrics, dataset provenance, business KPIs.
- **Transaction Risk Checker**: Interactive scenario simulator with `CustomerStateStore` baselines.
- **Investigation Queue**: Analyst workflow connected to SQLite database (`data/processed/fraud_intelligence.db`).
- **Model Analytics**: Validation threshold-cost curves, PR curves, calibration statistics.
- **Graph & Synthetic UPI**: MultiDiGraph node centrality visualization & mule ring detection.

---

## SQL Analytics & Database Manager

The project includes a SQL warehouse manager (`sql/db_manager.py`, `sql/schema.sql`, `sql/fraud_analysis.sql`) supporting:

```text
sql/
├── schema.sql           # DDL schema for transactions, alerts, and investigation queue
├── db_manager.py        # SQLite manager logging evaluations & managing analyst workflow
└── fraud_analysis.sql   # Analytical queries for velocity, fraud loss, and mule accounts
```

---

## Repository Structure

```text
Digital Payment Fraud & Risk Intelligence System/
├── api/                                # RESTful FastAPI Inference Service
│   ├── main.py                         # FastAPI app & endpoints
│   ├── schemas.py                      # Pydantic request/response models
│   └── dependencies.py                  # Service dependencies
├── app/                                # Streamlit Analyst Dashboard
│   ├── streamlit_app.py                # Dashboard application
│   └── style.css                       # Dark glassmorphism styling
├── src/                                # Core Engine Modules
│   ├── data_processing.py              # Data loading, validation, temporal split
│   ├── feature_engineering.py          # Strictly causal feature pipeline
│   ├── customer_state.py               # Stateful velocity & baseline store
│   ├── train.py                        # Model training, Optuna, calibration, IF
│   ├── predict.py                      # Single & batch inference predictor
│   ├── risk_engine.py                  # Hybrid risk score synthesizer
│   ├── synthetic_upi.py                # Persona-based UPI transaction generator
│   └── graph_fraud.py                  # NetworkX MultiDiGraph analytics
├── sql/                                # Database & Warehouse Analytics
│   ├── schema.sql                      # DDL schema for SQLite/PostgreSQL
│   ├── db_manager.py                   # SQLite database manager
│   └── fraud_analysis.sql              # Analytical queries
├── monitoring/                         # MLOps & Performance Monitoring
│   ├── drift_detector.py               # Population Stability Index (PSI) & KS test
│   └── business_kpis.py                # Production KPI tracking
├── models/                             # Model Artifacts (Single Source of Truth)
│   ├── xgboost_model.pkl               # Calibrated XGBoost binary
│   ├── xgboost_model.json              # Native XGBoost JSON model
│   ├── isolation_forest.pkl            # Fitted Isolation Forest binary
│   └── model_metadata.json             # Parameters, metrics, cost model, versions
├── reports/                            # Pipeline Artifacts & Reports
│   ├── model_card.md                   # Formal Model Card & provenance
│   ├── dataset_summary.json            # Volume & split statistics
│   ├── threshold_cost_table.csv        # Validation threshold optimization curve
│   ├── model_comparison.csv            # Cross-model benchmark table
│   ├── ablation_study.csv              # PaySim balance error ablation results
│   └── rbi_india_context.md            # RBI payment intelligence background
├── tests/                              # Unit & Integration Tests (40 tests)
│   ├── test_features.py                # Causal logic & temporal leakage tests
│   ├── test_risk_engine.py             # Risk tier & weight validation tests
│   ├── test_prediction.py              # Inference pipeline & safety tests
│   ├── test_causal_state.py            # Customer state store & DB tests
│   ├── test_graph_fraud.py             # MultiDiGraph & cycle detection tests
│   └── test_monitoring.py             # Drift detector & business KPI tests
├── tools/                              # Helper Tools & Generators
│   └── create_merged_notebook.py       # Standalone notebook generator
├── .github/workflows/                  # CI/CD Pipeline
│   └── ci.yml                          # GitHub Actions CI workflow
├── Dockerfile                          # FastAPI container definition
├── Dockerfile.streamlit                # Dashboard container definition
├── docker-compose.yml                  # Multi-container service definition
├── requirements.txt                    # Pinned production dependencies
├── requirements-dev.txt                # Testing & linting dependencies
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Run Test Suite (40 Unit Tests)

```bash
pytest tests/ -v
```

### 3. Run Model Training Pipeline

```bash
python src/train.py
```

### 4. Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Access local dashboard at `http://localhost:8501`.

### 5. Launch FastAPI REST Service

```bash
python -m uvicorn api.main:app --port 8000 --reload
```

Access interactive API docs at `http://localhost:8000/docs`.

### 6. Run with Docker Compose

```bash
docker-compose up --build
```

---

## Testing

Automated testing covers key parts of the system:

```text
tests/
├── test_features.py        # Causal feature validation & temporal leakage tests
├── test_risk_engine.py     # Risk tier mapping & weight validation tests
├── test_prediction.py      # Inference safety & error handling tests
├── test_causal_state.py    # Customer state store & SQLite logging tests
├── test_graph_fraud.py     # NetworkX MultiDiGraph & cycle tests
└── test_monitoring.py     # PSI drift detector & business KPI tests
```

Run test suite:

```bash
pytest tests/ -v
```

---

## Model Governance & Limitations

A dedicated model card is included at `reports/model_card.md`.

### Important Limitations

- **Synthetic Training Data**: PaySim is a synthetic mobile-money simulation.
- **PaySim Balance Artifacts**: The model relies partly on PaySim balance discrepancy artifacts (quantified via `reports/ablation_study.csv`).
- **Scope Limit**: Model trained exclusively on `TRANSFER` and `CASH_OUT` rails.
- **Synthetic UPI Layer**: The UPI dataset generator is explicitly synthetic.
- **Calibration & Governance**: Production deployment requires validation on real banking transaction streams.

---

## Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11+** | Core platform development |
| **FastAPI / Uvicorn** | Production RESTful API service |
| **Docker / Docker Compose** | Containerized deployment |
| **GitHub Actions** | Automated CI/CD pipeline |
| **Pandas / NumPy** | Causal feature engineering & numerical computation |
| **Scikit-Learn** | CalibratedClassifierCV & Isolation Forest anomaly detection |
| **XGBoost** | Optuna-tuned fraud classification |
| **Optuna** | Bayesian hyperparameter optimization |
| **SHAP** | Explainable AI (XAI) feature contribution |
| **NetworkX** | MultiDiGraph transaction network analytics |
| **SQLite / SQL** | Investigation queue warehouse & analytical queries |
| **Streamlit** | Analyst risk intelligence dashboard |
| **Pytest** | Automated unit & integration testing (40 tests) |

---

## Business Impact

The platform translates raw machine learning probability into actionable operational decision directives:

```text
Raw Prediction Only  ──>  Calibrated Probability + Anomaly Score + Rule Explanations
                                                │
                                                ▼
                                    Unified Risk Score (0-100)
                                                │
                                                ▼
                                    ALLOW / REVIEW / BLOCK Decision
```

---

## Author

**Sanman Kadam**  
MSc Statistics | Data Analytics and Data Science

---

<p align="center">
  <strong>Digital Payment Fraud and Risk Intelligence Platform</strong>
</p>

<p align="center">
  Detection. Explanation. Decision.
</p>
