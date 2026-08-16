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

```text
Business Problem
        ↓
Objective
        ↓
Solution
        ↓
Architecture
        ↓
Dataset
        ↓
Methodology
        ↓
Results
        ↓
Risk Engine
        ↓
Deployment
        ↓
Monitoring
```

---

## 1. Business Problem

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

## 2. Objective

The primary objective is to develop an end-to-end fraud intelligence platform capable of answering three key operational questions:

```text
1. Is the transaction fraudulent or suspicious?

2. Why is the transaction considered risky?

3. What action should be taken (ALLOW / REVIEW / BLOCK)?
```

### Core Capabilities

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

## 3. Solution

The project uses a hybrid fraud intelligence framework that combines supervised ML, unsupervised anomaly detection, behavioral baselines, and deterministic business rules into a single operational decision directive.

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
│   Fraud ML (80%)       Anomaly (10%)     Engine (10%)
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

## 4. Architecture

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

The system architecture flows from incoming transaction payloads to stateful feature enrichment (`CustomerStateStore`), multi-component risk score synthesis (`RiskEngine`), and automated decision directives (`ALLOW`, `REVIEW`, `BLOCK`).

---

## 5. Dataset

### Data Provenance & Scope

| Data Source | Purpose | Nature |
| :--- | :--- | :--- |
| **PaySim** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)) | `TRANSFER` and `CASH_OUT` fraud model training & evaluation | Synthetic mobile money simulation based on aggregated real logs |
| **RBI Payment Intelligence** ([rbi.org.in](https://www.rbi.org.in/)) | Indian digital payment ecosystem context | Official aggregate statistics |
| **Synthetic UPI Generator** (`src/synthetic_upi.py`) | Indian UPI-like behavioral simulation | Explicitly synthetic |

> [!IMPORTANT]
> **Transaction Rail Scope**: The machine learning model is trained exclusively on `TRANSFER` and `CASH_OUT` transaction types, as PaySim fraud labels are concentrated in these categories (0% fraud in `PAYMENT`, `CASH_IN`, `DEBIT`).

### Reserve Bank of India (RBI) Ecosystem Context

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

---

## 6. Methodology

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

### Feature Engineering

All features are constructed using **strictly backward-looking (causal) logic** to prevent temporal data leakage:

| Feature Category | Examples | Leakage Mitigation |
| :--- | :--- | :--- |
| **Transaction** | `log_amount`, One-hot payment types | Same-row non-leaking |
| **Velocity** | `transactions_last_1h`, `6h`, `24h` | Causal `searchsorted` rolling count |
| **Behavioral** | `amount_to_orig_prior_mean_ratio` | Expanding prior mean excluding current row |
| **Balance Errors** | `orig_balance_err`, `dest_balance_err` | PaySim artifact (evaluated via ablation study) |
| **Beneficiary** | `is_new_beneficiary` | Sequential `cumcount` tracking |
| **Temporal** | `hour_of_day`, `is_night_time` | Extracted directly from step index |
| **Network** | PageRank, Degree Centrality, Mule Score | Derived from NetworkX MultiDiGraph |

Implemented in [`src/feature_engineering.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/feature_engineering.py).

### Cost-Sensitive Threshold Optimization

A fraud detection model should not automatically use an arbitrary threshold of 0.50. Operating thresholds are selected on the **Validation Set** using an explicit financial loss function:

$$\text{Loss} = \sum_{\text{each missed fraud}} \text{actual\_amount}_i + (\text{False Positive Count} \times \text{INR } 200)$$

---

## 7. Results

Models were evaluated on a strict **temporal test split** (Train 70%, Validation 15%, Test 15%) where earlier observations were used for training and later observations were used for testing.

> **Single Source of Truth**: All reported metrics flow directly from `models/model_metadata.json` generated during training.

### Benchmark Comparison (Untouched Test Set)

| Model | PR-AUC | ROC-AUC | Threshold | Precision | Recall | F1 | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7959 | 0.9830 | 0.50 | 0.1840 | 0.7634 | 0.2966 | 0.0412 |
| Random Forest | 0.8660 | 0.9940 | 0.50 | 0.4180 | 0.8817 | 0.5670 | 0.0215 |
| **XGBoost (Calibrated, Optuna)** | **0.9471** | **0.9962** | **0.0100 (Validation Locked)** | **0.4876** | **0.9516** | **0.6448** | **0.0067** |

### Key Model Results

```text
PR-AUC (Test Set)               0.9471
ROC-AUC (Test Set)              0.9962
Recall (Test Set)               0.9516
Brier Score (Calibration)       0.0067
Expected Calibration Error      0.0091
Optimal Locked Threshold        0.0100
```

### Why PR-AUC?

Digital payment fraud detection is a highly imbalanced classification task (~0.2% fraud rate). Accuracy provides a misleading view of performance. **PR-AUC (Average Precision)** is the primary metric because it measures the precision-recall trade-off directly under severe class imbalance.

---

## 8. Risk Engine

The Risk Engine synthesizes three independent analytical signals into a composite **Risk Score (0 to 100)**:

$$\text{Risk Score} = (P_{\text{ML}} \times 100 \times w_{\text{ML}}) + (\text{Score}_{\text{Anomaly}} \times 100 \times w_{\text{Anomaly}}) + (\text{Score}_{\text{Rules}} \times w_{\text{Rules}})$$

| Component | Scale | Default Weight | Description |
| :--- | :---: | :---: | :--- |
| **Calibrated XGBoost ML** | 0 to 1 $\rightarrow$ 0 to 100 | **80%** | Calibrated fraud probability ($P_{\text{ML}}$) |
| **Isolation Forest Anomaly** | 0 to 1 $\rightarrow$ 0 to 100 | **10%** | Normalized anomaly score (fitted model) |
| **Business Rules Engine** | 0 to 100 | **10%** | Deterministic business rule triggers |

### Risk Classification Tiers & Actions

| Risk Score | Risk Tier | Action | Operational Action Directive |
| :---: | :--- | :--- | :--- |
| **0 to 30** | LOW | ALLOW | Automated payment approval |
| **31 to 60** | MEDIUM | REVIEW | Step-up authentication requested |
| **61 to 80** | HIGH | REVIEW | Step-up authentication & analyst queue review |
| **81 to 100** | CRITICAL | BLOCK | Transaction blocked; escalate for investigation |

### Explainable AI & Rule Triggers

The platform provides multi-layered explainability:
- **Rule 01**: High Velocity Spike ($\ge 5$ tx/1h)
- **Rule 02**: High Value + New Beneficiary ($\ge ₹50,000$)
- **Rule 03**: Off-Hours Nocturnal Transfer (00:00–05:00)
- **Rule 04**: Complete Balance Drain (100% liquidated)
- **Rule 05**: Extreme Amount Ratio vs Customer Baseline

### Graph-Based Network Analysis

Uses **NetworkX MultiDiGraph** (`src/graph_fraud.py`) to compute node PageRank, in/out-degree centrality, and detect circular payment cycles ($A \rightarrow B \rightarrow C \rightarrow A$).

---

## 9. Deployment

### RESTful FastAPI Microservice (`api/`)

Provides low-latency REST endpoints with Pydantic schema validation:
- `POST /predict`: Single transaction fraud scoring
- `POST /risk-score`: Full hybrid risk engine evaluation
- `GET /model-info`: Model metadata & software version provenance
- `GET /health`: Health check endpoint

### Interactive Streamlit Dashboard (`app/`)

Analyst investigation dashboard featuring:
- **Executive Overview**: High-level metrics, dataset summary, business KPIs.
- **Transaction Risk Checker**: Scenario simulator with stateful baselines.
- **Investigation Queue**: Analyst decision queue connected to SQLite warehouse (`data/processed/fraud_intelligence.db`).
- **Model Analytics**: Validation threshold-cost curves, PR curves, calibration stats.

### Containerization & CI/CD

- **`Dockerfile`**: Multi-stage Python 3.11 container for FastAPI.
- **`Dockerfile.streamlit`**: Dedicated container for the Streamlit dashboard.
- **`docker-compose.yml`**: Multi-container service definition with read-only volume mounts and health checks.
- **`.github/workflows/ci.yml`**: Automated GitHub Actions pipeline executing `flake8`, `black --check`, 40 `pytest` unit tests, and Docker build verification.

---

## 10. Monitoring

### Population Stability Index (PSI) Drift Detector

Implemented in [`monitoring/drift_detector.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/monitoring/drift_detector.py):
- Calculates PSI across 10 quantile bins and Kolmogorov-Smirnov (KS) test statistics to monitor feature distribution shifts.
- Classifies features as `STABLE` ($\text{PSI} < 0.10$), `MODERATE_DRIFT` ($0.10 \le \text{PSI} < 0.25$), or `SIGNIFICANT_DRIFT` ($\text{PSI} \ge 0.25$).

### Production Business KPI Tracker

Implemented in [`monitoring/business_kpis.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/monitoring/business_kpis.py):
- Tracks fraud capture rate, review rate, blocked fraud value (INR), missed fraud loss (INR), false positive investigation cost, and cost per detected fraud.

### Model Governance & Technical Documentation

- **Formal Model Card**: [reports/model_card.md](reports/model_card.md)
- **Comprehensive System & File Explanation**: [docs/TECHNICAL_EXPLANATION.md](docs/TECHNICAL_EXPLANATION.md) — Detailed step-by-step technical guide explaining every file, algorithm, rationale, and implementation pattern.

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
├── docs/                               # Comprehensive Technical Documentation
│   └── TECHNICAL_EXPLANATION.md        # File-by-file & concept-by-concept deep dive
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

## Technology Stack & Author

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

**Sanman Kadam**  
MSc Statistics | Data Analytics and Data Science

---

<p align="center">
  <strong>Digital Payment Fraud and Risk Intelligence Platform</strong>
</p>

<p align="center">
  Detection. Explanation. Decision.
</p>
