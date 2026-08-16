# Digital Payment Fraud & Risk Intelligence System — Comprehensive Technical & Architectural Guide

| Metadata | Details |
| :--- | :--- |
| **Author** | Sanman Kadam |
| **Document Purpose** | Comprehensive file-by-file and concept-by-concept technical explanation |
| **Target Audience** | Senior ML Engineers, Risk Analytics Managers, Technical Recruiters |
| **System Version** | v3.0.0 (Production-Grade Overhaul) |

---

## 1. Executive Overview & Design Rationale

The **Digital Payment Fraud & Risk Intelligence System** is an end-to-end, production-grade fraud decision platform. Unlike basic toy machine learning models that simply output `0` or `1`, this platform reflects how **tier-1 financial institutions, payment gateways, and UPI payment switches** actually evaluate transaction risk in production environments.

### Core Design Philosophy
1. **Never Predict in a Vacuum**: Machine learning probability alone is not enough to block a transaction. Real-world systems fuse ML predictions with unsupervised anomaly scores, stateful customer baselines, and deterministic compliance/business rules.
2. **Prevent Temporal Data Leakage**: In financial fraud modeling, training on future data to predict the past leads to artificially inflated offline metrics that collapse in production. All features, splits, and thresholds in this system are strictly **causal (backward-looking)** and **chronologically ordered**.
3. **Optimize for Financial Impact, Not Accuracy**: With massive class imbalance (~0.2% fraud rate), accuracy is meaningless. Operating thresholds are selected by minimizing **actual expected financial loss** (cost of missed fraud + cost of false positive analyst investigations).
4. **Production Architecture First**: ML code is accompanied by typed REST APIs (FastAPI), containerization (Docker & Compose), automated CI/CD workflows (GitHub Actions), and MLOps drift monitoring (Population Stability Index).

---

## 2. Core Topics & Methodological Rationale (Why & How)

### 2.1 Causal Feature Engineering & Preventing Data Leakage

#### **WHY?**
Standard machine learning pipelines often calculate global dataset statistics (e.g., `df['amount'].mean()` or `df.groupby('customer')['amount'].transform('mean')`). Doing this over an entire dataset causes **data leakage** because a transaction at time $T_1$ uses mean information from future transactions at time $T_{100}$. In a real-time payment gateway, future transactions do not exist yet.

#### **HOW?**
All features in [`src/feature_engineering.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/feature_engineering.py) are calculated using **strictly backward-looking (causal) logic**:
- **Expanding Customer Mean (`amount_to_orig_prior_mean_ratio`)**: Calculated using `cumsum()` and `cumcount()` shifted by 1 row (`shift(1)`), ensuring current transaction amount is compared *only* to the customer's prior transaction history.
- **Rolling Time-Window Velocity (`transactions_last_1h/6h/24h`)**: Computed using `np.searchsorted` on sorted step timestamps for each customer. It counts transactions occurring strictly within $[T - \Delta t, T)$ before the current event.
- **First-Time Beneficiary Tracking (`is_new_beneficiary`)**: Tracks cumulative occurrences of $(u, v)$ originator-beneficiary pairs up to time $T$.

---

### 2.2 Temporal Train / Validation / Test Split Strategy

#### **WHY?**
Random $k$-fold cross-validation or random `train_test_split` scrambles transaction time steps. If step 500 is in the training set and step 100 is in the test set, the model memorizes future patterns to predict the past.

#### **HOW?**
In [`src/data_processing.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/data_processing.py), data is sorted strictly by the simulation step index (`step` = hour index from 1 to 743):
- **Training Set (70%)**: Earliest steps (Steps 1 to 520) — used exclusively for model fitting and Isolation Forest training.
- **Validation Set (15%)**: Middle steps (Steps 521 to 631) — used for Optuna Bayesian hyperparameter search, probability calibration fitting, and cost-sensitive threshold selection.
- **Test Set (15%)**: Latest steps (Steps 632 to 743) — **locked and untouched** until the final one-time evaluation.

---

### 2.3 Optuna Bayesian Hyperparameter Optimization

#### **WHY?**
Grid search and random search are inefficient for gradient boosted trees. Optuna uses Tree-structured Parzen Estimator (TPE) Bayesian optimization to intelligently sample hyperparameter combinations that maximize the objective metric.

#### **HOW?**
In [`src/train.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/train.py), Optuna runs 15 trials maximizing **PR-AUC (Average Precision)** on the **Validation Set**:
- Hyperparameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`.
- Handles imbalanced data dynamically by setting `scale_pos_weight = (negative_count / positive_count)`.

---

### 2.4 Probability Calibration (Platt Sigmoid Scaling)

#### **WHY?**
Standard gradient boosted trees output raw margin scores passed through a sigmoid, which are often uncalibrated (clustering near 0 and 1). For risk engines that blend ML probabilities with other scores, raw model scores cannot be trusted as true statistical probabilities.

#### **HOW?**
In [`src/train.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/train.py), `sklearn.calibration.CalibratedClassifierCV` fits a logistic regression (Platt scaling) on the validation set predictions (`cv='prefit'`).
- Measures calibration quality using **Brier Score** ($\frac{1}{N}\sum (y_i - \hat{p}_i)^2$) and **Expected Calibration Error (ECE)** across 10 probability bins.

---

### 2.5 Cost-Sensitive Threshold Locking (Financial Loss Minimization)

#### **WHY?**
The standard binary decision threshold of `0.50` is completely arbitrary. In fraud detection, missing a ₹84,500 fraud transfer (False Negative) is exponentially more expensive than spending ₹200 on an analyst investigating a legitimate transaction (False Positive).

#### **HOW?**
In [`src/train.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/train.py), the threshold cost table evaluates 99 threshold candidates from `0.01` to `0.99` on the **Validation Set**:

$$\text{Total Expected Loss} = \sum_{\text{each False Negative}} \text{actual\_amount}_i + (\text{FP Count} \times \text{INR } 200)$$

The threshold minimizing total financial loss is **locked** and evaluated once on the untouched test set.

---

### 2.6 Unsupervised Anomaly Detection (Fitted Isolation Forest)

#### **WHY?**
Supervised ML models only recognize fraud patterns present in historical training labels (known-knowns). Novel fraud vectors, zero-day exploits, or account takeover anomalies (known-unknowns) require an unsupervised signal that flags unusual structural behavior regardless of past labels.

#### **HOW?**
In [`src/train.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/train.py) and [`src/risk_engine.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/risk_engine.py):
- An `IsolationForest` (100 estimators, 1% contamination) is fitted on training features and saved to `models/isolation_forest.pkl`.
- Raw decision function scores ($\approx [-0.5, 0.5]$) are inverted and clipped to a normalized score $\in [0, 1]$, where higher values indicate severe structural anomalies.

---

### 2.7 Multi-Component Hybrid Risk Score Synthesis

#### **WHY?**
Single-model fraud systems fail when model endpoints experience downtime or when complex social engineering bypasses pure ML signals. Real-world payment gateways synthesize multiple risk perspectives into a unified composite score.

#### **HOW?**
In [`src/risk_engine.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/risk_engine.py), three independent signals are scaled to 0–100 and weighted:

$$\text{Composite Risk Score} = (P_{\text{ML}} \times 100 \times w_{\text{ML}}) + (\text{Score}_{\text{Anomaly}} \times 100 \times w_{\text{Anomaly}}) + (\text{Score}_{\text{Rules}} \times w_{\text{Rules}})$$

- **Component Weights**: Loaded from `model_metadata.json` (optimized on validation set: e.g., ML 80%, Anomaly 10%, Rules 10%).
- **Operational Action Mapping**:
  - `0 – 30`: **LOW Tier** $\rightarrow$ `ALLOW` (Automated approval)
  - `31 – 60`: **MEDIUM Tier** $\rightarrow$ `REVIEW` (Step-up authentication requested)
  - `61 – 80`: **HIGH Tier** $\rightarrow$ `REVIEW` (Step-up authentication + analyst queue)
  - `81 – 100`: **CRITICAL Tier** $\rightarrow$ `BLOCK` (Transaction blocked + account escalation)

---

### 2.8 Stateful Real-Time Velocity & Baseline Tracking (`CustomerStateStore`)

#### **WHY?**
When evaluating a single incoming API transaction payload, batch rolling windows (`df.groupby()`) cannot be executed across millions of historical rows without severe latency. A stateful in-memory store keeps running customer statistics updated in $O(1)$ time.

#### **HOW?**
In [`src/customer_state.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/customer_state.py):
- Maintains in-memory sliding window deques of customer transaction timestamps, amounts, and seen beneficiary sets.
- `compute_realtime_features(tx)` calculates exact 1h/6h/24h velocity counts and expanding amount ratios in real time.
- `update_state(tx)` updates sliding deques post-evaluation.

---

### 2.9 Transaction Network Topology & Mule Detection (NetworkX MultiDiGraph)

#### **WHY?**
Organized fraud rings operate money mule networks — multi-tier layered transfers where stolen funds are rapidly split across intermediate drop accounts before cash-out. Tabular transaction models miss graph topology.

#### **HOW?**
In [`src/graph_fraud.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/src/graph_fraud.py):
- Constructs a directed multi-edge graph (`nx.MultiDiGraph`) preserving multiple transfers between originators and beneficiaries.
- Computes **PageRank**, **In-Degree / Out-Degree ratios**, and **Mule Risk Scores** ($\text{In-Degree} / \text{Out-Degree}$).
- Identifies **circular payment chains** ($A \rightarrow B \rightarrow C \rightarrow A$) using `nx.simple_cycles`.

---

### 2.10 RESTful Microservice Architecture (FastAPI & Pydantic)

#### **WHY?**
Machine learning models locked inside Jupyter notebooks cannot be consumed by payment switches or external microservices. A production deployment requires a low-latency REST API with schema validation.

#### **HOW?**
In [`api/main.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/api/main.py) and [`api/schemas.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/api/schemas.py):
- Built using **FastAPI** with `asynccontextmanager` lifespan initialization for loading predictor and risk engine models.
- Uses **Pydantic v2** models (`TransactionRequest`, `RiskScoreResponse`, `ModelInfoResponse`) for strict data type validation.
- Automatically serves interactive OpenAPI documentation at `/docs`.

---

### 2.11 MLOps & Production Drift Monitoring (PSI & KS Test)

#### **WHY?**
Payment patterns shift over time due to seasonal spending, economic changes, or new fraud tactics (concept drift & covariate shift). Models deployed without monitoring silently degrade.

#### **HOW?**
In [`monitoring/drift_detector.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/monitoring/drift_detector.py) and [`monitoring/business_kpis.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/monitoring/business_kpis.py):
- **Population Stability Index (PSI)**: Quantifies feature distribution drift across 10 quantile bins:
  - $\text{PSI} < 0.10$: Stable
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift (Warning)
  - $\text{PSI} \ge 0.25$: Significant drift (Triggers retraining alert)
- **Kolmogorov-Smirnov (KS) Test**: Evaluates continuous feature distribution differences.
- **Business KPI Tracker**: Calculates capture rate, blocked fraud value (INR), missed loss (INR), false positive investigation costs, and cost per detected fraud.

---

### 2.12 Containerization & CI/CD Pipelines (Docker & GitHub Actions)

#### **WHY?**
Ensures environment consistency across local development, testing, staging, and cloud production environments ("works on my machine" prevention).

#### **HOW?**
- **[`Dockerfile`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/Dockerfile)**: Multi-stage slim Python 3.11 container for FastAPI inference service.
- **[`Dockerfile.streamlit`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/Dockerfile.streamlit)**: Container for the Streamlit analyst dashboard.
- **[`docker-compose.yml`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/docker-compose.yml)**: Orchestrates API (port 8000) and Dashboard (port 8501) with shared model volume mounts and health checks.
- **[`.github/workflows/ci.yml`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/.github/workflows/ci.yml)**: Automated GitHub Actions pipeline executing `flake8`, `black --check`, 40 `pytest` unit tests, and Docker image build verification on every push.

---

## 3. Detailed File-by-File Breakdown

### Core Modules (`src/`)

#### **1. [`src/data_processing.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/data_processing.py)**
- **Why**: Centralizes dataset loading, basic validation, and strict temporal splitting.
- **How**:
  - `load_raw_data()`: Loads PaySim CSV dataset from `data/`.
  - `get_data_summary()`: Calculates record counts, total fraud, fraud rate per rail type, and step range.
  - `temporal_train_val_test_split()`: Sorts by `step` column and splits into Train (70%), Validation (15%), and Test (15%) without shuffling.

#### **2. [`src/feature_engineering.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/feature_engineering.py)**
- **Why**: Constructs 22 causal behavioral, velocity, structural, and temporal risk features.
- **How**:
  - Implements `build_features(df, is_training=True)` using causal pandas/numpy operations.
  - `get_feature_matrix()`: Separates $X$ feature DataFrame and $y$ target series.
  - `get_feature_schema()`: Exports structured metadata (`FEATURE_SCHEMA_VERSION = "2.0.0"`) defining each feature's type, description, and leakage mitigation rationale.

#### **3. [`src/customer_state.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/customer_state.py)**
- **Why**: Enables stateful single-row inference by maintaining customer transaction history in memory.
- **How**:
  - `CustomerStateStore` class keeps sliding `deque` windows per `nameOrig`.
  - `compute_realtime_features(tx)`: Computes rolling 1h/6h/24h counts, amount ratios, and new beneficiary flags in $O(1)$ time.
  - `update_state(tx)`: Appends transaction payload to sliding history.

#### **4. [`src/train.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/train.py)**
- **Why**: Orchestrates end-to-end model training, hyperparameter optimization, probability calibration, threshold locking, ablation, and artifact export.
- **How**:
  - `train_and_evaluate_all()`: Coordinates the full pipeline.
  - Optuna study runs 15 trials tuning XGBoost hyperparameters on validation PR-AUC.
  - Fits `CalibratedClassifierCV` (Platt scaling) on validation predictions.
  - Fits and exports `IsolationForest` to `models/isolation_forest.pkl`.
  - Runs feature ablation study (WITH vs WITHOUT PaySim balance errors) $\rightarrow$ `reports/ablation_study.csv`.
  - Runs Risk Engine weight optimization $\rightarrow$ `reports/weight_optimization.csv`.
  - Exports `models/xgboost_model.pkl`, `models/xgboost_model.json`, `models/model_metadata.json`, and `reports/dataset_summary.json`.

#### **5. [`src/predict.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/predict.py)**
- **Why**: Handles model inference for single transactions or batch DataFrames.
- **How**:
  - `FraudPredictor` class loads model binaries and metadata at startup.
  - **Safety Fix**: Raises explicit `FileNotFoundError` if model binaries are missing (no silent dummy fallback).
  - `predict_single()`: Merges transaction input with `CustomerStateStore` real-time state, computes feature matrix, returns calibrated probability, threshold flag, and version metadata.

#### **6. [`src/risk_engine.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/risk_engine.py)**
- **Why**: Synthesizes supervised ML, unsupervised anomaly scores, and deterministic business rules into a composite 0–100 risk score and decision directive.
- **How**:
  - `RiskEngine` loads fitted Isolation Forest (`isolation_forest.pkl`) and optimized weights (`model_metadata.json`).
  - `compute_anomaly_score()`: Normalizes IF decision function to $[0, 1]$.
  - `evaluate_business_rules()`: Checks 5 deterministic rules (velocity spike, high value new beneficiary, nocturnal transfer, balance drain, extreme amount ratio).
  - `calculate_risk()`: Produces composite score, risk tier (LOW/MEDIUM/HIGH/CRITICAL), action (ALLOW/REVIEW/BLOCK), action badge string, and score contribution details.

#### **7. [`src/synthetic_upi.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/synthetic_upi.py)**
- **Why**: Generates realistic Indian UPI transaction datasets with native UPI attributes for risk experimentation.
- **How**:
  - Uses 3 behavioral customer personas (`normal`, `suspicious`, `mule`).
  - Generates Indian UPI attributes (VPA handles `@okaxis`, `@ybl`, UPI apps GPay, PhonePe, Paytm, merchant categories, device IDs, Indian cities).
  - Scales persona allocation dynamically based on target `fraud_rate` parameter.
  - Exports output to `data/processed/synthetic_upi_transactions.csv`.

#### **8. [`src/graph_fraud.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/src/graph_fraud.py)**
- **Why**: Analyzes transaction network topology to detect money mule rings and circular payment loops.
- **How**:
  - `FraudGraphAnalyzer` builds `nx.MultiDiGraph` from transaction DataFrame.
  - Provides `@property G` alias for NetworkX graph access.
  - `compute_network_metrics()`: Calculates PageRank, in/out degree, and mule risk scores.
  - `detect_suspicious_subgraphs()`: Finds circular payment cycles ($A \rightarrow B \rightarrow C \rightarrow A$) and high-density clusters.

---

### Production API (`api/`)

#### **9. [`api/main.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/api/main.py)**
- **Why**: Serves RESTful inference endpoints for microservice integration.
- **How**:
  - Instantiates FastAPI app with lifespan context manager loading `FraudPredictor`, `RiskEngine`, and `DatabaseManager`.
  - Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /risk-score`, `GET /metrics`.

#### **10. [`api/schemas.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/api/schemas.py)**
- **Why**: Defines strict Pydantic v2 data models for API input payloads and responses.
- **How**:
  - `TransactionRequest`: Validates transaction amount $>0$, payment rail type enum, non-empty account strings, step integer, balances.
  - `PredictionResponse`, `RiskScoreResponse`, `ModelInfoResponse`, `HealthResponse`: Typed output structures.

---

### Dashboard & UI (`app/`)

#### **11. [`app/streamlit_app.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/app/streamlit_app.py)**
- **Why**: Interactive analyst dashboard for risk inspection, scenario simulation, and case investigation.
- **How**:
  - Uses `@st.cache_resource` to cache predictor, risk engine, and database manager.
  - 5 Navigation Tabs: Executive Overview, Transaction Risk Checker, Investigation Queue, Model Analytics, Synthetic UPI & Graph Analytics.
  - Loads metrics dynamically from `model_metadata.json`, `threshold_cost_table.csv`, and `dataset_summary.json` (zero hardcoded values).

#### **12. [`app/style.css`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/app/style.css)**
- **Why**: Custom dark glassmorphism visual styling for Streamlit UI.
- **How**: Defines CSS metric cards, status badges (`badge-low`, `badge-critical`), custom fonts, and dark theme colors.

---

### SQL Warehouse & Analyst Queue (`sql/`)

#### **13. [`sql/schema.sql`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/sql/schema.sql)**
- **Why**: Database DDL schema defining warehouse tables.
- **How**: Defines DDL for `transactions`, `fraud_alerts`, `investigation_queue`, and `rbi_macro_stats` with primary keys, foreign keys, and indexes.

#### **14. [`sql/db_manager.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/sql/db_manager.py)**
- **Why**: Manages SQLite connection (`data/processed/fraud_intelligence.db`) for logging evaluations and analyst decisions.
- **How**:
  - `log_evaluation()`: Inserts raw transaction, alert record, and creates investigation case for HIGH/CRITICAL risk tiers.
  - `get_investigation_queue()`: Reads pending/processed analyst cases.
  - `update_case_decision()`: Updates case status (CONFIRMED_FRAUD, FALSE_POSITIVE, etc.) and analyst notes.

#### **15. [`sql/fraud_analysis.sql`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/sql/fraud_analysis.sql)**
- **Why**: Analytical queries for offline warehouse reporting.
- **How**: SQL queries for executive fraud rates by rail, hourly risk concentration, top high-risk mule beneficiary accounts, and velocity CTEs.

---

### MLOps & Monitoring (`monitoring/`)

#### **16. [`monitoring/drift_detector.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/monitoring/drift_detector.py)**
- **Why**: Detects distribution drift between baseline training data and production inference data.
- **How**:
  - `calculate_psi()`: Computes Population Stability Index across 10 quantile bins.
  - `DriftDetector`: Computes PSI and KS-test statistics per feature column, classifying features as `STABLE`, `MODERATE_DRIFT`, or `SIGNIFICANT_DRIFT`.

#### **17. [`monitoring/business_kpis.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/monitoring/business_kpis.py)**
- **Why**: Tracks production financial KPIs and investigation efficiency.
- **How**:
  - `BusinessKPITracker`: Calculates fraud capture rate, review rate, blocked fraud value (INR), missed fraud loss (INR), false positive investigation cost, and cost per detected fraud.

---

### Test Suite (`tests/`)

#### **18. [`tests/test_features.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_features.py)** (10 tests)
- Tests feature shapes, null checks, causal customer means, new beneficiary flags, night time flags, integer velocity counts, feature schema structure, and **temporal leakage prevention**.

#### **19. [`tests/test_risk_engine.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_risk_engine.py)** (11 tests)
- Tests risk tier classification (CRITICAL $\rightarrow$ BLOCK, LOW $\rightarrow$ ALLOW), score bounds $[0, 100]$, action wording, weight validation (weights must sum to 1.0), engine config output, and anomaly model active status.

#### **20. [`tests/test_prediction.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_prediction.py)** (8 tests)
- Tests predictor loading, probability array bounds $[0, 1]$, threshold validity, `FileNotFoundError` handling when model is missing, single prediction output structure, and version tracking.

#### **21. [`tests/test_causal_state.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_causal_state.py)** (2 tests)
- Tests `CustomerStateStore` real-time sliding window calculations and `DatabaseManager` SQLite logging.

#### **22. [`tests/test_graph_fraud.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_graph_fraud.py)** (5 tests)
- Tests MultiDiGraph building, parallel edge preservation, network metrics calculation, circular cycle detection, and fan-in/fan-out degree checks.

#### **23. [`tests/test_monitoring.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20%26%20Risk%20Intelligence%20System/tests/test_monitoring.py)** (4 tests)
- Tests PSI calculation for identical vs. shifted distributions, `DriftDetector` feature evaluation, and `BusinessKPITracker` financial metric output.

---

### Infrastructure & Operations

#### **24. [`Dockerfile`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/Dockerfile)**
- Multi-stage Docker build using `python:3.11-slim` for serving the FastAPI REST microservice with embedded HTTP health checks.

#### **25. [`Dockerfile.streamlit`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/Dockerfile.streamlit)**
- Dedicated Docker container definition for serving the Streamlit analyst dashboard on port 8501.

#### **26. [`docker-compose.yml`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/docker-compose.yml)**
- Multi-container compose file coordinating `api` (port 8000) and `dashboard` (port 8501) with shared read-only volume mounts (`models/`, `reports/`) and health check dependencies.

#### **27. [`.github/workflows/ci.yml`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/.github/workflows/ci.yml)**
- Automated GitHub Actions workflow triggering on push/PR to `main`. Executes `flake8`, `black --check`, 40 `pytest` unit tests, and Docker container build verification.

#### **28. [`tools/create_merged_notebook.py`](file:///Users/the_irritater/Downloads/Digital%20Payment%20Fraud%20&%20Risk%20Intelligence%20System/tools/create_merged_notebook.py)**
- Helper script located in `tools/` that automatically compiles source modules into a single, executable Jupyter Notebook at `notebooks/Digital_Payment_Fraud_Intelligence_Complete.ipynb`.

---

## 4. End-to-End Data & Decision Flow

```text
1. Transaction Event Received
   (via Streamlit Form UI or REST API POST /risk-score)
                           │
                           ▼
2. Stateful Feature Construction (CustomerStateStore)
   - Computes rolling 1h/6h/24h transaction counts via searchsorted
   - Computes ratio of current amount to customer's expanding prior mean
   - Checks if beneficiary account is new for this customer
                           │
                           ▼
3. Machine Learning Inference (FraudPredictor)
   - Aligns feature columns with model expectations
   - Passes vector to CalibratedClassifierCV (Platt-scaled XGBoost)
   - Obtains calibrated fraud probability (P_ML)
                           │
                           ▼
4. Unsupervised Anomaly Scoring (RiskEngine)
   - Passes feature vector to fitted Isolation Forest
   - Inverts decision function to normalized score in [0, 1]
                           │
                           ▼
5. Deterministic Business Rules Evaluation (RiskEngine)
   - Evaluates 5 business rules (velocity, new beneficiary high-val, nocturnal transfer, balance drain, amount ratio)
   - Generates rule score (0 to 100) and list of triggered rule alerts
                           │
                           ▼
6. Risk Score & Decision Directive Synthesis
   - Risk Score = (P_ML * 100 * 0.80) + (Score_Anomaly * 100 * 0.10) + (Score_Rules * 0.10)
   - Maps score to Risk Tier (LOW / MEDIUM / HIGH / CRITICAL)
   - Assigns Action Directive (ALLOW / REVIEW / BLOCK)
                           │
                           ▼
7. SQLite Warehouse Logging & Analyst Queue Routing
   - Inserts transaction payload & alert evaluation into SQLite DB
   - Automatically routes HIGH and CRITICAL alerts to analyst investigation queue
```

---

## 5. Summary & Portfolio Presentation Guide

When explaining this project to interviewers or senior engineers, highlight these key design achievements:

1. **"We built a hybrid decision system, not just a binary classifier."** Explain how ML probabilities, Isolation Forest anomaly scores, and deterministic business rules synthesize into a unified 0–100 risk score with action directives (`ALLOW`, `REVIEW`, `BLOCK`).
2. **"We strictly eliminated temporal data leakage."** Explain how feature construction uses backward-looking expanding windows and how threshold optimization was locked strictly on the validation set before a single test set evaluation.
3. **"We optimized for financial business impact."** Explain how operating thresholds were selected by minimizing actual monetary loss (missed fraud amount + analyst review costs) rather than arbitrary 0.50 cutoffs.
4. **"We built for production deployment."** Point to the FastAPI REST API, Docker Compose setup, 40 passing unit tests, GitHub Actions CI pipeline, and Population Stability Index (PSI) drift monitoring layer.
