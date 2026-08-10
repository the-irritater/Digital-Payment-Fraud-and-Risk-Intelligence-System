-- Schema DDL for Digital Payment Fraud Intelligence Warehouse

-- 1. Raw & Ingested Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    step INT NOT NULL,
    type VARCHAR(32) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    nameOrig VARCHAR(64) NOT NULL,
    oldbalanceOrg DECIMAL(15, 2),
    newbalanceOrig DECIMAL(15, 2),
    nameDest VARCHAR(64) NOT NULL,
    oldbalanceDest DECIMAL(15, 2),
    newbalanceDest DECIMAL(15, 2),
    isFraud TINYINT DEFAULT 0,
    isFlaggedFraud TINYINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tx_step ON transactions(step);
CREATE INDEX IF NOT EXISTS idx_tx_orig ON transactions(nameOrig);
CREATE INDEX IF NOT EXISTS idx_tx_dest ON transactions(nameDest);
CREATE INDEX IF NOT EXISTS idx_tx_fraud ON transactions(isFraud);

-- 2. Fraud Alerts & Risk Engine Evaluation Output Table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    step INT NOT NULL,
    nameOrig VARCHAR(64) NOT NULL,
    nameDest VARCHAR(64) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    ml_probability DECIMAL(5, 4) NOT NULL,
    anomaly_score DECIMAL(5, 4) NOT NULL,
    rule_score DECIMAL(5, 2) NOT NULL,
    risk_score DECIMAL(5, 1) NOT NULL,
    risk_tier VARCHAR(16) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    recommended_action VARCHAR(16) NOT NULL, -- 'ALLOW', 'REVIEW', 'BLOCK'
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_alert_risk_tier ON fraud_alerts(risk_tier);
CREATE INDEX IF NOT EXISTS idx_alert_action ON fraud_alerts(recommended_action);

-- 3. Analyst Fraud Investigation Queue Table
CREATE TABLE IF NOT EXISTS investigation_queue (
    case_id VARCHAR(64) PRIMARY KEY,
    alert_id VARCHAR(64) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    risk_score DECIMAL(5, 1) NOT NULL,
    risk_tier VARCHAR(16) NOT NULL,
    investigation_status VARCHAR(32) DEFAULT 'PENDING', -- 'PENDING', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'UNDER_REVIEW'
    assigned_analyst VARCHAR(64) DEFAULT 'UNASSIGNED',
    decision_notes TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES fraud_alerts(alert_id)
);

-- 4. RBI Macroeconomic Fraud & Payment Statistics Table
CREATE TABLE IF NOT EXISTS rbi_macro_stats (
    stat_id INT PRIMARY KEY AUTO_INCREMENT,
    period_quarter VARCHAR(16) NOT NULL,
    upi_volume_million DECIMAL(12, 2) NOT NULL,
    upi_value_inr_crore DECIMAL(15, 2) NOT NULL,
    reported_digital_frauds INT NOT NULL,
    reported_fraud_value_crore DECIMAL(10, 2) NOT NULL,
    dominant_fraud_vector VARCHAR(64),
    threat_level VARCHAR(16)
);
