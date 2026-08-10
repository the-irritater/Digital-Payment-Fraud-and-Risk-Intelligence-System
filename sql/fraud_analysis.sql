-- Analytical Queries for Digital Payment Fraud & Risk Intelligence System

-- 1. Executive Summary KPIs: Total Volume, Fraud Rate, Financial Loss by Transaction Type
SELECT 
    type AS transaction_type,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_volume_inr,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss_inr,
    ROUND(AVG(CASE WHEN isFraud = 1 THEN amount ELSE NULL END), 2) AS avg_fraud_amount_inr
FROM transactions
GROUP BY type
ORDER BY fraud_count DESC;

-- 2. Hourly Risk Concentration (Simulated Hour step % 24)
SELECT 
    (step % 24) AS hour_of_day,
    COUNT(*) AS tx_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_tx_count,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS hourly_fraud_rate_pct
FROM transactions
GROUP BY (step % 24)
ORDER BY hour_of_day ASC;

-- 3. Top High-Risk Mule Beneficiary Accounts (Received Multiple High-Value Fraud Transfers)
SELECT 
    nameDest AS beneficiary_account,
    COUNT(*) AS inbound_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transfers_received,
    ROUND(SUM(amount), 2) AS total_amount_received_inr,
    ROUND(MAX(amount), 2) AS max_single_transfer_inr
FROM transactions
WHERE isFraud = 1
GROUP BY nameDest
HAVING COUNT(*) > 1
ORDER BY total_amount_received_inr DESC
LIMIT 20;

-- 4. High-Velocity Originator Accounts (More than 5 transactions in a short timeframe)
WITH VelocityCTE AS (
    SELECT 
        nameOrig,
        step,
        amount,
        isFraud,
        COUNT(*) OVER(PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS tx_velocity_2h
    FROM transactions
)
SELECT 
    nameOrig,
    MAX(tx_velocity_2h) AS max_2h_velocity,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txs,
    SUM(amount) AS total_spent
FROM VelocityCTE
WHERE tx_velocity_2h >= 4
GROUP BY nameOrig
ORDER BY max_2h_velocity DESC
LIMIT 15;

-- 5. Risk Tier Distribution from Fraud Alerts
SELECT 
    risk_tier,
    recommended_action,
    COUNT(*) AS alert_count,
    ROUND(AVG(risk_score), 1) AS avg_risk_score,
    ROUND(AVG(ml_probability), 4) AS avg_ml_prob,
    ROUND(SUM(amount), 2) AS total_exposure_inr
FROM fraud_alerts
GROUP BY risk_tier, recommended_action
ORDER BY avg_risk_score DESC;
