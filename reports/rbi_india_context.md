# India Digital Payment Fraud and Ecosystem Context

| Metadata | Details |
| :--- | :--- |
| **Author** | Sanman Kadam |
| **Topic** | Macroeconomic RBI Payment Intelligence & Fraud Vectors |
| **Scope** | India Digital Payments (UPI, IMPS, Cards, NEFT, RTGS) |

## Executive Summary

India's digital payment ecosystem has experienced rapid growth, led primarily by the Unified Payments Interface (UPI), digital wallets, and immediate payment rails (IMPS, NEFT, RTGS). According to official Reserve Bank of India (RBI) reports and payment system indicators, while the overall fraud-to-transaction ratio remains small by volume, absolute reported fraud incidents and values have scaled, driven by social engineering, phishing, fake merchant handles, and rapid multi-tier fund transfers.

This document bridges macroeconomic RBI fraud statistics with micro-level transaction risk scoring.

## 1. RBI Payment System Indicators and UPI Scale

### Annual UPI Transaction Volume and Value Growth

- **UPI Daily Volumes**: Hundreds of millions of transactions processed daily across NPCI payment switches.
- **UPI Monthly Value**: Trillions of rupees processed monthly.
- **Dominant Payment Rails**: UPI accounts for over 80% of total retail digital payment volumes in India.

### Payment Rail Comparison

| Payment Rail | Primary Use Case | Settlement Type | Risk Profile | Primary Fraud Vectors |
| :--- | :--- | :--- | :--- | :--- |
| **UPI (P2P / P2M)** | Micro and Retail Transfers | Real-Time Immediate | High Velocity / High Frequency | Phishing, Fraudulent Collect Requests, VPA Spoofing |
| **IMPS** | Instant Interbank Account Transfer | Real-Time Immediate | High Value Instant | Credential stuffing, SIM swapping |
| **Cards (Debit/Credit)** | E-Commerce and POS | Batch / Gateway | Moderate Velocity | CNP (Card-Not-Present) fraud, BIN attack |
| **NEFT / RTGS** | Wholesale and Bulk Transfer | Batch / Continuous | High Value | Business Email Compromise, Mule accounts |

## 2. RBI Fraud Registry and Reporting Nuances

### Reported Frauds vs. Fraud Attempts

The RBI Fraud Monitoring Cell collects reported domestic payment fraud data from commercial banks, small finance banks, and payment system operators.

> **Note on RBI Statistics**: RBI statistics capture actual reported financial losses after bank investigation. They do not capture real-time blocked fraud attempts or un-reported micro-frauds below reporting thresholds. Therefore, real-time risk engines must operate with strict detection sensitivities.

### Primary Fraud Categories in Digital Payments

```text
                             Digital Payment Threats
                                        │
      ┌─────────────────────────────────┼─────────────────────────────────┐
      ▼                                 ▼                                 ▼
 Social Engineering              Mule Account Rings               Technical Exploits
 (Phishing/VPA Spoofing)       (Rapid Layered Transfers)        (App Clones/Screen Sharing)
```

1. **Social Engineering and Phishing**: Impersonation of customer service personnel, fake QR codes requiring PIN entry to receive money.
2. **Money Mule Networks**: Accounts opened using compromised KYC documents, used as intermediate drop accounts for instant cash-outs.
3. **Velocity Spikes and Off-Hour Transfers**: Midnight transaction surges, new beneficiary additions followed immediately by high-value drain transfers.

## 3. Integrating Macro Context with Micro Model Risk Scoring

Our platform connects macroeconomic RBI risk intelligence with real-time transaction machine learning:

1. **Macro Risk Weighting**: Dynamic adjustment of decision thresholds during holiday seasons, major e-commerce sales, or high-fraud risk time windows (e.g. 01:00 AM to 05:00 AM).
2. **Behavioral Baseline Engine**: Benchmarking user transactions against national average velocity, typical transaction size distributions, and merchant category codes.
3. **Mule Detection via Network Analysis**: Identifying rapid multi-hop transfers mimicking mule network topologies reported in RBI intelligence briefings.
