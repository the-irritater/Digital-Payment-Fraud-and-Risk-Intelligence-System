"""
Synthetic UPI Transaction Generator for Digital Payment Fraud Intelligence System.
Generates realistic Indian UPI digital payment transactions with native Indian payment attributes,
merchant categories, VPA handles, device metadata, and synthetic fraud patterns.

Transactions are generated from three behavioral personas:
    1. Normal User — Low velocity, stable amounts, rare device/location changes.
    2. Suspicious User — High velocity, large amounts, frequent beneficiary churn.
    3. Mule Account — Many incoming transfers, few outgoing, acts as a drop account.

These personas provide a generative rationale for the synthetic data rather than
purely random attribute assignment.

NOTE: This dataset is explicitly synthetic. It does not represent real Indian
banking or UPI customer transactions.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

UPI_APPS = ['GPay', 'PhonePe', 'Paytm', 'BHIM', 'Cred', 'iMobile']
VPA_DOMAINS = ['@okaxis', '@ybl', '@paytm', '@upi', '@icici', '@sbi']
CITIES = ['Mumbai', 'Bengaluru', 'Delhi_NCR', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata', 'Ahmedabad']
MERCHANT_CATS = ['Electronics', 'Peer_to_Peer', 'Gaming_Betting', 'Jewelry', 'Grocery', 'Travel', 'Utilities']
DEVICE_TYPES = ['Android', 'iOS']

# Behavioral Persona Definitions
PERSONAS = {
    "normal": {
        "description": "Normal user with stable transaction patterns",
        "amount_mean": 1200.0,
        "amount_std": 800.0,
        "tx_per_day_mean": 4,
        "tx_per_day_std": 2,
        "night_tx_prob": 0.05,
        "new_beneficiary_prob": 0.12,
        "device_change_prob": 0.03,
        "location_change_prob": 0.05,
        "merchant_cats": ['Grocery', 'Utilities', 'Travel', 'Peer_to_Peer'],
        "fraud_prob": 0.0
    },
    "suspicious": {
        "description": "Suspicious user exhibiting high-risk behavioral signals",
        "amount_mean": 28000.0,
        "amount_std": 18000.0,
        "tx_per_day_mean": 18,
        "tx_per_day_std": 8,
        "night_tx_prob": 0.55,
        "new_beneficiary_prob": 0.80,
        "device_change_prob": 0.65,
        "location_change_prob": 0.60,
        "merchant_cats": ['Gaming_Betting', 'Jewelry', 'Electronics', 'Peer_to_Peer'],
        "fraud_prob": 0.85
    },
    "mule": {
        "description": "Mule account — many incoming transfers, few outgoing, acts as collection hub",
        "amount_mean": 35000.0,
        "amount_std": 20000.0,
        "tx_per_day_mean": 25,
        "tx_per_day_std": 10,
        "night_tx_prob": 0.45,
        "new_beneficiary_prob": 0.90,
        "device_change_prob": 0.70,
        "location_change_prob": 0.50,
        "merchant_cats": ['Peer_to_Peer', 'Electronics', 'Jewelry'],
        "fraud_prob": 0.90
    }
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "synthetic_upi_transactions.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def generate_synthetic_upi_dataset(num_records: int = 25000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Indian UPI transaction dataset using behavioral personas.
    
    Persona assignment probability:
        - 97% of customers are "normal" users.
        - 2% are "suspicious" users (account takeover, social engineering victims).
        - 1% are "mule" accounts (drop accounts in layered transfer chains).
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Generating {num_records:,} synthetic Indian UPI transactions (Fraud rate target: ~{fraud_rate*100}%)...")

    start_date = datetime(2026, 8, 1, 0, 0, 0)
    
    # Pre-assign personas to a pool of customers
    num_customers = max(num_records // 6, 500)
    customer_ids = [f"CUST_{1000 + i}" for i in range(num_customers)]
    customer_personas = {}
    for cid in customer_ids:
        roll = random.random()
        if roll < 0.01:
            customer_personas[cid] = "mule"
        elif roll < 0.03:
            customer_personas[cid] = "suspicious"
        else:
            customer_personas[cid] = "normal"
    
    records = []
    for i in range(num_records):
        tx_id = f"TX_UPI_{100000 + i}"
        sec_offset = random.randint(0, 10 * 24 * 3600)
        dt = start_date + timedelta(seconds=sec_offset)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        hour = dt.hour

        cust_id = random.choice(customer_ids)
        persona = PERSONAS[customer_personas[cust_id]]
        
        upi_app = random.choice(UPI_APPS)
        vpa_domain = random.choice(VPA_DOMAINS)
        city = random.choice(CITIES)
        device_type = random.choice(DEVICE_TYPES)
        device_id = f"DEV_{random.randint(1000, 3000)}"
        beneficiary_id = f"BEN_{random.randint(1000, 7000)}"

        # Generate amount from persona distribution
        amount = float(np.round(max(np.random.normal(persona['amount_mean'], persona['amount_std']), 10.0), 2))
        amount = float(min(amount, 100000.0))  # UPI daily limit ₹1 Lakh

        # Determine fraud based on persona probability
        is_fraud = 1 if (random.random() < persona['fraud_prob']) else 0
        
        # Apply persona behavioral signals
        is_night = hour in [0, 1, 2, 3, 4, 5]
        is_new_beneficiary = 1 if random.random() < persona['new_beneficiary_prob'] else 0
        device_change = 1 if random.random() < persona['device_change_prob'] else 0
        location_change = 1 if random.random() < persona['location_change_prob'] else 0
        merchant_cat = random.choice(persona['merchant_cats'])
        
        # Transaction velocity from persona's daily rate
        daily_rate = max(int(np.random.normal(persona['tx_per_day_mean'], persona['tx_per_day_std'])), 1)
        tx_last_1h = max(int(daily_rate / 24 * random.uniform(0.5, 2.0)), 1)
        tx_last_24h = daily_rate
        
        # Historical average from persona's amount distribution (with noise)
        avg_amount_30d = float(np.round(np.random.normal(persona['amount_mean'] * 0.9, persona['amount_std'] * 0.3), 2))
        avg_amount_30d = max(avg_amount_30d, 50.0)

        records.append({
            'transaction_id': tx_id,
            'timestamp': timestamp_str,
            'hour': hour,
            'customer_id': cust_id,
            'customer_persona': customer_personas[cust_id],
            'beneficiary_id': beneficiary_id,
            'amount_inr': amount,
            'upi_app': upi_app,
            'vpa_domain': vpa_domain,
            'merchant_category': merchant_cat,
            'city': city,
            'device_id': device_id,
            'device_type': device_type,
            'is_new_beneficiary': is_new_beneficiary,
            'transactions_last_1h': tx_last_1h,
            'transactions_last_24h': tx_last_24h,
            'avg_amount_30d': avg_amount_30d,
            'amount_ratio_vs_30d': round(amount / max(avg_amount_30d, 1.0), 2),
            'device_change': device_change,
            'location_change': location_change,
            'is_fraud': is_fraud
        })

    df_upi = pd.DataFrame(records)
    df_upi.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved synthetic UPI dataset to {OUTPUT_PATH}")
    print(f"Shape: {df_upi.shape} | Fraud cases: {df_upi['is_fraud'].sum():,} ({df_upi['is_fraud'].mean()*100:.2f}%)")
    
    # Print persona distribution summary
    persona_counts = df_upi['customer_persona'].value_counts()
    print("\nCustomer Persona Distribution:")
    for p, count in persona_counts.items():
        fraud_count = df_upi[df_upi['customer_persona'] == p]['is_fraud'].sum()
        print(f"  {p:>12s}: {count:,} txs | Fraud: {fraud_count:,}")
    
    return df_upi

if __name__ == "__main__":
    generate_synthetic_upi_dataset()
