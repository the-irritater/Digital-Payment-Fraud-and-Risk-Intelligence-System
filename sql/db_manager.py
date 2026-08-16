"""
SQL Warehouse Manager for Digital Payment Fraud Intelligence System.
Connects Streamlit and prediction pipeline to an integrated SQLite database
(data/processed/fraud_intelligence.db) executing DDL schema and analytical queries.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional

DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "fraud_intelligence.db"
    )
)


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Execute DDL schema to set up database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                nameOrig TEXT NOT NULL,
                oldbalanceOrg REAL,
                newbalanceOrig REAL,
                nameDest TEXT NOT NULL,
                oldbalanceDest REAL,
                newbalanceDest REAL,
                isFraud INTEGER DEFAULT 0,
                isFlaggedFraud INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_alerts (
                alert_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                nameOrig TEXT NOT NULL,
                nameDest TEXT NOT NULL,
                amount REAL NOT NULL,
                ml_probability REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                rule_score REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS investigation_queue (
                case_id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                nameOrig TEXT,
                nameDest TEXT,
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                investigation_status TEXT DEFAULT 'PENDING',
                assigned_analyst TEXT DEFAULT 'UNASSIGNED',
                decision_notes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

            # Seed initial sample queue if empty
            cursor.execute("SELECT COUNT(*) FROM investigation_queue")
            if cursor.fetchone()[0] == 0:
                self._seed_sample_cases(cursor)
                conn.commit()

    def _seed_sample_cases(self, cursor):
        """Seed initial investigation cases into database."""
        initial_cases = [
            (
                "CASE_92810",
                "ALT_92810",
                "TX92831",
                "C928310482",
                "C102938",
                84500.0,
                96.4,
                "CRITICAL",
                "BLOCK",
                "PENDING",
                "UNASSIGNED",
                "High nocturnal transfer to unseen beneficiary",
            ),
            (
                "CASE_92811",
                "ALT_92811",
                "TX92832",
                "C392019481",
                "C938201",
                52100.0,
                88.2,
                "CRITICAL",
                "BLOCK",
                "PENDING",
                "UNASSIGNED",
                "Velocity spike with complete balance liquidation",
            ),
            (
                "CASE_92812",
                "ALT_92812",
                "TX92833",
                "C482019482",
                "C849201",
                23500.0,
                68.5,
                "HIGH",
                "REVIEW",
                "UNDER_REVIEW",
                "Analyst_042",
                "Step-up authentication requested",
            ),
            (
                "CASE_92813",
                "ALT_92813",
                "TX92834",
                "C192039482",
                "C192039",
                73200.0,
                91.0,
                "CRITICAL",
                "BLOCK",
                "CONFIRMED_FRAUD",
                "Analyst_018",
                "Confirmed phishing fraud",
            ),
            (
                "CASE_92814",
                "ALT_92814",
                "TX92835",
                "C281920482",
                "M281920",
                1850.0,
                24.1,
                "LOW",
                "ALLOW",
                "FALSE_POSITIVE",
                "Analyst_042",
                "Verified normal merchant payment",
            ),
        ]
        cursor.executemany(
            """
        INSERT INTO investigation_queue
        (case_id, alert_id, transaction_id, nameOrig, nameDest, amount, risk_score, risk_tier, recommended_action, investigation_status, assigned_analyst, decision_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            initial_cases,
        )

    def log_evaluation(
        self, tx_dict: Dict[str, Any], risk_result: Dict[str, Any]
    ) -> str:
        """Log raw transaction, risk engine alert, and create investigation case in SQLite."""
        import uuid

        tx_id = f"TX_{uuid.uuid4().hex[:8].upper()}"
        alert_id = f"ALT_{uuid.uuid4().hex[:8].upper()}"
        case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Insert Transaction
            cursor.execute(
                """
            INSERT INTO transactions (transaction_id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tx_id,
                    int(tx_dict.get("step", 1)),
                    str(tx_dict.get("type", "TRANSFER")),
                    float(tx_dict.get("amount", 0.0)),
                    str(tx_dict.get("nameOrig", "")),
                    float(tx_dict.get("oldbalanceOrg", 0.0)),
                    float(tx_dict.get("newbalanceOrig", 0.0)),
                    str(tx_dict.get("nameDest", "")),
                    float(tx_dict.get("oldbalanceDest", 0.0)),
                    float(tx_dict.get("newbalanceDest", 0.0)),
                ),
            )
            # 2. Insert Alert
            cursor.execute(
                """
            INSERT INTO fraud_alerts (alert_id, transaction_id, step, nameOrig, nameDest, amount, ml_probability, anomaly_score, rule_score, risk_score, risk_tier, recommended_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert_id,
                    tx_id,
                    int(tx_dict.get("step", 1)),
                    str(tx_dict.get("nameOrig", "")),
                    str(tx_dict.get("nameDest", "")),
                    float(tx_dict.get("amount", 0.0)),
                    float(risk_result["components"]["ml_probability"]),
                    float(risk_result["components"]["normalized_anomaly_score"]),
                    float(risk_result["components"]["rule_score"]),
                    float(risk_result["risk_score"]),
                    str(risk_result["risk_tier"]),
                    str(risk_result["action"]),
                ),
            )
            # 3. Create Investigation Queue Case if HIGH or CRITICAL
            if risk_result["risk_tier"] in ["HIGH", "CRITICAL"]:
                cursor.execute(
                    """
                INSERT INTO investigation_queue (case_id, alert_id, transaction_id, nameOrig, nameDest, amount, risk_score, risk_tier, recommended_action, investigation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                    (
                        case_id,
                        alert_id,
                        tx_id,
                        str(tx_dict.get("nameOrig", "")),
                        str(tx_dict.get("nameDest", "")),
                        float(tx_dict.get("amount", 0.0)),
                        float(risk_result["risk_score"]),
                        str(risk_result["risk_tier"]),
                        str(risk_result["action"]),
                    ),
                )
            conn.commit()

        return case_id

    def get_investigation_queue(self) -> pd.DataFrame:
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                """
            SELECT case_id, transaction_id, nameOrig AS Originator, nameDest AS Beneficiary, amount AS Amount_INR, risk_score AS Risk_Score, risk_tier AS Tier, recommended_action AS Recommended, investigation_status AS Status, assigned_analyst AS Analyst, decision_notes AS Notes
            FROM investigation_queue
            ORDER BY case_id DESC
            """,
                conn,
            )
        return df

    def update_case_decision(
        self, case_id: str, status: str, analyst_id: str, notes: str
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            UPDATE investigation_queue
            SET investigation_status = ?, assigned_analyst = ?, decision_notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE case_id = ?
            """,
                (status, analyst_id, notes, case_id),
            )
            conn.commit()


_db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    return _db_manager
