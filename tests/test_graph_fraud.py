"""
Unit tests for Graph Fraud Analytics.
Validates MultiDiGraph construction, network metrics, and cycle detection.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_fraud import FraudGraphAnalyzer


def _make_sample_graph_df():
    """Create a sample transaction DataFrame for graph construction."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3", "C1", "C4", "C5"],
            "beneficiary_id": ["C2", "C3", "C3", "C4", "C5", "C1", "C2"],
            "amount_inr": [5000, 10000, 3000, 15000, 2000, 8000, 1000],
            "is_fraud": [0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_graph_builds_successfully():
    """Graph should be built without errors from sample data."""
    analyzer = FraudGraphAnalyzer()
    df = _make_sample_graph_df()
    analyzer.build_graph_from_dataframe(
        df,
        orig_col="customer_id",
        dest_col="beneficiary_id",
        amount_col="amount_inr",
        fraud_col="is_fraud",
    )
    assert analyzer.G is not None
    assert analyzer.G.number_of_nodes() > 0
    assert analyzer.G.number_of_edges() > 0


def test_multidigraph_allows_parallel_edges():
    """MultiDiGraph should preserve parallel edges between the same nodes."""
    analyzer = FraudGraphAnalyzer()
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "beneficiary_id": ["C2", "C2"],
            "amount_inr": [1000, 2000],
            "is_fraud": [0, 1],
        }
    )
    analyzer.build_graph_from_dataframe(
        df,
        orig_col="customer_id",
        dest_col="beneficiary_id",
        amount_col="amount_inr",
        fraud_col="is_fraud",
    )
    assert analyzer.G.number_of_edges() == 2, "MultiDiGraph should keep both edges"


def test_network_metrics_computed():
    """Network metrics DataFrame should contain centrality columns."""
    analyzer = FraudGraphAnalyzer()
    df = _make_sample_graph_df()
    analyzer.build_graph_from_dataframe(
        df,
        orig_col="customer_id",
        dest_col="beneficiary_id",
        amount_col="amount_inr",
        fraud_col="is_fraud",
    )
    metrics = analyzer.compute_network_metrics()
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) > 0
    assert "account_id" in metrics.columns or "node" in metrics.columns


def test_cycle_detection():
    """Graph with cycles should detect them."""
    analyzer = FraudGraphAnalyzer()
    # C1 → C2 → C3 → C1 forms a cycle
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "beneficiary_id": ["C2", "C3", "C1"],
            "amount_inr": [5000, 3000, 8000],
            "is_fraud": [1, 1, 1],
        }
    )
    analyzer.build_graph_from_dataframe(
        df,
        orig_col="customer_id",
        dest_col="beneficiary_id",
        amount_col="amount_inr",
        fraud_col="is_fraud",
    )

    import networkx as nx

    cycles = list(nx.simple_cycles(analyzer.G))
    assert len(cycles) > 0, "Should detect at least one cycle in circular graph"


def test_fan_in_fan_out():
    """Mule account receiving from many senders should have high in-degree."""
    analyzer = FraudGraphAnalyzer()
    # C1, C2, C3 all send to C_MULE
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "beneficiary_id": ["C_MULE", "C_MULE", "C_MULE"],
            "amount_inr": [5000, 3000, 8000],
            "is_fraud": [1, 0, 1],
        }
    )
    analyzer.build_graph_from_dataframe(
        df,
        orig_col="customer_id",
        dest_col="beneficiary_id",
        amount_col="amount_inr",
        fraud_col="is_fraud",
    )

    assert analyzer.G.in_degree("C_MULE") == 3, "Mule should have in-degree 3"
    assert (
        analyzer.G.out_degree("C_MULE") == 0
    ), "Mule should have out-degree 0 (cash-out node)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
