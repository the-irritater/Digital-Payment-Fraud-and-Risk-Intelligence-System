"""
Graph-Based Fraud Risk Analysis Module for Digital Payment Fraud Intelligence System.
Uses NetworkX to build transaction flow graphs, compute network centrality metrics,
and identify structural patterns (high in-degree hubs, circular chains) that may be
associated with suspicious behavior such as money mule networks.

IMPORTANT: Graph metrics (PageRank, degree centrality, etc.) identify network
structures that correlate with known fraud topologies. They do NOT independently
prove fraud. These features should be used as supplementary risk signals alongside
supervised ML and business rules.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class FraudGraphAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph_from_dataframe(
        self, 
        df: pd.DataFrame, 
        orig_col: str = 'nameOrig', 
        dest_col: str = 'nameDest', 
        amount_col: str = 'amount',
        fraud_col: str = 'isFraud'
    ):
        """Construct directed multi-attribute transaction graph from DataFrame."""
        print(f"[FraudGraphAnalyzer] Constructing transaction network graph from {len(df):,} transactions...")
        self.graph.clear()
        
        for idx, row in df.iterrows():
            u = str(row[orig_col])
            v = str(row[dest_col])
            amt = float(row[amount_col])
            is_fraud = int(row.get(fraud_col, 0))
            
            # Add nodes
            if not self.graph.has_node(u):
                self.graph.add_node(u, node_type='Customer', total_sent=0.0, total_received=0.0, tx_count=0)
            if not self.graph.has_node(v):
                self.graph.add_node(v, node_type='Beneficiary', total_sent=0.0, total_received=0.0, tx_count=0)
                
            # Update node statistics
            self.graph.nodes[u]['total_sent'] += amt
            self.graph.nodes[u]['tx_count'] += 1
            self.graph.nodes[v]['total_received'] += amt
            self.graph.nodes[v]['tx_count'] += 1
            
            # Add directed edge
            self.graph.add_edge(u, v, amount=amt, is_fraud=is_fraud)
            
        print(f"[FraudGraphAnalyzer] Graph built: {self.graph.number_of_nodes():,} nodes | {self.graph.number_of_edges():,} directed edges.")

    def compute_network_metrics(self) -> pd.DataFrame:
        """Calculate network centrality metrics to identify structurally suspicious nodes."""
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=200)
        except Exception:
            pagerank = {n: 0.0 for n in self.graph.nodes()}
            
        metrics = []
        for node in self.graph.nodes():
            in_d = in_degree.get(node, 0)
            out_d = out_degree.get(node, 0)
            pr = pagerank.get(node, 0.0)
            sent = self.graph.nodes[node].get('total_sent', 0.0)
            recv = self.graph.nodes[node].get('total_received', 0.0)
            
            # Mule risk heuristic: high in-degree + low out-degree suggests collection hub
            # This is a structural indicator, not proof of fraudulent behavior
            mule_ratio = (in_d + 1.0) / (out_d + 1.0)
            mule_risk_score = min((mule_ratio * 10.0) + (pr * 1000.0), 100.0)
            
            metrics.append({
                'account_id': node,
                'node_type': self.graph.nodes[node].get('node_type', 'Unknown'),
                'in_degree': in_d,
                'out_degree': out_d,
                'pagerank': round(pr, 6),
                'total_sent_inr': round(sent, 2),
                'total_received_inr': round(recv, 2),
                'mule_risk_score': round(mule_risk_score, 1)
            })
            
        df_metrics = pd.DataFrame(metrics).sort_values('mule_risk_score', ascending=False).reset_index(drop=True)
        return df_metrics

    def detect_suspicious_subgraphs(self) -> List[Dict[str, Any]]:
        """Identify connected communities and potential circular payment chains."""
        undirected_g = self.graph.to_undirected()
        components = list(nx.connected_components(undirected_g))
        
        suspicious_clusters = []
        cluster_id = 1
        
        for comp in components:
            if len(comp) >= 3:
                subg = self.graph.subgraph(comp)
                fraud_edges = sum(1 for u, v, d in subg.edges(data=True) if d.get('is_fraud', 0) == 1)
                total_edges = subg.number_of_edges()
                total_volume = sum(d.get('amount', 0.0) for u, v, d in subg.edges(data=True))
                
                # Check for simple cycles (circular money laundering chains)
                try:
                    cycles = list(nx.simple_cycles(subg))
                except Exception:
                    cycles = []
                    
                if fraud_edges > 0 or len(cycles) > 0 or total_edges >= 5:
                    suspicious_clusters.append({
                        'cluster_id': f"RING_{cluster_id:03d}",
                        'node_count': len(comp),
                        'edge_count': total_edges,
                        'fraud_transaction_count': fraud_edges,
                        'total_volume_inr': round(total_volume, 2),
                        'circular_chain_count': len(cycles),
                        'sample_nodes': list(comp)[:5]
                    })
                    cluster_id += 1
                    
        return suspicious_clusters

if __name__ == "__main__":
    from synthetic_upi import generate_synthetic_upi_dataset
    df_upi = generate_synthetic_upi_dataset(num_records=5000)
    
    analyzer = FraudGraphAnalyzer()
    analyzer.build_graph_from_dataframe(df_upi, orig_col='customer_id', dest_col='beneficiary_id', amount_col='amount_inr', fraud_col='is_fraud')
    
    df_metrics = analyzer.compute_network_metrics()
    print("\nTop 5 Potential Money Mule Accounts:")
    print(df_metrics.head())
    
    clusters = analyzer.detect_suspicious_subgraphs()
    print(f"\nDiscovered {len(clusters)} Suspicious Fraud Rings/Clusters:")
    for cl in clusters[:3]:
        print(f"  {cl['cluster_id']}: {cl['node_count']} accounts | Volume: ₹{cl['total_volume_inr']:,} | Fraud txs: {cl['fraud_transaction_count']}")
