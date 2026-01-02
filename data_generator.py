import networkx as nx
import pandas as pd
import random
import numpy as np
from faker import Faker
import os

# --- CONFIGURATION ---
# This sets the path to the current folder, wherever the script is running
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Ensure the directory exists
os.makedirs(BASE_PATH, exist_ok=True)
print(f"📂 Working Directory set to: {BASE_PATH}")

# Initialize
fake = Faker()
np.random.seed(42) 

def generate_intelligence_network(num_citizens=1000, num_threats=20):
    print(f"Generating network with {num_citizens} citizens and {num_threats} threat actors...")
    
    # 1. Background (Scale-Free)
    G = nx.barabasi_albert_graph(n=num_citizens, m=2, seed=42)
    for node in G.nodes():
        G.nodes[node]['type'] = 'civilian'
        G.nodes[node]['risk_label'] = 0 
        G.nodes[node]['name'] = fake.name()
        G.nodes[node]['country'] = fake.country()

    # 2. Covert Cell (Star Topology)
    covert_start_id = num_citizens
    threat_nodes = range(covert_start_id, covert_start_id + num_threats)
    handler_id = threat_nodes[0]
    covert_edges = []
    
    for i in range(1, len(threat_nodes)):
        covert_edges.append((handler_id, threat_nodes[i]))
        if random.random() > 0.8:
            covert_edges.append((threat_nodes[i], threat_nodes[i-1]))

    G.add_nodes_from(threat_nodes)
    G.add_edges_from(covert_edges)

    for node in threat_nodes:
        G.nodes[node]['type'] = 'covert_operative'
        G.nodes[node]['risk_label'] = 1 
        G.nodes[node]['name'] = fake.name()
        G.nodes[node]['country'] = random.choice(['Country A', 'Country B', 'Country C'])

    # 3. Bridge
    civilian_contacts = np.random.choice(range(num_citizens), size=3, replace=False)
    for civilian in civilian_contacts:
        G.add_edge(handler_id, civilian)

    # 4. Weights
    for u, v in G.edges():
        if G.nodes[u]['risk_label'] == 1 and G.nodes[v]['risk_label'] == 1:
            G.edges[u, v]['weight'] = np.random.randint(50, 100)
            G.edges[u, v]['communication_type'] = 'encrypted'
        else:
            G.edges[u, v]['weight'] = np.random.randint(1, 20)
            G.edges[u, v]['communication_type'] = 'standard'

    return G

def export_data(G):
    # Prepare Paths
    nodes_path = os.path.join(BASE_PATH, 'nodes.csv')
    edges_path = os.path.join(BASE_PATH, 'edges.csv')

    # Export Nodes
    node_data = []
    for node, attrs in G.nodes(data=True):
        node_data.append({
            'id': node,
            'name': attrs['name'],
            'country': attrs['country'],
            'type': attrs['type'],
            'risk_label': attrs['risk_label']
        })
    pd.DataFrame(node_data).to_csv(nodes_path, index=False)
    
    # Export Edges
    edge_data = []
    for u, v, attrs in G.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'weight': attrs['weight'],
            'type': attrs['communication_type']
        })
    pd.DataFrame(edge_data).to_csv(edges_path, index=False)
    
    print(f"✅ Data generated successfully!")
    print(f"   Nodes: {nodes_path}")
    print(f"   Edges: {edges_path}")

if __name__ == "__main__":
    G = generate_intelligence_network()
    export_data(G)