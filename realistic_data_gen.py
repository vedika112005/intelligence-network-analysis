import pandas as pd
import networkx as nx
import random
from faker import Faker
import numpy as np
import os

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

fake = Faker()
np.random.seed(42)

def generate_realistic_intel_data(num_employees=1500, num_bad_actors=40):
    print(f"Generating 'Project Gotham' Dataset ({num_employees} entities)...")
    
    G = nx.Graph()
    
    # Departments
    departments = ['HR', 'IT', 'Shipping', 'Executive', 'Sales', 'Legal', 'R&D']
    
    # --- 1. Generate Employees (The Haystack) ---
    print("   - Creating civilian profiles...")
    for i in range(num_employees):
        dept = random.choice(departments)
        person = {
            'id': i,
            'name': fake.name(),
            'role': f"{dept} Specialist",
            'department': dept,
            'clearance': 'Low',
            'risk_label': 0,
            # Normal 'Civilian' Chatter
            'last_log': random.choice([
                f"Sent quarterly report to {dept} head.",
                "Requested vacation time for family trip.",
                "Submitted expense report for office supplies.",
                "Updated server firmware logs.",
                "Lunch meeting with sales team.",
                "Badge scanned at main entrance.",
                "Logged into workstation.",
                "Attended town hall meeting."
            ])
        }
        G.add_node(i, **person)

    # --- 2. Inject the Threat (The Smuggling Ring) ---
    print(f"   - Injecting {num_bad_actors} covert operatives...")
    bad_actor_ids = random.sample(range(num_employees), num_bad_actors)
    
    # The Handler (Executive) - The Boss
    handler = bad_actor_ids[0]
    G.nodes[handler]['risk_label'] = 1
    G.nodes[handler]['role'] = "Vice President of Operations"
    G.nodes[handler]['department'] = "Executive"
    G.nodes[handler]['last_log'] = "Authorized urgent off-book shipment to shell company."

    # Split the rest of the bad actors
    # 50% Couriers (Shipping), 50% Money Men (Sales/Legal)
    split_index = len(bad_actor_ids) // 2
    couriers = bad_actor_ids[1:split_index]
    money_men = bad_actor_ids[split_index:]

    # Configure Couriers
    for c in couriers:
        G.nodes[c]['risk_label'] = 1
        G.nodes[c]['role'] = "Shipping Manager"
        G.nodes[c]['department'] = "Shipping"
        G.nodes[c]['last_log'] = "Message Intercepted: 'Package arrived at Dock 4. Waiting for pickup.'"

    # Configure Money Men
    for m in money_men:
        G.nodes[m]['risk_label'] = 1
        G.nodes[m]['role'] = "Account Manager"
        G.nodes[m]['department'] = random.choice(['Sales', 'Legal'])
        G.nodes[m]['last_log'] = "Flagged Transaction: Large crypto transfer to unverified wallet."

    # --- 3. Create Edges (Communications) ---
    print("   - Generating communication logs (this may take a moment)...")
    edges = []
    
    # A. Normal Traffic (Sparse connectivity to keep graph clean)
    # Strategy: People connect to ~2-3 others in their dept, and rarely outside.
    for u in G.nodes():
        # Connect to random people in SAME department
        # We perform a random sample to speed up loop
        potential_friends = random.sample(range(num_employees), k=4) 
        
        for v in potential_friends:
            if u != v:
                u_dept = G.nodes[u]['department']
                v_dept = G.nodes[v]['department']
                
                # High chance if same dept
                if u_dept == v_dept:
                     if random.random() > 0.4: # 60% chance to connect to selected peer
                        edges.append((u, v, random.randint(5, 20)))
                
                # Very Low chance if different dept (The "Silo" effect)
                else:
                    if random.random() > 0.98: # 2% chance
                        edges.append((u, v, random.randint(1, 5)))

    # B. Covert Traffic (The "Star" Pattern)
    # The Handler talks to ALL operatives (High Volume)
    for agent in couriers:
        edges.append((handler, agent, random.randint(60, 100)))
        
    for agent in money_men:
        edges.append((handler, agent, random.randint(60, 100)))

    G.add_weighted_edges_from(edges)
    return G

def export_to_csv(G):
    # Nodes
    node_data = [attr for n, attr in G.nodes(data=True)]
    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv(os.path.join(BASE_PATH, "nodes.csv"), index=False)
    
    # Edges
    edge_data = [{'source': u, 'target': v, 'weight': attr['weight']} for u, v, attr in G.edges(data=True)]
    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv(os.path.join(BASE_PATH, "edges.csv"), index=False)
    
    print(f"✅ Large Scale Dataset Generated!")
    print(f"   - Nodes: {len(df_nodes)}")
    print(f"   - Edges: {len(df_edges)}")

if __name__ == "__main__":
    G = generate_realistic_intel_data(num_employees=1500, num_bad_actors=40)
    export_to_csv(G)