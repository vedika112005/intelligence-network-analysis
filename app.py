import streamlit as st
import pandas as pd
import networkx as nx
import joblib
import os
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
# This sets the path to the current folder, wherever the script is running
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Intel Network Hunter", layout="wide")

# --- LOAD RESOURCES ---
@st.cache_data
def load_data():
    nodes = pd.read_csv(os.path.join(BASE_PATH, 'nodes.csv'))
    edges = pd.read_csv(os.path.join(BASE_PATH, 'edges.csv'))
    features = pd.read_csv(os.path.join(BASE_PATH, 'extracted_features.csv'))
    
    # Try to load the model, but don't crash if it's missing (allows running on real data)
    model_path = os.path.join(BASE_PATH, 'covert_network_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
        
    return nodes, edges, features, model

def build_graph(nodes, edges):
    G = nx.Graph()
    for _, row in nodes.iterrows():
        # Handle cases where real data might miss columns
        node_type = row['type'] if 'type' in row else 'Unknown'
        risk = row['risk_label'] if 'risk_label' in row else 0
        
        G.add_node(row['id'], label=str(row['name']), type=node_type, risk_label=risk, title=f"ID: {row['id']}")
    
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    return G

# --- VISUALIZATION ENGINE ---
def visualize_interactive(G, output_path="network.html"):
    """
    Creates an interactive HTML graph using PyVis.
    """
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Convert nx graph to pyvis
    net.from_nx(G)
    
    # Color code nodes
    for node in net.nodes:
        # Get the original node data from NetworkX graph
        try:
            nx_node = G.nodes[node['id']]
            if nx_node.get('risk_label', 0) == 1:
                node['color'] = '#ff4d4d'  # Red for Threat
                node['size'] = 25
            else:
                node['color'] = '#00bfff'  # Blue for Civilian
                node['size'] = 10
        except:
            node['color'] = '#00bfff'

    # Physics Options
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    # Save and return path
    path = os.path.join(BASE_PATH, output_path)
    net.save_graph(path)
    return path

# --- UI LAYOUT ---
st.title("🕵️ Intelligence Hunter: Advanced Network Analysis")

# Sidebar
st.sidebar.header("Operations Center")
view_mode = st.sidebar.radio("Select Module:", ["Dashboard", "Interactive Map", "Pathfinder"])

try:
    df_nodes, df_edges, df_features, model = load_data()
    G = build_graph(df_nodes, df_edges)
    st.sidebar.success(f"System Online: {len(df_nodes)} Entities Tracked")
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- MODULE 1: DASHBOARD ---
if view_mode == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Subjects", len(df_nodes))
    
    # Count threats if label exists, else 0
    threat_count = len(df_nodes[df_nodes['risk_label']==1]) if 'risk_label' in df_nodes.columns else 0
    col2.metric("Confirmed Threats", threat_count)
    col3.metric("Network Density", f"{nx.density(G):.4f}")
    
    # Advanced Metric
    if len(G) > 0:
        assortativity = nx.degree_assortativity_coefficient(G)
        col4.metric("Homophily Score", f"{assortativity:.2f}")
    
    st.markdown("### 🚨 Threat Prediction Unit")
    
    # Choose Model Type
    model_type = st.radio("AI Detection Mode:", ["Supervised (Pre-trained)", "Unsupervised (Anomaly Scan)"])

    if st.button("Run AI Diagnostics"):
        # Prepare input features (dropping non-numeric)
        X_input = df_features.drop(columns=['id', 'name', 'country', 'type', 'risk_label'], errors='ignore')
        
        results = df_nodes.copy()
        
        if model_type == "Supervised (Pre-trained)":
            if model:
                # Use the loaded Random Forest
                probs = model.predict_proba(X_input)[:, 1]
                results['Risk_Score'] = probs
            else:
                st.error("No pre-trained model found. Please train the model or use Unsupervised mode.")
                st.stop()
                
        elif model_type == "Unsupervised (Anomaly Scan)":
            # Use Isolation Forest (No labels needed)
            iso = IsolationForest(contamination=0.02, random_state=42)
            iso.fit(X_input)
            
            # Decision function: lower = more anomalous. We invert it for a 0-1 score.
            scores = -iso.decision_function(X_input)
            min_s, max_s = scores.min(), scores.max()
            results['Risk_Score'] = (scores - min_s) / (max_s - min_s)

        # Filter High Risk
        high_risk = results[results['Risk_Score'] > 0.65].sort_values('Risk_Score', ascending=False)
        
        st.warning(f"AI Detected {len(high_risk)} Anomalies in the network.")
        st.dataframe(
            high_risk[['id', 'name', 'Risk_Score']].style.background_gradient(cmap='Reds'),
            use_container_width=True
        )

# --- MODULE 2: INTERACTIVE MAP ---
elif view_mode == "Interactive Map":
    st.subheader("🌍 Geospatial Link Analysis")
    st.markdown("Use your mouse to drag nodes, zoom, and explore connections.")
    
    # Generate the HTML file
    html_path = visualize_interactive(G)
    
    # Load HTML into Streamlit
    with open(html_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    components.html(source_code, height=650)

# --- MODULE 3: PATHFINDER ---
elif view_mode == "Pathfinder":
    st.subheader("🔍 Connection Tracer")
    st.write("Find the shortest route between two suspects.")
    
    col1, col2 = st.columns(2)
    # Safely get max ID
    max_id = len(df_nodes)-1
    
    source_id = col1.number_input("Suspect A (ID)", min_value=0, max_value=max_id, value=0)
    target_id = col2.number_input("Suspect B (ID)", min_value=0, max_value=max_id, value=min(1000, max_id))
    
    if st.button("Trace Connection"):
        try:
            path = nx.shortest_path(G, source=source_id, target=target_id)
            st.success(f"Connection Found! Distance: {len(path)-1} hops")
            st.write(f"**Path:** {' ➡️ '.join(map(str, path))}")
            
            # Show details
            path_details = df_nodes[df_nodes['id'].isin(path)]
            st.dataframe(path_details)
            
        except nx.NetworkXNoPath:
            st.error("No connection exists between these subjects.")
        except nx.NodeNotFound:
            st.error("One of the IDs does not exist in the graph.")