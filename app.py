import streamlit as st
import pandas as pd
import networkx as nx
import joblib
import os
import sqlite3
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="ShadowLink AI: Threat Hunter", layout="wide", page_icon="🕵️‍♂️")

# --- IMPORT GRAPH RAG MODULE ---
try:
    from graph_rag import build_intelligence_briefing, simulate_llm_response
except ImportError:
    st.error("⚠️ 'graph_rag.py' not found. Please make sure the file is in the same directory.")
    st.stop()

# --- LOAD RESOURCES (HYBRID SQL/CSV) ---
@st.cache_data
def load_data():
    db_path = os.path.join(BASE_PATH, 'shadowlink.db')
    nodes_path = os.path.join(BASE_PATH, 'nodes.csv')
    
    # OPTION A: Try loading from SQLite (Preferred)
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            nodes = pd.read_sql_query("SELECT * FROM nodes", conn)
            edges = pd.read_sql_query("SELECT * FROM edges", conn)
            features = pd.read_sql_query("SELECT * FROM features", conn)
            conn.close()
        except Exception as e:
            st.warning(f"⚠️ SQL Load failed ({e}). Falling back to CSV.")
            return load_csv_data()
            
    # OPTION B: Fallback to CSV (If DB missing or failed)
    elif os.path.exists(nodes_path):
        return load_csv_data()
    else:
        st.error("❌ No Data Found! Please upload 'shadowlink.db' OR the CSV files.")
        st.stop()
        
    # Load Model (Optional)
    model_path = os.path.join(BASE_PATH, 'covert_network_model.pkl')
    model = joblib.load(model_path) if os.path.exists(model_path) else None
        
    return nodes, edges, features, model

def load_csv_data():
    """Helper to load CSVs if SQL fails"""
    nodes = pd.read_csv(os.path.join(BASE_PATH, 'nodes.csv'))
    edges = pd.read_csv(os.path.join(BASE_PATH, 'edges.csv'))
    features = pd.read_csv(os.path.join(BASE_PATH, 'extracted_features.csv'))
    model_path = os.path.join(BASE_PATH, 'covert_network_model.pkl')
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    return nodes, edges, features, model

def build_graph(nodes, edges):
    """Reconstructs the graph object for visualization"""
    G = nx.Graph()
    for _, row in nodes.iterrows():
        # Handle Attributes
        role = row['role'] if 'role' in row else 'Unknown'
        dept = row['department'] if 'department' in row else 'Unknown'
        
        # Tooltip title
        title_html = f"ID: {row['id']}\nName: {row['name']}\nRole: {role}\nDept: {dept}"
        
        # FIX: Force Convert EVERYTHING to standard Python types for PyVis
        node_id = int(row['id'])
        risk_val = int(row['risk_label']) if 'risk_label' in row else 0
        
        G.add_node(node_id, 
                   label=str(node_id), 
                   title=title_html,
                   group=dept, 
                   risk_label=risk_val)
    
    for _, row in edges.iterrows():
        # FIX: Force Convert Edge data to standard Python types
        source_id = int(row['source'])
        target_id = int(row['target'])
        weight_val = int(row['weight'])
        
        G.add_edge(source_id, target_id, weight=weight_val)
    return G

# --- VISUALIZATION ENGINE (PyVis) ---
def visualize_interactive(G, output_path="network.html"):
    net = Network(height="700px", width="100%", bgcolor="#1e1e1e", font_color="white")
    net.from_nx(G)
    
    # Custom Logic for ShadowLink Visualization
    for node in net.nodes:
        node_id = node['id']
        nx_node = G.nodes[node_id]
        
        # Highlight Bad Actors (Ground Truth) in Red
        if nx_node.get('risk_label', 0) == 1:
            node['color'] = '#ff4b4b' # Red
            node['size'] = 20
            node['shape'] = 'dot'
        else:
            # Color Civilians by Department
            dept = nx_node.get('group', '')
            if dept == 'HR':
                node['color'] = '#4caf50' 
            elif dept == 'IT':
                node['color'] = '#2196f3'
            elif dept == 'Shipping':
                node['color'] = '#ff9800'
            elif dept == 'Executive':
                node['color'] = '#9c27b0'
            else:
                node['color'] = '#9e9e9e' # Default Grey
            
            node['size'] = 8

    # Physics Options (Optimized for 1500 nodes)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -30,
          "centralGravity": 0.005,
          "springLength": 230,
          "springConstant": 0.18
        },
        "maxVelocity": 146,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 200 }
      }
    }
    """)
    
    path = os.path.join(BASE_PATH, output_path)
    net.save_graph(path)
    return path

# --- UI LAYOUT ---
st.title("🕵️‍♂️ ShadowLink AI")
st.markdown("### Advanced Insider Threat Detection System")

# Sidebar
st.sidebar.header("Operations Center")
view_mode = st.sidebar.radio("Select Module:", 
    ["Dashboard Overview", "Interactive Map", "Graph RAG Dossier", "Pathfinder"])

try:
    df_nodes, df_edges, df_features, model = load_data()
    G = build_graph(df_nodes, df_edges)
    st.sidebar.success(f"System Online: {len(df_nodes)} Entities Tracked")
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- MODULE 1: DASHBOARD OVERVIEW ---
if view_mode == "Dashboard Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", len(df_nodes))
    col2.metric("Confirmed Threats", len(df_nodes[df_nodes['risk_label']==1]))
    col3.metric("Network Density", f"{nx.density(G):.5f}")
    
    if len(G) > 0:
        assortativity = nx.degree_assortativity_coefficient(G)
        col4.metric("Homophily Score", f"{assortativity:.2f}")

    st.divider()

    st.subheader("🚨 Unsupervised Anomaly Detection")
    st.write("Using **Isolation Forest** to detect statistical outliers in network topology.")
    
    if st.button("Run Anomaly Scan"):
        # Explicitly select ONLY numeric feature columns
        numeric_cols = ['degree_centrality', 'betweenness', 'closeness', 'eigenvector', 'clustering']
        available_cols = [c for c in numeric_cols if c in df_features.columns]
        X_input = df_features[available_cols]
        
        # Run Isolation Forest
        iso = IsolationForest(contamination=0.03, random_state=42)
        iso.fit(X_input)
        
        # Get Scores
        scores = -iso.decision_function(X_input)
        df_nodes['Anomaly_Score'] = scores
        
        # Filter Top Anomalies
        top_threats = df_nodes.sort_values('Anomaly_Score', ascending=False).head(20)
        
        st.warning(f"Scan Complete. Top {len(top_threats)} Anomalies Detected.")
        
        # Show Results
        try:
            st.dataframe(
                top_threats[['id', 'name', 'role', 'department', 'Anomaly_Score']].style.background_gradient(cmap='Reds'),
                use_container_width=True
            )
        except ImportError:
            # Fallback if matplotlib is missing on server
            st.dataframe(top_threats[['id', 'name', 'role', 'department', 'Anomaly_Score']], use_container_width=True)

# --- MODULE 2: INTERACTIVE MAP ---
elif view_mode == "Interactive Map":
    st.subheader("🌍 Geospatial Link Analysis")
    st.caption("🔴 Red = Confirmed/Suspected Threat | 🔵 Blue/Green = Standard Employees")
    
    if st.button("Generate Map (May take a moment)"):
        with st.spinner("Simulating Force-Directed Graph..."):
            html_path = visualize_interactive(G)
            
            with open(html_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=750)

# --- MODULE 3: GRAPH RAG DOSSIER ---
elif view_mode == "Graph RAG Dossier":
    st.subheader("📁 AI-Augmented Intelligence Dossier")
    st.markdown("Analyze a specific subject using **Graph RAG (Retrieval Augmented Generation)**.")
    st.info("The AI reads the 'Intercepted Logs' (Text) + 'Network Topology' (Math) to assess risk.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_id = st.number_input("Enter Subject ID:", min_value=0, max_value=len(df_nodes)-1, value=0)
        
        try:
            person = df_nodes[df_nodes['id'] == target_id].iloc[0]
            st.markdown(f"**Subject:** {person['name']}")
            st.markdown(f"**Role:** {person['role']}")
            st.markdown(f"**Dept:** {person['department']}")
            
            if person['risk_label'] == 1:
                st.error("⚠️ FLAGGED IN DATABASE")
            else:
                st.success("✅ CLEAN RECORD")
                
        except:
            st.error("ID not found.")

    with col2:
        if st.button("Generate AI Assessment"):
            with st.spinner("Decrypting logs & analyzing topology..."):
                briefing = build_intelligence_briefing(target_id, df_nodes, df_features, df_edges)
                ai_response = simulate_llm_response(briefing)
                
                st.markdown("### 📝 AI Generated Intelligence Report")
                st.markdown(ai_response)
                
                with st.expander("📂 View Classified Source Data (Raw Context)"):
                    st.code(briefing)

# --- MODULE 4: PATHFINDER ---
elif view_mode == "Pathfinder":
    st.subheader("🔍 Connection Tracer")
    st.write("Find the shortest route between two subjects.")
    
    c1, c2 = st.columns(2)
    start_id = c1.number_input("Source ID", min_value=0, max_value=len(df_nodes)-1, value=0)
    end_id = c2.number_input("Target ID", min_value=0, max_value=len(df_nodes)-1, value=10)
    
    if st.button("Trace Connection"):
        try:
            path = nx.shortest_path(G, source=start_id, target=end_id)
            st.success(f"Path Found: {len(path)-1} Hops")
            
            path_str = ""
            for node in path:
                role = df_nodes[df_nodes['id'] == node]['role'].values[0]
                path_str += f"**{node}** ({role}) ➡️ "
            st.markdown(path_str[:-3])
            
            path_df = df_nodes[df_nodes['id'].isin(path)]
            st.dataframe(path_df[['id', 'name', 'role', 'department']])
            
        except nx.NetworkXNoPath:
            st.error("No connection exists between these subjects.")
        except Exception as e:
            st.error(f"Error tracing path: {e}")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("ShadowLink AI v2.0 | High-Fidelity Insider Threat Simulation")
