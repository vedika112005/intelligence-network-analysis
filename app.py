import streamlit as st
import pandas as pd
import networkx as nx
import joblib
import os
import sqlite3
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.ensemble import IsolationForest

# --- IMPORT GRAPH RAG MODULE ---
try:
    from graph_rag import build_intelligence_briefing, simulate_llm_response
except ImportError:
    st.error("⚠️ 'graph_rag.py' not found. Please make sure the file is in the same directory.")
    st.stop()

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="ShadowLink AI: Threat Hunter", layout="wide", page_icon="🕵️‍♂️")

# --- DATABASE MANAGEMENT (CRUD) ---

def add_suspect_to_db(name, role, dept, risk, log):
    """
    Inserts a new record into BOTH nodes and features tables.
    """
    db_path = os.path.join(BASE_PATH, 'shadowlink.db')
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Get Auto-Increment ID
    cursor.execute("SELECT MAX(id) FROM nodes")
    max_id = cursor.fetchone()[0]
    new_id = int(max_id) + 1 if max_id is not None else 0
    
    # 2. Insert into Nodes
    cursor.execute("""
        INSERT INTO nodes (id, name, role, department, clearance, risk_label, last_log)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (new_id, name, role, dept, "Low", risk, log))
    
    # 3. Insert into Features (Initialize with 0.0 to prevent AI crash)
    cursor.execute("""
        INSERT INTO features (id, degree_centrality, betweenness, closeness, eigenvector, clustering)
        VALUES (?, 0.0, 0.0, 0.0, 0.0, 0.0)
    """, (new_id,))
    
    conn.commit()
    conn.close()
    return new_id

def delete_suspect_from_db(target_id):
    """
    Deletes a record from nodes, features, AND edges (Clean Up).
    """
    db_path = os.path.join(BASE_PATH, 'shadowlink.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if ID exists
    cursor.execute("SELECT name FROM nodes WHERE id=?", (target_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return False, "ID not found."
    
    name = result[0]
    
    # 1. Delete from Nodes
    cursor.execute("DELETE FROM nodes WHERE id=?", (target_id,))
    # 2. Delete from Features
    cursor.execute("DELETE FROM features WHERE id=?", (target_id,))
    # 3. Delete from Edges (Remove connections to avoid graph errors)
    cursor.execute("DELETE FROM edges WHERE source=? OR target=?", (target_id, target_id))
    
    conn.commit()
    conn.close()
    return True, name

# --- LOAD RESOURCES ---
@st.cache_data
def load_data():
    db_path = os.path.join(BASE_PATH, 'shadowlink.db')
    nodes_path = os.path.join(BASE_PATH, 'nodes.csv')
    
    # OPTION A: SQL Loading (Preferred)
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
            
    # OPTION B: CSV Fallback
    elif os.path.exists(nodes_path):
        return load_csv_data()
    else:
        st.error("❌ No Data Found! Please upload 'shadowlink.db' OR the CSV files.")
        st.stop()
        
    model_path = os.path.join(BASE_PATH, 'covert_network_model.pkl')
    model = joblib.load(model_path) if os.path.exists(model_path) else None
        
    return nodes, edges, features, model

def load_csv_data():
    nodes = pd.read_csv(os.path.join(BASE_PATH, 'nodes.csv'))
    edges = pd.read_csv(os.path.join(BASE_PATH, 'edges.csv'))
    features = pd.read_csv(os.path.join(BASE_PATH, 'extracted_features.csv'))
    model_path = os.path.join(BASE_PATH, 'covert_network_model.pkl')
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    return nodes, edges, features, model

def build_graph(nodes, edges):
    G = nx.Graph()
    for _, row in nodes.iterrows():
        role = row['role'] if 'role' in row else 'Unknown'
        dept = row['department'] if 'department' in row else 'Unknown'
        title_html = f"ID: {row['id']}\nName: {row['name']}\nRole: {role}\nDept: {dept}"
        
        node_id = int(row['id'])
        risk_val = int(row['risk_label']) if 'risk_label' in row else 0
        
        G.add_node(node_id, label=str(node_id), title=title_html, group=dept, risk_label=risk_val)
    
    for _, row in edges.iterrows():
        source_id = int(row['source'])
        target_id = int(row['target'])
        weight_val = int(row['weight'])
        G.add_edge(source_id, target_id, weight=weight_val)
    return G

# --- VISUALIZATION ENGINE ---
def visualize_interactive(G, output_path="network.html"):
    net = Network(height="700px", width="100%", bgcolor="#1e1e1e", font_color="white")
    net.from_nx(G)
    for node in net.nodes:
        node_id = node['id']
        nx_node = G.nodes[node_id]
        if nx_node.get('risk_label', 0) == 1:
            node['color'] = '#ff4b4b' 
            node['size'] = 20
            node['shape'] = 'dot'
        else:
            dept = nx_node.get('group', '')
            if dept == 'HR': node['color'] = '#4caf50' 
            elif dept == 'IT': node['color'] = '#2196f3'
            elif dept == 'Shipping': node['color'] = '#ff9800'
            elif dept == 'Executive': node['color'] = '#9c27b0'
            else: node['color'] = '#9e9e9e'
            node['size'] = 8

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -30, "centralGravity": 0.005, "springLength": 230, "springConstant": 0.18 },
        "maxVelocity": 146, "solver": "forceAtlas2Based", "timestep": 0.35, "stabilization": { "enabled": true, "iterations": 200 }
      }
    }
    """)
    path = os.path.join(BASE_PATH, output_path)
    net.save_graph(path)
    return path

# --- UI LAYOUT ---
st.title("🕵️‍♂️ ShadowLink AI")
st.markdown("### Advanced Insider Threat Detection System")

# --- SIDEBAR: OPERATIONS ---
st.sidebar.header("Operations Center")
view_mode = st.sidebar.radio("Select Module:", 
    ["Dashboard Overview", "Interactive Map", "Graph RAG Dossier", "Pathfinder"])

st.sidebar.divider()

# 1. ADD SUSPECT
with st.sidebar.expander("📝 Log New Intel (SQL)"):
    new_name = st.text_input("Subject Name")
    new_role = st.text_input("Role", value="Contractor")
    new_dept = st.selectbox("Department", ["HR", "IT", "Shipping", "Executive", "Sales", "External"])
    new_risk = st.selectbox("Risk Level", [0, 1], format_func=lambda x: "High Threat" if x==1 else "Civilian")
    new_log = st.text_area("Intercepted Intel")
    
    if st.button("Submit Report"):
        if new_name and new_log:
            new_id = add_suspect_to_db(new_name, new_role, new_dept, new_risk, new_log)
            if new_id:
                st.success(f"Subject {new_id} added.")
                st.cache_data.clear()
                st.rerun()
        else:
            st.error("Name and Intel required.")

# 2. DELETE SUSPECT (SMARTER VERSION)
with st.sidebar.expander("🗑️ Delete Record"):
    del_id = st.number_input("Target ID to Delete", min_value=0, step=1)
    
    # Check if ID exists before showing button
    try:
        # Quick check if df_nodes is loaded to avoid error before main logic
        if 'df_nodes' not in locals():
             df_nodes, _, _, _ = load_data()
             
        if del_id < len(df_nodes):
            # Find name for confirmation
            name_rows = df_nodes[df_nodes['id'] == del_id]['name'].values
            if len(name_rows) > 0:
                st.sidebar.warning(f"Selected: **{name_rows[0]}**")
                
                if st.button("Confirm Delete"):
                    success, msg = delete_suspect_from_db(del_id)
                    if success:
                        st.success(f"Deleted: {msg}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.sidebar.caption("ID not found in current view.")
        else:
            st.sidebar.caption("ID out of range.")
    except:
        pass

# --- MAIN LOGIC ---
try:
    df_nodes, df_edges, df_features, model = load_data()
    G = build_graph(df_nodes, df_edges)
    st.sidebar.success(f"System Online: {len(df_nodes)} Entities Tracked")
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- MODULE 1: DASHBOARD ---
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
        # STEP 1: Safe Merge (Fixes 'Length Mismatch' error)
        # We merge nodes and features so we only analyze people who exist in both tables
        combined_df = pd.merge(df_nodes, df_features, on='id', how='inner')
        
        # STEP 2: Select Features
        numeric_cols = ['degree_centrality', 'betweenness', 'closeness', 'eigenvector', 'clustering']
        # Check which columns actually exist to avoid 'KeyError'
        available_features = [c for c in numeric_cols if c in combined_df.columns]
        
        if not available_features:
            st.error("No valid numeric features found. Check database.")
        else:
            X_input = combined_df[available_features]
            
            # STEP 3: Run AI
            iso = IsolationForest(contamination=0.03, random_state=42)
            iso.fit(X_input)
            
            # STEP 4: Score and Sort
            combined_df['Anomaly_Score'] = -iso.decision_function(X_input)
            top_threats = combined_df.sort_values('Anomaly_Score', ascending=False).head(20)
            
            st.warning(f"Scan Complete. Top {len(top_threats)} Anomalies Detected.")
            
            # STEP 5: Safe Display (Fixes KeyError for 'department' if using old DB)
            desired_cols = ['id', 'name', 'role', 'department', 'Anomaly_Score']
            final_cols = [c for c in desired_cols if c in top_threats.columns]
            
            try:
                st.dataframe(
                    top_threats[final_cols].style.background_gradient(cmap='Reds'),
                    use_container_width=True
                )
            except ImportError:
                st.dataframe(top_threats[final_cols], use_container_width=True)

# --- MODULE 2: MAP ---
elif view_mode == "Interactive Map":
    st.subheader("🌍 Geospatial Link Analysis")
    st.caption("🔴 Red = Confirmed/Suspected Threat | 🔵 Blue/Green = Standard Employees")
    if st.button("Generate Map"):
        with st.spinner("Simulating..."):
            html_path = visualize_interactive(G)
            with open(html_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=750)

# --- MODULE 3: GRAPH RAG ---
elif view_mode == "Graph RAG Dossier":
    st.subheader("📁 AI-Augmented Intelligence Dossier")
    col1, col2 = st.columns([1, 2])
    with col1:
        target_id = st.number_input("Enter Subject ID:", min_value=0, max_value=len(df_nodes)-1, value=0)
        try:
            person = df_nodes[df_nodes['id'] == target_id].iloc[0]
            st.markdown(f"**Subject:** {person['name']}")
            st.markdown(f"**Role:** {person.get('role', 'Unknown')}")
            st.markdown(f"**Dept:** {person.get('department', 'Unknown')}")
            if person.get('risk_label', 0) == 1: st.error("⚠️ FLAGGED IN DATABASE")
            else: st.success("✅ CLEAN RECORD")
        except:
            st.error("ID not found.")
    with col2:
        if st.button("Generate AI Assessment"):
            briefing = build_intelligence_briefing(target_id, df_nodes, df_features, df_edges)
            ai_response = simulate_llm_response(briefing)
            st.markdown("### 📝 AI Report")
            st.markdown(ai_response)
            with st.expander("📂 Raw Data"): st.code(briefing)

# --- MODULE 4: PATHFINDER ---
elif view_mode == "Pathfinder":
    st.subheader("🔍 Connection Tracer")
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
            st.error("No connection.")
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("ShadowLink AI v2.2 | Insider Threat System")
