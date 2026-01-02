import pandas as pd
import networkx as nx
import os

# This sets the path to the current folder, wherever the script is running
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_graph():
    """
    Loads the graph from the CSV files we created earlier.
    """
    nodes_path = os.path.join(BASE_PATH, 'nodes.csv')
    edges_path = os.path.join(BASE_PATH, 'edges.csv')
    
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    
    G = nx.Graph()
    # Add nodes with basic attributes
    for _, row in df_nodes.iterrows():
        G.add_node(row['id'], 
                   risk_label=row['risk_label'], 
                   country=row['country'])
        
    # Add edges
    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
        
    return G, df_nodes

def calculate_graph_metrics(G, df_nodes):
    """
    Calculates centrality and structural metrics for every node.
    """
    print("🚀 Starting Feature Extraction...")
    
    # 1. Degree Centrality (Connection Volume)
    print("   - Calculating Degree Centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    # 2. Betweenness Centrality (Bridge/Gatekeeper Score)
    print("   - Calculating Betweenness Centrality (this involves pathfinding)...")
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # 3. Closeness Centrality (Speed of information flow)
    print("   - Calculating Closeness Centrality...")
    closeness_centrality = nx.closeness_centrality(G)
    
    # 4. Eigenvector Centrality (Influence of neighbors)
    print("   - Calculating Eigenvector Centrality...")
    # max_iter increased because sparse graphs sometimes take longer to converge
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # 5. Clustering Coefficient (The "Secrecy" Score)
    print("   - Calculating Clustering Coefficient...")
    clustering_coefficient = nx.clustering(G)
    
    # --- COMBINE INTO A DATAFRAME ---
    print("📊 Compiling features into a dataset...")
    
    # Convert dictionaries to DataFrame columns
    df_features = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['degree_centrality'])
    df_features['betweenness'] = df_features.index.map(betweenness_centrality)
    df_features['closeness'] = df_features.index.map(closeness_centrality)
    df_features['eigenvector'] = df_features.index.map(eigenvector_centrality)
    df_features['clustering'] = df_features.index.map(clustering_coefficient)
    
    # Reset index to get 'id' as a column
    df_features = df_features.reset_index().rename(columns={'index': 'id'})
    
    # Merge with original node data (to keep the Risk Label and Country)
    final_df = pd.merge(df_nodes, df_features, on='id')
    
    return final_df

if __name__ == "__main__":
    # 1. Load Data
    G, df_nodes = load_graph()
    
    # 2. Calculate Features
    df_final = calculate_graph_metrics(G, df_nodes)
    
    # 3. Save to CSV
    output_path = os.path.join(BASE_PATH, 'extracted_features.csv')
    df_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ Feature Engineering Complete!")
    print(f"📂 File saved to: {output_path}")
    print("-" * 30)
    print("Preview of the data:")
    print(df_final[['name', 'risk_label', 'degree_centrality', 'clustering']].head(10))