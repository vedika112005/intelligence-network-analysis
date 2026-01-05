import sqlite3
import pandas as pd
import os

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_NAME = "shadowlink.db"

def init_database():
    print("⚙️ Initializing SQLite Database...")
    
    # 1. Connect to Database (Creates file if missing)
    conn = sqlite3.connect(os.path.join(BASE_PATH, DB_NAME))
    cursor = conn.cursor()
    
    # 2. Load CSV Data
    try:
        nodes = pd.read_csv(os.path.join(BASE_PATH, 'nodes.csv'))
        edges = pd.read_csv(os.path.join(BASE_PATH, 'edges.csv'))
        features = pd.read_csv(os.path.join(BASE_PATH, 'extracted_features.csv'))
        print("   - CSV files loaded.")
    except FileNotFoundError:
        print("❌ Error: CSV files not found. Run 'realistic_data_gen.py' first.")
        return

    # 3. Write Data to SQL Tables
    # if_exists='replace' means it overwrites old tables every time you run this
    nodes.to_sql('nodes', conn, if_exists='replace', index=False)
    edges.to_sql('edges', conn, if_exists='replace', index=False)
    features.to_sql('features', conn, if_exists='replace', index=False)
    
    print(f"✅ Success! Database created at: {DB_NAME}")
    print(f"   - Table 'nodes': {len(nodes)} rows")
    print(f"   - Table 'edges': {len(edges)} rows")
    
    # 4. Verify with a Test Query
    print("\n🔬 Testing SQL Query (Selecting first 3 Executives):")
    cursor.execute("SELECT name, role FROM nodes WHERE department='Executive' LIMIT 3")
    for row in cursor.fetchall():
        print(f"   - {row}")

    conn.close()

if __name__ == "__main__":
    init_database()