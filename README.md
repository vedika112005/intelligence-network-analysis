# 🕵️‍♂️ ShadowLink AI
### Advanced Insider Threat Detection & Graph RAG Analysis

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Graph Theory](https://img.shields.io/badge/Graph%20Theory-NetworkX-purple)
![ML](https://img.shields.io/badge/Unsupervised-Isolation%20Forest-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 📌 Project Overview
**ShadowLink AI** is a next-generation Counter-Intelligence platform designed to detect **Insider Threats** and **Covert Cells** hidden within legitimate corporate networks. 

Simulating a high-security logistics firm (**"Project Gotham"**) containing **1,500 entities**, this system combines **Graph Topology Mathematics** with **Automated Forensics** to identify "The Needle in the Haystack"—covert smuggling rings hiding inside standard departmental traffic.

Unlike traditional tools that look at rows of data, ShadowLink analyzes the **structure of relationships** to find anomalies like "Star Topologies" (Command & Control) and "Bridge Nodes" (Cut-outs).

## 🚀 Key Features

### 1. 📂 Graph RAG Dossier (Retrieval Augmented Generation)
*   **Contextual Intelligence:** The system generates a "Classified Briefing" by retrieving a suspect's mathematical metrics (Centrality, Clustering) and combining them with their intercepted communication logs.
*   **Hybrid AI Engine:**
    *   **Default Mode (Heuristic):** Uses a deterministic logic engine to scan logs for trigger phrases (*"off-book"*, *"crypto"*) and combines them with topology to issue a threat verdict.
    *   **Production Mode (GPT-4):** Architecture supports live integration with OpenAI's GPT-4 API for dynamic, non-deterministic reasoning (configured via secrets).

### 2. 🌍 Interactive Geospatial Map
*   **Physics Engine:** A force-directed graph visualization (PyVis) allowing analysts to drag nodes, zoom into clusters, and visually inspect connections.
*   **Visual Forensics:** Automatically colors nodes by Department (HR, IT, Shipping) and highlights Threats in Red.

### 3. 🚨 Unsupervised Anomaly Detection
*   **Zero-Day Discovery:** Uses **Isolation Forest** algorithms to scan the network for statistical outliers without needing prior training labels.
*   **Behavioral Scoring:** Flags entities that mathematically deviate from the corporate norm (e.g., a Shipping Manager with excessive influence scores).

## 🛠️ Tech Stack
*   **Core Logic:** Python 3.9
*   **Graph Processing:** NetworkX
*   **Machine Learning:** Scikit-learn (Isolation Forest), Pandas
*   **NLP Logic:** Hybrid (Heuristic / OpenAI API)
*   **Visualization:** PyVis, Streamlit Components
*   **Frontend:** Streamlit

## 📂 Project Structure
```text
├── app.py                  # Main ShadowLink Dashboard
├── graph_rag.py            # AI Engine (Briefing & Analysis Logic)
├── realistic_data_gen.py   # "Project Gotham" Generator (1,500 Nodes)
├── feature_extraction.py   # Math Engine (Centrality, PageRank, Clustering)
├── requirements.txt        # Dependencies
├── nodes.csv               # Generated Intelligence Data (Identities & Logs)
├── edges.csv               # Generated Communication Logs
└── README.md               # Project Documentation

📊 The Intelligence Logic
The system relies on three core behavioral indicators to flag suspects:
The Secrecy Index (Clustering Coefficient):
Normal: High clustering (Friends of friends know each other).
Threat: Low clustering (Operatives communicate with a Handler but are isolated from each other to protect the cell).
The Bridge Score (Betweenness Centrality):
Identifies "Gatekeepers" who connect two otherwise disconnected departments (e.g., a VP secretly coordinating with the loading dock).
Content Forensics:
Scans intercepted logs for "Trigger Keywords" associated with money laundering (Crypto) or unauthorized logistics (Off-book shipments).

💻 How to Run Locally
1.Clone the Repository
git clone https://github.com/your-username/shadowlink-ai.git
cd shadowlink-ai

2.Install Dependencies
pip install -r requirements.txt

3.Generate Fresh Data (Optional)
python realistic_data_gen.py
python feature_extraction.py

4.Run the Dashboard
streamlit run app.py

📈 Future Scope
Real-Time Ingestion: Connecting to live Slack/Email APIs for enterprise deployment.
Temporal Analysis: Adding a "Time Slider" to watch the criminal network evolve over weeks.
Blockchain Tracking: Integrating crypto-wallet tracking to correlate financial flows with communication graphs.


Created by Vedika as a project in Applied Graph Machine Learning & Counter-Intelligence.