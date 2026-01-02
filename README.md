# Intelligence Network Analysis System
### Detecting Covert Criminal Cells using Graph Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%2F%20Isolation%20Forest-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## Project Overview
This project is an advanced **Counter-Terrorism & Intelligence Analysis Tool**. It uses **Graph Theory** and **Machine Learning** to identify hidden covert cells within unstructured communication networks.

Unlike traditional data analysis which looks at individual behavior, this system analyzes **relationships (topology)** to detect specific structural anomalies associated with clandestine operations, such as "Star Topologies" (Hub-and-Spoke command structures) and low-clustering "Bridge" nodes.


## Key Features
*   **🕸️ Interactive Graph Visualization:** A physics-based network map (using PyVis) allowing analysts to drag, zoom, and inspect node connections.
*   **🧠 Dual-Mode AI Engine:**
    *   **Supervised Learning (Random Forest):** Detects known threat patterns based on training data (Recall: 95%).
    *   **Unsupervised Learning (Isolation Forest):** Detects "Zero-Day" anomalies and outliers without prior labeling.
*   **📍 Pathfinding Algorithms:** Built-in "Trace" tool to find the shortest path between a suspect and a target.
*   **📊 Automatic Feature Engineering:** Calculates complex metrics like *Betweenness Centrality*, *PageRank*, and *Clustering Coefficients* on the fly.


## 🛠️ Tech Stack
*   **Language:** Python
*   **Graph Processing:** NetworkX
*   **Machine Learning:** Scikit-learn (Random Forest, Isolation Forest)
*   **Visualization:** PyVis, Matplotlib, Seaborn
*   **Web Framework:** Streamlit

## 📂 Project Structure

├── app.py # Main Dashboard Application
├── data_generator.py # Script to create synthetic intelligence data
├── feature_extraction.py # Calculates Graph Metrics (Math Engine)
├── train_model.py # Trains the ML Classifiers
├── requirements.txt # Dependencies
├── nodes.csv # Node Data (People)
├── edges.csv # Edge Data (Calls/Connections)
└── covert_network_model.pkl # Pre-trained AI Model

## The Intelligence Logic
The system relies on three core behavioral indicators to flag suspects:
1.  **The Secrecy Index (Clustering Coefficient):** Normal social circles form triangles (friends of friends know each other). Covert operatives rarely introduce their contacts to one another to minimize exposure.
2.  **The Bridge Score (Betweenness Centrality):** Identifying "Handlers" who act as the sole connection between a hidden cell and the outside world.
3.  **Homophily Analysis:** Detecting disassortative mixing (High-degree leaders connecting exclusively to low-degree operatives).

## How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/intelligence-network-analysis.git
    cd intelligence-network-analysis
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## Future Scope
*   Integration with NLP to analyze the *content* of messages, not just the connections.
*   Temporal Analysis to track how the network evolves over time (Dynamic Graphs).
*   Deployment on secure on-premise servers for real-world agency use.


