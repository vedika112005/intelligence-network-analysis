import pandas as pd
import streamlit as st
from openai import OpenAI

def build_intelligence_briefing(target_id, df_nodes, df_features, df_edges):
    """
    Constructs the 'Context Window' (The Prompt) for the AI.
    It retrieves the Target's Identity + Math Metrics + Associate Network.
    """
    # 1. Fetch Subject Identity
    try:
        subject = df_nodes[df_nodes['id'] == target_id].iloc[0]
        metrics = df_features[df_features['id'] == target_id].iloc[0]
    except IndexError:
        return "Error: Subject ID not found."

    # 2. Fetch Network Context (Who do they talk to?)
    # Filter edges where this person is either the source or the target
    connections = df_edges[(df_edges['source'] == target_id) | (df_edges['target'] == target_id)]
    
    # Get top 4 most frequent contacts
    top_contacts = connections.sort_values(by='weight', ascending=False).head(4)
    
    contact_list = []
    for _, row in top_contacts.iterrows():
        # Identify the 'other' person in the connection
        neighbor_id = row['target'] if row['source'] == target_id else row['source']
        try:
            # Look up neighbor's role
            neighbor = df_nodes[df_nodes['id'] == neighbor_id].iloc[0]
            contact_list.append(f"- ID {neighbor_id}: {neighbor['role']} ({neighbor['department']}) [Vol: {row['weight']}]")
        except:
            pass

    # 3. Construct the Classified Prompt
    briefing = f"""
    *** CLASSIFIED INTELLIGENCE DOSSIER ***
    
    TARGET IDENTITY:
    - Name: {subject['name']}
    - Role: {subject['role']} ({subject['department']})
    - Clearance: {subject['clearance']}
    - Ground Truth Label: {'Risk' if subject['risk_label']==1 else 'Clear'}
    
    INTERCEPTED COMM LOG:
    "{subject['last_log']}"
    
    TOPOLOGICAL ANALYSIS:
    - Betweenness (Bridge Score): {metrics['betweenness']:.4f} (High = Connector)
    - Clustering (Secrecy): {metrics['clustering']:.4f} (Low = Hidden Cell)
    - Eigenvector (Influence): {metrics['eigenvector']:.4f}
    
    KNOWN ASSOCIATES:
    {chr(10).join(contact_list)}
    
    MISSION:
    Assess if this subject is an Insider Threat (Terrorist Support / Smuggling).
    """
    
    return briefing

def simulate_llm_response(briefing):
    """
    THE INTELLIGENT ROUTER:
    1. Tries to call Real GPT-4 (via Streamlit Secrets) if available.
    2. If no key is found (Default for GitHub), it falls back to Heuristic Simulation.
    """
    
    # --- ATTEMPT 1: REAL GPT-4 (Production Mode) ---
    try:
        # Check if key exists in .streamlit/secrets.toml
        if "OPENAI_API_KEY" in st.secrets:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            system_instruction = """
            You are "ShadowLink", an advanced Counter-Intelligence AI.
            Analyze the provided dossier. 
            1. Compare the Role vs. The Log (e.g., Is a Shipping Manager authorizing crypto transfers?).
            2. Analyze Topology (High Betweenness + Low Clustering = Covert Bridge).
            
            Output Format:
            ### [THREAT LEVEL: LOW / SUSPICIOUS / CRITICAL]
            **Executive Summary:** [Analysis]
            **Recommendation:** [Action]
            """

            response = client.chat.completions.create(
                model="gpt-4", 
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": briefing}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

    except Exception:
        # Silently fail over to simulation if no key is found or API fails
        pass

    # --- ATTEMPT 2: HEURISTIC SIMULATION (Demo/GitHub Mode) ---
    # This logic mimics AI reasoning using keyword triggers from our dataset.
    # It ensures the project works perfectly for recruiters without needing an API Key.
    
    briefing_lower = briefing.lower()
    
    if "authorized urgent" in briefing_lower or "package arrived" in briefing_lower:
        status = "🔴 CRITICAL THREAT"
        reason = "Subject is authorizing off-book logistics flows bypassing standard protocols."
        action = "Immediate detention and device seizure."
    elif "crypto" in briefing_lower:
        status = "🟠 HIGH RISK"
        reason = "Financial anomalies detected (Crypto/Shell Companies) inconsistent with reported income."
        action = "Freeze assets and initiate forensic audit."
    elif "low" in briefing_lower and "0.0000" not in briefing_lower:
        status = "🟢 LOW RISK"
        reason = "Activity consistent with civilian duties. No topological anomalies detected."
        action = "Continue standard periodic review."
    else:
        status = "🟡 SUSPICIOUS"
        reason = "Topology indicates potential bridge node behavior (High Betweenness), though content is ambiguous."
        action = "Flag for enhanced monitoring."

    return f"""
    ### {status}
    
    **AI Assessment:**
    {reason}
    
    **Topology Note:**
    The subject's connection pattern suggests they are { "a central hub" if "CRITICAL" in status else "a standard employee" }.
    
    **Recommendation:**
    {action}
    
    *(Note: Running in Offline Heuristic Mode. Add OpenAI Key to secrets for Live GenAI.)*
    """