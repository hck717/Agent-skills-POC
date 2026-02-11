import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from orchestrator_react import ReActOrchestrator
from skills.web_search_agent.agent import WebSearchAgent
from skills.business_analyst_standard.agent import BusinessAnalystStandard
try:
    from skills.business_analyst_crag import BusinessAnalystCRAG
except ImportError:
    BusinessAnalystCRAG = None

# Load .env initially to populate defaults
load_dotenv()

def main():
    st.set_page_config(
        page_title="Agent Skills POC",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Enterprise Agent Skills POC")
    st.markdown("ReAct Orchestrator with Specialist Agents")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model Selection
    model_options = ["deepseek-r1:8b", "llama3", "mistral"]
    selected_model = st.sidebar.selectbox("Analysis Model", model_options, index=0)

    # 🔐 Database Credentials (UI Overrides)
    with st.sidebar.expander("🔐 Database Credentials", expanded=False):
        st.caption("Override .env defaults here")
        
        qdrant_url = st.text_input(
            "Qdrant URL", 
            value=os.getenv("QDRANT_URL", ""),
            type="default",
            placeholder="https://xyz.qdrant.io"
        )
        
        qdrant_key = st.text_input(
            "Qdrant API Key", 
            value=os.getenv("QDRANT_API_KEY", ""),
            type="password"
        )
        
        neo4j_uri = st.text_input(
            "Neo4j URI", 
            value=os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        
        neo4j_pass = st.text_input(
            "Neo4j Password", 
            value=os.getenv("NEO4J_PASSWORD", ""),
            type="password"
        )

    # Agent Selection
    st.sidebar.subheader("Active Agents")
    use_crag = st.sidebar.checkbox("Business Analyst CRAG (Deep Reader)", value=True, disabled=BusinessAnalystCRAG is None)
    use_standard = st.sidebar.checkbox("Business Analyst Standard", value=False)
    use_web = st.sidebar.checkbox("Web Search Agent", value=True)
    
    if BusinessAnalystCRAG is None:
        st.sidebar.warning("BusinessAnalystCRAG not found/installed")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a research question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Re-initialize/Update orchestrator with current settings
            # We initialize with the selected model
            orchestrator = ReActOrchestrator(model=selected_model)
            
            # Register Selected Agents with UI Credentials
            if use_crag and BusinessAnalystCRAG:
                # Instantiate with UI values
                crag_agent = BusinessAnalystCRAG(
                    qdrant_url=qdrant_url if qdrant_url else None,
                    qdrant_key=qdrant_key if qdrant_key else None,
                    neo4j_uri=neo4j_uri,
                    neo4j_pass=neo4j_pass
                )
                orchestrator.register_specialist("business_analyst_crag", crag_agent)
            
            if use_standard:
                 orchestrator.register_specialist("business_analyst", BusinessAnalystStandard(model=selected_model))
            
            if use_web:
                 try:
                    orchestrator.register_specialist("web_search_agent", WebSearchAgent())
                 except Exception as e:
                    st.error(f"Failed to load Web Agent: {e}")

            with st.spinner("Orchestrating agents..."):
                try:
                    # Run research
                    final_report = orchestrator.research(prompt)
                    full_response = final_report
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    full_response = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
