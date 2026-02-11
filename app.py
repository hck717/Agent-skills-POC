import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from orchestrator_react import ReActOrchestrator
from skills.web_search_agent.agent import WebSearchAgent
# 🔥 FIX: Import the correct class name from graph_agent
from skills.business_analyst_standard.graph_agent import BusinessAnalystGraphAgent as BusinessAnalystStandard
from scripts.seed_neo4j_ba_graph import seed

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
    model_options = ["deepseek-r1:8b", "llama3", "mistral", "qwen2.5:7b"]
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
        
        neo4j_user = st.text_input(
            "Neo4j User",
            value=os.getenv("NEO4J_USER", "neo4j")
        )
        
        neo4j_pass = st.text_input(
            "Neo4j Password", 
            value=os.getenv("NEO4J_PASSWORD", ""),
            type="password"
        )

    # Data Management
    with st.sidebar.expander("💾 Data Management", expanded=False):
        st.caption("Manage Local Knowledge Graph")
        if st.button("🌱 Seed Graph DB (All Data)"):
            data_dir = "./data"
            if os.path.exists(data_dir):
                tickers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
                
                if not tickers:
                     st.sidebar.warning("No ticker folders found in ./data")
                else:
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()
                    
                    for i, ticker in enumerate(tickers):
                        status_text.text(f"Seeding {ticker}...")
                        try:
                            seed(
                                uri=neo4j_uri,
                                user=neo4j_user,
                                password=neo4j_pass if neo4j_pass else "password",
                                ticker=ticker,
                                reset=True
                            )
                        except Exception as e:
                            st.sidebar.error(f"Failed to seed {ticker}: {e}")
                        
                        progress_bar.progress((i + 1) / len(tickers))
                    
                    status_text.text("✅ Seeding Complete!")
                    st.sidebar.success(f"Seeded {len(tickers)} tickers to Neo4j")
            else:
                st.sidebar.error("Data directory not found")

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
            orchestrator = ReActOrchestrator() # Orchestrator uses its own models defined in class
            
            # Register Selected Agents with UI Credentials
            if use_crag and BusinessAnalystCRAG:
                # Instantiate with UI values
                crag_agent = BusinessAnalystCRAG(
                    qdrant_url=qdrant_url if qdrant_url else None,
                    qdrant_key=qdrant_key if qdrant_key else None,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_pass=neo4j_pass
                )
                orchestrator.register_specialist("business_analyst_crag", crag_agent)
            
            if use_standard:
                 # 🔥 FIX: Pass model_name instead of model
                 orchestrator.register_specialist("business_analyst", BusinessAnalystStandard(model_name=selected_model))
            
            if use_web:
                 try:
                    orchestrator.register_specialist("web_search_agent", WebSearchAgent())
                 except Exception as e:
                    st.error(f"Failed to load Web Agent: {e}")

            # 🔥 UI: Chain of Thought Visualization
            status_container = st.status("🤖 Orchestrator working...", expanded=True)
            
            def ui_callback(step: str, detail: str, status: str):
                """Callback to update Streamlit UI from Orchestrator"""
                # Log to console for debugging
                print(f"[UI] {step}: {detail} ({status})")
                
                if status == "running":
                    status_container.write(f"**{step}**: {detail}")
                elif status == "complete":
                    status_container.write(f"✅ **{step}**: {detail}")
                elif status == "error":
                    status_container.error(f"❌ **{step}**: {detail}")
            
            try:
                # Run research with callback
                final_report = orchestrator.research(prompt, callback=ui_callback)
                
                # Close status container
                status_container.update(label="✅ Analysis Complete", state="complete", expanded=False)
                
                full_response = final_report
                message_placeholder.markdown(full_response)
            except Exception as e:
                status_container.update(label="❌ Analysis Failed", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                full_response = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
