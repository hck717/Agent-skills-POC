#!/usr/bin/env python3
"""
Streamlit UI for ReAct Multi-Agent Equity Research System

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import re
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import ReActOrchestrator
# 🔥 UPDATED: Use business_analyst_standard instead of business_analyst
from skills.business_analyst_standard.graph_agent import BusinessAnalystGraphAgent
from skills.web_search_agent.agent import WebSearchAgent


# Page config
st.set_page_config(
    page_title="ReAct Equity Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .trace-box {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 2px solid #444;
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        max-height: 600px;
        overflow-y: auto;
        line-height: 1.8;
    }
    .citation {
        display: inline-block;
        background-color: #1976d2;
        color: #ffffff;
        padding: 3px 8px;
        margin: 0 3px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'business_analyst' not in st.session_state:
    st.session_state.business_analyst = None
if 'web_search_agent' not in st.session_state:
    st.session_state.web_search_agent = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'show_trace_default' not in st.session_state:
    st.session_state.show_trace_default = True
if 'tavily_api_key' not in st.session_state:
    st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")


def format_citations(text: str) -> str:
    """Format citations to be highly visible styled badges"""
    citation_pattern = r'\[([^\]]+?)\]'
    
    def replace_citation(match):
        citation_text = match.group(1)
        if citation_text.isdigit() or ':' in citation_text:
            return f'<span class="citation">[{citation_text}]</span>'
        else:
            return match.group(0)
    
    formatted_text = re.sub(citation_pattern, replace_citation, text)
    return formatted_text


def initialize_orchestrator(max_iterations: int = 3, ollama_url: str = "http://localhost:11434", tavily_key: str = None):
    """Initialize the ReAct orchestrator and register agents"""
    try:
        # Create orchestrator (using local Ollama, no API key needed)
        orchestrator = ReActOrchestrator(
            ollama_url=ollama_url,
            max_iterations=max_iterations
        )
        
        # Test connection
        if not orchestrator.test_connection():
            return False, "Failed to connect to Ollama. Make sure Ollama is running: `ollama serve`"
        
        # Try to register Business Analyst
        try:
            business_analyst = BusinessAnalystGraphAgent(
                data_path="./data",
                db_path="./storage/chroma_db"
            )
            orchestrator.register_specialist("business_analyst", business_analyst)
            st.session_state.business_analyst = business_analyst
            st.session_state.business_analyst_status = "✅ Active (Standard RAG)"
        except Exception as e:
            st.session_state.business_analyst_status = f"⚠️ Error: {str(e)[:50]}"
        
        # Try to register Web Search Agent (if Tavily key provided)
        if tavily_key and tavily_key.strip():
            try:
                web_search_agent = WebSearchAgent(
                    tavily_api_key=tavily_key,
                    ollama_model="deepseek-r1:8b"
                )
                orchestrator.register_specialist("web_search_agent", web_search_agent)
                st.session_state.web_search_agent = web_search_agent
                st.session_state.web_search_status = "✅ Active"
            except Exception as e:
                st.session_state.web_search_status = f"⚠️ Error: {str(e)[:50]}"
        else:
            st.session_state.web_search_status = "⏳ No Tavily API key"
        
        st.session_state.orchestrator = orchestrator
        st.session_state.initialized = True
        return True, "System initialized successfully!"
        
    except Exception as e:
        return False, f"Failed to initialize: {str(e)}"


# Sidebar
with st.sidebar:
    st.markdown("### 🔬 ReAct Research System")
    st.markdown("---")
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("🔧 System Settings", expanded=not st.session_state.initialized):
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="URL of your Ollama instance"
        )
        
        st.markdown("---")
        
        # Tavily API Key (for Web Search Agent)
        st.markdown("**🌐 Web Search (Optional)**")
        tavily_key = st.text_input(
            "Tavily API Key",
            value=st.session_state.tavily_api_key,
            type="password",
            help="Get free API key at https://tavily.com\nLeave empty to use only document analysis"
        )
        st.session_state.tavily_api_key = tavily_key
        
        if not tavily_key:
            st.info("💡 Without Tavily key, system will only use document analysis (Business Analyst)")
        
        st.markdown("---")
        st.markdown("""
        **Required Ollama Models:**
        ```bash
        ollama pull deepseek-r1:8b
        ollama pull nomic-embed-text
        ```
        
        **Start Ollama:**
        ```bash
        ollama serve
        ```
        """)
    
    # System Status
    st.markdown("---")
    st.markdown("### System Status")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Connecting to Ollama and initializing orchestrator..."):
                success, message = initialize_orchestrator(
                    ollama_url=ollama_url,
                    tavily_key=st.session_state.tavily_api_key
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.success("✅ System Ready")
        
        # Agent Status
        st.markdown("---")
        st.markdown("### Specialist Agents")
        
        from orchestrator_react import ReActOrchestrator as RO
        
        for agent_name, info in RO.SPECIALIST_AGENTS.items():
            if agent_name in st.session_state.orchestrator.specialist_agents:
                priority = info.get('priority', 99)
                st.success(f"✅ {agent_name} (P{priority})")
            else:
                st.info(f"⏳ {agent_name}")
        
        # Show agent execution order
        st.caption("""
        **Execution Order:**
        1. Business Analyst (documents - Standard RAG)
        2. Web Search Agent (supplements)
        3. Synthesis (combines all)
        """)
        
        # Business Analyst Data Management
        if st.session_state.business_analyst:
            st.markdown("---")
            st.markdown("### 📚 Business Analyst Data")
            
            with st.expander("🔧 Data Management", expanded=False):
                st.markdown("""
                **Supported Formats:**
                - 📄 PDF (.pdf)
                - 📝 Word (.docx)
                - 📃 Text (.txt)
                - 📋 Markdown (.md)
                """)
                
                # Get database stats
                if st.button("📊 Check Database Stats", use_container_width=True):
                    with st.spinner("Checking database..."):
                        stats = st.session_state.business_analyst.get_database_stats()
                        
                        if 'error' in stats:
                            st.error(f"❌ {stats['error']}")
                        else:
                            st.success("📈 Database Statistics:")
                            for ticker, count in stats.items():
                                if ticker != 'TOTAL':
                                    st.metric(f"{ticker}", f"{count:,} chunks")
                            st.markdown("---")
                            st.metric("**Total Chunks**", f"{stats.get('TOTAL', 0):,}")
                
                st.markdown("---")
                
                # Reingest button
                st.markdown("**🔄 Re-ingest Documents**")
                st.caption("Scan ./data folder and embed all documents")
                if st.button("🔄 Reingest All Data", use_container_width=True, type="primary"):
                    with st.spinner("Re-ingesting documents from ./data folder..."):
                        try:
                            st.session_state.business_analyst.ingest_data()
                            st.success("✅ Documents re-ingested successfully!")
                            st.info("💡 Click 'Check Database Stats' to see updated counts")
                        except Exception as e:
                            st.error(f"❌ Error during ingestion: {str(e)}")
                
                st.markdown("---")
                
                # Reset button
                st.markdown("**⚠️ Reset Vector Database**")
                st.caption("⚠️ This will DELETE all embedded documents!")
                
                reset_confirmed = st.checkbox(
                    "I understand this will delete all data",
                    key="reset_confirm"
                )
                
                if st.button(
                    "🗑️ Reset Database", 
                    use_container_width=True, 
                    disabled=not reset_confirmed
                ):
                    with st.spinner("Resetting vector database..."):
                        try:
                            success, message = st.session_state.business_analyst.reset_vector_db()
                            if success:
                                st.success(f"✅ {message}")
                                st.warning("⚠️ Database cleared. Run 'Reingest All Data' to reload.")
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        # Settings
        st.markdown("---")
        st.markdown("### Settings")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum number of ReAct loop iterations (3 = Business + Web + Synthesis)"
        )
        st.session_state.orchestrator.max_iterations = max_iterations
        
        st.session_state.show_trace_default = st.checkbox(
            "Auto-show ReAct Trace",
            value=st.session_state.show_trace_default,
            help="Automatically display reasoning trace"
        )
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.orchestrator = None
            st.session_state.business_analyst = None
            st.session_state.web_search_agent = None
            st.session_state.history = []
            st.session_state.initialized = False
            st.rerun()
    
    # Info
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **ReAct Framework**
    - 🧠 Think
    - ⚡ Act  
    - 👁️ Observe
    - 🔁 Repeat
    
    **Data Sources:**
    - 📄 Local Documents (10-K, PDFs)
    - 🌐 Web Search (Tavily + local LLM)
    - 🔒 All synthesis done locally
    
    **Business Analyst:** Standard RAG v24.0
    """)


# Main content
st.markdown('<div class="main-header">🔬 ReAct Equity Research System</div>', unsafe_allow_html=True)

if not st.session_state.initialized:
    st.info("👈 Please initialize the system using the sidebar.")
    
    # Setup instructions
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Step 1: Install Ollama**")
        st.code("""
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from ollama.com
        """, language="bash")
        
        st.markdown("**Step 2: Pull Models**")
        st.code("""
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
        """, language="bash")
    
    with col2:
        st.markdown("**Step 3: Start Ollama**")
        st.code("""
ollama serve
        """, language="bash")
        
        st.markdown("**Step 4: (Optional) Get Tavily Key**")
        st.markdown("""
        For web search: [tavily.com](https://tavily.com)
        """)
        
        st.markdown("**Step 5: Initialize System**")
        st.markdown("""
        Click **"🚀 Initialize System"** in the sidebar
        """)
    
    st.markdown("---")
    st.markdown("### ✅ Benefits")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("💰 **Zero API Costs**")
        st.caption("Local LLM synthesis")
    with col2:
        st.success("🔒 **Full Privacy**")
        st.caption("Data stays local")
    with col3:
        st.success("📚 **Document Citations**")
        st.caption("Local file references")
    with col4:
        st.success("🌐 **Web Supplement**")
        st.caption("Current market data")

else:
    # Query interface
    st.markdown("### 💬 Ask Your Research Question")
    
    # Example queries
    with st.expander("📌 Example Queries"):
        if st.session_state.web_search_agent:
            st.markdown("""
            **With Web Search:**
            - What are Apple's latest competitive developments? (documents + web)
            - Analyze Tesla's recent news and 10-K risk factors (hybrid)
            - Compare Microsoft's strategy in documents vs analyst opinions (multi-source)
            
            **Document-Only:**
            - Evaluate Apple's supply chain vulnerabilities from their 10-K
            - What risks does Apple face according to their SEC filings?
            """)
        else:
            st.markdown("""
            - What are Apple's key competitive risks?
            - Analyze Tesla's market position
            - Evaluate Apple's supply chain vulnerabilities from their 10-K
            - What risks does Apple face according to their SEC filings?
            """)
    
    # Query input
    query = st.text_area(
        "Your Question:",
        placeholder="e.g., Analyze Apple's competitive positioning from their 10-K and supplement with recent market developments.",
        height=100,
        key="query_input"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        submit_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.history:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
            if clear_button:
                st.session_state.history = []
                st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🧠 ReAct loop running..."):
            try:
                start_time = datetime.now()
                report = st.session_state.orchestrator.research(query)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                num_iterations = len(st.session_state.orchestrator.trace.thoughts)
                specialists_called = st.session_state.orchestrator.trace.get_specialist_calls()
                
                st.session_state.history.append({
                    'query': query,
                    'report': report,
                    'trace': st.session_state.orchestrator.get_trace_summary(),
                    'duration': duration,
                    'iterations': num_iterations,
                    'specialists': specialists_called,
                    'timestamp': datetime.now()
                })
                
                st.success(f"✅ Complete in {duration:.1f}s ({num_iterations} iterations)")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
                with st.expander("🐛 Debug Information"):
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        latest = st.session_state.history[-1]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Iterations", latest['iterations'])
        with col2:
            st.metric("Duration", f"{latest['duration']:.1f}s")
        with col3:
            st.metric("Specialists", len(latest.get('specialists', [])))
        with col4:
            if latest['iterations'] > 0:
                st.metric("Time/Iter", f"{latest['duration'] / latest['iterations']:.1f}s")
        
        if latest.get('specialists'):
            st.info(f"🤖 **Specialists Called:** {', '.join(latest['specialists'])}")
        
        st.markdown("---")
        
        # Query
        st.markdown("**🔍 Query:**")
        st.info(latest['query'])
        
        # Report with formatted citations
        st.markdown("**📄 Research Report:**")
        formatted_report = format_citations(latest['report'])
        st.markdown(formatted_report, unsafe_allow_html=True)
        
        # ReAct Trace
        st.markdown("---")
        st.markdown("### 🧠 ReAct Reasoning Trace")
        
        with st.expander("📋 View Detailed Trace", expanded=st.session_state.show_trace_default):
            st.markdown(f'<div class="trace-box">{latest["trace"]}</div>', unsafe_allow_html=True)
        
        # Download
        st.markdown("---")
        st.download_button(
            "💾 Download Report",
            data=f"# Research Report\n\n**Query:** {latest['query']}\n\n## Report\n\n{latest['report']}\n\n## Trace\n\n{latest['trace']}",
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )


# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🔬 **ReAct Framework**")
with col2:
    st.markdown("🤖 **Local Ollama LLM**")
with col3:
    st.markdown("📊 **Document Citations**")
with col4:
    st.markdown("🌐 **Web Supplement**")
