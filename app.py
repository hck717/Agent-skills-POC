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
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent


# Page config
st.set_page_config(
    page_title="ReAct Equity Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark trace background and visible citations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    /* DARK TRACE BOX with light text */
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
    /* Make emojis and special chars more visible in trace */
    .trace-box strong {
        color: #ffd700;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    /* CITATION STYLING - Blue badges */
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
    /* Report section with better readability */
    .report-section {
        line-height: 1.9;
        font-size: 1.05rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'perplexity_api_key' not in st.session_state:
    st.session_state.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")
if 'eodhd_api_key' not in st.session_state:
    st.session_state.eodhd_api_key = os.getenv("EODHD_API_KEY", "")
if 'show_trace_default' not in st.session_state:
    st.session_state.show_trace_default = True


def format_citations(text: str) -> str:
    """
    Format citations in the report to be highly visible styled badges
    Converts [1], [2], [web:3], [cite:4], etc. to styled HTML
    """
    # Pattern to match citations like [1], [2], [web:3], [cite:4]
    # More aggressive pattern to catch all bracket notations
    citation_pattern = r'\[([^\]]+?)\]'
    
    def replace_citation(match):
        citation_text = match.group(1)
        # Only format if it looks like a citation (number or source:number)
        if citation_text.isdigit() or ':' in citation_text:
            return f'<span class="citation">[{citation_text}]</span>'
        else:
            # Return unchanged if it's not a citation
            return match.group(0)
    
    formatted_text = re.sub(citation_pattern, replace_citation, text)
    return formatted_text


def initialize_orchestrator(perplexity_api_key: str, max_iterations: int = 5):
    """Initialize the ReAct orchestrator and register agents"""
    try:
        # Validate API key
        if not perplexity_api_key or len(perplexity_api_key) < 10:
            return False, "Invalid Perplexity API key. Please enter a valid key."
        
        # Create orchestrator with API key
        orchestrator = ReActOrchestrator(
            perplexity_api_key=perplexity_api_key,
            max_iterations=max_iterations
        )
        
        # Try to register Business Analyst
        try:
            business_analyst = BusinessAnalystGraphAgent(
                data_path="./data",
                db_path="./storage/chroma_db"
            )
            orchestrator.register_specialist("business_analyst", business_analyst)
            st.session_state.business_analyst_status = "✅ Active"
        except Exception as e:
            st.session_state.business_analyst_status = f"⚠️ Error: {str(e)[:50]}"
        
        st.session_state.orchestrator = orchestrator
        st.session_state.initialized = True
        st.session_state.perplexity_api_key = perplexity_api_key
        return True, "System initialized successfully!"
        
    except Exception as e:
        return False, f"Failed to initialize: {str(e)}"


# Sidebar
with st.sidebar:
    st.markdown("### 🔬 ReAct Research System")
    st.markdown("---")
    
    # API Key Configuration
    st.markdown("### 🔑 API Configuration")
    
    with st.expander("⚙️ Configure API Keys", expanded=not st.session_state.initialized):
        perplexity_key = st.text_input(
            "Perplexity API Key *",
            value=st.session_state.perplexity_api_key,
            type="password",
            help="Required for ReAct orchestration and synthesis"
        )
        
        eodhd_key = st.text_input(
            "EODHD API Key (Optional)",
            value=st.session_state.eodhd_api_key,
            type="password",
            help="Optional - for market data access"
        )
        
        st.markdown("---")
        st.markdown("""
        **Get API Keys:**
        - [Perplexity AI](https://www.perplexity.ai/settings/api) (Required)
        - [EODHD](https://eodhd.com/register) (Optional)
        """)
    
    # System Status
    st.markdown("---")
    st.markdown("### System Status")
    
    if perplexity_key:
        st.success("✅ Perplexity API Key Set")
    else:
        st.error("❌ Perplexity API Key Required")
    
    if eodhd_key:
        st.success("✅ EODHD API Key Set")
    else:
        st.info("ℹ️ EODHD API Key Optional")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True, disabled=not perplexity_key):
            with st.spinner("Initializing ReAct orchestrator..."):
                success, message = initialize_orchestrator(perplexity_key)
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
                st.success(f"✅ {agent_name}")
            else:
                st.info(f"⏳ {agent_name}")
        
        # Settings
        st.markdown("---")
        st.markdown("### Settings")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of ReAct loop iterations"
        )
        st.session_state.orchestrator.max_iterations = max_iterations
        
        # Show trace by default checkbox
        st.session_state.show_trace_default = st.checkbox(
            "Auto-show ReAct Trace",
            value=st.session_state.show_trace_default,
            help="Automatically display reasoning trace after each query"
        )
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.orchestrator = None
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
    
    Iterative, adaptive multi-agent orchestration for equity research.
    """)


# Main content
st.markdown('<div class="main-header">🔬 ReAct Equity Research System</div>', unsafe_allow_html=True)

if not st.session_state.initialized:
    st.info("👈 Please configure API keys and initialize the system using the sidebar.")
    
    # Setup instructions
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option 1: Using UI (Recommended)**")
        st.markdown("""
        1. Click **"⚙️ Configure API Keys"** in sidebar
        2. Enter your Perplexity API key
        3. Click **"🚀 Initialize System"**
        4. Start asking questions!
        """)
        
        st.markdown("**Option 2: Using Environment Variables**")
        st.code("""
export PERPLEXITY_API_KEY="your-key"
streamlit run app.py
        """, language="bash")
    
    with col2:
        st.markdown("**Setup Ollama (for Business Analyst)**")
        st.code("""
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
        """, language="bash")
        
        st.markdown("**Get Your API Keys**")
        st.markdown("""
        - [Perplexity API](https://www.perplexity.ai/settings/api) - Required
        - [EODHD](https://eodhd.com/register) - Optional
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.markdown("""
    - [QUICKSTART.md](QUICKSTART.md) - 5-minute setup guide ⭐
    - [docs/REACT_FRAMEWORK.md](docs/REACT_FRAMEWORK.md) - Complete ReAct guide
    - [docs/SPECIALIST_AGENTS.md](docs/SPECIALIST_AGENTS.md) - Agent specifications
    - [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Debug guide
    - [README.md](README.md) - Project overview
    """)

else:
    # Query interface
    st.markdown("### 💬 Ask Your Research Question")
    
    # Example queries
    with st.expander("📌 Example Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Simple Queries:**
            - What are Apple's key competitive risks?
            - Analyze Tesla's market position
            - What is Microsoft's profit margin?
            """)
        with col2:
            st.markdown("""
            **Complex Queries (triggers full 5 iterations):**
            - Analyze Apple's competitive positioning, key risk factors, and business model sustainability based on their latest 10-K filing
            - Compare Microsoft and Google's business models, financial performance, and competitive advantages
            - Evaluate Tesla's growth strategy, market risks, and financial health from their 10-K
            """)
    
    # Query input
    query = st.text_area(
        "Your Question:",
        placeholder="e.g., Analyze Apple's competitive positioning, key risk factors, and business model sustainability based on their latest 10-K filing.",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    with col3:
        if st.session_state.history:
            download_button = st.download_button(
                "💾 Download",
                data="\n\n---\n\n".join([f"**Q:** {h['query']}\n\n{h['report']}" for h in st.session_state.history]),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    if clear_button:
        st.session_state.history = []
        st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🧠 ReAct loop running..."):
            try:
                # Execute research
                start_time = datetime.now()
                report = st.session_state.orchestrator.research(query)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                num_iterations = len(st.session_state.orchestrator.trace.thoughts)
                specialists_called = st.session_state.orchestrator.trace.get_specialist_calls()
                
                # Add to history
                st.session_state.history.append({
                    'query': query,
                    'report': report,
                    'trace': st.session_state.orchestrator.get_trace_summary(),
                    'duration': duration,
                    'iterations': num_iterations,
                    'specialists': specialists_called,
                    'timestamp': datetime.now()
                })
                
                st.success(f"✅ Analysis complete in {duration:.1f}s ({num_iterations} iterations, {len(specialists_called)} specialists)")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                
                # Show helpful error messages
                error_str = str(e).lower()
                if "401" in error_str or "unauthorized" in error_str:
                    st.warning("⚠️ **API Key Issue**: Your Perplexity API key may be invalid. Please check and update it in the sidebar.")
                elif "400" in error_str or "bad request" in error_str:
                    st.warning("⚠️ **API Request Issue**: The request format may be incorrect. This could be due to an invalid model name or API key.")
                elif "429" in error_str or "rate limit" in error_str:
                    st.warning("⚠️ **Rate Limit**: You've exceeded the API rate limit. Please wait a moment and try again.")
                
                with st.expander("🐛 Debug Information"):
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Tabs for current and history
        tabs = st.tabs(["📄 Latest Result", "📚 History"])
        
        with tabs[0]:
            if st.session_state.history:
                latest = st.session_state.history[-1]
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iterations", latest['iterations'])
                with col2:
                    st.metric("Duration", f"{latest['duration']:.1f}s")
                with col3:
                    num_specialists = len(latest.get('specialists', []))
                    st.metric("Specialists", num_specialists)
                with col4:
                    # Fix division by zero
                    if latest['iterations'] > 0:
                        time_per_iter = latest['duration'] / latest['iterations']
                        st.metric("Time/Iter", f"{time_per_iter:.1f}s")
                    else:
                        st.metric("Time/Iter", "N/A")
                
                # Show which specialists were called
                if latest.get('specialists'):
                    st.info(f"🤖 **Specialists Called:** {', '.join(latest['specialists'])}")
                
                st.markdown("---")
                
                # Query
                st.markdown("**🔍 Query:**")
                st.info(latest['query'])
                
                # Report with formatted citations - MUST use unsafe_allow_html=True
                st.markdown("**📄 Research Report:**")
                formatted_report = format_citations(latest['report'])
                st.markdown(formatted_report, unsafe_allow_html=True)
                
                # ReAct Trace - DARK BACKGROUND FOR VISIBILITY
                st.markdown("---")
                st.markdown("### 🧠 ReAct Reasoning Trace")
                st.markdown("*Step-by-step reasoning and actions taken during the analysis*")
                
                # Show trace in an expander (expanded by default if setting is on)
                with st.expander("📋 View Detailed Trace", expanded=st.session_state.show_trace_default):
                    st.markdown(f'<div class="trace-box">{latest["trace"]}</div>', unsafe_allow_html=True)
                
                # Option to download just this report
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "💾 Download Report",
                        data=f"# Research Report\n\n**Query:** {latest['query']}\n\n## Report\n\n{latest['report']}\n\n## ReAct Trace\n\n{latest['trace']}",
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        
        with tabs[1]:
            if len(st.session_state.history) > 1:
                for idx, item in enumerate(reversed(st.session_state.history[:-1]), 1):
                    with st.expander(f"📝 Query {len(st.session_state.history) - idx}: {item['query'][:80]}..."):
                        st.markdown(f"**⏰ Timestamp:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Metrics for this query
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("Duration", f"{item['duration']:.1f}s")
                        with mcol2:
                            st.metric("Iterations", item['iterations'])
                        with mcol3:
                            num_specs = len(item.get('specialists', []))
                            st.metric("Specialists", num_specs)
                        
                        if item.get('specialists'):
                            st.info(f"🤖 **Specialists:** {', '.join(item['specialists'])}")
                        
                        st.markdown("---")
                        
                        # Report with citations - MUST use unsafe_allow_html=True
                        st.markdown("**📄 Report:**")
                        formatted_hist_report = format_citations(item['report'])
                        st.markdown(formatted_hist_report, unsafe_allow_html=True)
                        
                        # Trace
                        if st.checkbox(f"🧠 Show ReAct Trace", key=f"trace_{idx}"):
                            st.markdown(f'<div class="trace-box">{item["trace"]}</div>', unsafe_allow_html=True)
            else:
                st.info("No history yet. Run another query to see history.")


# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔬 **ReAct Framework**")
with col2:
    st.markdown("🤖 **Multi-Agent System**")
with col3:
    st.markdown("📊 **Equity Research**")
