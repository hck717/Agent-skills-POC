#!/usr/bin/env python3
"""
Streamlit UI for ReAct Multi-Agent Equity Research System

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
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
    .trace-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
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


def check_environment():
    """Check if all required environment variables are set"""
    issues = []
    
    if not os.getenv("PERPLEXITY_API_KEY"):
        issues.append("❌ PERPLEXITY_API_KEY not set")
    
    if not os.getenv("EODHD_API_KEY"):
        issues.append("⚠️ EODHD_API_KEY not set (optional)")
    
    return issues


def initialize_orchestrator():
    """Initialize the ReAct orchestrator and register agents"""
    try:
        # Create orchestrator
        orchestrator = ReActOrchestrator(max_iterations=5)
        
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
        return True, "System initialized successfully!"
        
    except Exception as e:
        return False, f"Failed to initialize: {str(e)}"


def format_trace(trace_summary):
    """Format ReAct trace for display"""
    return f"```\n{trace_summary}\n```"


# Sidebar
with st.sidebar:
    st.markdown("### 🔬 ReAct Research System")
    st.markdown("---")
    
    # System Status
    st.markdown("### System Status")
    
    env_issues = check_environment()
    if not env_issues:
        st.success("✅ Environment OK")
    else:
        for issue in env_issues:
            if "❌" in issue:
                st.error(issue)
            else:
                st.warning(issue)
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Initializing ReAct orchestrator..."):
                success, message = initialize_orchestrator()
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
        
        show_trace_by_default = st.checkbox(
            "Auto-show ReAct Trace",
            value=False,
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
    st.info("👈 Please initialize the system using the sidebar.")
    
    # Setup instructions
    st.markdown("### 🛠️ Setup Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Set Environment Variables**")
        st.code("""
export PERPLEXITY_API_KEY="your-key"
export EODHD_API_KEY="your-key"
        """, language="bash")
        
        st.markdown("**2. Start Ollama**")
        st.code("""
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
        """, language="bash")
    
    with col2:
        st.markdown("**3. Install Dependencies**")
        st.code("""
pip install streamlit
pip install -r requirements.txt
        """, language="bash")
        
        st.markdown("**4. Run Streamlit**")
        st.code("""
streamlit run app.py
        """, language="bash")
    
    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.markdown("""
    - [REACT_FRAMEWORK.md](REACT_FRAMEWORK.md) - Complete ReAct guide
    - [SPECIALIST_AGENTS.md](SPECIALIST_AGENTS.md) - Agent specifications
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
            **Company Analysis:**
            - What are Apple's key competitive risks?
            - Analyze Tesla's market position
            - Compare Microsoft and Google
            """)
        with col2:
            st.markdown("""
            **Financial Metrics:**
            - What is Apple's profit margin?
            - Analyze Amazon's growth trajectory  
            - Compare Netflix and Disney's valuations
            """)
    
    # Query input
    query = st.text_area(
        "Your Question:",
        placeholder="e.g., What are Apple's key competitive risks and profit margins?",
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
                # Progress tracking
                progress_placeholder = st.empty()
                
                # Execute research
                start_time = datetime.now()
                report = st.session_state.orchestrator.research(query)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                num_iterations = len(st.session_state.orchestrator.trace.thoughts)
                
                # Add to history
                st.session_state.history.append({
                    'query': query,
                    'report': report,
                    'trace': st.session_state.orchestrator.get_trace_summary(),
                    'duration': duration,
                    'iterations': num_iterations,
                    'timestamp': datetime.now()
                })
                
                st.success(f"✅ Analysis complete in {duration:.1f}s ({num_iterations} iterations)")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
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
                    st.metric("Agents Called", latest['iterations'])
                with col4:
                    st.metric("Time/Iter", f"{latest['duration']/latest['iterations']:.1f}s")
                
                st.markdown("---")
                
                # Query
                st.markdown("**🔍 Query:**")
                st.info(latest['query'])
                
                # Report
                st.markdown("**📄 Research Report:**")
                st.markdown(latest['report'])
                
                # ReAct Trace
                st.markdown("---")
                show_trace = st.checkbox(
                    "🔍 Show ReAct Trace",
                    value=show_trace_by_default,
                    key="show_trace_latest"
                )
                
                if show_trace:
                    st.markdown("**🧠 ReAct Reasoning Trace:**")
                    st.markdown(f'<div class="trace-box">{latest["trace"]}</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            if len(st.session_state.history) > 1:
                for idx, item in enumerate(reversed(st.session_state.history[:-1]), 1):
                    with st.expander(f"Query {len(st.session_state.history) - idx}: {item['query'][:60]}..."):
                        st.markdown(f"**Timestamp:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"**Duration:** {item['duration']:.1f}s | **Iterations:** {item['iterations']}")
                        st.markdown("---")
                        st.markdown(item['report'])
                        
                        if st.checkbox(f"Show trace", key=f"trace_{idx}"):
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
