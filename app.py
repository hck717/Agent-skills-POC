import streamlit as st
import time
from orchestrator_react import ReActOrchestrator

# Config
st.set_page_config(
    page_title="Agentic Equity Research (DeepSeek R1)", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .report-font { font-family: 'Times New Roman', serif; font-size: 1.1rem; }
    .stCodeBlock { background-color: #f8f9fa; }
    .agent-thought { border-left: 3px solid #0066cc; padding-left: 1rem; margin: 0.5rem 0; color: #404040; font-style: italic; }
    .agent-action { border-left: 3px solid #28a745; padding-left: 1rem; margin: 0.5rem 0; background-color: #f0fff4; }
    .metric-card { background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'orchestrator' not in st.session_state:
    try:
        st.session_state.orchestrator = ReActOrchestrator(max_iterations=3)
    except Exception as e:
        st.error(f"Failed to initialize: {e}")

if 'trace_logs' not in st.session_state:
    st.session_state.trace_logs = []

if 'current_status' not in st.session_state:
    st.session_state.current_status = "Ready"

def update_callback(step: str, detail: str, status: str):
    """Callback to update UI from orchestrator"""
    # Map status to icon
    icon = "⏳"
    if status == "running": icon = "🔄"
    elif status == "complete": icon = "✅"
    elif status == "error": icon = "❌"
    
    # Add log entry
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.trace_logs.append({
        "time": timestamp,
        "step": step,
        "detail": detail,
        "icon": icon
    })
    
    # Update status bar
    st.session_state.current_status = f"{icon} {step}: {detail}"
    
    # Force refresh (using empty container trick)
    # Note: Streamlit doesn't support easy background updates, so we rely on re-renders
    pass

# Sidebar
with st.sidebar:
    st.title("📈 Agent Settings")
    
    model_choice = st.selectbox(
        "Reasoning Model",
        ["DeepSeek-R1 8B (Recommended)", "Qwen 2.5 7B"],
        index=0
    )
    
    st.info("""
    **Architecture:**
    - **ReAct Orchestrator**: Rule-based + LLM
    - **Deep Reader (CRAG)**: 10-K Analysis (Neo4j + Qdrant)
    - **Web Agent**: Real-time Market Data
    """)
    
    st.divider()
    st.markdown("### 🔍 Transparency")
    st.checkbox("Show Raw Trace", value=True, key="show_trace")
    st.checkbox("Show Citations", value=True, key="show_cites")

# Main Interface
st.title("📈 Agentic Equity Research Analyst")
st.markdown("*Institutional-grade analysis powered by GraphRAG & DeepSeek R1*")

# Input
query = st.text_input(
    "Research Query", 
    placeholder="e.g., Analyze Apple's 2026 business model and risks...",
    help="Enter a ticker or company name. The system will auto-seed the graph DB."
)

col1, col2 = st.columns([1, 4])
with col1:
    search_btn = st.button("🚀 Start Research", use_container_width=True, type="primary")

# Layout
left_col, right_col = st.columns([1.5, 1])

if search_btn and query:
    # Clear logs
    st.session_state.trace_logs = []
    
    # Running State
    with left_col:
        status_container = st.empty()
        report_container = st.container()
    
    with right_col:
        st.subheader("🧠 Thought Process (Live)")
        log_container = st.empty()
        
        # Display logs function
        def render_logs():
            logs_md = ""
            for log in reversed(st.session_state.trace_logs):
                logs_md += f"**{log['time']}** {log['icon']} **{log['step']}**\n\n{log['detail']}\n\n---\n\n"
            log_container.markdown(logs_md)

    # Execute Research
    try:
        # We need to wrap the orchestrator call to capture updates
        # Since the orchestrator is synchronous, we pass the callback
        
        # Define a wrapper that updates the specific UI elements
        def ui_callback(step, detail, status):
            update_callback(step, detail, status)
            status_container.info(f"**{step}**: {detail}")
            render_logs()
            # time.sleep(0.1) # UI visual pacing
        
        # Run
        with st.spinner("🤖 Agents orchestrating..."):
            final_report = st.session_state.orchestrator.research(query, callback=ui_callback)
        
        # Success
        status_container.success("✅ Analysis Complete")
        
        # Render Report
        with report_container:
            st.markdown("## 📄 Final Research Report")
            st.markdown(f"<div class='report-font'>{final_report}</div>", unsafe_allow_html=True)
            
    except Exception as e:
        status_container.error(f"❌ Critical Error: {str(e)}")
        st.exception(e)

else:
    with left_col:
        st.info("👋 Ready to research. Enter a query above.")
        
    with right_col:
        st.markdown("### 🧠 System Capabilities")
        st.markdown("""
        **1. Auto-Seeding Graph DB**
        - Detects company from query (e.g., "Apple" -> AAPL)
        - Wipes & re-seeds Neo4j with specific company data
        
        **2. Deep Reader Agent (CRAG)**
        - Retrieval: Hybrid (Vector + Graph)
        - Ingestion: Semantic Chunking (Atomic Propositions)
        - Evaluation: Self-Corrective RAG
        
        **3. Web Search Agent**
        - Live market data & news
        - Fallback for missing 10-K data
        """)
