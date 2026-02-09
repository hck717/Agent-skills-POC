# Agent-skills-POC

**Multi-agent equity research system with ReAct (Reasoning + Acting) orchestration.**

## 🎯 Quick Start

### ReAct-Based Multi-Agent System (Recommended)

```bash
# 1. Setup environment
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API keys
export PERPLEXITY_API_KEY="your-key"    # For ReAct orchestrator
export EODHD_API_KEY="your-key"         # Optional, for market data

# 3. Start Ollama (for Business Analyst agent)
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. Run ReAct orchestrator
python main_orchestrated.py
```

### Single-Agent Mode (Business Analyst Only)

```bash
python main.py
```

## 🔄 What is ReAct?

**ReAct (Reasoning + Acting)** is an iterative framework where the orchestrator:

1. **Thinks** 💭 - Reasons about what to do next
2. **Acts** ⚡ - Executes specialist agents  
3. **Observes** 👁️ - Analyzes results
4. **Repeats** 🔁 - Refines strategy based on observations

This enables **dynamic adaptation**, **self-correction**, and **early stopping**.

📚 **See:** [REACT_FRAMEWORK.md](REACT_FRAMEWORK.md) for complete documentation.

## 📁 Project Structure

```
Agent-skills-POC/
├── main_orchestrated.py             # 🔥 ReAct multi-agent entry point
├── orchestrator_react.py            # ReAct orchestration engine
├── orchestrator.py                  # Legacy planner
├── main.py                          # Single agent mode
│
├── REACT_FRAMEWORK.md               # 📚 ReAct guide
├── SPECIALIST_AGENTS.md             # Agent specifications
├── ORCHESTRATOR_README.md           # Legacy docs
│
├── skills/business_analyst/         # ✅ Implemented specialist
├── prompts/                         # Persona templates
├── data/                            # PDF storage
└── storage/chroma_db/               # Vector DB
```

## 🏗️ ReAct Architecture

```
User Query
    ↓
╭─────────────────────────────────────╮
│     ReAct Loop (max 5 iterations)   │
│                                     │
│  Iteration 1:                       │
│    💭 Thought → ⚡ Action → 👁️ Observation │
│                                     │
│  Iteration 2:                       │  
│    💭 Thought → ⚡ Action → 👁️ Observation │
│                                     │
│  ... (adapts based on results)      │
│                                     │
│  Iteration N:                       │
│    💭 "Sufficient" → 🏁 Finish        │
╰─────────────────────────────────────╯
    ↓
[Synthesis]
    ↓
Final Report + Trace
```

**Key Advantages:**
- ✅ Adaptive - Changes strategy based on observations
- ✅ Efficient - Stops early when sufficient
- ✅ Self-correcting - Calls additional agents if needed  
- ✅ Transparent - Full reasoning trace

## 🤖 The 6 Specialist Agents

| Agent | Status | Capabilities |
|-------|--------|-------------|
| **Business Analyst** | ✅ | 10-K analysis, risk assessment, competitive intel |
| **Quantitative Analyst** | 📋 | Financial ratios, DCF, trend forecasting |
| **Market Analyst** | 📋 | Sentiment, technicals, price data |
| **Industry Analyst** | 📋 | Sector trends, peer comparison |
| **ESG Analyst** | 📋 | ESG scoring, sustainability |
| **Macro Analyst** | 📋 | Economic indicators, FX exposure |

## 🚀 Usage

### ReAct Orchestration

```python
from orchestrator_react import ReActOrchestrator

orchestrator = ReActOrchestrator(max_iterations=5)

# Register specialists
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
business_analyst = BusinessAnalystGraphAgent()
orchestrator.register_specialist("business_analyst", business_analyst)

# Execute research
report = orchestrator.research(
    "Analyze Apple's competitive risks and profit margins"
)

print(report)
print(orchestrator.get_trace_summary())  # View reasoning
```

**Output:**
```
💭 [THOUGHT 1] Need qualitative risks AND quantitative margins
⚡ [ACTION 1] call_specialist → business_analyst
👁️ [OBSERVATION 1] Extracted 5 competitive risks...

💭 [THOUGHT 2] Have risks, need margin calculations
⚡ [ACTION 2] call_specialist → quantitative_analyst  
👁️ [OBSERVATION 2] Net margin 25.3%, Operating 30.1%...

💭 [THOUGHT 3] Sufficient information gathered
⚡ [ACTION 3] finish
```

## 📊 ReAct vs Traditional

| Feature | Traditional | ReAct |
|---------|-------------|-------|
| Planning | One-shot | Iterative |
| Adaptation | ❌ No | ✅ Yes |
| Self-correct | ❌ No | ✅ Yes |
| Early stop | ❌ No | ✅ Yes |
| Transparency | Limited | Full trace |
| Efficiency | Fixed | Variable (2-5 iter) |

**Example:** Query "What does Apple do?"

- **Traditional:** Calls 3-4 agents (overkill)
- **ReAct:** 2 iterations → Business Analyst → Finish
- **Result:** 2x faster

## 🧠 Key Features

### Business Analyst (✅ Implemented)
- ReAct loop with LangGraph
- ChromaDB + BERT reranking
- Persona-based analysis
- Page-level citations

### ReAct Orchestration (✅ Implemented)
- Iterative reasoning: Think → Act → Observe
- Dynamic agent selection
- Self-correction capabilities
- Early stopping optimization
- Complete reasoning trace
- Context-aware synthesis

## 🛠️ Tech Stack

**Core:** LangGraph, LangChain, Ollama (Qwen 2.5), ChromaDB, Perplexity API

**ML/NLP:** BERT Cross-Encoder, Nomic Embeddings, Sentence Transformers

**Data:** PyPDF, Pandas, EODHD API

## 📈 Performance

- **Single Agent:** ~15-30s
- **ReAct Simple (1-2 agents):** ~30-45s
- **ReAct Complex (3-4 agents):** ~50-70s
- **Per iteration:** ~8-12s
- **Synthesis:** ~10-15s

**Efficiency:** ReAct saves ~40% on simple queries via early stopping

## 📚 Documentation

- **[REACT_FRAMEWORK.md](REACT_FRAMEWORK.md)** - Complete ReAct guide
- **[SPECIALIST_AGENTS.md](SPECIALIST_AGENTS.md)** - Agent specs
- **[ORCHESTRATOR_README.md](ORCHESTRATOR_README.md)** - Legacy docs

## 🔧 Commands

In `main_orchestrated.py`:

- Normal query - Ask research questions
- `trace` - Show ReAct reasoning from last query
- `ingest` - Process documents in `/data`
- `quit` - Exit

## 🗺️ Roadmap

- [x] Business Analyst (RAG + Reranking)
- [x] Multi-agent orchestration
- [x] **ReAct framework** 🎉
- [ ] Quantitative Analyst
- [ ] Market Analyst (real-time)
- [ ] Industry Analyst (web search)
- [ ] ESG Analyst
- [ ] Macro Analyst
- [ ] Parallel execution
- [ ] Multi-turn memory
- [ ] Cost tracking

## 💡 Why ReAct?

### Traditional
```python
plan = planner.plan(query)  # Fixed
results = execute_all(plan)  # Cannot adapt
```

### ReAct
```python
while not done:
    thought = reason(query, history)
    action = decide(thought)
    result = execute(action)
    
    if sufficient(history):
        done = True  # Early stop
```

**Benefits:** Adapts, self-corrects, efficient

## 🎓 Learning Path

1. **ReAct Framework** - Iterative reasoning
2. **Multi-Agent Systems** - Orchestration
3. **Agentic RAG** - Advanced retrieval
4. **LangGraph** - Stateful workflows
5. **Hybrid LLMs** - Local + Cloud

## 📝 Design Philosophy

從單一 Agent 嘅「直線流程」升級到 **ReAct Loop** 真正識思考，再加 Multi-Agent Orchestration 模擬完整 Research Team：ReAct Orchestrator 做 Project Manager，各 Specialist 做專家，Synthesizer 寫 Final Report。

## 📄 License

MIT
