# Agent-skills-POC

Multi-agent equity research system with intelligent orchestration.

## 🎯 Quick Start

### Single-Agent Mode (Business Analyst)

```bash
# 1. Setup environment
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API keys
export EODHD_API_KEY=""

# 3. Start Ollama
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. Run single agent
python main.py
```

### Multi-Agent Orchestration Mode

```bash
# Additional setup for orchestrator
export PERPLEXITY_API_KEY="your-key"

# Run orchestrated system
python main_orchestrated.py
```

## 📁 Project Structure

```
Agent-skills-POC/
├── main.py                          # Single Business Analyst agent
├── main_orchestrated.py             # Multi-agent orchestration entry point
├── orchestrator.py                  # Planner & Synthesis agents
├── SPECIALIST_AGENTS.md             # Detailed agent specifications
├── ORCHESTRATOR_README.md           # Full orchestration documentation
│
├── skills/
│   └── business_analyst/
│       ├── graph_agent.py           # ✅ Implemented: RAG + LangGraph
│       └── agent.py
│
├── prompts/                         # Persona templates
│   ├── chief_strategy_officer.md
│   ├── competitive_intel.md
│   ├── risk_officer.md
│   └── ...
│
├── data/                            # PDF storage (10-Ks by ticker)
└── storage/chroma_db/               # Vector database
```

## 🤖 Architecture

### Two Modes of Operation

#### Mode 1: Single Specialist Agent
Direct interaction with Business Analyst for 10-K analysis.

```
User → Business Analyst → RAG + Reranking → LLM → Response
```

#### Mode 2: Multi-Agent Orchestration
Intelligent coordination of 6 specialist agents.

```
User Query
    ↓
[Planner Agent] ──→ Selects & tasks specialist agents
    ↓
┌───────────────────────────────────────────────┐
│  Business Analyst  │  Quantitative Analyst    │
│  Market Analyst    │  Industry Analyst        │
│  ESG Analyst       │  Macro Analyst           │
└───────────────────────────────────────────────┘
    ↓
[Synthesis Agent] ──→ Combines insights
    ↓
Final Report
```

## 🔧 The 6 Specialist Agents

| Agent | Status | Capabilities |
|-------|--------|-------------|
| **Business Analyst** | ✅ Implemented | 10-K analysis, risk assessment, competitive intelligence |
| **Quantitative Analyst** | 📋 Planned | Financial ratios, DCF valuation, trend forecasting |
| **Market Analyst** | 📋 Planned | Sentiment analysis, technical indicators, price data |
| **Industry Analyst** | 📋 Planned | Sector trends, peer comparison, regulatory analysis |
| **ESG Analyst** | 📋 Planned | ESG scoring, sustainability, governance evaluation |
| **Macro Analyst** | 📋 Planned | Economic indicators, rate sensitivity, FX exposure |

## 📚 Documentation

- **[SPECIALIST_AGENTS.md](SPECIALIST_AGENTS.md)** - Detailed specifications for each agent (helps Planner make better decisions)
- **[ORCHESTRATOR_README.md](ORCHESTRATOR_README.md)** - Complete orchestration system guide
- **[skills/business_analyst/SKILL.md](skills/business_analyst/SKILL.md)** - Business Analyst implementation details

## 🧠 Key Features

### Business Analyst (Implemented)
- **ReAct Loop Architecture**: LangGraph-based reasoning and action cycle
- **Advanced RAG**: ChromaDB + BERT Cross-Encoder reranking
- **Persona-Based Analysis**: Auto-selects analyst persona (Strategy, Risk, Competitive)
- **Citation Tracking**: Page-level source attribution

### Orchestration System (Implemented)
- **Intelligent Planning**: Perplexity-powered agent selection
- **Dynamic Task Assignment**: Specific tasks for each specialist
- **Smart Synthesis**: Combines multi-agent outputs into coherent reports
- **Extensible Design**: Easy to add new specialist agents

## 🚀 Usage Examples

### Single Agent
```python
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent

analyst = BusinessAnalystGraphAgent()
analyst.ingest_data()  # Process PDFs
result = analyst.analyze("What are Apple's key competitive risks?")
```

### Multi-Agent Orchestration
```python
from orchestrator import EquityResearchOrchestrator

orchestrator = EquityResearchOrchestrator()
report = orchestrator.research(
    "Compare Apple and Microsoft's profit margins and competitive positioning"
)
# Automatically deploys Business Analyst + Quantitative Analyst
```

## 🔄 Workflow Comparison

### Old Architecture (v1)
```
Query → Search → Answer (Single-pass)
```

### New Architecture (v2)
```
Query → Plan → Execute Multi-Agents → Synthesize → Report
       ↓
   [Planner decides which experts to consult]
```

## 🛠️ Tech Stack

**Core:**
- **LangGraph** - Agent workflow orchestration
- **LangChain** - LLM framework
- **Ollama** - Local LLM inference (Qwen 2.5)
- **ChromaDB** - Vector storage
- **Perplexity API** - Planner & Synthesis agents

**ML/NLP:**
- BERT Cross-Encoder (Reranking)
- Nomic Embed Text (Embeddings)
- Sentence Transformers

**Data:**
- PyPDF (Document loading)
- Pandas (Data analysis)
- EODHD API (Market data)

## 📊 Performance

- **Single Agent**: ~15-30 seconds (RAG + local LLM)
- **Multi-Agent (2-3 agents)**: ~40-60 seconds
- **Planner overhead**: ~5-10 seconds
- **Synthesis overhead**: ~10-15 seconds

## 🎓 Learning Path

This project demonstrates:
1. **Agentic RAG** - Beyond simple retrieval
2. **Multi-Agent Systems** - Orchestration patterns
3. **ReAct Loops** - Reasoning + Acting cycles
4. **LangGraph** - Stateful agent workflows
5. **Hybrid Architectures** - Local + Cloud LLMs

## 🔮 Roadmap

- [x] Business Analyst with RAG + Reranking
- [x] Multi-agent orchestration framework
- [x] Planner & Synthesis agents
- [ ] Implement Quantitative Analyst
- [ ] Implement Market Analyst (real-time data)
- [ ] Implement Industry Analyst (web search)
- [ ] Implement ESG Analyst
- [ ] Implement Macro Analyst
- [ ] Parallel agent execution
- [ ] Agent memory for multi-turn conversations
- [ ] Cost tracking and optimization

## 📝 Notes

### Why Multi-Agent?
- **Specialization**: Domain experts > generalists
- **Scalability**: Parallel execution + independent development
- **Accuracy**: Cross-validated insights from multiple perspectives
- **Flexibility**: Dynamic agent selection per query

### Design Philosophy
從單一 Agent 嘅「直線流程」升級到真正識思考嘅 **ReAct Loop**，而家再加埋 Multi-Agent Orchestration，模擬一個完整嘅 Research Team：Planner 做 Project Manager，各個 Specialist 做專家，Synthesizer 做 Senior Analyst 寫 Final Report。

## 📄 License

MIT
