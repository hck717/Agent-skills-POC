# 🔬 Agent-Skills-POC

**Multi-agent equity research system with ReAct (Reasoning + Acting) orchestration.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Quick Start

### Option 1: Streamlit UI (Recommended)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API keys
export PERPLEXITY_API_KEY="your-key"
export EODHD_API_KEY="your-key"  # Optional

# 3. Start Ollama
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. Launch UI
streamlit run app.py
# → Opens at http://localhost:8501
```

### Option 2: CLI with ReAct

```bash
# Same setup, then:
python main_orchestrated.py
```

### Option 3: Single Agent CLI

```bash
python main.py
```

---

## 📁 Project Structure

```
Agent-skills-POC/
├── README.md                    # You are here
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── 🌐 app.py                   # Streamlit UI entry point
├── 🔥 main_orchestrated.py      # ReAct CLI entry point
├── main.py                      # Single agent CLI
│
├── 🧠 orchestrator/            # Orchestration engine
│   ├── __init__.py
│   ├── react.py                 # ReAct framework
│   └── legacy.py                # Legacy planner
│
├── orchestrator_react.py        # ReAct implementation
├── orchestrator.py              # Legacy implementation
│
├── 🤖 skills/                  # Specialist agents
│   └── business_analyst/
│       ├── graph_agent.py       # ✅ Main implementation
│       ├── agent.py
│       └── SKILL.md
│
├── 📖 docs/                    # Documentation
│   ├── REACT_FRAMEWORK.md       # ReAct architecture
│   ├── SPECIALIST_AGENTS.md     # Agent specifications
│   ├── UI_GUIDE.md              # Streamlit guide
│   └── ORCHESTRATOR.md          # Orchestration docs
│
├── 🎭 prompts/                # Persona templates
│   ├── chief_strategy_officer.md
│   ├── competitive_intel.md
│   └── risk_officer.md
│
├── 📂 data/                   # PDF storage (10-Ks by ticker)
└── 💾 storage/                # Vector database
    └── chroma_db/
```

---

## 🔄 What is ReAct?

ReAct (Reasoning + Acting) enables **iterative, adaptive** decision-making:

```
╭───────────────────────────────────╮
│   ReAct Loop (max 5 iterations)   │
│                                   │
│  1. 🧠 Think → What to do next? │
│  2. ⚡ Act → Call specialist agent  │
│  3. 👁️ Observe → Analyze results   │
│  4. 🔁 Repeat → Until sufficient    │
╰───────────────────────────────────╯
```

**Advantages:**
- ✅ **Adaptive** - Changes strategy based on observations
- ✅ **Efficient** - Stops early when sufficient info gathered
- ✅ **Self-correcting** - Can call additional agents if needed
- ✅ **Transparent** - Complete reasoning trace available

📚 **Deep Dive:** [docs/REACT_FRAMEWORK.md](docs/REACT_FRAMEWORK.md)

---

## 🤖 The 6 Specialist Agents

| Agent | Status | Capabilities | Keywords |
|-------|--------|--------------|----------|
| **Business Analyst** | ✅ | 10-K analysis, risk assessment, competitive intel | `10-K`, `risk`, `competitive` |
| **Quantitative Analyst** | 📋 | Financial ratios, DCF, trend forecasting | `calculate`, `ratio`, `DCF` |
| **Market Analyst** | 📋 | Sentiment, technicals, price data | `sentiment`, `price`, `technical` |
| **Industry Analyst** | 📋 | Sector trends, peer comparison | `industry`, `peers`, `sector` |
| **ESG Analyst** | 📋 | ESG scoring, sustainability | `ESG`, `carbon`, `sustainability` |
| **Macro Analyst** | 📋 | Economic indicators, FX exposure | `rates`, `FX`, `geopolitical` |

📚 **Detailed Specs:** [docs/SPECIALIST_AGENTS.md](docs/SPECIALIST_AGENTS.md)

---

## 🚀 Usage Examples

### Streamlit UI

1. Run `streamlit run app.py`
2. Click "🚀 Initialize System"
3. Type query: "What are Apple's competitive risks?"
4. Click "🔍 Analyze"
5. View results + toggle ReAct trace

**Features:**
- 🖱️ Point-and-click interface
- 📊 Real-time metrics (iterations, duration)
- 🔍 Toggle ReAct trace visibility
- 📁 Session history with expand/collapse
- 💾 Download reports as markdown
- ⚙️ Adjustable max iterations

📚 **UI Guide:** [docs/UI_GUIDE.md](docs/UI_GUIDE.md)

### Python API

```python
from orchestrator.react import ReActOrchestrator
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent

# Initialize
orchestrator = ReActOrchestrator(max_iterations=5)

# Register specialists
business_analyst = BusinessAnalystGraphAgent()
orchestrator.register_specialist("business_analyst", business_analyst)

# Execute research
report = orchestrator.research(
    "Analyze Apple's competitive risks and profit margins"
)

print(report)
print(orchestrator.get_trace_summary())  # View reasoning
```

**Output Example:**
```
🧠 [THOUGHT 1] Need qualitative risks AND quantitative margins
⚡ [ACTION 1] call_specialist → business_analyst
👁️ [OBSERVATION 1] Extracted 5 competitive risks...

🧠 [THOUGHT 2] Have risks, need margin calculations
⚡ [ACTION 2] call_specialist → quantitative_analyst
👁️ [OBSERVATION 2] Net margin 25.3%, Operating 30.1%...

🧠 [THOUGHT 3] Sufficient information gathered
⚡ [ACTION 3] finish
```

---

## 🏗️ Architecture

### System Overview

```
╭───────────────────────────────────────────────────────╮
│                    USER LAYER                         │
│                                                       │
│  🌐 Streamlit UI  │  💻 CLI (ReAct)  │  💻 CLI (Single) │
│      (app.py)      │ (main_orchestrated) │   (main.py)    │
╰───────────────────────────────────────────────────────╯
                         │
                         ↓
╭───────────────────────────────────────────────────────╮
│              ORCHESTRATION LAYER                        │
│                                                       │
│  🧠 ReAct Orchestrator (orchestrator/react.py)       │
│  - Iterative reasoning: Think → Act → Observe         │
│  - Dynamic agent selection                             │
│  - Self-correction & early stopping                    │
╰───────────────────────────────────────────────────────╯
                         │
                         ↓
╭───────────────────────────────────────────────────────╮
│               SPECIALIST AGENTS LAYER                   │
│                                                       │
│  🤖 Business Analyst      (skills/business_analyst/)  │
│  📊 Quantitative Analyst  (📋 planned)                 │
│  💹 Market Analyst        (📋 planned)                 │
│  🏗️ Industry Analyst      (📋 planned)                 │
│  🌱 ESG Analyst           (📋 planned)                 │
│  🌍 Macro Analyst         (📋 planned)                 │
╰───────────────────────────────────────────────────────╯
                         │
                         ↓
╭───────────────────────────────────────────────────────╮
│                 DATA LAYER                            │
│                                                       │
│  💾 ChromaDB Vector Store  (storage/chroma_db/)       │
│  📂 PDF Documents         (data/)                      │
│  🎭 Persona Templates      (prompts/)                   │
│  🌐 External APIs         (Perplexity, EODHD)          │
╰───────────────────────────────────────────────────────╯
```

### ReAct vs Traditional

| Feature | Traditional Planner | ReAct Framework |
|---------|--------------------|-----------------|
| **Planning** | One-shot (fixed) | Iterative (adaptive) |
| **Agent Selection** | All predetermined | Dynamic per iteration |
| **Self-Correction** | ❌ No | ✅ Yes |
| **Early Stopping** | ❌ No | ✅ Yes |
| **Reasoning Transparency** | Limited | Full trace available |
| **Efficiency** | Fixed cost | Variable (2-5 iterations avg) |

---

## 🛠️ Tech Stack

### Core
- **LangGraph** - Agent workflow orchestration
- **LangChain** - LLM framework
- **Ollama** - Local LLM inference (Qwen 2.5:7b)
- **ChromaDB** - Vector storage
- **Perplexity API** - ReAct reasoning & synthesis
- **Streamlit** - Web UI

### ML/NLP
- **BERT Cross-Encoder** - Document reranking
- **Nomic Embeddings** - Text embeddings
- **Sentence Transformers** - Similarity search

### Data & APIs
- **PyPDF** - Document loading
- **Pandas** - Data analysis
- **EODHD API** - Market data (optional)

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Single Agent | ~15-30s |
| ReAct Simple (1-2 agents) | ~30-45s |
| ReAct Complex (3-4 agents) | ~50-70s |
| Per Iteration Overhead | ~8-12s |
| Synthesis | ~10-15s |

**Efficiency Gain:** ReAct saves ~40% time on simple queries via early stopping

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/REACT_FRAMEWORK.md](docs/REACT_FRAMEWORK.md) | Complete ReAct architecture guide |
| [docs/SPECIALIST_AGENTS.md](docs/SPECIALIST_AGENTS.md) | Detailed agent specifications |
| [docs/UI_GUIDE.md](docs/UI_GUIDE.md) | Streamlit interface guide |
| [docs/ORCHESTRATOR.md](docs/ORCHESTRATOR.md) | Orchestration system docs |
| [skills/business_analyst/SKILL.md](skills/business_analyst/SKILL.md) | Business Analyst implementation |

---

## 🗺️ Roadmap

### Completed ✅
- [x] Business Analyst with RAG + BERT reranking
- [x] Multi-agent orchestration framework
- [x] **ReAct framework for iterative reasoning**
- [x] **Streamlit web UI**
- [x] ReAct trace visualization
- [x] Session history & download

### In Progress 🚧
- [ ] Quantitative Analyst implementation
- [ ] Market Analyst (real-time data)
- [ ] Industry Analyst (web search)

### Planned 📋
- [ ] ESG Analyst
- [ ] Macro Analyst
- [ ] Parallel agent execution
- [ ] Multi-turn memory system
- [ ] Cost tracking per iteration
- [ ] Chart visualization in UI
- [ ] Agent performance analytics

---

## 🎓 Learning Resources

This project demonstrates:

1. **ReAct Framework** - Iterative reasoning + acting pattern
2. **Multi-Agent Systems** - Coordinating specialist agents
3. **Agentic RAG** - Beyond simple retrieval
4. **LangGraph** - Stateful agent workflows
5. **Hybrid LLMs** - Local (Ollama) + Cloud (Perplexity)
6. **Streamlit** - Interactive data applications

**Academic Reference:**
- [ReAct Paper (Yao et al. 2023)](https://arxiv.org/abs/2210.03629) - *ReAct: Synergizing Reasoning and Acting in Language Models*

---

## 🔧 Development

### Setup Development Environment

```bash
# Clone
git clone https://github.com/hck717/Agent-skills-POC.git
cd Agent-skills-POC

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install
pip install -r requirements.txt

# Set environment variables
export PERPLEXITY_API_KEY="your-key"
export EODHD_API_KEY="your-key"

# Start Ollama
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Adding New Specialist Agents

1. Create agent in `skills/<agent_name>/`
2. Implement `analyze(query)` method
3. Register in `orchestrator_react.py`:
   ```python
   SPECIALIST_AGENTS = {
       "your_agent": {
           "description": "...",
           "capabilities": [...],
           "keywords": [...]
       }
   }
   ```
4. Update `docs/SPECIALIST_AGENTS.md`

### Running Tests

```bash
# Test single agent
python main.py

# Test ReAct orchestration
python main_orchestrated.py

# Test UI
streamlit run app.py
```

---

## ❓ FAQ

**Q: Why ReAct instead of traditional planning?**  
A: ReAct adapts based on intermediate results, self-corrects, and stops early when sufficient info is gathered. Traditional planning commits upfront and cannot adjust.

**Q: Which interface should I use?**  
A: Streamlit UI for demos and exploration. CLI for development and debugging. Python API for integration.

**Q: Can I add my own specialist agents?**  
A: Yes! Follow the development guide above. Agents just need an `analyze(query)` method.

**Q: Do I need all 6 agents implemented?**  
A: No. The system works with any subset. Currently only Business Analyst is implemented.

**Q: Is this production-ready?**  
A: The framework is solid. Business Analyst is production-ready. Other agents are planned.

---

## 💡 Design Philosophy

> 從單一 Agent 嘅「直線流程」升級到 **ReAct Loop** 真正識思考，再加 Multi-Agent Orchestration 模擬完整 Research Team：ReAct Orchestrator 做 Project Manager，各 Specialist 做專家，Synthesizer 寫 Final Report。而家仲有 Streamlit UI 畀人方便用！

Translation: *"Upgraded from single agent 'linear flow' to ReAct Loop with real reasoning, plus Multi-Agent Orchestration simulating a complete Research Team: ReAct Orchestrator as Project Manager, specialists as experts, Synthesizer writing the final report. Now with Streamlit UI for easy use!"*

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

---

## 📧 Contact

Built by [@hck717](https://github.com/hck717)

For questions or suggestions, open an issue on GitHub.

---

**🔬 Built for Transaction Banking & Equity Research**  
**🤖 Powered by ReAct + Multi-Agent Orchestration**  
**🌐 Streamlit UI + Python CLI**
