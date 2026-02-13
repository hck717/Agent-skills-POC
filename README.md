# 🏦 Multi-Agent Autonomous Investment Research Platform

> **From SEC Filings to Institutional-Grade Research: A Modular AI System for Equity Analysis**

## 🎯 Vision: The "Lean 6" Autonomous Investment Team

This platform is evolving toward a **6-agent autonomous research system** that replicates the workflow of a professional investment team, combining deep document analysis, quantitative forensics, network intelligence, macroeconomic context, behavioral signals, and real-time awareness.

### The Complete Architecture (Target)

| Agent | Role | RAG Strategy | Status |
|-------|------|--------------|--------|
| **1. Business Analyst** | Deep Reader | Graph-Augmented CRAG with Proposition Chunking | ✅ **Implemented (4 variants)** |
| **2. Quantitative Fundamental** | Math Auditor + Quant | Chain-of-Table with Dual-Path Verification | 🚧 **Planned** |
| **3. Supply Chain Graph** | Network Detective | GraphRAG with Centrality Detection | 🚧 **Planned** |
| **4. Macro Economic** | The Economist | Text-to-SQL + Time-Series RAG | 🚧 **Planned** |
| **5. Insider & Sentiment** | The Psychologist | Temporal Contrastive RAG | 🚧 **Planned** |
| **6. Web Search** | News Desk | HyDE + Step-Back Prompting + Reranking | ✅ **Implemented** |

***

## 🚀 Current Implementation (v3.6)

### What's Working Today

**✅ Core Orchestration**
- **ReAct Framework**: Rule-based multi-agent coordination with iterative reasoning loops
- **Hybrid LLM Strategy**: DeepSeek-R1 8B (analysis) + Qwen 2.5 7B (synthesis backup)
- **Senior PM Persona**: Investment memo generation with conviction-based insights
- **Auto-Ingestion**: Automatic document processing and knowledge graph seeding

**✅ Business Analyst Agent (4 RAG Variants)**
1. **Standard RAG** (`business_analyst_standard/`) - Baseline vector search
2. **Corrective RAG (CRAG)** (`business_analyst_crag/`) - Quality evaluation with web fallback
3. **Self-RAG** (`business_analyst_selfrag/`) - Self-reflection and adaptive retrieval
4. **GraphRAG** (`business_analyst_graphrag/`) - Neo4j knowledge graph integration

**✅ Web Search Agent**
- Real-time market intelligence via Tavily API
- Temporal context enhancement (adds "2026", "latest", "Q1" to queries)
- Citation preservation with strict temperature control (0.0)
- Automatic fallback injection when LLM fails to cite

**✅ Professional UI (Streamlit)**
- Real-time execution metrics (iterations, duration, specialists called)
- Interactive ReAct trace visualization
- Quality scoring and citation coverage analysis
- Markdown report export

***

## 📊 System Architecture (Current)

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
│          "What are Microsoft's AI risks?"                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│             ReAct Orchestrator (v3.6)                       │
│  • Rule-based reasoning (max 3 iterations)                  │
│  • Auto-ticker extraction + metadata parsing                │
│  • Hybrid synthesis (DeepSeek → Qwen fallback)             │
│  • Senior PM persona synthesis                              │
└──────┬────────────────────────────────────┬─────────────────┘
       │                                    │
       │  Iteration 1: Rule 1               │  Iteration 2: Rule 2
       ▼                                    ▼
┌──────────────────────┐           ┌──────────────────────┐
│  Business Analyst    │           │  Web Search Agent    │
│  (CRAG Variant)      │           │                      │
│  ──────────────────  │           │  ──────────────────  │
│  • Neo4j Graph       │           │  • Tavily API        │
│  • ChromaDB Vector   │           │  • Real-time News    │
│  • BERT Reranking    │           │  • Temporal Context  │
│  • CRAG Evaluator    │──Fallback─→  • Citation Strict   │
│  • 10-K Citations    │    ↓      │  • URL Citations     │
│                      │  Web Agent│                      │
│  MODEL:              │           │  MODEL:              │
│  DeepSeek-R1 8B      │           │  DeepSeek-R1 8B      │
│                      │           │                      │
│  Sources: [1-7]      │           │  Sources: [8-12]     │
└──────────────────────┘           └──────────────────────┘
       │                                    │
       └──────────┬────────────────────────┘
                  │
                  │  Iteration 3: finish
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              SYNTHESIS ENGINE (Hybrid)                      │
│  MODEL: DeepSeek-R1 8B (Primary) / Qwen 2.5 7B (Backup)   │
│  ─────────────────────────────────────────────────────────  │
│  • Merges document (1-7) + web (8-12) sources              │
│  • Senior PM investment memo structure                      │
│  • Enforces citation coverage [X] format                    │
│  • Dynamic structure based on query type                    │
│  • Duration: 30-60s (synthesis only)                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│             PROFESSIONAL INVESTMENT MEMO                    │
│  ───────────────────────────────────────────────────────── │
│  • The Setup: Market consensus vs. reality [cited]          │
│  • The Edge/Thesis: Key drivers market misses [8+ cites]   │
│  • The Bear Case: What kills the trade [cited]             │
│  • Catalyst Path: Events to watch [cited]                   │
│  • References: All sources with clean formatting           │
│  ───────────────────────────────────────────────────────── │
│  Duration: 1.5-2.5 min | Citations: 30-50 | Quality: 75-85│
└─────────────────────────────────────────────────────────────┘
```

***

## 🏗️ Technology Stack

### Current Infrastructure

| Component | Technology | Purpose | Status |
|-----------|------------|---------|--------|
| **Vector DB** | ChromaDB | 10-K/10-Q document embeddings | ✅ Production |
| **Knowledge Graph** | Neo4j | Structured entity relationships | ✅ Production |
| **Analysis LLM** | Ollama (DeepSeek-R1 8B) | Deep financial reasoning | ✅ Production |
| **Synthesis LLM** | Ollama (Qwen 2.5 7B) | Fast report generation | ✅ Production |
| **Embeddings** | nomic-embed-text | Vector search | ✅ Production |
| **Reranking** | sentence-transformers BERT | Relevance scoring | ✅ Production |
| **Web Search** | Tavily API | Real-time intelligence | ✅ Production |
| **UI** | Streamlit | Interactive dashboard | ✅ Production |

### Planned Infrastructure (Phase 2-4)

| Component | Technology | Purpose | Timeline |
|-----------|------------|---------|----------|
| **Vector DB** | Qdrant | Central bank docs + propositions | Phase 2 |
| **Structured DB** | Postgres + TimescaleDB | Financial statements + insider trades | Phase 2 |
| **OLAP DB** | DuckDB | 30+ years pricing + factor analysis | Phase 2 |
| **Data Pipeline** | Airflow | Automated multi-source ingestion | Phase 2 |
| **Data Sources** | EODHD API + FMP API | 30+ years fundamentals, macro, EOD prices | Phase 2 |

***

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** - For local LLMs ([Download](https://ollama.ai/))
- **Docker** (Optional) - For Neo4j graph database
- **10-K PDFs** - SEC filings in `data/{TICKER}/` folders

### 1. Clone & Install

```bash
git clone https://github.com/hck717/Agent-skills-POC.git
cd Agent-skills-POC

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama Models

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull required models
ollama pull deepseek-r1:8b      # Analysis model (5.0 GB)
ollama pull qwen2.5:7b           # Synthesis backup (4.7 GB)
ollama pull nomic-embed-text     # Embeddings (274 MB)
```

**💡 Model Strategy:**
- **DeepSeek-R1 8B**: Superior reasoning for 10-K analysis + synthesis (primary)
- **Qwen 2.5 7B**: 10x faster backup for synthesis timeouts
- **Result**: Best quality + reliability

### 3. Setup Neo4j (Optional but Recommended)

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Access browser: http://localhost:7474
# Credentials: neo4j / password
```

### 4. Add Your Data

```bash
# Structure your 10-K filings
data/
├── AAPL/
│   └── APPL 10-k Filings.pdf
├── MSFT/
│   └── MSFT 10-K 2024.pdf
└── TSLA/
    └── TSLA 10-K 2024.pdf
```

### 5. Configure Environment (Optional)

```bash
# Create .env file
cat > .env << EOF
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Tavily API (for web search)
TAVILY_API_KEY=your-api-key-here
EOF
```

### 6. Launch

```bash
# Option 1: Interactive UI
streamlit run app.py

# Option 2: Command-line testing
python orchestrator_react.py
```

🎉 **Open browser at `http://localhost:8501`**

***

## 📖 Usage Examples

### Example 1: Risk Analysis
```
Query: "What are Apple's supply chain risks from their 2025 10-K?"

Output Structure:
- The Setup: Apple's reported supply chain concentration (67% China)
- The Edge: Hidden dependencies market overlooks (chip packaging bottleneck)
- Bear Case: Geopolitical escalation scenarios
- Catalysts: Taiwan tension indicators, supplier diversification timeline
```

### Example 2: Investment Thesis
```
Query: "Analyze Microsoft's AI monetization strategy"

Output Structure:
- Current Consensus: Azure AI growth expectations
- Our View: Copilot adoption curve vs. pricing power (market underestimates margin expansion)
- Risks: Open-source LLM commoditization
- Catalysts: Q2 2026 Copilot revenue disclosure
```

### Example 3: Quick Facts
```
Query: "What is Tesla's gross margin trend?"

Output: Direct answer with citations from 10-K + recent earnings calls
```

***

## 🧠 Business Analyst RAG Variants

### 1. Standard RAG (`business_analyst_standard/`)
**Architecture**: Classic vector search → LLM synthesis

```python
Query → Embedding → ChromaDB (Top 50) → BERT Rerank (Top 10) → LLM Analysis
```

**Use Case**: Baseline performance, fastest execution

### 2. Corrective RAG - CRAG (`business_analyst_crag/`)
**Architecture**: Quality evaluation + adaptive refinement + web fallback

```python
Query → Retrieval → CRAG Evaluator (0-1 score)
  ├─ Score > 0.7: High confidence → Use docs
  ├─ Score 0.5-0.7: Ambiguous → Refine query, retry
  └─ Score < 0.5: Low confidence → Trigger Web Search Agent
```

**Use Case**: Zero hallucinations, production quality (recommended)

**Key Innovation**: Automatic fallback chain ensures comprehensive coverage

### 3. Self-RAG (`business_analyst_selfrag/`)
**Architecture**: Self-reflection with adaptive retrieval

```python
Query → Generate Draft → Self-Critique → Retrieve Additional Context → Refine
```

**Use Case**: Complex multi-hop reasoning, evolving queries

### 4. GraphRAG (`business_analyst_graphrag/`)
**Architecture**: Neo4j knowledge graph + vector search

```python
Query → Vector Search (semantic) + Cypher Queries (structural) → Combined Context
```

**Use Case**: Entity relationships, network analysis (e.g., "How are Apple's suppliers connected?")

**Current State**: Basic implementation, will be enhanced in Phase 3 (Supply Chain Graph Agent)

***

## 🎨 Streamlit UI Features

### Dashboard Components

1. **Query Interface**
   - Natural language input
   - Ticker auto-detection
   - Metadata extraction (years, topics)

2. **Real-Time Metrics**
   - Iterations completed
   - Total duration
   - Specialists called
   - Time per iteration

3. **Report Viewer**
   - Markdown rendering
   - Clickable citations [X]
   - Clean reference formatting

4. **ReAct Trace (Expandable)**
   ```
   🧠 Iteration 1: Rule-Based Reasoning
      └─ Action: Call business_analyst_crag (Ticker: MSFT)
         └─ Observation: Retrieved 10-K Risk Factors... (1,247 chars)
   
   🧠 Iteration 2: Rule-Based Reasoning
      └─ Action: Call web_search_agent (Ticker: MSFT)
         └─ Observation: Found 5 recent articles... (892 chars)
   
   🧠 Iteration 3: Rule-Based Reasoning
      └─ Action: FINISH (Analysis complete)
   ```

5. **Export Options**
   - Download as Markdown (.md)
   - Copy to clipboard

***

## 📁 Repository Structure

```
Agent-skills-POC/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
│
├── 🎨 app.py                       # Streamlit UI (main entry point)
├── 🧠 orchestrator_react.py        # ReAct orchestrator (v3.6)
│
├── 🤖 skills/                      # Specialist agents
│   ├── business_analyst_standard/  # Baseline RAG
│   ├── business_analyst_crag/      # Corrective RAG (recommended)
│   ├── business_analyst_selfrag/   # Self-reflective RAG
│   ├── business_analyst_graphrag/  # Graph-augmented RAG
│   └── web_search_agent/           # Real-time web intelligence
│
├── 📂 scripts/                     # Utility scripts
│   ├── ingest_documents_ba.py      # Document → Neo4j + ChromaDB
│   └── ...
│
├── 📂 prompts/                     # Agent system prompts
│
├── 📂 data/                        # SEC filings (your 10-K PDFs)
│   ├── AAPL/
│   ├── MSFT/
│   ├── TSLA/
│   └── .gitkeep
│
├── 📂 docs/                        # Documentation
│   ├── architecture.md             # Detailed system design
│   ├── lean6_vision.md             # Full 6-agent roadmap
│   └── ...
│
├── 🐳 docker-compose.yml           # Multi-container setup (future)
├── 🐳 Dockerfile.airflow           # Airflow data pipeline (future)
└── 💾 storage/                     # Auto-generated databases
    └── chroma_db/                  # Vector embeddings
```

***

## 🛣️ Roadmap: From POC to Production

### ✅ Phase 1: Core Agents (COMPLETE)
- [x] ReAct orchestration framework
- [x] Business Analyst (4 RAG variants)
- [x] Web Search Agent
- [x] Hybrid LLM synthesis
- [x] Streamlit UI with trace visualization
- [x] Auto-ingestion pipeline

### 🚧 Phase 2: Data Infrastructure (Weeks 1-3)
**Goal**: Multi-database foundation + API integrations

- [ ] Deploy Qdrant (replace/augment ChromaDB)
- [ ] Setup Postgres + TimescaleDB extension
- [ ] Setup DuckDB for OLAP queries
- [ ] Integrate EODHD API (30+ years fundamentals, macro, prices)
- [ ] Integrate FMP API (detailed statements, transcripts, 13F)
- [ ] Configure Airflow data pipelines (using existing Dockerfile)

**Why**: Current system limited to local PDFs. Need automated, comprehensive data.

### 🚧 Phase 3: Quantitative Fundamental Agent (Weeks 4-6)
**Goal**: Math Auditor + Quant Factor Analysis

**New Capabilities**:
- Chain-of-Table reasoning (SELECT → FILTER → AGGREGATE → RANK → OUTLIER_DETECT)
- Dual-path verification (Python vs SQL for accuracy)
- Forensic scoring:
  - Beneish M-score (earnings manipulation detection)
  - Altman Z-score (bankruptcy risk)
  - Piotroski F-score (fundamental quality)
- Factor analysis across 10K+ tickers:
  - Value (P/E, P/B, EV/EBITDA, FCF Yield)
  - Quality (ROE, ROA, Margins, Debt/Equity)
  - Momentum (1M/3M/6M/12M returns)
  - Growth (3Y/5Y CAGR, consistency)
  - Volatility (std dev, beta, Sharpe)

**Tech Stack**: Postgres (normalized statements) + DuckDB (time-series OLAP)

### 🚧 Phase 4: Supply Chain Graph Agent (Weeks 7-9)
**Goal**: Network Detective for hidden dependencies

**New Capabilities**:
- NER extraction from 10-K Item 1 (suppliers, customers, competitors)
- Neo4j graph construction with relationships:
  - SUPPLIES_TO, CUSTOMER_OF, HOLDS, COMPETES_IN, EXPOSED_TO
- Graph algorithms:
  - PageRank (systemically important suppliers)
  - Betweenness Centrality (chokepoint detection)
  - Louvain Community (supply chain clusters)
- Risk detection:
  - Concentration risk (single supplier > 20% COGS)
  - Geographic risk (revenue in unstable regions)
  - Contagion risk (high betweenness + low redundancy)

**Tech Stack**: Enhanced Neo4j (upgrade from current basic GraphRAG)

### 🚧 Phase 5: Macro Economic Agent (Weeks 10-11)
**Goal**: The Economist - Cycle & FX Analysis

**New Capabilities**:
- Economic cycle classification (Expansion, Peak, Contraction, Trough)
  - Based on GDP growth, unemployment, yield curve slope, ISM PMI
- FX & interest rate impact modeling
  - Interest rate differential (Fed vs ECB vs BOJ)
  - Carry trade detection
  - Company-specific currency exposure analysis
- Central bank RAG (FOMC minutes, ECB/BOJ statements)
  - Tone scoring (hawkish vs dovish)

**Tech Stack**: DuckDB (macro time-series 1960+) + Qdrant (central bank docs)

### 🚧 Phase 6: Insider & Sentiment Agent (Weeks 12-13)
**Goal**: The Psychologist - Words vs Actions

**New Capabilities**:
- Temporal Contrastive RAG (Q4 2025 guidance vs Q3 2025)
- Vector difference analysis (semantic drift detection)
- Cross-modal divergence:
  - Bullish guidance + insider selling = Red flag
  - Negative drift + insider buying = Contrarian signal
- Conviction scoring: (1 - similarity) × insider_buy_ratio × volume_percentile

**Tech Stack**: Postgres TimescaleDB (Form 4 data) + Qdrant (earnings transcripts)

### 🚧 Phase 7: Advanced RAG Techniques (Weeks 14-15)
**Goal**: Enhance retrieval quality across all agents

**Upgrades**:
- Proposition-based chunking (25-30% better context preservation)
- Hybrid search (Vector + BM25 + Graph Cypher)
- HyDE for Web Search (Hypothetical Document Embeddings)
- Step-Back Prompting (broaden context)
- Cohere/BGE Reranking (filter noise)

### 🚧 Phase 8: Multi-Agent Consensus (Weeks 16-17)
**Goal**: Orchestrator 2.0 - Conflict Resolution

**New Logic**:
- Agent Consensus Matrix (Signal × Confidence scoring)
- 5 Conflict Resolution Patterns:
  1. Growth vs Quality (forensic scores override narrative)
  2. Momentum vs Insider Conviction (contrarian signals)
  3. Concentration Risk vs Diversification Progress
  4. Optimistic Guidance vs Macro Headwinds
  5. Strong Fundamentals vs Supply Chain Vulnerability
- Dynamic synthesis: Final recommendation with conviction level

***

## 🔧 Configuration

### Environment Variables

```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

TAVILY_API_KEY=your-tavily-key          # Web search
EODHD_API_KEY=your-eodhd-key            # Phase 2
FMP_API_KEY=your-fmp-key                # Phase 2

OLLAMA_URL=http://localhost:11434
```

### Orchestrator Settings

```python
# orchestrator_react.py (v3.6)

# Model configuration
ANALYSIS_MODEL = "deepseek-r1:8b"   # For specialist analysis
SYNTHESIS_MODEL = "deepseek-r1:8b"   # For final report synthesis

# Iteration limits
max_iterations = 3                   # Default orchestration depth

# Synthesis parameters
temperature = 0.3                    # Senior PM synthesis creativity
num_predict = 4000                   # Token limit for comprehensive reports
```

### Business Analyst Settings

```python
# skills/business_analyst_crag/

# RAG parameters
top_k_retrieval = 50         # Initial vector search results
top_k_rerank = 10           # After BERT reranking
model = "deepseek-r1:8b"    # Analysis model
temperature = 0.2           # Analysis temperature

# CRAG thresholds
high_confidence = 0.7       # Use docs directly
low_confidence = 0.5        # Trigger web fallback
ambiguous_range = (0.5, 0.7) # Refine query and retry
```

***

## 📊 Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 12GB | 16GB+ |
| CPU | 4 cores | 8+ cores (Apple Silicon ideal) |
| Storage | 15GB | 25GB+ |
| GPU | None | Apple Silicon or NVIDIA (auto-detected) |

**💡 Note**: System loads one model at a time (8-10GB RAM peak)

### Speed Benchmarks (v3.6)

| Task | Duration | Model | Notes |
|------|----------|-------|-------|
| Ticker Extraction | 2-5s | DeepSeek-R1 8B | JSON parsing |
| Auto-Ingestion | 30-60s | nomic-embed | One-time per 10-K |
| Business Analyst Call | 60-90s | DeepSeek-R1 8B | RAG + Neo4j + LLM |
| Web Search Agent Call | 30-45s | DeepSeek-R1 8B | Tavily + synthesis |
| Final Synthesis | 30-60s | DeepSeek-R1 8B | Senior PM memo |
| **Total Query** | **1.5-2.5 min** | Hybrid | End-to-end |

### Quality Metrics (Current)

| Metric | Target | Typical Output |
|--------|--------|----------------|
| Citation Coverage | 95%+ | 80-90% |
| Citations per Report | 30+ | 30-50 |
| Investment Thesis Citations | 8+ | 10-15 |
| Generation Time | <3 min | 1.5-2.5 min |
| ReAct Iterations | 2-3 | 2 (fixed rules) |

***

## 🧪 Testing

### Quick System Check

```bash
# 1. Verify Ollama models
ollama list
# Expected: deepseek-r1:8b, qwen2.5:7b, nomic-embed-text

# 2. Test Neo4j connection
# Open http://localhost:7474 (should see browser interface)

# 3. Test orchestrator
python orchestrator_react.py
# Try: "What are Apple's key products?"

# 4. Launch UI
streamlit run app.py
```

### Example Queries by Complexity

**Simple (30-60s):**
```
"What is Microsoft's revenue for 2025?"
"List Tesla's main risk factors"
```

**Medium (1-1.5 min):**
```
"Analyze Apple's competitive risks from their 10-K"
"What is NVIDIA's AI strategy?"
```

**Complex (1.5-2.5 min):**
```
"Based on Microsoft's FY2025 10-K:
1. What are the key AI monetization risks?
2. How has their cloud strategy evolved?
3. What are recent competitive developments?
4. Provide specific page references and market context."
```

***

## 🤝 Contributing

Contributions welcome! Focus areas:

### High Priority
1. **Data Source Integrations** - EODHD, FMP, Bloomberg APIs
2. **New Specialist Agents** - Quantitative, Supply Chain, Macro, Insider
3. **Advanced RAG** - Proposition chunking, hybrid search, HyDE
4. **Quality Improvements** - Citation extraction, fact-checking

### Medium Priority
5. **Performance Optimization** - Parallel agent execution, caching
6. **UI Enhancements** - Chart generation, Excel export, multi-document comparison
7. **Testing** - Unit tests, integration tests, benchmarks

### Documentation
8. **Tutorials** - Agent development guide, RAG technique comparisons
9. **Examples** - Industry-specific analyses, use case studies

**How to Contribute:**
```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and test
python -m pytest tests/

# 4. Submit pull request with detailed description
```

***

## 📚 Documentation

- **[Architecture Deep Dive](docs/architecture.md)** - System design and data flow
- **[Lean 6 Vision](docs/lean6_vision.md)** - Complete 6-agent specification
- **[RAG Techniques Comparison](docs/rag_comparison.md)** - Benchmarks across 4 variants
- **[API Integration Guide](docs/api_integration.md)** - EODHD, FMP setup
- **[Agent Development Guide](docs/agent_development.md)** - How to add new specialists

***

## 🙏 Acknowledgments

- **Ollama** - Local LLM infrastructure that makes this possible
- **DeepSeek** - Superior reasoning models for financial analysis
- **Qwen Team** - Fast, efficient synthesis models
- **LangChain** - Agent orchestration framework
- **ChromaDB** - Vector database foundation
- **Neo4j** - Knowledge graph platform
- **Streamlit** - Rapid UI development
- **Tavily** - High-quality web search API

***

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

***

## 📞 Support & Community

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/hck717/Agent-skills-POC/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/hck717/Agent-skills-POC/discussions)
- 📧 **Contact**: Open an issue for questions

***

## 🎯 Key Differentiators

### Why This System?

1. **Local-First**: No cloud costs, full data privacy
2. **Modular Architecture**: Swap agents, RAG techniques, or LLMs independently
3. **Production-Ready Orchestration**: ReAct framework with rule-based reliability
4. **4 RAG Variants**: Compare Standard, CRAG, Self-RAG, GraphRAG side-by-side
5. **Hybrid LLM Strategy**: DeepSeek quality + Qwen speed backup
6. **Senior PM Persona**: Investment memos, not corporate summaries
7. **Zero Hallucinations**: CRAG evaluation + web fallback chain
8. **Clear Roadmap**: From 2-agent POC to 6-agent autonomous team

***

## 🔮 Vision: The Future (Phase 8)

```
User: "Analyze Microsoft - full team assessment"

[6 Agents Execute in Parallel]

Business Analyst: "Cloud segment 39% margin [1], AI Copilot mentioned 47 times in 10-K [2]"
Quant Fundamental: "Beneish M-score 0.8 (clean) [3], ROE 42% (top decile) [4]"
Supply Chain: "Azure infra concentrated in 3 GPU suppliers (risk) [5]"
Macro: "USD strength reducing international revenue by 4% [6]"
Insider: "CFO purchased $2M shares (high conviction) [7]"
Web Search: "OpenAI partnership extended to 2030 [8]"

[Orchestrator Synthesizes]

## MSFT Investment Memo

**The Setup**: Market pricing 15% AI revenue growth. Reality: Copilot adoption 
accelerating 40% QoQ [8], but margin expansion underestimated (Copilot 70% GM vs 
Azure 39% [1]).

**The Edge**: 
1. Margin Inflection: Copilot mix shift drives 300bps expansion by FY2027 [model]
2. Insider Conviction: CFO buying at $420 (10x normal size) [7] = floor established
3. Quality Screen: Top decile ROE [4] + clean forensics [3] = durable compounder

**Bear Case**:
1. GPU Dependency: 60% Nvidia H100 exposure [5] = supply constraint Q2-Q3 2026
2. FX Headwind: USD strength cutting $4B revenue [6]
3. Competition: Google Gemini pricing 30% below [8]

**Catalysts**:
- Feb 28: FY Q2 earnings (watch Copilot revenue disclosure)
- March 15: GPU supply agreement renewal (check diversification)
- April 1: Azure price increase (margin test)

**Conviction: STRONG BUY | Target: $485 | Stop: $395**

[All 8 citations auto-formatted with clickable links]
```

***

**Built with ❤️ for institutional-grade equity research**

⭐ **Star this repo if you're building the future of AI-powered investing!**

***

## 📈 Version History

- **v3.6** (Current) - Senior PM persona, auto-ingestion, metadata extraction
- **v3.0** - CRAG web fallback chain
- **v2.2** - Hybrid LLM strategy (DeepSeek + Qwen)
- **v2.0** - Business Analyst + Web Search agents
- **v1.0** - Initial ReAct orchestration POC

***

This README reflects your **current production-ready 2-agent system** while clearly outlining the path to your **"Lean 6" vision**. It's honest about what works today and ambitious about what's coming next. Ready to commit to your repo?
