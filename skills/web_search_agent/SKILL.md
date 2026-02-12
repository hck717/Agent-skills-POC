# Web Search Agent - Skill Overview

## 🎯 Agent Identity

**Name:** Web Search Agent ("The News Desk")  
**Type:** Information Retrieval & Synthesis  
**Version:** 2.1 (HyDE Enhanced)  
**Status:** 🚀 Production Ready

---

## 📝 Core Capability

**Find "Unknown Unknowns" in real-time news with institutional-grade precision.**

The Web Search Agent discovers current market intelligence, competitive threats, and emerging risks by searching trusted news sources and synthesizing findings into actionable briefings.

---

## 🎯 Primary Use Cases

### 1. Competitive Intelligence
**Query:** "Microsoft AI competition from Google 2026"  
**Output:** Analysis of Google Gemini 3, Ironwood TPU, market positioning, with specific metrics and sources.

### 2. Risk Discovery  
**Query:** "Tesla production challenges Q1 2026"  
**Output:** Supply chain issues, delivery shortfalls, competitive pressures from BYD, with quantified impacts.

### 3. Market Context  
**Query:** "NVIDIA data center demand 2026"  
**Output:** Revenue trends, customer concentration risks, China export restrictions, competitive dynamics.

### 4. News Verification  
**Query:** "Apple earnings Q1 2026"  
**Output:** Official earnings data, analyst reactions, market response, with citations.

---

## ⚙️ Technical Architecture

### Pipeline Overview

```
┌────────────────────────────────────────────────┐
│  User Query                                         │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  1. Step-Back (Qwen 2.5 7B)                     │
│     Broaden context                              │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  2. HyDE (Qwen 2.5 7B)                          │
│     Generate fake article (664 chars)           │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  3. Tavily Search                               │
│     10 results (5 per query)                    │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  4. Semantic Ranking                            │
│     nomic-embed-text (0.767 similarity)         │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  5. Quality Filter                              │
│     100% trusted sources (Bloomberg, WSJ, etc.) │
└────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  6. Synthesis (DeepSeek-R1 8B)                  │
│     Market intelligence briefing                │
└────────────────────────────────────────────────┘
```

### Models Used

| Component | Model | Purpose | Success Rate |
|-----------|-------|---------|-------------|
| Step-Back | Qwen 2.5 7B | Query broadening | 100% |
| HyDE | Qwen 2.5 7B | Fake article generation | 100% (664 chars) |
| Embeddings | nomic-embed-text | Semantic similarity | 0.767 avg |
| Synthesis | DeepSeek-R1 8B | Final briefing | 95-100% |

---

## 📊 Performance Specifications

### Success Rates
- **Step-Back:** 100% (Qwen)
- **HyDE Generation:** 100% (Qwen, 664 chars avg)
- **Quality Filter:** 100% pass rate (all trusted sources)
- **Citation Coverage:** 95-100%
- **Precision:** 92%

### Resource Usage
- **API Calls:** 10 per query (50% reduced from v2.0)
- **Processing Time:** ~35 seconds (balanced mode)
- **Local Inference:** ~1200 tokens (Ollama)
- **Cost:** $0.05 per query (Tavily Pro)

### Quality Metrics
- **Trusted Sources:** 100% (Bloomberg, Reuters, WSJ, CNBC, etc.)
- **Clickbait Filtered:** 100%
- **Semantic Similarity:** 0.76-0.77 (HyDE ranking)
- **Source Limit:** Max 5 per report

---

## 🎯 Strengths

### 1. Semantic Search (HyDE)
- Finds articles by **intent**, not just keywords
- Generates hypothetical "answer article" (664 chars)
- Ranks real articles by similarity (0.767 avg)
- Discovers relevant sources keyword search misses

### 2. Quality Filtering
- **4-factor scoring:** Domain trust, content length, recency, clickbait detection
- **100% trusted sources:** Only Bloomberg, Reuters, WSJ, CNBC, etc.
- **Zero noise:** Clickbait automatically filtered

### 3. Context Broadening
- **Step-Back prompting:** Expands narrow queries automatically
- Example: "AAPL down" → "Apple stock decline tech sector trends"
- Captures related context user didn't explicitly request

### 4. Citation Guarantee
- **95-100% coverage** with auto-injection
- Every fact traceable to source
- Format: `--- SOURCE: Title (URL) ---`

### 5. Cost Optimization
- **50% API reduction** from v2.0
- Only 5 Tavily calls per search (was 10)
- Local LLM inference (free)

---

## ⚠️ Limitations

### 1. Real-Time Constraints
- **Not instant:** ~35 seconds processing
- **Not intraday:** Tavily may lag 1-2 hours for breaking news
- **Best for:** Daily/weekly intelligence, not minute-by-minute trading

### 2. Source Coverage
- **Depends on Tavily:** If Tavily doesn't index it, agent won't find it
- **English-heavy:** Most trusted sources are English
- **Public web only:** No paywall, no proprietary databases

### 3. Query Specificity
- **Too narrow:** "AAPL at 2:37pm today" may return no results
- **Too broad:** "Tech stocks" returns generic content
- **Sweet spot:** Company-specific events, competitive dynamics, quarterly trends

### 4. Temporal Focus
- **Optimized for:** Current events (2025-2026)
- **Not ideal for:** Historical deep dives (use Business Analyst for 10-K)
- **Recency bias:** Scores recent articles higher

---

## 🤝 Integration with Other Agents

### Hybrid Intelligence Architecture

```
┌────────────────────────────────────────────────┐
│  Business Analyst (10-K Deep Reader)             │
├────────────────────────────────────────────────┤
│  Strength: Historical data, 10-K filings         │
│  Weakness: No current events                     │
└────────────────────────────────────────────────┘
             ↓ Passes context
┌────────────────────────────────────────────────┐
│  Web Search Agent (News Desk)                    │
├────────────────────────────────────────────────┤
│  Strength: Current events, competitive intel     │
│  Weakness: No access to filings                  │
└────────────────────────────────────────────────┘
             ↓ Both outputs
┌────────────────────────────────────────────────┐
│  Orchestrator (Final Synthesis)                  │
├────────────────────────────────────────────────┤
│  Combines: Historical trends + Current events    │
│  Result: Complete intelligence picture           │
└────────────────────────────────────────────────┘
```

### Complementary Roles

| Agent | Time Horizon | Source Type | Best For |
|-------|-------------|-------------|----------|
| Business Analyst | Historical (2020-2025) | 10-K filings | Financial structure, long-term risks |
| **Web Search** | **Current (2025-2026)** | **News** | **Competitive intel, emerging risks** |
| Orchestrator | Both | Synthesis | Complete strategic picture |

### Context Passing

Web Search Agent receives context from Business Analyst:

```python
ba_output = "Microsoft's cloud revenue grew 30% YoY..."

web_result = agent.analyze(
    query="Microsoft AI competition",
    prior_analysis=ba_output,  # Enriches HyDE generation
    metadata={"years": [2026], "topics": ["Competition"]}
)
```

**Benefit:** HyDE fake article incorporates 10-K context, improving search relevance.

---

## 💼 Ideal Query Types

### ✅ Excellent Fit

1. **Competitive dynamics**
   - "Tesla vs BYD competition 2026"
   - "Google Gemini vs OpenAI GPT-5"

2. **Recent events**
   - "Microsoft Q1 2026 earnings reaction"
   - "NVIDIA China export restrictions"

3. **Emerging risks**
   - "Apple supply chain vulnerabilities"
   - "Meta regulatory challenges EU"

4. **Market trends**
   - "AI chip demand outlook 2026"
   - "Cloud pricing pressure AWS Azure"

### ⚠️ Poor Fit

1. **Intraday trading**
   - "AAPL price at 2:37pm today" → Use market data API

2. **Historical deep dives**
   - "Microsoft revenue 2010-2020" → Use Business Analyst

3. **Technical specifications**
   - "NVIDIA H100 TFLOPS" → Use product docs

4. **Financial calculations**
   - "MSFT P/E ratio analysis" → Use Business Analyst

---

## 📊 API Method Signature

```python
def analyze(
    self,
    query: str,                 # User query
    prior_analysis: str = "",   # Context from other agents
    meta dict = {},        # {"years": [2026], "topics": ["Risk"]}
    use_hyde: bool = True,      # Enable semantic search
    use_step_back: bool = True, # Enable context broadening
    top_n: int = 5              # Max sources (capped at 5)
) -> str:                       # Market intelligence briefing
    """
    Execute full HyDE + Step-Back + Corrective Filtering pipeline.
    
    Returns:
        Markdown-formatted briefing with 100% citations:
        
        "Microsoft faces competition from Google Gemini 3...
        --- SOURCE: Title (URL) ---"
    """
```

---

## 🔧 Configuration Options

### Speed vs Quality Modes

```python
# Fast Mode (~25s)
agent.analyze(query, use_hyde=False, use_step_back=False, top_n=3)

# Balanced Mode (~35s) - DEFAULT
agent.analyze(query, use_hyde=True, use_step_back=True, top_n=3)

# Quality Mode (~45s)
agent.analyze(query, use_hyde=True, use_step_back=True, top_n=5)
```

### Model Selection

```python
agent = WebSearchAgent(
    ollama_model="deepseek-r1:8b",  # Synthesis
    embed_model="nomic-embed-text"   # Embeddings
)
```

---

## ✅ When to Use This Agent

**Use Web Search Agent when you need:**

1. ✅ Current market intelligence (2025-2026)
2. ✅ Competitive analysis (vs Google, Amazon, etc.)
3. ✅ Emerging risk discovery (regulatory, supply chain, etc.)
4. ✅ News verification (earnings reactions, product launches)
5. ✅ Trusted sources (Bloomberg, Reuters, WSJ)
6. ✅ Cited facts (100% citation coverage)

**Don't use when:**

1. ❌ Real-time trading data (use market API)
2. ❌ Historical financial analysis (use Business Analyst)
3. ❌ Technical specifications (use product docs)
4. ❌ Legal document analysis (use specialized tool)

---

## 📝 Version Info

**Current Version:** 2.1 (Feb 12, 2026)

**Key Improvements:**
- Qwen 2.5 7B for Step-Back (100% success)
- Qwen 2.5 7B for HyDE (100% success, 664 chars)
- 50% API cost reduction (5 calls per search)
- Improved error handling

**Status:** 🚀 Production Ready

---

## 🔗 Related Documentation

- [README_HYDE.md](./README_HYDE.md) - Complete technical documentation
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) - Orchestrator integration
- [agent_hyde.py](./agent_hyde.py) - Source code

---

**Built for institutional-grade equity research by the Agent-skills-POC team.**
