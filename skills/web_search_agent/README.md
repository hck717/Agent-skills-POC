# 🌐 Web Search Agent

## Overview

The Web Search Agent supplements document-based analysis with **current web information** using:
- **Tavily API** for web search
- **Local Ollama LLM** for synthesis

### Role in the System

```
┌─────────────────────┐
│  User Query         │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ 1. Business Analyst │  ← Analyzes 10-K documents (local)
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ 2. Web Search Agent │  ← Supplements with current web data
│    (THIS AGENT)     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ 3. Orchestrator     │  ← Synthesizes both sources
│    Synthesis        │
└─────────────────────┘
```

**Key Design Principle:** Web Search Agent **always runs AFTER** Business Analyst to identify and fill information gaps.

---

## Why Web Search Supplement?

### 10-K Documents Are Historical

| Information Type | 10-K Filing | Web Search |
|------------------|-------------|------------|
| **Financial Data** | 6-12 months old | Current |
| **Stock Price** | ❌ Not included | ✅ Real-time |
| **Recent News** | ❌ Outdated | ✅ Latest |
| **Analyst Opinions** | ❌ Not included | ✅ Current consensus |
| **Market Sentiment** | ❌ Not included | ✅ Live data |
| **Competitor Moves** | Limited | ✅ Breaking news |

### Example Use Cases

**Query:** "Analyze Apple's competitive risks"

**Business Analyst Output (10-K):**
- Supply chain concentration in China [Page 23]
- Competition from Samsung and Huawei [Page 45]
- Regulatory risks in Europe [Page 67]

**Web Search Agent Supplement:**
- [Bloomberg] Apple stock down 5% on China iPhone ban reports
- [Reuters] EU antitrust investigation expanded to App Store
- [WSJ] Samsung gains market share in India with budget phones

**Final Synthesis:**
```markdown
## Competitive Risks

### Supply Chain Concentration
Apple's 10-K discloses heavy reliance on Chinese manufacturers [1].
Recent reports indicate potential iPhone restrictions in China [5],
heightening this documented risk [1][5].

References:
[1] APPL 10-k Filings.pdf - Page 23
[5] Bloomberg - https://bloomberg.com/...
```

---

## Setup

### 1. Get Tavily API Key

1. Go to [tavily.com](https://tavily.com)
2. Sign up (free tier: 1,000 searches/month)
3. Copy your API key

### 2. Install Dependencies

```bash
pip install tavily-python ollama
```

### 3. Configure

**Option A: Environment Variable (Recommended)**
```bash
export TAVILY_API_KEY="tvly-xxxxx"
```

**Option B: Streamlit UI**
- Open sidebar
- Enter Tavily API Key in "Web Search (Optional)" section

**Option C: Code**
```python
from skills.web_search_agent.agent import WebSearchAgent

agent = WebSearchAgent(tavily_api_key="tvly-xxxxx")
```

---

## Usage

### Standalone Testing

```python
from skills.web_search_agent.agent import WebSearchAgent

# Initialize
agent = WebSearchAgent(tavily_api_key="tvly-xxxxx")

# Test connection
if agent.test_connection():
    # Analyze with web supplement
    result = agent.analyze(
        query="What are Apple's latest competitive challenges?",
        prior_analysis=""  # Optional: pass Business Analyst output
    )
    print(result)
```

### In Orchestrator (Automatic)

When integrated in the orchestrator, the Web Search Agent:
1. Automatically receives Business Analyst output as context
2. Enhances query to focus on missing information
3. Searches web and synthesizes results
4. Returns analysis with `--- SOURCE: Title (URL) ---` markers

```python
from orchestrator_react import ReActOrchestrator
from skills.web_search_agent.agent import WebSearchAgent

orchestrator = ReActOrchestrator()

# Register agent (orchestrator handles execution order)
web_agent = WebSearchAgent(tavily_api_key="tvly-xxxxx")
orchestrator.register_specialist("web_search_agent", web_agent)

# Web agent will automatically run after Business Analyst
result = orchestrator.research("Analyze Apple's risks")
```

---

## Citation Format

### Output Format

The agent preserves Tavily citations in the same format as document citations:

```markdown
--- SOURCE: Bloomberg (https://bloomberg.com/apple-china-ban) ---
```

### Orchestrator Processing

1. **Web Search Agent Output:**
```
Apple faces new China restrictions.
--- SOURCE: Bloomberg (https://bloomberg.com/...) ---

Analysts downgrade on regulatory concerns.
--- SOURCE: Reuters (https://reuters.com/...) ---
```

2. **Orchestrator Extracts:**
```
[1] APPL 10-k Filings.pdf - Page 23  (from Business Analyst)
[2] Bloomberg - https://bloomberg.com/...  (from Web Search)
[3] Reuters - https://reuters.com/...  (from Web Search)
```

3. **Final Report:**
```markdown
## Risk Analysis

Supply chain risks documented in 10-K [1] have materialized 
with recent China restrictions [2]. Analysts project 15% 
revenue impact [3].

## References
[1] APPL 10-k Filings.pdf - Page 23
[2] Bloomberg - https://bloomberg.com/...
[3] Reuters - https://reuters.com/...
```

---

## Query Enhancement

The agent automatically enhances queries to focus on web-appropriate information:

| Original Query | Enhanced Query |
|----------------|----------------|
| "Apple risks" | "Apple risks recent breaking news 2025 2026" |
| "Competitive landscape" | "Competitive landscape market share analyst ratings recent 2025 2026" |
| "Financial performance" | "Financial performance latest earnings analyst estimates recent 2025 2026" |

This ensures web results complement (not duplicate) document analysis.

---

## Configuration

### Key Parameters

```python
WebSearchAgent(
    tavily_api_key="tvly-xxxxx",  # Required
    ollama_model="qwen2.5:7b"      # Local LLM for synthesis
)
```

### Tavily Search Depth

Currently set to **"advanced"** for detailed financial analysis:

```python
response = self.tavily.search(
    query=query,
    search_depth="advanced",  # More thorough than "basic"
    max_results=5
)
```

---

## Benefits

✅ **Fills Document Gaps** - Supplements historical 10-K data with current info  
✅ **Current Market Data** - Stock prices, analyst ratings, recent news  
✅ **Breaking Developments** - Regulatory changes, competitor moves  
✅ **Unified Citations** - Web sources cited alongside document sources  
✅ **Local Synthesis** - Tavily for search, Ollama for analysis (privacy)  
✅ **Optional** - System works without Tavily key (document-only mode)  

---

## Limitations

⚠️ **API Rate Limits** - Free tier: 1,000 searches/month  
⚠️ **Search Quality** - Depends on Tavily's index and ranking  
⚠️ **Synthesis Time** - Local LLM synthesis adds 30-60s  
⚠️ **Citation Accuracy** - Web sources may be less reliable than 10-Ks  

---

## Troubleshooting

### "Failed to initialize: TAVILY_API_KEY not found"

**Cause:** API key not set

**Fix:**
```bash
export TAVILY_API_KEY="tvly-xxxxx"
# Or enter in Streamlit UI sidebar
```

---

### "Search Error: Rate limit exceeded"

**Cause:** Exceeded Tavily free tier (1,000/month)

**Fix:**
- Wait for monthly reset
- Upgrade Tavily plan
- Temporarily disable web search (use document-only mode)

---

### "No web sources found in final report"

**Cause:** Web Search Agent not registered or failed

**Fix:**
1. Check Streamlit sidebar shows "✅ web_search_agent"
2. Verify Tavily API key is valid
3. Check console output for errors:
   ```
   🌐 Web Search Agent analyzing: '...'
   ✅ Found 5 web sources
   ```

---

### "Web citations not formatted correctly"

**Cause:** Agent not preserving SOURCE markers

**Check:** Agent output should contain:
```
--- SOURCE: Title (URL) ---
```

**Not:**
```
[Bloomberg](https://...)
```

---

## Architecture

```python
class WebSearchAgent:
    def analyze(query, prior_analysis):
        # 1. Enhance query for web focus
        enhanced = _enhance_query(query, prior_analysis)
        
        # 2. Search web via Tavily
        context, citations = _search_web(enhanced)
        
        # 3. Synthesize with local Ollama
        analysis = _synthesize_with_llm(query, context, prior_analysis)
        
        # 4. Return with SOURCE markers
        return analysis  # Contains "--- SOURCE: ... ---"
```

---

## Examples

### Example 1: Risk Analysis

**Query:** "Analyze Apple's regulatory risks"

**Business Analyst (10-K):**
```
EU regulatory scrutiny over App Store [Page 67]
```

**Web Search Agent:**
```
EU expands antitrust probe, $10B fine possible [Bloomberg]
Apple appeals DMA classification [Reuters]
```

**Final Synthesis:**
```
## Regulatory Risks

Apple's 10-K flags EU App Store scrutiny [1]. This risk has
escalated with expanded antitrust investigations [2] and 
potential $10B fines [2]. Apple is appealing DMA classification [3].

[1] APPL 10-k Filings.pdf - Page 67
[2] Bloomberg - https://...
[3] Reuters - https://...
```

---

### Example 2: Competitive Analysis

**Query:** "Compare Apple and Samsung competition"

**Business Analyst (10-K):**
```
Samsung offers competitive alternatives at lower prices [Page 45]
```

**Web Search Agent:**
```
Samsung Q4 market share up 3% in India [TechCrunch]
iPhone 16 launch delayed in key markets [WSJ]
```

**Final Synthesis:**
```
## Competitive Dynamics

Apple's 10-K acknowledges Samsung pricing pressure [1].
Recent data shows Samsung gaining 3% market share in India [2],
while iPhone 16 delays may widen the gap [3].

[1] APPL 10-k Filings.pdf - Page 45
[2] TechCrunch - https://...
[3] WSJ - https://...
```

---

## Summary

**Purpose:** Supplement document analysis with current web data  
**Execution:** Always runs AFTER Business Analyst  
**Search:** Tavily API (cloud)  
**Synthesis:** Local Ollama LLM (privacy preserved)  
**Citations:** Unified format with document sources  
**Optional:** System works without Tavily key  

**Result:** Comprehensive reports combining historical 10-K analysis with current market intelligence.
