# Specialist Agent Specifications

This document provides detailed specifications for each specialist agent in the equity research system. The Planner Agent uses this information to make intelligent decisions about agent deployment.

---

## 1. Business Analyst Agent

**Status:** ✅ Implemented  
**Implementation:** `skills/business_analyst/graph_agent.py`  
**Tech Stack:** LangGraph, Ollama (Qwen 2.5), ChromaDB, BERT Reranker

### Core Capabilities
- Extract and analyze financial statements from 10-K/10-Q filings
- Identify competitive positioning and market share insights
- Assess strategic and operational risks
- Evaluate business model strength and unit economics
- Analyze management discussion & analysis (MD&A) sections

### Data Sources
- SEC EDGAR filings (10-K, 10-Q, 8-K)
- Local PDF document storage (indexed in ChromaDB)
- Company annual reports

### Strengths
- High accuracy on qualitative business analysis
- Strong at identifying risks buried in filings
- Excellent citation tracking with page numbers
- BERT reranking ensures retrieval precision

### Limitations
- Requires pre-ingested documents (no real-time filing fetch)
- Limited to companies with documents in `/data` folder
- Does not perform complex financial calculations
- No access to real-time market data

### Best Use Cases
- "What are the key risks facing [Company]?"
- "Explain [Company]'s business model and competitive advantages"
- "Compare competitive positioning of [Company A] vs [Company B]"
- "What did management say about [specific topic] in the 10-K?"
- "Analyze [Company]'s supply chain risks"

### Avoid Using For
- Real-time stock prices or market data
- Complex financial ratio calculations (use Quantitative Analyst)
- Industry-wide trends without specific company context
- Sentiment analysis from news/social media

---

## 2. Quantitative Analyst Agent

**Status:** 📋 Planned  
**Planned Tech Stack:** Python, Pandas, NumPy, Financial APIs

### Core Capabilities
- Calculate financial ratios (P/E, ROE, ROIC, debt ratios, etc.)
- Perform DCF (Discounted Cash Flow) valuation models
- Compute growth rates (CAGR, YoY, QoQ)
- Statistical analysis and trend forecasting
- Comparative peer benchmarking with quantitative metrics
- Sensitivity analysis and scenario modeling

### Data Sources
- Financial statements (from Business Analyst or APIs)
- Market data APIs (Yahoo Finance, Alpha Vantage, EODHD)
- Historical price data
- Earnings estimates and analyst consensus

### Strengths
- 100% computational accuracy (no LLM hallucinations)
- Can handle complex multi-step calculations
- Produces formatted tables and charts
- Statistical rigor in trend analysis

### Limitations
- Requires structured financial data as input
- Cannot interpret qualitative factors
- Dependent on data quality from upstream sources

### Best Use Cases
- "Calculate [Company]'s P/E ratio and compare to sector average"
- "What is [Company]'s 5-year revenue CAGR?"
- "Perform a DCF valuation of [Company] with 10% discount rate"
- "Compare profit margins of [Company A] vs [Company B] over 3 years"
- "Calculate [Company]'s return on invested capital trend"

### Avoid Using For
- Qualitative business analysis
- Extracting data from unstructured documents
- Market sentiment or news interpretation

---

## 3. Market Analyst Agent

**Status:** 📋 Planned  
**Planned Tech Stack:** Python, TA-Lib, News APIs, Sentiment Models

### Core Capabilities
- Track and analyze market sentiment from news and social media
- Technical analysis (moving averages, RSI, MACD, support/resistance)
- Monitor price movements, volume, and volatility
- Identify trading patterns and anomalies
- Track institutional ownership and insider trading
- Analyze analyst rating changes and price target movements

### Data Sources
- Real-time and historical price data (Yahoo Finance, Bloomberg)
- News APIs (NewsAPI, Benzinga, Reuters)
- Social media sentiment (Twitter/X, Reddit, StockTwits)
- Analyst ratings databases
- Options flow data

### Strengths
- Real-time market pulse
- Captures sentiment not in official filings
- Technical pattern recognition
- Early warning on momentum shifts

### Limitations
- Sentiment can be noisy and misleading
- Technical analysis not predictive of fundamentals
- Requires real-time data subscriptions

### Best Use Cases
- "What is the current market sentiment on [Company]?"
- "Show me the technical chart analysis for [Ticker]"
- "Has there been unusual trading volume in [Company] recently?"
- "What are analysts saying about [Company]'s earnings?"
- "Identify support and resistance levels for [Ticker]"

### Avoid Using For
- Long-term fundamental valuation
- Detailed financial statement analysis
- Regulatory or compliance questions

---

## 4. Industry Analyst Agent

**Status:** 📋 Planned  
**Planned Tech Stack:** LangChain, Web Search APIs, Sector Databases

### Core Capabilities
- Provide sector-specific context and industry dynamics
- Benchmark company performance against peers
- Analyze regulatory landscape and policy impacts
- Track industry trends, disruptions, and emerging technologies
- Identify competitive intensity (Porter's Five Forces)
- Monitor M&A activity and consolidation trends

### Data Sources
- Industry research reports (IBISWorld, Statista)
- Trade publications and industry news
- Government databases (e.g., SEC, FDA, FCC)
- Competitive intelligence platforms
- Academic research on industry dynamics

### Strengths
- Deep sector expertise and context
- Peer comparison frameworks
- Regulatory change monitoring
- Identifies industry tailwinds/headwinds

### Limitations
- May lack company-specific granularity
- Industry reports can be expensive/paywalled
- Analysis may lag real-time developments

### Best Use Cases
- "How does [Company] compare to competitors in the [Industry]?"
- "What are the key trends shaping the [Sector]?"
- "What regulatory changes could impact [Industry]?"
- "Who are the main competitors of [Company] and their market shares?"
- "Is the [Industry] growing or declining?"

### Avoid Using For
- Company-specific financial details
- Real-time stock prices
- Individual stock buy/sell recommendations

---

## 5. ESG Analyst Agent

**Status:** 📋 Planned  
**Planned Tech Stack:** ESG Data APIs, NLP Models, Sustainability Databases

### Core Capabilities
- Evaluate environmental impact and carbon footprint
- Assess social responsibility (labor, diversity, community)
- Analyze corporate governance structures and practices
- Score ESG performance against frameworks (SASB, GRI, TCFD)
- Identify ESG-related risks and controversies
- Track sustainability initiatives and commitments

### Data Sources
- ESG rating agencies (MSCI, Sustainalytics, S&P)
- Sustainability reports and CSR disclosures
- Carbon disclosure databases (CDP)
- Controversy databases
- Regulatory filings (proxy statements, diversity reports)

### Strengths
- Comprehensive ESG coverage
- Risk identification beyond financial metrics
- Alignment with responsible investing mandates
- Long-term risk assessment

### Limitations
- ESG scores lack standardization
- Data availability varies by company size
- Qualitative factors hard to quantify
- May not directly correlate with financial performance

### Best Use Cases
- "What is [Company]'s ESG rating and score breakdown?"
- "Assess [Company]'s carbon reduction commitments and progress"
- "Are there any ESG controversies or red flags for [Company]?"
- "How does [Company]'s board diversity compare to peers?"
- "Evaluate [Company]'s supply chain labor practices"

### Avoid Using For
- Traditional financial valuation
- Short-term trading decisions
- Technical chart analysis

---

## 6. Macro Analyst Agent

**Status:** 📋 Planned  
**Planned Tech Stack:** Economic APIs, Macro Models, Geopolitical Databases

### Core Capabilities
- Analyze macroeconomic indicators (GDP, inflation, unemployment)
- Assess interest rate sensitivity and monetary policy impacts
- Evaluate currency exposure and FX risks
- Monitor geopolitical risks and trade policy changes
- Track commodity price impacts
- Analyze fiscal policy effects on sectors/companies

### Data Sources
- Economic data APIs (FRED, World Bank, IMF)
- Central bank statements and projections
- Currency and commodity price feeds
- Geopolitical risk databases
- Trade and tariff data

### Strengths
- Macro context for company performance
- Interest rate sensitivity analysis
- Global exposure assessment
- Early warning on macro headwinds/tailwinds

### Limitations
- Macro forecasts are inherently uncertain
- Company-specific impacts vary widely
- Requires deep economic modeling expertise

### Best Use Cases
- "How would a 2% Fed rate increase affect [Company]?"
- "What is [Company]'s exposure to currency fluctuations?"
- "How sensitive is [Sector] to oil price changes?"
- "What geopolitical risks could impact [Company]'s operations?"
- "Analyze [Company]'s debt in the context of rising interest rates"

### Avoid Using For
- Company-specific operational details
- Technical stock analysis
- Individual financial statement items

---

## Agent Selection Decision Tree

### Query Analysis Framework for Planner Agent

```
Query Intent Detection:

1. FINANCIAL STATEMENTS / FILINGS
   Keywords: "10-K", "risk factors", "business model", "competitive", "MD&A"
   → Deploy: Business Analyst

2. CALCULATIONS / RATIOS / VALUATION
   Keywords: "calculate", "ratio", "P/E", "DCF", "growth rate", "margin", "CAGR"
   → Deploy: Quantitative Analyst

3. MARKET DATA / SENTIMENT / TECHNICALS
   Keywords: "sentiment", "price", "trading", "volume", "chart", "analyst rating"
   → Deploy: Market Analyst

4. INDUSTRY TRENDS / PEER COMPARISON
   Keywords: "industry", "sector", "competitors", "market share", "peer", "regulation"
   → Deploy: Industry Analyst

5. ESG / SUSTAINABILITY / GOVERNANCE
   Keywords: "ESG", "carbon", "sustainability", "diversity", "governance", "ethical"
   → Deploy: ESG Analyst

6. MACRO / ECONOMIC / GEOPOLITICAL
   Keywords: "interest rate", "inflation", "FX", "currency", "GDP", "geopolitical"
   → Deploy: Macro Analyst
```

### Multi-Agent Deployment Guidelines

**Use 2-3 agents when:**
- Query requires both qualitative and quantitative analysis
- Comparison involves different data sources
- Context requires macro + micro perspectives

**Example:** "Analyze Apple's profit margin trends and assess competitive threats"
- Business Analyst: Extract competitive threats from 10-K
- Quantitative Analyst: Calculate margin trends and peer comparisons

**Use 1 agent when:**
- Query is narrow and domain-specific
- Single data source can answer completely
- Speed is prioritized over comprehensiveness

---

## Integration Notes

### For Planner Agent Implementation

When creating the planning prompt, include:
1. Full capability descriptions (not just summaries)
2. Data source availability for each agent
3. Current implementation status (some agents may not be available)
4. Example queries that match each agent's strengths

### For Synthesis Agent Integration

When combining outputs:
1. Prioritize quantitative data from Quantitative Analyst
2. Use Business Analyst for context and risk framing
3. Market Analyst provides current sentiment overlay
4. Industry Analyst validates competitive assumptions
5. ESG Analyst adds long-term risk perspective
6. Macro Analyst provides economic context

### Performance Optimization

- **Parallel Execution**: Agents with independent data sources can run simultaneously
- **Caching**: Business Analyst document retrieval can be cached for multiple queries
- **Fallback**: If specialist unavailable, Planner should select alternative or notify user

---

**Last Updated:** February 9, 2026  
**Maintained by:** Agent Orchestration System
