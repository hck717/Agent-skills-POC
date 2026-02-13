# Web Search Agent (The "News Desk")

## Agent ID
`web_search_agent`

## Primary Role
Find "unknown unknowns" and breaking developments using HyDE (Hypothetical Document Embeddings) + Step-Back Prompting + Reranking. Provides real-time context when historical data (10-K/10-Q) is outdated.

---

## Core Capabilities

### 1. Real-Time Market Intelligence
- **Breaking News**: Company-specific developments within last 24-48 hours
- **Market-Moving Events**: Earnings surprises, M&A announcements, regulatory decisions
- **Analyst Actions**: Upgrades/downgrades, price target changes, initiation coverage
- **Economic Releases**: GDP, CPI, employment data, Fed announcements

### 2. Gap-Filling for Historical Agents
- **CRAG Fallback**: Triggered when Business Analyst confidence <0.5
- **Temporal Bridge**: Connect stale 10-K/10-Q (90 days old) to current events
- **Contradiction Detection**: Identify when recent news contradicts historical filings

### 3. Sentiment & Credibility Filtering
- **Source Credibility Scoring**: Reuters/Bloomberg (high) vs unknown blogs (low)
- **Clickbait Detection**: Filter sensational headlines with no substance
- **Duplicate Removal**: Consolidate identical press releases across sources
- **Freshness Prioritization**: Weight articles by recency (<24h = highest priority)

---

## Search Architecture

### Three-Phase Query Transformation

#### Phase 1: Step-Back Prompting
**Purpose**: Broaden narrow queries to capture related context

**Example**:
```
User Query: "Why is AAPL down today?"

Step-Back Prompt:
"What are the major recent news events affecting large-cap tech stocks 
or Apple specifically in the past 48 hours?"

Benefit: Catches related events like "Tech selloff" or "iPhone supply issues" 
that may not explicitly mention stock price.
```

**Templates**:
```python
step_back_templates = {
    'stock_movement': "What are recent news events affecting {ticker} or {sector} stocks in the past {timeframe}?",
    'earnings': "What are the latest earnings results and analyst reactions for {ticker} and its peers?",
    'industry_trend': "What are recent developments in the {industry} industry that could impact {ticker}?",
    'regulatory': "What are recent regulatory or legal developments affecting {ticker} or {industry}?"
}
```

---

#### Phase 2: HyDE (Hypothetical Document Embeddings)
**Purpose**: Search for intent and context, not just keywords

**Workflow**:
```
1. Generate Hypothetical Article:
   User Query: "Why is TSLA down 5% today?"
   
   LLM Generates Fake Article:
   "Tesla shares fell 5% today following reports of production delays at 
   the Gigafactory Berlin due to supply chain constraints. Analysts noted 
   concerns about Q1 delivery targets being missed, with Wedbush lowering 
   their price target from $350 to $320. The decline was compounded by 
   broader EV sector weakness as competitors reported slowing demand in China."

2. Embed Hypothetical Article:
   - Convert fake article to 768-dim embedding
   - This embedding captures: production issues, supply chain, analyst downgrades, China demand

3. Semantic Search Real News:
   - Query news APIs with hypothetical embedding
   - Returns articles with similar themes/context
   - Matches intent even if exact keywords differ

4. Results:
   ✓ "Tesla Berlin factory halts production for 3 days" (Reuters)
   ✓ "Wedbush cuts TSLA target on delivery concerns" (Bloomberg)
   ✓ "China EV sales disappoint in January" (CNBC)
   ✗ "Elon Musk tweets about Mars mission" (filtered out - not relevant to stock decline)
```

**HyDE Templates**:
```python
hyde_templates = {
    'stock_decline': """
        Generate a news article explaining why {ticker} stock declined {pct}% today:
        
        Headline: "{ticker} Shares Fall {pct}% on {reason}"
        
        Article: "{ticker} stock dropped {pct}% today following {catalyst}. 
        Analysts noted {analysis}. The decline was driven by {driver}. 
        Trading volume was {volume_desc}, with {technical_context}."
    """,
    
    'earnings_beat': """
        Generate an earnings report summary for {ticker}:
        
        "{ticker} reported Q{quarter} earnings that beat expectations. 
        Revenue of ${revenue}M exceeded consensus of ${consensus}M. 
        EPS of ${eps} beat estimates. Key highlights included {highlights}. 
        Management raised guidance, projecting {forward_guidance}."
    """
}
```

---

#### Phase 3: Reranking & Filtering
**Purpose**: Remove noise, prioritize quality sources

**Reranking Criteria**:
```json
{
  "source_credibility": {
    "tier_1": ["Reuters", "Bloomberg", "WSJ", "FT"],
    "tier_2": ["CNBC", "MarketWatch", "Barron's"],
    "tier_3": ["Seeking Alpha", "Yahoo Finance"],
    "filtered": ["unknown blogs", "promotional sites"]
  },
  
  "freshness_weight": {
    "<1_hour": 1.0,
    "1-6_hours": 0.9,
    "6-24_hours": 0.7,
    "1-7_days": 0.4,
    ">7_days": 0.1
  },
  
  "noise_filters": {
    "clickbait_patterns": [
      "You won't believe",
      "This one trick",
      "Shocking news"
    ],
    "duplicate_threshold": 0.95,
    "min_article_length": 200
  }
}
```

**Cohere Rerank Integration**:
```python
import cohere

co = cohere.Client(api_key)

# Initial search returns 50 articles
raw_results = search_api.query(hyde_embedding, limit=50)

# Rerank with query relevance
reranked = co.rerank(
    query="Why is AAPL down today?",
    documents=[r['text'] for r in raw_results],
    top_n=10,
    model='rerank-english-v2.0'
)

# Apply credibility + freshness weighting
final_results = []
for result in reranked:
    score = result.relevance_score
    score *= credibility_weight[result.source]
    score *= freshness_weight[result.age]
    final_results.append((result, score))

final_results.sort(key=lambda x: x[1], reverse=True)
```

---

## Data Sources

### Primary APIs (No Persistence)
| Source | Data Type | Coverage | Rate Limit |
|--------|-----------|----------|------------|
| **EODHD** | Real-time quotes, economic calendar | Global markets | 1000 req/day |
| **FMP** | Latest news, press releases, analyst upgrades | US stocks | 250 req/day |
| **Tavily Search** | General web search with relevance scoring | Global news | 1000 req/month |
| **SEC EDGAR RSS** | Latest 8-K filings (material events) | US public companies | Unlimited |

### API Configuration
```json
{
  "eodhd": {
    "real_time_quotes": "https://eodhd.com/api/real-time/{ticker}?api_token={token}",
    "economic_calendar": "https://eodhd.com/api/economic-events?from={date}&to={date}",
    "news_api": "https://eodhd.com/api/news?s={ticker}&limit=50"
  },
  
  "fmp": {
    "stock_news": "https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=50",
    "press_releases": "https://financialmodelingprep.com/api/v3/press-releases/{ticker}",
    "analyst_upgrades": "https://financialmodelingprep.com/api/v3/upgrades-downgrades?symbol={ticker}"
  },
  
  "tavily": {
    "search": "https://api.tavily.com/search",
    "params": {
      "search_depth": "advanced",
      "max_results": 20,
      "include_domains": ["reuters.com", "bloomberg.com", "wsj.com"]
    }
  },
  
  "sec_edgar": {
    "rss_feed": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom"
  }
}
```

---

## Query Patterns & Examples

### Query Type 1: Real-Time Stock Movement
**User Query**: "Why is NVDA up 8% today?"

**Agent Workflow**:
```
1. Step-Back Expansion:
   "What are recent news events affecting NVDA or semiconductor stocks in the past 24 hours?"

2. HyDE Article Generation:
   "NVIDIA shares surged 8% today following better-than-expected earnings results. 
   Revenue of $22.1B beat consensus of $20.4B, driven by strong data center GPU demand. 
   Management raised Q1 guidance, citing accelerating AI infrastructure spending..."

3. Parallel API Calls:
   - FMP Stock News: 15 articles (past 24h)
   - EODHD News: 12 articles
   - Tavily Search: 8 articles from Tier-1 sources

4. Reranking (Cohere):
   Top 5 Results:
   [1] "Nvidia Beats Estimates on AI Chip Demand" (Reuters, 2h ago, score: 0.95)
   [2] "NVDA Raises Q1 Guidance by 15%" (Bloomberg, 3h ago, score: 0.92)
   [3] "AMD Shares Rise on Nvidia's Strong Results" (WSJ, 4h ago, score: 0.78)
   [4] "Semiconductor Rally: NVDA, AMD, INTC All Up" (CNBC, 5h ago, score: 0.71)
   [5] "Wedbush Raises NVDA Target to $1200" (MarketWatch, 6h ago, score: 0.68)

5. Response:
   ## Why NVDA is Up 8% Today
   
   ### Primary Catalyst: Earnings Beat
   **NVIDIA reported Q4 2025 earnings after market close yesterday that 
   significantly exceeded analyst expectations** [Reuters, 2h ago]
   
   **Key Metrics:**
   - Revenue: $22.1B (vs consensus $20.4B, +8.3% beat)
   - EPS: $5.16 (vs consensus $4.64, +11.2% beat)
   - Data Center Revenue: $18.4B (+27% QoQ)
   
   ### Forward Guidance
   Management raised Q1 2026 revenue guidance to $24-26B, above Street 
   expectations of $22B, citing "unprecedented demand for AI infrastructure" 
   [Bloomberg, 3h ago]
   
   ### Analyst Reactions
   - **Wedbush**: Raised price target $950 → $1200 (Outperform)
   - **Morgan Stanley**: Increased estimates, called AI demand "sustainable"
   - **JPMorgan**: Maintained Overweight, raised target $1050 → $1150
   
   ### Sector Impact
   Broader semiconductor rally: AMD +5%, INTC +3%, AVGO +4% on positive 
   read-through from NVDA's AI commentary [WSJ, 4h ago]
```

---

### Query Type 2: CRAG Fallback (Low Historical Confidence)
**User Query**: "Analyze AAPL's strategy"

**Business Analyst Agent**:
```
1. Retrieval from 10-K (filed 2025-10-31, 3.5 months old)
   - CRAG Evaluator Score: 0.42 (Low confidence)
   - Reason: Query likely seeking recent developments not in 10-K

2. Trigger Web Search Agent:
   - Query: "Apple strategy 2026 recent developments"
```

**Web Search Agent Workflow**:
```
1. Step-Back:
   "What are recent strategic announcements or pivots by Apple in Q4 2025 and Q1 2026?"

2. HyDE:
   "Apple announced a major strategic shift toward AI-powered services in January 2026. 
   The company unveiled a new subscription tier for enhanced Siri capabilities, 
   integrating advanced language models across iOS 19..."

3. Search Results (Reranked):
   [1] "Apple Announces 'Apple Intelligence+' Subscription" (Bloomberg, Jan 15, 2026)
   [2] "AAPL Shifts Strategy: Services Over Hardware" (WSJ, Jan 22, 2026)
   [3] "Apple Vision Pro 2 Delayed to 2027" (Reuters, Feb 5, 2026)

4. Hybrid Response (Business Analyst + Web Search):
   ## Apple's Current Strategy (Feb 2026)
   
   ### Historical Foundation (10-K 2025-10-31)
   Apple's long-term strategy focuses on:
   - Ecosystem lock-in via Services (30% of revenue)
   - Premium hardware positioning (avg selling price >$900)
   - Geographic expansion in India (+40% YoY growth)
   
   ### Recent Strategic Shifts (Q1 2026) 🆕
   **January 2026: Major AI Services Push**
   - Launched "Apple Intelligence+" subscription ($19.99/month)
   - Enhanced Siri with on-device LLMs (privacy-focused positioning)
   - Bundles with Apple One for $29.99/month
   - Analyst estimate: Could add $8-10B annual recurring revenue by 2027
   [Bloomberg, Jan 15, 2026]
   
   **Strategy Pivot: Services Prioritization**
   - Hardware delays (Vision Pro 2 pushed to 2027) to focus on AI services
   - Services gross margin: 72% vs Hardware 36%
   - CFO commentary: "Prioritizing high-margin recurring revenue"
   [WSJ, Jan 22, 2026]
   
   ### Synthesis
   Apple is accelerating its Services-first strategy disclosed in 2025 10-K, 
   with AI subscriptions as the new growth vector. This aligns with historical 
   trend (Services revenue +15% CAGR 2020-2025) but represents faster execution 
   than previously guided.
```

---

### Query Type 3: Contradiction Detection
**User Query**: "Is TSLA on track with production targets?"

**Workflow**:
```
1. Business Analyst (10-K 2025-10-31):
   "Management projects 2026 deliveries of 2.0-2.2M vehicles, representing 
   25% growth over 2025."
   CRAG Score: 0.88 (High confidence)

2. Web Search (Recent News):
   [1] "Tesla Cuts Production at Shanghai Plant 20%" (Reuters, Feb 10, 2026)
   [2] "TSLA Analysts Lower Delivery Estimates After China Slowdown" (Bloomberg, Feb 11, 2026)

3. Contradiction Detected:
   Historical Guidance (Oct 2025): 2.0-2.2M deliveries (+25%)
   Recent Reality (Feb 2026): Shanghai production cuts, analyst downgrades

4. Response:
   ## TSLA Production Target Analysis
   
   ### Official Guidance (10-K filed Oct 31, 2025)
   Tesla management projected 2026 deliveries of **2.0-2.2 million vehicles**, 
   representing 25% growth over 2025. [10-K, Item 7]
   
   ### ⚠️ Recent Developments Suggest Risk to Targets
   
   **February 2026: Shanghai Production Cuts**
   - Tesla reduced Gigafactory Shanghai output by 20% in February
   - Reason: "Weak demand in China" amid EV price war
   - Shanghai represents ~50% of Tesla's global production
   [Reuters, Feb 10, 2026]
   
   **Analyst Downgrades**
   - Goldman Sachs: Lowered 2026 delivery estimate to 1.85M (from 2.1M)
   - Morgan Stanley: Cut to 1.9M, citing "China headwinds"
   - Consensus now: 1.88M (-9% vs management guidance)
   [Bloomberg, Feb 11, 2026]
   
   ### Interpretation
   **Risk to Guidance**: Recent production cuts contradict October guidance. 
   Unless Tesla compensates with US/EU ramp, 2.0M target appears challenging. 
   Watch for revised guidance in Q1 2026 earnings call (April).
```

---

## Output Format

### Standard Response Structure
```markdown
## [Topic] - Real-Time Update

### Breaking News Summary
[1-2 sentence headline with source + timestamp]

### Key Developments (Past 24-48 Hours)
1. **[Event 1]** [Source, Time]
   - Detail
   - Impact

2. **[Event 2]** [Source, Time]
   - Detail
   - Impact

### Analyst Commentary
- [Firm]: [Action] ([Source, Time])
- [Firm]: [Action] ([Source, Time])

### Historical Context (if available from other agents)
[Connect to 10-K/10-Q data or forensic scores]

### Sources
- [Source 1]: [URL] (Credibility: Tier-1, Freshness: 2h)
- [Source 2]: [URL] (Credibility: Tier-1, Freshness: 4h)
```

---

## Integration with Other Agents

### Upstream Triggers
| Agent | Trigger Condition | Integration Pattern |
|-------|-------------------|---------------------|
| **Business Analyst** | CRAG Score <0.5 | Fallback for real-time context |
| **Insider & Sentiment** | Narrative drift detected | Validate with recent news |
| **Quantitative** | Forensic red flag | Check for breaking scandal/news |

### Downstream Consumers
| Agent | Data Provided | Use Case |
|-------|---------------|----------|
| **Business Analyst** | Recent strategic announcements | Update 10-K context |
| **Insider & Sentiment** | Breaking news sentiment | Cross-validate transcript tone |

---

## Limitations & Constraints

### No Data Persistence
- **Ephemeral**: Results NOT stored (cache <24h optional)
- **Implication**: Cannot perform historical news analysis
- **Mitigation**: For historical research, user must query Business Analyst (10-K/10-Q)

### API Rate Limits
- EODHD: 1000 req/day → ~40 queries/hour
- FMP: 250 req/day → ~10 queries/hour
- Tavily: 1000 req/month → ~30 queries/day
- **Risk**: High query volume can exhaust daily limits

### Source Credibility Variance
- Tier-1 sources (Reuters, Bloomberg) have paywall/API access limits
- Tier-3 sources (Seeking Alpha, blogs) may lack verification
- **Mitigation**: Multi-source confirmation required for critical claims

---

## Performance Metrics

### Query Latency
- **Step-Back + HyDE**: 500-800ms (LLM generation)
- **Parallel API Calls**: 1-2 seconds (3 sources)
- **Reranking**: 300-500ms (Cohere)
- **Total End-to-End**: 2.5-4 seconds

### Accuracy
- **Source Credibility**: 95% of Tier-1 sources factually accurate
- **Freshness**: 90% of results <6 hours old
- **Noise Reduction**: 85% of clickbait/duplicates filtered

---

## When to Use This Agent

### ✅ Ideal For:
- "Why is [ticker] up/down today?"
- "What's the latest news on [company]?"
- "Breaking developments for [ticker]"
- "Recent analyst actions on [ticker]"
- "What happened to [ticker] this week?"

### ❌ Not Suitable For:
- Historical analysis (>1 month old) - use Business Analyst
- Financial calculations - use Quantitative Agent
- Strategy deep-dives - use Business Analyst
- Supply chain analysis - use Supply Chain Graph

---

## Example Planner Routing Logic

```python
def route_query_to_web_search(query: str) -> bool:
    """Determine if query requires real-time web search"""
    
    real_time_keywords = [
        'today', 'now', 'currently', 'latest', 'recent', 'breaking',
        'this week', 'this month', 'just announced', 'yesterday'
    ]
    
    movement_keywords = [
        'up', 'down', 'fell', 'rose', 'surged', 'plunged', 'rallied',
        'dropped', 'climbed', 'tanked'
    ]
    
    news_keywords = [
        'news', 'announcement', 'headline', 'report', 'development',
        'update', 'press release'
    ]
    
    # Check for real-time query patterns
    if any(kw in query.lower() for kw in real_time_keywords):
        return True
    
    # Check for stock movement queries
    if any(kw in query.lower() for kw in movement_keywords):
        if 'today' in query.lower() or 'now' in query.lower():
            return True
    
    # Check for news-specific requests
    if any(kw in query.lower() for kw in news_keywords):
        return True
    
    return False


def is_crag_fallback_needed(business_analyst_score: float) -> bool:
    """Check if Business Analyst needs Web Search fallback"""
    return business_analyst_score < 0.5
```

---

## Error Handling

### Scenario 1: No Recent News Found
```
User Query: "Latest news on [obscure ticker]"
Issue: All APIs return empty results

Response:
"I couldn't find any recent news (past 7 days) for [ticker] from major 
financial news sources (Reuters, Bloomberg, WSJ, CNBC).

This could indicate:
1. Low media coverage (small-cap or private company)
2. No material events recently
3. API coverage gaps for this ticker

Would you like me to:
- Search the company's investor relations page?
- Check SEC EDGAR for recent 8-K filings?
- Analyze historical 10-K/10-Q instead?"
```

### Scenario 2: API Rate Limit Exceeded
```
User Query: "Why is TSLA down today?"
Issue: FMP API rate limit exceeded (250 req/day)

Response:
"I've reached the daily API rate limit for FMP news (250 requests). 
Falling back to alternative sources:

✓ EODHD News API (still available)
✓ Tavily web search (Tier-1 sources)
✗ FMP press releases (unavailable)

Results may be less comprehensive but still provide recent context.

[Proceed with partial results...]"
```

---

## Version History
- **v2.0** (2026-02-13): HyDE + Step-Back + Cohere Rerank
- **v1.5** (2025-09-01): Added CRAG fallback integration
- **v1.0** (2025-07-15): Initial keyword-based search implementation
