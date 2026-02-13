# Business Analyst Agent (The "Deep Reader")

## Agent ID
`business_analyst_agent`

## Primary Role
Extract strategic insights, business model analysis, and risk factors from SEC filings (10-K/10-Q) with zero hallucinations using Graph-Augmented Corrective RAG (CRAG).

---

## Core Capabilities

### 1. Strategic Analysis
- **Business Model Deep Dive**: Revenue streams, customer segments, competitive positioning
- **Growth Strategy Extraction**: Market expansion plans, M&A strategy, product roadmap
- **Competitive Landscape**: Key competitors, market share dynamics, differentiation factors
- **Technology Stack Analysis**: Infrastructure investments, R&D focus areas, digital transformation

### 2. Risk Factor Analysis
- **Regulatory Risks**: Compliance challenges, pending legislation, antitrust concerns
- **Operational Risks**: Supply chain dependencies, key personnel risks, cybersecurity threats
- **Market Risks**: Customer concentration, geographic exposure, commodity price sensitivity
- **Financial Risks**: Debt covenants, liquidity constraints, FX exposure

### 3. Management Discussion & Analysis (MD&A)
- **Performance Attribution**: Revenue growth drivers, margin expansion/contraction factors
- **Forward Guidance**: Management outlook, growth projections, capital allocation plans
- **Segment Performance**: Geographic/product segment analysis, cross-segment trends
- **Critical Accounting Policies**: Revenue recognition changes, impairment tests, contingencies

---

## RAG Architecture

### Retrieval Strategy: Graph-Augmented Corrective RAG (CRAG)

#### Phase 1: Proposition-Based Retrieval
- **Chunking Method**: LLM decomposes 10-K/10-Q into atomic, standalone propositions
- **Example Proposition**: "The company derives 65% of revenue from semiconductor customers, creating concentration risk"
- **Vector Store**: Qdrant collections (`business_analyst_10k`, `business_analyst_10q`)
- **Embedding Model**: `nomic-embed-text` (768 dimensions)

#### Phase 2: Hybrid Search
```
Query: "What is the AI strategy?"

1. Dense Retrieval (Vector Search):
   - Semantic similarity on proposition embeddings
   - Returns: Top-5 conceptually similar propositions

2. Sparse Retrieval (BM25):
   - Keyword matching on exact terms ("Generative AI", "LLM")
   - Returns: Top-5 keyword matches

3. Graph Traversal (Neo4j):
   MATCH (:Strategy)-[:DEPENDS_ON]->(:Technology)
   WHERE Technology.name CONTAINS "AI"
   - Returns: Related concepts, dependencies
```

#### Phase 3: Corrective RAG (CRAG)
```
Evaluator Score → Action:

• Score > 0.7 (High Confidence):
  → Use retrieved propositions directly
  
• Score 0.5-0.7 (Ambiguous):
  → Rewrite query with more context
  → Retry retrieval with refined query
  
• Score < 0.5 (Low Confidence):
  → Discard retrieved documents
  → Trigger Web Search Agent for recent developments
  → Combine historical (10-K) + real-time (web) context
```

---

## Data Sources

### Primary Sources
| Source | Data Type | Coverage | Update Frequency |
|--------|-----------|----------|------------------|
| **EODHD** | 10-K/10-Q sections (Business Overview, MD&A, Risk Factors) | 3 years | Quarterly |
| **FMP** | Full 10-K/10-Q text, MD&A sections, earnings transcripts | 3 years | Quarterly |

### Database Architecture
```
Qdrant (Vector Store):
├── Collection: business_analyst_10k
│   ├── Propositions: ~50,000 per ticker (3 years)
│   ├── Metadata: {ticker, filing_date, section, proposition_id, confidence_score}
│   └── Indexing: HNSW + BM25 sparse index

Neo4j (Knowledge Graph):
├── Nodes:
│   ├── Concepts (e.g., "Cloud Revenue Growth")
│   ├── Strategies (e.g., "Vertical Integration")
│   ├── Risks (e.g., "Regulatory Scrutiny")
│   ├── Technologies (e.g., "Generative AI")
│   └── Markets (e.g., "China", "North America")
│
└── Relationships:
    ├── DEPENDS_ON (Strategy → Technology)
    ├── IMPACTS (Risk → Revenue)
    └── AFFECTS (Market → Strategy)
```

---

## Query Patterns & Examples

### Query Type 1: Strategic Deep Dive
**User Query**: "Analyze AAPL's AI strategy"

**Agent Workflow**:
```
1. Hybrid Retrieval:
   - Vector search: "artificial intelligence strategy investments"
   - BM25 search: "AI", "machine learning", "LLM", "generative AI"
   - Graph traversal: MATCH (s:Strategy)-[:DEPENDS_ON]->(t:Technology {name:"AI"})

2. CRAG Evaluation:
   - Score: 0.85 (High confidence)
   - Action: Use retrieved propositions

3. Response Structure:
   ## AI Strategy Overview
   - Core Focus: [extracted from Item 1]
   - R&D Investments: [extracted from MD&A]
   - Competitive Positioning: [graph relationships]
   - Risk Factors: [extracted from Item 1A]
   
   ## Supporting Evidence
   - 10-K Filing Date: 2025-10-31, Page 12
   - MD&A Section: "Research and Development"
   - Graph Relationship: Strategy "AI Development" DEPENDS_ON Technology "LLM Infrastructure"
```

---

### Query Type 2: Risk Assessment
**User Query**: "What are TSLA's supply chain risks?"

**Agent Workflow**:
```
1. Section-Specific Retrieval:
   - Filter: section="Risk Factors", ticker="TSLA"
   - Keywords: "supply chain", "supplier", "shortage", "dependency"

2. Graph Traversal:
   MATCH (r:Risk)-[:AFFECTS]->(c:Concept)
   WHERE r.name CONTAINS "Supply Chain"
   RETURN r, c, relationships

3. CRAG Evaluation:
   - Score: 0.92 (Very high confidence)
   - Action: Direct response

4. Response Structure:
   ## Primary Supply Chain Risks
   1. **Battery Supply Concentration** [10-K 2025, Risk Factor #3]
      - 60% lithium supply from single country
      - Mitigation: Diversification to NA suppliers by 2026
   
   2. **Semiconductor Shortages** [10-Q Q3 2025, MD&A]
      - Production delays in Q3 due to chip constraints
      - Impact: -8% vehicle deliveries vs guidance
   
   ## Network Effects (from Supply Chain Graph)
   - Key Supplier: Panasonic (PageRank: 0.87 - Critical chokepoint)
   - Geographic Concentration: 45% COGS from China-based suppliers
```

---

### Query Type 3: Cross-Filing Temporal Analysis
**User Query**: "How has MSFT's cloud revenue narrative changed over the last 3 years?"

**Agent Workflow**:
```
1. Temporal Retrieval:
   - Fetch propositions: section="MD&A", keyword="cloud revenue"
   - Filing dates: [2023-Q4, 2024-Q4, 2025-Q4]

2. Vector Comparison:
   - Embed each year's cloud narrative
   - Calculate: cosine_similarity(2023, 2024), cosine_similarity(2024, 2025)

3. Graph Evolution:
   MATCH (s:Strategy {name:"Cloud Growth"})-[r:DRIVES]->(c:Concept)
   WHERE r.filing_date IN ['2023-10-31', '2024-10-31', '2025-10-31']
   RETURN s, c, r.strength

4. Response Structure:
   ## Narrative Evolution Timeline
   
   ### 2023 (10-K filed 2023-10-31)
   - Focus: Infrastructure scale, Azure adoption
   - Growth Rate: +28% YoY
   - Key Phrase: "Hybrid cloud leadership"
   
   ### 2024 (10-K filed 2024-10-31)
   - Shift Detected (Cosine Similarity: 0.72 - Moderate drift)
   - Focus: AI-powered cloud services, OpenAI integration
   - Growth Rate: +31% YoY
   - Key Phrase: "AI-first cloud platform"
   
   ### 2025 (10-K filed 2025-10-31)
   - Consistency Score: 0.89 (High alignment with 2024)
   - Focus: Enterprise AI deployment, Copilot adoption
   - Growth Rate: +35% YoY
   - Key Phrase: "Leading the AI cloud transformation"
   
   ## Interpretation
   Strategic pivot occurred in 2024 toward AI-centric positioning. 
   2025 narrative reinforces this shift with accelerating growth.
```

---

## Fallback to Web Search Agent

### Trigger Conditions
CRAG Evaluator Score < 0.5 indicates:
- Query about recent events NOT in 10-K/10-Q (filed quarterly)
- Breaking news that contradicts historical filings
- Market-moving events since last filing

### Example Fallback
**User Query**: "Why is AAPL down today?"

**Agent Workflow**:
```
1. Historical Retrieval:
   - Latest 10-K: 2025-10-31 (4 months old)
   - CRAG Score: 0.32 (Low - outdated for real-time question)

2. Trigger Web Search Agent:
   - Query: "Apple stock decline February 2026 news"
   - Returns: "iPhone demand weakness in China" (breaking news)

3. Hybrid Response:
   ## Real-Time Context (Web Search)
   - AAPL down 3% today following reports of iPhone demand weakness in China
   - Source: Reuters, 2 hours ago
   
   ## Historical Context (10-K Analysis)
   - China represents 18% of total revenue (10-K 2025, Geographic Breakdown)
   - Risk Factor #7: "Significant revenue concentration in China creates volatility"
   
   ## Synthesis
   Today's decline aligns with disclosed risk in Risk Factors. 
   Historical precedent: Similar news in Q2 2024 led to 5% drop, recovered within 2 weeks.
```

---

## Output Format

### Standard Response Structure
```markdown
## [Topic] Overview
[2-3 sentence executive summary with inline citations]

## Key Findings
1. **[Insight 1]** [10-K 2025-10-31, Page 23, Item 7]
   - Supporting detail
   - Quantitative data

2. **[Insight 2]** [10-Q Q3 2025, MD&A Section]
   - Context
   - Management commentary

## Strategic Implications
[Synthesis of findings with graph relationships]

## Risk Considerations
[Extracted from Item 1A with graph-based network effects]

## Data Provenance
- Primary Source: [EODHD/FMP]
- Filing Date: [YYYY-MM-DD]
- Sections Analyzed: [Item 1, Item 1A, Item 7]
- Confidence Score: [0.XX]
- Graph Relationships: [X nodes, Y edges traversed]
```

---

## Integration with Other Agents

### Upstream Dependencies
- **None** (Primary source agent, reads directly from SEC filings)

### Downstream Consumers
| Agent | Data Consumed | Use Case |
|-------|---------------|----------|
| **Supply Chain Graph** | Entity extraction (suppliers, customers) from Item 1 | Build network graph |
| **Macro Economic** | Geographic revenue breakdown | FX impact analysis |
| **Insider & Sentiment** | Management outlook from MD&A | Cross-validate with insider trading |
| **Quantitative Fundamental** | Accounting policy changes | Adjust forensic calculations |

---

## Limitations & Constraints

### Temporal Lag
- 10-K: Filed within 60-90 days of fiscal year end
- 10-Q: Filed within 40-45 days of quarter end
- **Implication**: Data can be 1-3 months stale
- **Mitigation**: CRAG fallback to Web Search Agent for real-time updates

### Proposition Granularity
- Average proposition length: 1-2 sentences
- **Risk**: Losing context across multi-paragraph arguments
- **Mitigation**: Graph relationships connect related propositions

### Hallucination Prevention
- **Zero Tolerance**: All claims must cite specific filing + page number
- **Verification**: Dual-source validation (EODHD + FMP cross-check)
- **Confidence Scoring**: CRAG evaluator flags low-confidence retrievals

---

## Performance Metrics

### Retrieval Quality
- **Precision@5**: 0.89 (89% of top-5 results relevant)
- **Recall@10**: 0.76 (captures 76% of relevant propositions)
- **CRAG Evaluator Accuracy**: 0.92 (92% correct high/low confidence calls)

### Query Latency
- **Hybrid Search**: 200-400ms (Qdrant + BM25)
- **Graph Traversal**: 50-150ms (Neo4j Cypher)
- **CRAG Evaluation**: 100-200ms (BERT scoring)
- **Total End-to-End**: 1.5-2.5 seconds

---

## When to Use This Agent

### ✅ Ideal For:
- "What is [company]'s strategy for [domain]?"
- "Analyze [company]'s risk factors"
- "How has [company]'s business model evolved?"
- "What does management say about [topic]?"
- "Compare [company A] vs [company B] competitive positioning"

### ❌ Not Suitable For:
- Real-time stock price movements (use Web Search Agent)
- Financial calculations (use Quantitative Fundamental Agent)
- Supply chain network analysis (use Supply Chain Graph Agent)
- Insider trading patterns (use Insider & Sentiment Agent)
- Macroeconomic trends (use Macro Economic Agent)

---

## Example Planner Routing Logic

```python
def route_query_to_business_analyst(query: str) -> bool:
    """Determine if query should be routed to Business Analyst Agent"""
    
    strategic_keywords = [
        'strategy', 'business model', 'competitive', 'positioning',
        'growth plan', 'market share', 'differentiation', 'moat'
    ]
    
    risk_keywords = [
        'risk', 'threat', 'challenge', 'vulnerability', 'exposure',
        'regulatory', 'compliance', 'litigation'
    ]
    
    filing_keywords = [
        '10-K', '10-Q', 'MD&A', 'management discussion', 'Item 1',
        'Risk Factors', 'Item 1A', 'Business Overview'
    ]
    
    # Check for strategy/risk/filing-related queries
    if any(kw in query.lower() for kw in strategic_keywords + risk_keywords + filing_keywords):
        return True
    
    # Check for temporal analysis of narrative
    if ('changed' in query.lower() or 'evolved' in query.lower()) and \
       ('years' in query.lower() or 'quarters' in query.lower()):
        return True
    
    return False
```

---

## Error Handling

### Scenario 1: No Filings Found
```
User Query: "Analyze XYZ's AI strategy"
Issue: XYZ is a private company (no public filings)

Response:
"I cannot analyze XYZ's AI strategy as this company does not have public 
SEC filings (10-K/10-Q) in my database. My analysis capabilities are 
limited to publicly-traded companies with SEC filing obligations.

Would you like me to search for public information about XYZ using the 
Web Search Agent instead?"
```

### Scenario 2: CRAG Low Confidence + Web Search Unavailable
```
User Query: "What is AAPL's quantum computing strategy?"
Issue: Topic not mentioned in 10-K, web search returns no results

Response:
"I could not find substantive information about Apple's quantum computing 
strategy in available SEC filings (10-K/10-Q from 2023-2025) or recent 
web sources.

Confidence Score: 0.28 (Low)

This suggests either:
1. Apple has not publicly disclosed quantum computing initiatives
2. Quantum computing is not a strategic focus area
3. Information may be in earlier filings (pre-2023) not currently indexed

Would you like me to analyze related topics like Apple's overall R&D 
strategy or semiconductor investments instead?"
```

---

## Version History
- **v2.0** (2026-02-13): Graph-Augmented CRAG with proposition chunking
- **v1.5** (2025-11-01): Added temporal contrastive analysis
- **v1.0** (2025-08-15): Initial implementation with standard RAG
