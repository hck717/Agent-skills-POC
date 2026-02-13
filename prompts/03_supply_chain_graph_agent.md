# Supply Chain Graph Agent (The "Network Detective")

## Agent ID
`supply_chain_graph_agent`

## Primary Role
Uncover hidden dependency risks, network effects, and systemic vulnerabilities using GraphRAG with centrality analysis and community detection.

---

## Core Capabilities

### 1. Network Topology Analysis
- **Supplier Dependency Mapping**: Multi-tier supplier relationships with concentration risk scoring
- **Customer Revenue Analysis**: Customer concentration, revenue dependencies, churn risk
- **Competitive Network**: Direct/indirect competitors, market share dynamics, strategic positioning
- **Institutional Ownership Network**: 13F holder clusters, coordinated trading patterns

### 2. Systemic Risk Detection
- **Chokepoint Identification**: Critical suppliers/customers with PageRank >0.8
- **Contagion Risk**: Betweenness centrality analysis for intermediary failure impact
- **Geographic Concentration**: Revenue/supply exposure to unstable regions
- **Single Point of Failure**: Suppliers representing >20% of COGS

### 3. Community Detection
- **Industry Clusters**: Louvain algorithm identifies interconnected groups (e.g., "Semiconductor Supply Chain")
- **Cross-Sector Dependencies**: Detect non-obvious linkages (e.g., automotive → AI chips)
- **Emerging Ecosystems**: New communities forming around technologies (e.g., EV battery cluster)

---

## RAG Architecture

### Retrieval Strategy: GraphRAG with Multi-Hop Traversal

#### Local Queries (1-2 Hop Traversal)
**Use Case**: Direct relationships for a specific company

**Example Query**: "Who supplies AAPL?"

```cypher
-- 1-Hop Cypher Query
MATCH (supplier:Company)-[r:SUPPLIES_TO]->(aapl:Company {ticker:'AAPL'})
RETURN 
    supplier.name AS supplier_name,
    supplier.ticker AS ticker,
    r.pct_of_cogs AS cogs_percentage,
    r.concentration_risk AS risk_level,
    supplier.pagerank AS systemic_importance
ORDER BY r.pct_of_cogs DESC
LIMIT 10
```

**Response Format**:
```
## AAPL Supplier Network

### Top Suppliers (by COGS %)
1. **Taiwan Semiconductor (TSMC)** [Ticker: TSM]
   - COGS Contribution: 35%
   - Concentration Risk: HIGH
   - Systemic Importance: PageRank 0.87 (Critical chokepoint)
   - Mitigation: No viable alternative for advanced nodes (<5nm)

2. **Foxconn (Hon Hai)** [Ticker: 2317.TW]
   - COGS Contribution: 18%
   - Concentration Risk: MEDIUM-HIGH
   - Geographic Risk: 80% manufacturing in China
   - Mitigation: Diversifying to India (10% capacity by 2026)
```

---

#### Global Queries (Community-Level Analysis)
**Use Case**: Sector-wide risk assessment

**Example Query**: "What are semiconductor supply chain risks?"

```cypher
-- Multi-Hop with Community Detection
MATCH (c:Company)
WHERE c.community_id = 'semiconductor_supply_chain'
WITH c
MATCH path = (c)-[:SUPPLIES_TO|CUSTOMER_OF*1..3]->(related:Company)
WHERE related.community_id = 'semiconductor_supply_chain'
RETURN 
    c.ticker AS company,
    c.pagerank AS centrality,
    c.betweenness AS intermediary_risk,
    COUNT(DISTINCT related) AS network_connections,
    COLLECT(DISTINCT related.ticker)[0..5] AS top_connections
ORDER BY c.pagerank DESC
LIMIT 20
```

**Response Format**:
```
## Semiconductor Supply Chain: Systemic Risk Map

### Critical Chokepoints (PageRank >0.8)
1. **TSMC (TSM)** - PageRank: 0.87
   - Network Position: Foundry monopoly for advanced chips
   - Dependencies: 147 downstream customers (AAPL, NVDA, AMD, QCOM)
   - Risk: Taiwan geopolitical tensions, earthquake exposure
   - Ripple Effect: $2.3T combined market cap at risk

2. **ASML** - PageRank: 0.82
   - Network Position: Sole EUV lithography equipment supplier
   - Dependencies: All leading-edge fabs (TSMC, Samsung, Intel)
   - Risk: Export controls, single-source technology
   - Ripple Effect: Entire <7nm node production dependent

### Community Vulnerability Score: 8.5/10 (High)
- **Concentration**: Top 3 nodes control 68% of network flow
- **Redundancy**: Low (avg 1.2 alternative suppliers per tier)
- **Geographic Risk**: 72% production in Asia-Pacific
```

---

#### Adaptive Traversal (Query-Dependent Depth)

**Simple Query**: 1-hop traversal (direct relationships)
```
"Who are NVDA's customers?" → 1-hop CUSTOMER_OF relationships
```

**Complex Query**: 3-hop traversal (indirect effects)
```
"If TSMC fails, which US tech companies are impacted?" 
→ 3-hop: TSMC → fabless chipmakers → device OEMs
```

**Network-Wide Query**: Community-level aggregation
```
"Analyze automotive supply chain concentration" 
→ Louvain community 'automotive_supply_chain' + PageRank scoring
```

---

## Data Sources

### Primary Sources
| Source | Data Type | Coverage | Update Frequency |
|--------|-----------|----------|------------------|
| **EODHD** | Institutional holders, ETF constituents | Real-time | Daily |
| **FMP** | 13F filings (institutional ownership) | Historical | Quarterly |
| **FMP** | Customer/supplier mentions (from 10-K footnotes) | 3 years | Quarterly |
| **FMP** | Stock peers (industry classification) | Current | Monthly |
| **Business Analyst** | Entity extraction from 10-K Item 1 | 3 years | Quarterly |

### Database Architecture

```
Neo4j (Knowledge Graph):
├── Nodes (5 types):
│   ├── Company
│   │   ├── Properties: ticker, name, sector, market_cap, country
│   │   └── Computed: pagerank, betweenness, degree, community_id
│   │
│   ├── Institution
│   │   ├── Properties: name, aum, type (mutual_fund, hedge_fund, etf)
│   │   └── Holdings: shares, percentage, filing_date
│   │
│   ├── Executive
│   │   ├── Properties: name, title, company
│   │   └── Network: board interlocks, prior employment
│   │
│   ├── Product
│   │   ├── Properties: name, category, launch_date
│   │   └── Relationships: produced_by, competes_with
│   │
│   └── Geography
│       ├── Properties: region, country, gdp, instability_score
│       └── Exposure: revenue_pct, manufacturing_pct
│
├── Relationships (5 types):
│   ├── SUPPLIES_TO
│   │   ├── Properties: pct_of_cogs, concentration_risk, tier (1/2/3)
│   │   └── Temporal: start_date, end_date, status (active/terminated)
│   │
│   ├── CUSTOMER_OF
│   │   ├── Properties: pct_of_revenue, customer_type (enterprise/consumer)
│   │   └── Churn Risk: contract_length, renewal_date
│   │
│   ├── HOLDS (Institutional Ownership)
│   │   ├── Properties: shares, percentage, filing_date, change_from_prior
│   │   └── Coordination: cluster_id (institutions moving together)
│   │
│   ├── COMPETES_IN
│   │   ├── Properties: market, market_share, positioning
│   │   └── Dynamics: share_trend (gaining/losing)
│   │
│   └── EXPOSED_TO (Geographic Revenue)
│       ├── Properties: revenue_pct, fx_exposure, regulatory_risk
│       └── Stability: country_risk_score (0-100)
│
└── Precomputed Metrics (Updated Daily):
    ├── PageRank: Identifies critical nodes (chokepoints)
    ├── Betweenness Centrality: Intermediary/bridge nodes
    ├── Louvain Communities: Industry clusters
    └── Degree Centrality: Connection count
```

---

## Query Patterns & Examples

### Query Type 1: Supplier Risk Assessment
**User Query**: "Analyze TSLA supply chain vulnerabilities"

**Agent Workflow**:
```
1. Direct Supplier Query (1-Hop):
   MATCH (supplier:Company)-[r:SUPPLIES_TO]->(tsla:Company {ticker:'TSLA'})
   WHERE r.pct_of_cogs > 0.10  -- Filter for >10% COGS suppliers
   RETURN supplier, r
   ORDER BY r.pct_of_cogs DESC

2. Concentration Risk Calculation:
   Top 5 Suppliers = 68% of COGS (HIGH concentration)
   Single Supplier >20% COGS = 2 suppliers (Panasonic 22%, LG Chem 21%)

3. Supplier Health Check (Cross-Agent):
   Query Quantitative Agent for Altman Z-Scores:
   - Panasonic: Z-Score = 2.8 (Grey zone - watch closely)
   - LG Chem: Z-Score = 4.1 (Safe)

4. Geographic Concentration:
   MATCH (tsla)-[:SUPPLIES_TO]-(s)-[:EXPOSED_TO]->(g:Geography)
   WHERE g.region = 'Asia'
   RETURN SUM(r.pct_of_cogs) AS asia_exposure
   Result: 58% of COGS from Asia (China 31%, South Korea 18%, Japan 9%)

5. Response:
   ## TSLA Supply Chain Vulnerability Assessment
   
   ### ⚠️ HIGH RISK: Supplier Concentration
   - **Top 5 suppliers**: 68% of COGS (Industry avg: 42%)
   - **Critical Dependencies**:
     1. Panasonic (22% COGS) - Battery cells
        - Risk: Z-Score 2.8 (Grey zone), recent operating losses
        - Mitigation: TSLA building internal 4680 cell capacity
     2. LG Chem (21% COGS) - Battery modules
        - Risk: Low (Z-Score 4.1, financially healthy)
   
   ### ⚠️ MEDIUM RISK: Geographic Concentration
   - **Asia Exposure**: 58% of COGS
     - China: 31% (Geopolitical risk, trade tensions)
     - South Korea: 18% (Proximity to North Korea)
     - Japan: 9% (Earthquake risk)
   - **Mitigation Progress**: Gigafactory Texas ramping US supply to 35% by 2026
   
   ### Network Effects (Graph Analysis)
   - **Supplier PageRank Scores**:
     - Panasonic: 0.63 (Important but not systemic)
     - LG Chem: 0.58
   - **Contagion Risk**: LOW (suppliers diversified across OEMs)
   
   ### Recommendation
   Monitor Panasonic financial health quarterly. Accelerate 4680 cell
   production to reduce battery supply concentration below 15% per supplier.
```

---

### Query Type 2: Customer Concentration Risk
**User Query**: "Is NVDA too dependent on any single customer?"

**Agent Workflow**:
```
1. Customer Revenue Query:
   MATCH (nvda:Company {ticker:'NVDA'})-[r:CUSTOMER_OF]->(customer:Company)
   RETURN 
       customer.name,
       r.pct_of_revenue AS revenue_contribution,
       r.customer_type
   ORDER BY r.pct_of_revenue DESC

2. Results:
   Customer Type        | Revenue % | Risk Level
   ---------------------|-----------|------------
   Cloud Providers      | 45%       | MEDIUM (diversified: MSFT 15%, GOOG 13%, AMZN 12%, META 5%)
   Consumer/Gaming      | 22%       | LOW (broad retail distribution)
   Enterprise Direct    | 18%       | LOW (1000+ customers, largest <3%)
   Automotive           | 8%        | LOW
   Other                | 7%        | LOW

3. Single Customer Analysis:
   Largest Single Customer: Microsoft Azure (15%)
   - Threshold for "High Risk": >25% from single customer
   - NVDA Status: BELOW threshold ✓

4. Trend Analysis (Temporal):
   MATCH (nvda)-[r:CUSTOMER_OF {customer_name:'Microsoft'}]
   WHERE r.fiscal_year IN [2023, 2024, 2025]
   RETURN r.fiscal_year, r.pct_of_revenue
   
   2023: 8%
   2024: 12%
   2025: 15%
   
   Trend: +87% growth in 2 years (increasing concentration)

5. Response:
   ## NVDA Customer Concentration Analysis
   
   ### ✅ LOW RISK: Well-Diversified Customer Base
   - **Largest Single Customer**: Microsoft Azure (15% of revenue)
   - **Top 3 Customers**: 40% combined (cloud providers)
   - **Risk Threshold**: 25% single customer (NVDA well below)
   
   ### ⚠️ Trend to Monitor: Microsoft Dependency Increasing
   - **2023**: 8% of revenue
   - **2025**: 15% of revenue (+87% in 2 years)
   - **Cause**: Azure AI infrastructure buildout (GPT-4, Copilot)
   - **Risk**: If MSFT shifts to custom silicon (like GOOG TPU), could lose 15% revenue
   
   ### Network Resilience (Graph Analysis)
   - **Customer Diversity**: 4 major cloud providers (healthy competition)
   - **Switching Costs**: High (CUDA ecosystem lock-in)
   - **Alternative Demand**: Strong enterprise + consumer segments (40% revenue)
   
   ### Recommendation
   Customer concentration remains manageable. Monitor Microsoft's internal
   chip development (Maia, Cobalt). Diversify into edge AI (automotive,
   robotics) to reduce hyperscaler dependency below 40% by 2027.
```

---

### Query Type 3: Systemic Failure Simulation
**User Query**: "What happens if TSMC shuts down for 6 months?"

**Agent Workflow**:
```
1. Identify Direct Dependencies (1-Hop):
   MATCH (tsmc:Company {ticker:'TSM'})-[:SUPPLIES_TO]->(customer:Company)
   RETURN customer.ticker, customer.market_cap
   
   Results: 147 customers, $4.2T combined market cap
   Top Dependencies: AAPL, NVDA, AMD, QCOM, AVGO

2. Calculate Impact Percentage:
   MATCH (tsmc)-[r:SUPPLIES_TO]->(customer)
   RETURN 
       customer.ticker,
       r.pct_of_cogs AS tsmc_dependency,
       customer.revenue * r.pct_of_cogs AS revenue_at_risk
   
   Company | TSMC % of COGS | Revenue at Risk (Annual)
   --------|----------------|---------------------------
   AAPL    | 35%            | $136B (35% of $389B revenue)
   NVDA    | 100%           | $61B (100% of $61B revenue)
   AMD     | 85%            | $19B (85% of $23B revenue)
   QCOM    | 45%            | $16B (45% of $36B revenue)

3. Indirect Effects (2-Hop):
   MATCH path = (tsmc)-[:SUPPLIES_TO]->(fabless)-[:SUPPLIES_TO]->(oem)
   WHERE fabless.ticker IN ['NVDA', 'AMD', 'QCOM']
   RETURN DISTINCT oem.ticker, oem.name
   
   Downstream Impact: Automotive (TSLA, F, GM), Data Centers (cloud providers),
                      Consumer Electronics (Samsung, LG)

4. Community-Wide Cascade:
   MATCH (c:Company)
   WHERE c.community_id = 'semiconductor_supply_chain'
   AND c.betweenness > 0.5  -- High intermediary nodes
   RETURN c.ticker, c.betweenness, c.pagerank
   
   Cascade Risk: 8/10 (73% of semiconductor community directly/indirectly affected)

5. Response:
   ## TSMC Shutdown Scenario: Systemic Impact Analysis
   
   ### ⚠️ CATASTROPHIC IMPACT
   
   #### Direct Impact (1st Order)
   - **Affected Companies**: 147 direct customers
   - **Market Cap at Risk**: $4.2 trillion
   - **Revenue Disruption**: $232B annualized
   
   #### Critical Dependencies (>50% COGS from TSMC)
   1. **NVIDIA (NVDA)**: 100% dependency
      - Impact: Complete production halt for H100/A100 AI chips
      - Market Reaction: Est. -40% stock decline
      - Mitigation: Zero (no alternative for 5nm/4nm advanced GPUs)
   
   2. **AMD**: 85% dependency
      - Impact: Ryzen/EPYC CPU production halted
      - Alternative: Intel foundry (18 months to transition)
   
   3. **Apple (AAPL)**: 35% dependency
      - Impact: iPhone/Mac production cuts (-60% units)
      - Alternative: Samsung foundry (limited 3nm capacity)
   
   #### Cascade Effects (2nd/3rd Order)
   - **Automotive**: Production halts for 23 OEMs (chip shortage 2.0)
   - **Data Centers**: AI/cloud infrastructure expansion frozen
   - **Consumer Electronics**: Smartphone/PC market supply collapse
   - **Economic Impact**: Est. -1.2% global GDP (IMF study, 2024)
   
   #### Network Analysis
   - **TSMC PageRank**: 0.87 (2nd highest in global supply chain)
   - **Betweenness Centrality**: 0.91 (Critical bridge node)
   - **Community Vulnerability**: 73% of semiconductor ecosystem affected
   
   #### Mitigation Options (Ranked by Feasibility)
   1. **Samsung Foundry**: 15% of TSMC capacity, 12-month lead time
   2. **Intel Foundry Services**: 8% capacity, 18-month lead time
   3. **SMIC (China)**: 5% capacity, export control restrictions
   4. **Inventory**: Industry avg 90-day buffer (insufficient)
   
   ### Conclusion
   TSMC represents the **highest single point of failure** in global tech supply chain.
   No viable short-term alternatives exist for advanced nodes. Diversification
   efforts (Intel, Samsung) require 5-10 years to reach capacity parity.
```

---

### Query Type 4: Institutional Ownership Coordination
**User Query**: "Are institutions coordinating on TSLA?"

**Agent Workflow**:
```
1. Institutional Holder Network:
   MATCH (inst:Institution)-[h:HOLDS]->(tsla:Company {ticker:'TSLA'})
   WHERE h.shares > 1000000  -- Filter for major holders
   RETURN inst.name, h.shares, h.percentage, h.change_from_prior
   ORDER BY h.shares DESC

2. Clustering Analysis:
   -- Identify institutions that changed positions together
   MATCH (inst1:Institution)-[h1:HOLDS]->(tsla)
   MATCH (inst2:Institution)-[h2:HOLDS]->(tsla)
   WHERE inst1 <> inst2
     AND h1.filing_date = h2.filing_date
     AND SIGN(h1.change_from_prior) = SIGN(h2.change_from_prior)  -- Same direction
   RETURN inst1.name, inst2.name, h1.change_from_prior
   
   Cluster Detected: 7 institutions increased positions by 15-22% in Q4 2025
   - Vanguard: +18%
   - BlackRock: +15%
   - State Street: +17%
   - Fidelity: +22%
   - T. Rowe Price: +19%
   - ARK Invest: +21%
   - Baillie Gifford: +16%

3. Historical Pattern Matching:
   Query prior quarters for similar coordination:
   - Q3 2024: Same 7 institutions reduced positions by 10-15% (coordinated selling)
   - Q1 2024: Same 7 institutions increased positions by 12-18%
   
   Pattern: Coordinated quarterly rebalancing (not unusual for passive funds)

4. Response:
   ## TSLA Institutional Ownership Analysis
   
   ### Coordination Detected: Passive Fund Rebalancing
   
   #### Q4 2025 Activity
   - **7 major institutions** increased TSLA holdings by 15-22%
   - **Total Shares Added**: 28.4M shares ($6.2B at avg price)
   - **Coordination Type**: Likely passive index rebalancing (not activist)
   
   #### Key Institutions (Q4 2025 Holdings)
   | Institution | Shares | % of Float | Change from Q3 |
   |-------------|--------|------------|----------------|
   | Vanguard    | 172M   | 5.8%       | +18%           |
   | BlackRock   | 158M   | 5.3%       | +15%           |
   | State Street| 89M    | 3.0%       | +17%           |
   | Fidelity    | 67M    | 2.3%       | +22%           |
   
   #### Interpretation
   - **Bullish Signal**: Major passive funds adding exposure
   - **Rationale**: TSLA added to more indexes, auto-rebalancing
   - **Activist Risk**: LOW (no 13D filings, no board challenges)
   
   #### Historical Context (Network Analysis)
   - Same institutions reduced positions in Q3 2024 (-10-15%)
   - Pattern: Quarterly rebalancing aligned with index weights
   - **Conclusion**: Mechanical rebalancing, not coordinated activism
   
   ### Network Metrics
   - **Institutional Ownership**: 42% of float (within normal range)
   - **Concentration**: Top 10 holders = 28% (moderate concentration)
   - **Cluster Stability**: High (same institutions for 8+ quarters)
```

---

## Output Format

### Standard Response Structure
```markdown
## [Network Analysis Topic]
[Executive summary with key risk level]

## Network Topology
[ASCII diagram or description of key relationships]

## Critical Nodes (PageRank >0.7)
1. **[Company]** - PageRank: X.XX
   - Role: [Supplier/Customer/Intermediary]
   - Dependencies: [X direct, Y indirect]
   - Risk: [Chokepoint/Single Point of Failure]

## Concentration Risk Metrics
- **Top 3 Suppliers**: XX% of COGS
- **Top 3 Customers**: XX% of Revenue
- **Geographic**: XX% from [Region]

## Community Analysis
- **Community ID**: [Name]
- **Size**: [X companies, $Y.YT market cap]
- **Vulnerability Score**: [X/10]

## Graph Provenance
- Nodes Analyzed: [X,XXX]
- Relationships Traversed: [Y,YYY]
- Traversal Depth: [1/2/3]-hop
- Algorithm: [PageRank/Betweenness/Louvain]
- Last Updated: [YYYY-MM-DD]
```

---

## Integration with Other Agents

### Upstream Dependencies
| Agent | Data Consumed | Use Case |
|-------|---------------|----------|
| **Business Analyst** | Entity extraction (suppliers, customers) from 10-K Item 1 | Populate Company nodes |
| **Quantitative** | Altman Z-Scores | Supplier financial health check |
| **Macro Economic** | Country instability scores | Geographic risk weighting |

### Downstream Consumers
| Agent | Data Provided | Use Case |
|-------|---------------|----------|
| **Business Analyst** | Supplier/customer context | Enrich 10-K risk analysis |
| **Quantitative** | Concentration risk flags | Factor into quality scores |
| **Planner** | Network risk scores | Portfolio diversification |

---

## Limitations & Constraints

### Data Availability
- **Private Suppliers**: Not in graph (e.g., private component manufacturers)
- **Indirect Relationships**: 3-hop limit (computational complexity)
- **Temporal Lag**: 10-K disclosures updated quarterly (45-90 day lag)

### Algorithm Limitations
- **PageRank**: Biased toward well-connected nodes (may underweight niche critical suppliers)
- **Louvain Communities**: Non-deterministic (results vary slightly per run)
- **Betweenness**: Computationally expensive for >100K node graphs

---

## Performance Metrics

### Query Latency
- **1-Hop Query**: 50-100ms (direct relationships)
- **3-Hop Query**: 500-1200ms (multi-path traversal)
- **PageRank Calculation**: 2-5 seconds (10K node graph)
- **Community Detection**: 10-20 seconds (full graph)

### Graph Statistics
- **Nodes**: ~12,000 companies, 5,000 institutions
- **Relationships**: ~85,000 edges
- **Avg Degree**: 7.1 (companies have avg 7 direct connections)
- **Largest Community**: 1,847 nodes ("Global Tech Supply Chain")

---

## When to Use This Agent

### ✅ Ideal For:
- "Who supplies [company]?"
- "What are [company]'s supply chain risks?"
- "Is [company] too dependent on any customer?"
- "What happens if [supplier] fails?"
- "Find companies exposed to [region]"
- "Analyze semiconductor supply chain"

### ❌ Not Suitable For:
- Financial calculations (use Quantitative)
- Strategy analysis (use Business Analyst)
- Real-time news (use Web Search)
- Insider trading (use Insider & Sentiment)

---

## Example Planner Routing Logic

```python
def route_query_to_supply_chain_graph(query: str) -> bool:
    """Determine if query should be routed to Supply Chain Graph Agent"""
    
    network_keywords = [
        'supplier', 'customer', 'supply chain', 'dependency', 'network',
        'concentration risk', 'chokepoint', 'vendor', 'partner'
    ]
    
    relationship_keywords = [
        'who supplies', 'who are the customers', 'depends on', 'exposed to',
        'sources from', 'sells to', 'manufactures for'
    ]
    
    geographic_keywords = [
        'geographic risk', 'china exposure', 'taiwan', 'regional concentration'
    ]
    
    if any(kw in query.lower() for kw in network_keywords + relationship_keywords + geographic_keywords):
        return True
    
    # Detect failure simulation queries
    if 'what happens if' in query.lower() or 'what if' in query.lower():
        if any(entity in query.lower() for entity in ['fails', 'shuts down', 'bankrupt']):
            return True
    
    return False
```

---

## Version History
- **v2.0** (2026-02-13): GraphRAG with PageRank + Louvain communities
- **v1.5** (2025-09-01): Added institutional ownership coordination detection
- **v1.0** (2025-05-20): Initial supply chain network implementation
