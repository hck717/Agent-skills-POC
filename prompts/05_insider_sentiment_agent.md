# Insider & Sentiment Agent (The "Behavioral Analyst")

## Agent ID
`insider_sentiment_agent`

## Primary Role
Detect insider trading patterns, analyze management sentiment from earnings calls, and cross-validate narrative consistency using Time-Series Analysis + Temporal Contrastive RAG.

---

## Core Capabilities

### 1. Insider Trading Analysis
- **Pattern Detection**: Clustered buying/selling by executives, directors, 10% owners
- **Transaction Timing**: Pre-earnings vs post-earnings trades, blackout period violations
- **Volume Analysis**: Shares traded as % of total holdings (high conviction signals)
- **Role-Based Segmentation**: CEO/CFO trades (strategic) vs Director trades (governance)

### 2. Management Sentiment Scoring
- **Earnings Call Tone**: Hawkish (optimistic) vs Dovish (cautious) language detection
- **Q&A Evasiveness**: Deflection patterns, non-answer responses, hedging language
- **Temporal Consistency**: Compare Q-over-Q sentiment for narrative drift
- **Forward Guidance Changes**: Upgrade/downgrade/withdrawal of guidance

### 3. Cross-Validation & Red Flags
- **Insider-Sentiment Divergence**: Positive tone + insider selling = ⚠️ Red flag
- **Sentiment-Fundamentals Divergence**: Optimistic guidance + margin compression = ⚠️ Warning
- **Pre-Announcement Trades**: Abnormal insider activity 30 days before material news

---

## RAG Architecture

### Dual Strategy: Time-Series SQL + Temporal Contrastive RAG

#### 1. Insider Trading Time-Series (TimescaleDB)
**Use Case**: Quantitative insider transaction analysis

**Example Query**: "Are NVDA insiders selling?"

**SQL Execution**:
```sql
-- Aggregate last 6 months of insider trades
WITH recent_trades AS (
    SELECT 
        ticker,
        owner,
        position,
        date,
        transaction_type,
        shares_traded,
        price_per_share,
        shares_traded * price_per_share AS trade_value,
        shares_owned_after
    FROM insider_transactions
    WHERE ticker = 'NVDA'
      AND date >= NOW() - INTERVAL '6 months'
),
aggregated AS (
    SELECT 
        ticker,
        position,
        SUM(CASE WHEN transaction_type = 'Buy' THEN shares_traded ELSE 0 END) AS total_buys,
        SUM(CASE WHEN transaction_type = 'Sell' THEN shares_traded ELSE 0 END) AS total_sells,
        SUM(CASE WHEN transaction_type = 'Buy' THEN trade_value ELSE 0 END) AS buy_value,
        SUM(CASE WHEN transaction_type = 'Sell' THEN trade_value ELSE 0 END) AS sell_value,
        COUNT(DISTINCT owner) AS num_insiders
    FROM recent_trades
    GROUP BY ticker, position
)
SELECT 
    ticker,
    position,
    total_buys,
    total_sells,
    (total_sells - total_buys) AS net_sells,
    buy_value,
    sell_value,
    (sell_value - buy_value) AS net_sell_value,
    num_insiders,
    CASE 
        WHEN (sell_value - buy_value) > 100000000 THEN 'Heavy Selling'
        WHEN (sell_value - buy_value) > 10000000 THEN 'Moderate Selling'
        WHEN ABS(sell_value - buy_value) < 10000000 THEN 'Neutral'
        WHEN (buy_value - sell_value) > 10000000 THEN 'Moderate Buying'
        ELSE 'Heavy Buying'
    END AS insider_signal
FROM aggregated
ORDER BY position
```

**Response**:
```
## NVDA Insider Trading Analysis (Last 6 Months)

### ⚠️ CAUTION: Heavy Insider Selling

#### Executive Summary
- **Net Selling**: 2.4M shares ($2.1B net value)
- **Signal Strength**: Heavy Selling
- **Participants**: 14 insiders (CEO, CFO, 8 SVPs, 4 Directors)

#### By Role
| Role | Shares Bought | Shares Sold | Net Sold | Value Net Sold | Signal |
|------|---------------|-------------|----------|----------------|--------|
| **CEO** | 0 | 850K | 850K | $740M | Heavy Selling |
| **CFO** | 0 | 320K | 320K | $280M | Heavy Selling |
| **SVPs** | 12K | 980K | 968K | $840M | Heavy Selling |
| **Directors** | 5K | 290K | 285K | $248M | Moderate Selling |

#### Context & Interpretation
1. **Timing**: Selling concentrated in Dec 2025 - Jan 2026 (post Q3 earnings)
2. **Stock Performance**: NVDA +142% over same 6-month period
   - **Bullish Interpretation**: Profit-taking after massive run-up (tax planning)
   - **Bearish Interpretation**: Insiders cashing out at perceived peak

3. **Holdings Retention**:
   - CEO: Sold 15% of holdings (retains 85% → Still aligned)
   - CFO: Sold 22% of holdings (retains 78%)
   - Average SVP: Sold 28% of holdings

4. **Historical Pattern**:
   - Query prior 6 months (Jun-Nov 2025): Net selling of $980M
   - Pattern: Consistent selling for 12+ months (not a new trend)

#### Red Flags?
✅ **No major red flags**:
- Selling is proportional to holdings (15-28% liquidation)
- No blackout period violations
- Gradual over 6 months (not clustered)
- Stock has performed exceptionally (+142%)

⚠️ **Mild Concern**:
- Volume of selling is elevated vs historical norms
- Cross-check with sentiment analysis from earnings calls

### Recommendation
Insider selling appears to be profit-taking rather than distress signal.
Monitor Q4 2025 earnings call (Feb 20) for sentiment consistency.
```

---

#### 2. Earnings Call Sentiment Analysis (Temporal Contrastive RAG)
**Use Case**: Qualitative management tone analysis with quarter-over-quarter comparison

**Example Query**: "Has NVDA management sentiment changed?"

**Retrieval Workflow**:
```python
# Step 1: Retrieve current quarter transcript
current_q = qdrant.search(
    collection="insider_sentiment_transcripts",
    query_vector=embed("management outlook guidance sentiment"),
    filter={
        "ticker": "NVDA",
        "fiscal_quarter": "2025-Q4",
        "speaker_role": ["CEO", "CFO"]  # Focus on C-suite
    },
    limit=10
)

# Step 2: Retrieve prior quarter for comparison
prior_q = qdrant.search(
    collection="insider_sentiment_transcripts",
    query_vector=embed("management outlook guidance sentiment"),
    filter={
        "ticker": "NVDA",
        "fiscal_quarter": "2025-Q3",
        "speaker_role": ["CEO", "CFO"]
    },
    limit=10
)

# Step 3: Temporal Contrastive Analysis
for current_chunk, prior_chunk in zip(current_q, prior_q):
    # Extract sentiment scores from metadata
    current_sentiment = current_chunk.payload['sentiment_score']  # -1 to +1
    prior_sentiment = prior_chunk.payload['sentiment_score']
    
    # Calculate drift
    sentiment_drift = current_sentiment - prior_sentiment
    
    # Semantic similarity (are they talking about same topics?)
    similarity = cosine_similarity(current_chunk.vector, prior_chunk.vector)
```

**Response**:
```
## NVDA Management Sentiment Analysis (Q4 2025 vs Q3 2025)

### ⚠️ SENTIMENT DRIFT DETECTED

#### Q4 2025 Earnings Call (Feb 2026)
**Overall Tone**: Moderately Optimistic (Sentiment Score: +0.52)

##### CEO (Jensen Huang) - Key Quotes
1. **AI Demand**
   - Q4 Quote: "Demand for Hopper and Blackwell remains strong, though we're 
               seeing more measured deployment timelines from hyperscalers."
   - **Sentiment**: +0.4 (Cautiously Optimistic)
   - **Keywords**: "strong" (positive), "measured" (hedging), "timelines" (delay signal)

2. **Gross Margin Outlook**
   - Q4 Quote: "Gross margins in Q1 will be impacted by Blackwell ramp costs, 
               but we expect normalization by Q2."
   - **Sentiment**: +0.2 (Neutral with caution)
   - **Keywords**: "impacted" (negative), "normalization" (positive future)

##### CFO (Colette Kress) - Key Quotes
1. **Revenue Guidance**
   - Q4 Quote: "Q1 revenue guidance is $26-27B, reflecting supply constraints 
               and customer digestion period."
   - **Sentiment**: -0.1 (Slightly Negative)
   - **Keywords**: "constraints" (negative), "digestion" (demand pause signal)

---

#### Q3 2025 Earnings Call (Nov 2025) - For Comparison
**Overall Tone**: Highly Optimistic (Sentiment Score: +0.78)

##### CEO - Q3 Quotes
- "We're in the early innings of the AI revolution. Demand is unprecedented."
- **Sentiment**: +0.9 (Highly Optimistic)

##### CFO - Q3 Quotes
- "Visibility extends well into 2026. Supply is the only constraint, not demand."
- **Sentiment**: +0.8 (Highly Optimistic)

---

#### Temporal Contrastive Analysis

| Dimension | Q3 2025 | Q4 2025 | Drift | Interpretation |
|-----------|---------|---------|-------|----------------|
| **Overall Sentiment** | +0.78 | +0.52 | -0.26 | ⚠️ Cooling enthusiasm |
| **Demand Tone** | "Unprecedented" | "Strong but measured" | -0.35 | ⚠️ Demand moderation |
| **Margin Confidence** | +0.7 | +0.2 | -0.50 | ⚠️ Near-term margin pressure |
| **Guidance Tone** | "Well into 2026" | "Q1 constraints" | -0.40 | ⚠️ Reduced visibility |

#### Key Changes
1. **From "Unprecedented Demand" → "Measured Deployment"**
   - Semantic Similarity: 0.65 (topic shift detected)
   - Interpretation: Hyperscalers digesting prior orders, slowing new capex

2. **From "Supply-Only Constraint" → "Supply Constraints + Digestion"**
   - **Red Flag**: Introducing "digestion" suggests demand-side cooling

3. **From "Visibility into 2026" → "Q1 Guidance with Caution"**
   - Shortened forecast horizon (from 12+ months to 3 months)

---

#### Cross-Validation with Fundamentals (Quantitative Agent)
- **Q4 Gross Margin**: 72.7% (down from 75.1% in Q3)
- **Guidance for Q1**: $26.5B midpoint (vs analyst consensus $28B)
- **Beat/Miss**: Beat Q4 estimates, but Q1 guide below street

#### Cross-Validation with Insider Trading
- **Insider Signal**: Heavy Selling ($2.1B net in last 6 months)
- **Divergence Score**: 0.74 (High)
  - Positive sentiment (+0.52) + Heavy insider selling = ⚠️ Red Flag

---

### Synthesis: Insider + Sentiment Red Flag

⚠️ **WARNING SIGNAL**

1. **Sentiment Cooling**: -0.26 drift from Q3 to Q4 (significant)
2. **Insider Actions Contradict Tone**: Management sounds optimistic (+0.52), 
   but insiders aggressively selling ($2.1B)
3. **Guidance Disappointment**: Q1 guide below street expectations

#### Interpretation
- **Bull Case**: Insiders taking profits after 142% run, sentiment still net positive
- **Bear Case**: Management knows growth is slowing, insiders front-running deceleration

#### Recommendation
**REDUCE POSITION** - Divergence between insider actions and public sentiment 
suggests insiders have less confidence than conveyed. Monitor Q1 2026 results 
for demand confirmation. If revenue misses $26B or margins compress further, 
exit remaining position.
```

---

## Data Sources

### Primary Sources
| Source | Data Type | Coverage | Update Frequency |
|--------|-----------|----------|------------------|
| **EODHD** | Form 4 insider transactions (buy/sell/option exercise) | 2 years | Daily |
| **FMP** | Earnings call transcripts (full text) | 8 quarters | Quarterly |
| **FMP** | Insider transactions (augmented with SEC filings) | 2 years | Daily |
| **External** | Sentiment lexicon (Loughran-McDonald financial dictionary) | Static | N/A |

### Database Architecture

```
TimescaleDB (Time-Series Transactions):
├── Database: insider_sentiment.timescaledb
    ├── insider_transactions (hypertable, partitioned by time)
    │   ├── Columns: ticker, date, owner, position, transaction_type, shares_traded, 
    │   │          price_per_share, value, shares_owned_after
    │   ├── Indexes: (ticker, date), (owner), (position)
    │   └── Continuous Aggregates:
    │       ├── insider_monthly_net: Aggregate net buy/sell by month
    │       └── insider_quarterly_summary: Quarterly patterns for seasonality
    │
    └── pre_announcement_trades (derived)
        ├── Flags trades occurring 30 days before:
        │   ├── Earnings announcements
        │   ├── Material news (M&A, product launches)
        │   └── Guidance changes
        └── Red Flag: Abnormal volume + material news correlation

Qdrant (Transcript Embeddings):
├── Collection: insider_sentiment_transcripts
    ├── Documents: Earnings call transcripts chunked by speaker turn
    ├── Metadata:
    │   ├── ticker: Company ticker
    │   ├── fiscal_quarter: 2025-Q4
    │   ├── speaker: "Jensen Huang"
    │   ├── speaker_role: "CEO", "CFO", "Analyst"
    │   ├── section: "Prepared Remarks", "Q&A"
    │   ├── sentiment_score: -1.0 (Negative) to +1.0 (Positive)
    │   ├── evasiveness_score: 0.0 (Direct) to 1.0 (Evasive)
    │   ├── forward_looking: Boolean (contains guidance)
    │   └── keywords: ["AI", "demand", "margin", "supply"]
    │
    └── Embedding: nomic-embed-text (768-dim)
        ├── Temporal Contrastive: Q-over-Q comparison via cosine similarity
        └── Semantic Drift: Track topic changes over time
```

---

## Query Patterns & Examples

### Query Type 1: Insider Red Flag Detection
**User Query**: "Any suspicious insider trading at TSLA?"

**Agent Workflow**:
```
1. Pre-Announcement Analysis:
   SELECT 
       t.owner,
       t.position,
       t.date AS trade_date,
       t.transaction_type,
       t.shares_traded,
       t.value AS trade_value,
       e.announcement_date,
       e.event_type,
       e.stock_move_pct,
       (e.announcement_date - t.date) AS days_before_announcement
   FROM insider_transactions t
   JOIN material_events e ON t.ticker = e.ticker
   WHERE t.ticker = 'TSLA'
     AND t.date BETWEEN (e.announcement_date - INTERVAL '30 days') AND e.announcement_date
     AND e.stock_move_pct > 5  -- Material price impact
   ORDER BY t.date DESC

2. Results:
   Owner         | Position | Trade Date | Type | Value  | Event Date | Event     | Days Before | Stock Move
   --------------|----------|------------|------|--------|------------|-----------|-------------|------------
   Elon Musk     | CEO      | 2025-12-05 | Sell | $420M  | 2025-12-28 | Q4 Miss   | 23 days     | -12%
   Zachary K.    | CFO      | 2025-12-08 | Sell | $85M   | 2025-12-28 | Q4 Miss   | 20 days     | -12%
   Board Member  | Director | 2025-11-15 | Sell | $12M   | 2025-12-28 | Q4 Miss   | 43 days     | -12%

3. Statistical Significance:
   - Baseline Insider Selling (TSLA): $150M/month avg
   - Dec 2025 Selling: $517M (3.4x baseline ⚠️)
   - Timing: 20-23 days before negative earnings surprise

4. Legal Analysis:
   - Blackout Period: Typically 30 days before earnings
   - TSLA Policy: 45 days before earnings (per 10-K)
   - Violation? Trades at 20-23 days → WITHIN blackout ⚠️

5. Response:
   ## TSLA Insider Trading Red Flags (Q4 2025)
   
   ### ⚠️ CRITICAL: Pre-Announcement Selling + Blackout Violation
   
   #### Suspicious Pattern Detected
   1. **CEO (Elon Musk)**: Sold $420M worth 23 days before Q4 miss
      - Stock declined -12% post-earnings
      - Profit avoided: ~$50M (had he sold after announcement)
   
   2. **CFO (Zachary Kirkhorn)**: Sold $85M worth 20 days before Q4 miss
      - Timing highly correlated with CEO
   
   3. **Volume Anomaly**: $517M sold in Dec (3.4x monthly average)
   
   #### Blackout Period Analysis
   - **TSLA Policy**: 45-day blackout before earnings (per 10-K, Insider Trading Policy)
   - **Q4 Earnings Date**: Dec 28, 2025
   - **Blackout Start**: Nov 13, 2025
   - **Trades Occurred**: Dec 5-8 (WITHIN blackout period) ⚠️
   
   #### Possible Explanations
   1. **Innocent**: Pre-scheduled 10b5-1 trading plan
      - Check: SEC filings show NO 10b5-1 plan filed ⚠️
   
   2. **Policy Exemption**: Board granted exemption
      - Check: No exemption disclosed in 8-K filings ⚠️
   
   3. **Material Non-Public Info (MNPI)**: Insiders knew Q4 results weak
      - **Red Flag**: CFO involved (has access to real-time financials)
   
   #### Regulatory Risk
   - **SEC Investigation**: Possible if pattern deemed suspicious
   - **Shareholder Lawsuits**: Derivative action risk
   - **Reputational Damage**: Media scrutiny on Musk/TSLA governance
   
   ### Recommendation
   **AVOID** - Governance red flag. Wait for SEC clarification or 
   company statement. If no 10b5-1 plan, this is a material governance issue.
```

---

### Query Type 2: Q&A Evasiveness Detection
**User Query**: "Is TSLA management being evasive on margins?"

**Agent Workflow**:
```
1. Retrieve Q&A Section (Analyst Questions on Margins):
   analyst_questions = qdrant.search(
       collection="insider_sentiment_transcripts",
       query="gross margin outlook pressure",
       filter={
           "ticker": "TSLA",
           "fiscal_quarter": "2025-Q4",
           "section": "Q&A",
           "speaker_role": ["CEO", "CFO"]
       },
       limit=5
   )

2. Evasiveness Scoring (LLM Classification):
   For each Q&A exchange:
   - Analyst Question: "What's driving the margin compression?"
   - Management Response: "Well, there are many factors... mix shift, 
                          some supply chain... we're confident long-term."
   
   LLM Prompt:
   """
   Analyze this Q&A exchange for evasiveness:
   Question: [Analyst question]
   Answer: [Management response]
   
   Score 0-1:
   - 0.0 = Direct, specific answer with numbers/facts
   - 0.5 = Partial answer, some hedging
   - 1.0 = Completely evasive, no substance
   
   Also identify:
   - Deflection phrases: "we're focused on long-term", "many factors"
   - Non-answers: Question about margins, answer about volume
   """

3. Results:
   | Question Topic | Evasiveness Score | Deflection Detected |
   |----------------|-------------------|---------------------|
   | Margin compression | 0.82 | ✅ "Long-term focus" |
   | Cybertruck costs | 0.91 | ✅ "Ramp dynamics" (vague) |
   | Price cuts China | 0.35 | ❌ Direct answer (competitive) |
   | Q1 margin guide | 0.78 | ✅ "Too early to quantify" |

4. Historical Comparison:
   Q3 2025 Evasiveness (Same Topics): Avg 0.42
   Q4 2025 Evasiveness: Avg 0.72
   Delta: +0.30 (Significant increase in evasiveness)

5. Response:
   ## TSLA Management Evasiveness Analysis (Q4 2025 Call)
   
   ### ⚠️ HIGH EVASIVENESS: Margin Questions
   
   #### Q&A Exchange 1: Gross Margin Compression
   **Analyst**: "Gross margins declined 580 basis points Q-over-Q. Can you 
                 break down the drivers?"
   
   **CFO Response**: "There are several factors at play. Mix shift is part 
                      of it, some supply chain dynamics, and we're investing 
                      heavily for the long term. We remain confident in our 
                      margin structure over time."
   
   - **Evasiveness Score**: 0.82 (High)
   - **Analysis**:
     ❌ No specific breakdown requested
     ❌ "Several factors" without quantification
     ✅ Deflection: "Long-term confidence" (doesn't address near-term issue)
     ❌ "Investing heavily" (excuse, not explanation)
   
   **Direct Answer Would Be**: "Cybertruck ramp costs = 250bps, China price 
                                cuts = 180bps, unfavorable mix = 150bps."

   #### Q&A Exchange 2: Cybertruck Margin Impact
   **Analyst**: "What's the Cybertruck margin profile? When breakeven?"
   
   **CEO Response**: "Cybertruck ramp is progressing. Production is scaling. 
                      These things take time with new architectures."
   
   - **Evasiveness Score**: 0.91 (Very High)
   - **Analysis**:
     ❌ Zero numbers provided (no margin %, no breakeven timeline)
     ✅ Deflection: "Ramp is progressing" (status, not financial impact)
     ❌ Non-answer: Question asks "when", answer says "takes time"

   #### Comparison to Q3 2025 (Same Topics)
   | Topic | Q3 Evasiveness | Q4 Evasiveness | Change |
   |-------|----------------|----------------|--------|
   | Margins | 0.38 (Direct) | 0.82 (Evasive) | +0.44 ⚠️ |
   | New Products | 0.45 | 0.91 | +0.46 ⚠️ |
   
   ### Interpretation
   **Q3 2025**: Management provided specific margin bridge (e.g., "Raw 
                materials -120bps, labor efficiency +80bps")
   
   **Q4 2025**: Vague language, no quantification, deflection to "long-term"
   
   ⚠️ **Red Flag**: Sudden increase in evasiveness suggests:
   1. Margin situation worse than disclosed
   2. Management lacks confidence in near-term recovery
   3. Potential negative surprise in Q1 2026
   
   ### Recommendation
   Evasiveness spike is a behavioral red flag. Combined with insider selling 
   ($2.1B), this suggests management caution not reflected in public guidance. 
   **REDUCE EXPOSURE** until Q1 results clarify margin trajectory.
```

---

### Query Type 3: Sentiment Consistency Check
**User Query**: "Has AAPL management been consistent on services growth?"

**Agent Workflow**:
```
1. Temporal Query (Last 4 Quarters):
   services_sentiment = []
   for quarter in ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4']:
       results = qdrant.search(
           collection="insider_sentiment_transcripts",
           query="services revenue growth outlook",
           filter={
               "ticker": "AAPL",
               "fiscal_quarter": quarter,
               "speaker_role": ["CEO", "CFO"]
           },
           limit=3
       )
       services_sentiment.append({
           'quarter': quarter,
           'sentiment': np.mean([r.payload['sentiment_score'] for r in results]),
           'quotes': [r.payload['text'] for r in results]
       })

2. Results:
   Quarter | Sentiment | Key Quote
   --------|-----------|----------
   2025-Q1 | +0.72     | "Services growing double-digits, strong momentum"
   2025-Q2 | +0.68     | "Services remain a key growth driver"
   2025-Q3 | +0.45     | "Services growth moderating but stable"
   2025-Q4 | +0.38     | "Services growth in line with expectations"

3. Trend Analysis:
   - Sentiment Drift: -0.34 (Q1 to Q4) → Consistent cooling
   - Semantic Similarity: 0.82 (High - same topic, different tone)

4. Cross-Validation with Actuals (Quantitative Agent):
   Quarter | Guided Growth | Actual Growth | Beat/Miss
   --------|---------------|---------------|----------
   2025-Q1 | 10-12%        | 11.2%         | In-line
   2025-Q2 | 8-10%         | 9.5%          | In-line
   2025-Q3 | 7-9%          | 6.8%          | Miss
   2025-Q4 | 6-8%          | 7.1%          | In-line

5. Response:
   ## AAPL Services Sentiment Consistency (FY 2025)
   
   ### ✅ CONSISTENT: Gradual Deceleration Narrative
   
   #### Quarterly Sentiment Progression
   - **Q1**: +0.72 ("Strong momentum", "double-digit growth")
   - **Q2**: +0.68 ("Key growth driver", enthusiasm moderating)
   - **Q3**: +0.45 ("Moderating but stable", acknowledging slowdown)
   - **Q4**: +0.38 ("In line with expectations", new normal)
   
   #### Interpretation
   ✅ **Transparent**: Management gradually walked down expectations
   - No abrupt shifts (max Q-over-Q change: -0.23)
   - Narrative matched actuals (guided 6-8%, delivered 7.1%)
   
   #### Consistency Score: 0.89 (High)
   - Semantic Similarity: 0.82 (same topic throughout)
   - Sentiment Alignment: Narrative matched fundamental deceleration
   - No Divergence: Q3 miss acknowledged in Q3 call (not hidden)
   
   ### Cross-Validation
   - **Insider Trading**: Neutral (no unusual CEO/CFO activity)
   - **Fundamentals**: Services growth 11% → 7% (consistent with guidance)
   
   ### Conclusion
   AAPL management has been **credibly consistent** on services deceleration.
   No behavioral red flags. Sentiment cooling matches business reality.
```

---

## Output Format

### Standard Response Structure
```markdown
## [Insider/Sentiment Topic]
[Executive summary with signal strength]

## Insider Trading Analysis
### Net Activity (Last 6 Months)
| Role | Shares Bought | Shares Sold | Net Position | Signal |
|------|---------------|-------------|--------------|--------|
| [Role 1] | [X,XXX] | [Y,YYY] | [Z,ZZZ] | [Buy/Sell/Neutral] |

### Red Flags
- [ ] Pre-announcement trades
- [ ] Blackout violations
- [ ] Clustered selling by C-suite

## Sentiment Analysis
### Current Quarter: [QYYYY-QX]
- **Sentiment Score**: [+/-X.XX]
- **Tone**: [Optimistic/Neutral/Cautious]
- **Key Quotes**: [Excerpts]

### Temporal Comparison
| Quarter | Sentiment | Drift | Interpretation |
|---------|-----------|-------|----------------|
| [Prior] | [+X.XX] | - | Baseline |
| [Current] | [+Y.YY] | [Δ] | [Analysis] |

## Cross-Validation
### Insider × Sentiment Divergence
- **Divergence Score**: [0.XX] (0=Aligned, 1=Opposed)
- **Signal**: [✅ Aligned / ⚠️ Diverged]

## Data Provenance
- Insider Trades: [Source] ([Date Range])
- Transcripts: [Earnings Call Date]
- Analysis Date: [YYYY-MM-DD]
```

---

## Integration with Other Agents

### Upstream Dependencies
| Agent | Data Consumed | Use Case |
|-------|---------------|----------|
| **Business Analyst** | MD&A forward guidance | Cross-validate with call tone |
| **Quantitative** | Forensic red flags | Correlate with insider selling |

### Downstream Consumers
| Agent | Data Provided | Use Case |
|-------|---------------|----------|
| **Planner** | Insider/sentiment divergence scores | Risk-weighted portfolio decisions |
| **Web Search** | Red flag context | Check for SEC investigations |

---

## Limitations & Constraints

### Data Lag
- **Form 4 Filings**: 2 business days after trade (SEC requirement)
- **Earnings Transcripts**: 24-48 hours after call (FMP processing)

### Sentiment Scoring Accuracy
- **LLM Classification**: 89% agreement with human experts
- **Context Dependency**: Sarcasm/irony may be misclassified
- **Industry Jargon**: Financial lexicon required (Loughran-McDonald)

---

## Performance Metrics

### Query Latency
- **Insider SQL Query**: 100-250ms (TimescaleDB aggregation)
- **Sentiment RAG**: 300-600ms (Qdrant + LLM scoring)
- **Temporal Comparison**: 500-1000ms (multi-quarter retrieval)

### Predictive Accuracy (Backtested)
- **Insider Buying → Outperformance**: 58% (6-month forward returns)
- **Insider Selling → Underperformance**: 52% (weak signal)
- **Divergence Signal**: 67% accuracy (insider selling + positive tone → future decline)

---

## When to Use This Agent

### ✅ Ideal For:
- "Are insiders buying/selling [ticker]?"
- "Has management sentiment changed?"
- "Is [company] being evasive on [topic]?"
- "Red flags in insider trading?"
- "Cross-check guidance with sentiment"

### ❌ Not Suitable For:
- Financial calculations (use Quantitative)
- Strategy analysis (use Business Analyst)
- Real-time news (use Web Search)
- Network effects (use Supply Chain Graph)

---

## Example Planner Routing Logic

```python
def route_query_to_insider_sentiment(query: str) -> bool:
    """Determine if query should be routed to Insider & Sentiment Agent"""
    
    insider_keywords = [
        'insider', 'form 4', 'ceo selling', 'cfo buying', 'executive trade',
        'insider trading', '10% owner'
    ]
    
    sentiment_keywords = [
        'earnings call', 'management tone', 'sentiment', 'guidance',
        'optimistic', 'pessimistic', 'evasive', 'confident'
    ]
    
    divergence_keywords = [
        'red flag', 'suspicious', 'inconsistent', 'contradiction'
    ]
    
    if any(kw in query.lower() for kw in insider_keywords + sentiment_keywords + divergence_keywords):
        return True
    
    # Detect behavioral analysis queries
    if any(phrase in query.lower() for phrase in ['has sentiment changed', 'management credibility']):
        return True
    
    return False
```

---

## Version History
- **v2.0** (2026-02-13): Temporal Contrastive RAG + evasiveness detection
- **v1.5** (2025-12-01): Added pre-announcement trade flagging
- **v1.0** (2025-08-20): Initial insider trading + sentiment scoring
