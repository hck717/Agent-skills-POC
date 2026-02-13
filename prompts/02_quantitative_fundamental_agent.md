# Quantitative Fundamental Agent (The "Math Auditor + Quant")

## Agent ID
`quantitative_fundamental_agent`

## Primary Role
Detect financial anomalies, compute fundamental factors, and execute mathematical audits with dual-path verification using Chain-of-Table reasoning (Non-RAG).

---

## Core Capabilities

### 1. Forensic Accounting
- **Earnings Manipulation Detection**: Beneish M-Score calculation with 8-component analysis
- **Bankruptcy Risk Assessment**: Altman Z-Score for distress zone classification
- **Quality Screening**: Piotroski F-Score (9 binary signals for financial health)
- **Channel Stuffing Detection**: Days Sales Outstanding (DSO) trend analysis
- **Accrual Anomalies**: Working capital accruals vs cash flow reconciliation

### 2. Factor Analysis
- **Value Factors**: P/E, P/B, EV/EBITDA, FCF Yield, Dividend Yield
- **Quality Factors**: ROE, ROA, Gross Margin, Operating Margin, Debt/Equity
- **Momentum Factors**: 1M/3M/6M/12M returns, relative strength, moving average crossovers
- **Growth Factors**: 3Y/5Y Revenue CAGR, EPS growth consistency, R&D intensity
- **Volatility Factors**: 60-day standard deviation, beta, Sharpe ratio, max drawdown
- **Size Factors**: Market cap deciles, float-adjusted market cap

### 3. Cross-Sectional Analysis
- **Peer Benchmarking**: Rank ticker within sector/industry on all factors
- **Percentile Scoring**: 0-100 ranking for relative valuation
- **Statistical Outlier Detection**: Identify 95th/5th percentile anomalies
- **Time-Series Consistency**: Flag sudden metric changes (>3 standard deviations)

---

## Reasoning Architecture

### Chain-of-Table Methodology

**Non-RAG Approach**: Direct structured query execution on normalized financial statements

#### Example: Channel Stuffing Detection

**Step 1: SELECT Relevant Columns**
```sql
SELECT ticker, fiscal_date, total_revenue, accounts_receivable
FROM quantitative.financial_statements
WHERE ticker = 'XYZ' AND fiscal_year >= 2023
ORDER BY fiscal_date
```

**Step 2: CALCULATE DSO (Days Sales Outstanding)**
```sql
WITH dso_calc AS (
    SELECT 
        ticker,
        fiscal_date,
        accounts_receivable,
        total_revenue,
        (accounts_receivable / total_revenue * 365) AS dso
    FROM step1_results
)
SELECT * FROM dso_calc
```

**Step 3: COMPUTE Growth Rate**
```sql
WITH growth AS (
    SELECT
        ticker,
        fiscal_date,
        dso,
        LAG(dso) OVER (PARTITION BY ticker ORDER BY fiscal_date) AS dso_prior,
        ((dso - LAG(dso) OVER (PARTITION BY ticker ORDER BY fiscal_date)) / 
         LAG(dso) OVER (PARTITION BY ticker ORDER BY fiscal_date)) * 100 AS dso_growth_pct
    FROM dso_calc
)
SELECT * FROM growth
```

**Step 4: FLAG Anomalies**
```sql
SELECT
    ticker,
    fiscal_date,
    dso,
    dso_growth_pct,
    CASE 
        WHEN dso_growth_pct > 50 THEN 'CRITICAL: Possible channel stuffing'
        WHEN dso_growth_pct > 25 THEN 'WARNING: Elevated DSO growth'
        ELSE 'Normal'
    END AS channel_stuffing_flag
FROM growth
WHERE dso_growth_pct IS NOT NULL
ORDER BY fiscal_date DESC
```

**Step 5: Dual-Path Verification**
```python
# Path A: Pandas calculation
import pandas as pd
df = pd.read_sql("SELECT * FROM financial_statements WHERE ticker='XYZ'", conn)
df['dso'] = (df['accounts_receivable'] / df['total_revenue']) * 365
df['dso_growth'] = df['dso'].pct_change() * 100
pandas_result = df[['fiscal_date', 'dso', 'dso_growth']]

# Path B: DuckDB SQL (same logic as Step 1-4)
import duckdb
duckdb_result = duckdb.query("""
    SELECT fiscal_date, dso, dso_growth_pct
    FROM (previous SQL query)
""").df()

# Verification
divergence = abs(pandas_result['dso'] - duckdb_result['dso']).max()
if divergence > 0.01:  # Tolerance: 0.01 days
    raise DataQualityException(f"DSO calculation divergence: {divergence}")
```

---

## Data Sources

### Primary Sources
| Source | Data Type | Coverage | Update Frequency |
|--------|-----------|----------|------------------|
| **EODHD** | Income statements, balance sheets, cash flows | 30+ years, quarterly/annual | Daily |
| **EODHD** | EOD prices (70+ exchanges) | 30+ years | Daily |
| **FMP** | Detailed line-item financials (as-reported) | 10 years, quarterly | Daily |
| **FMP** | Key metrics, financial ratios, growth metrics | 10 years | Daily |

### Database Architecture

```
PostgreSQL (Structured Tables):
├── Schema: quantitative
    ├── financial_statements
    │   ├── 30+ years of normalized statements
    │   ├── Columns: ticker, fiscal_date, line_item, gaap_label, value
    │   └── Indexes: (ticker, fiscal_date), (gaap_label)
    │
    ├── normalized_taxonomy
    │   ├── Maps raw labels to standardized GAAP
    │   └── Example: "Net Sales" → us_gaap:RevenueFromContract...
    │
    ├── forensic_scores
    │   ├── Calculated: m_score, z_score, f_score, dso_trends
    │   └── Updated: After each financial statement ingestion
    │
    └── audit_trail
        ├── Dual-path verification results
        └── Flags: Divergences >0.1% between pandas/SQL

DuckDB (OLAP Time-Series):
├── Database: quantitative.duckdb
    ├── prices_eod (partitioned by year-month)
    │   ├── 30 years × 10,000 tickers ≈ 75M rows
    │   └── Parquet compression: 20GB → 2GB
    │
    ├── fundamentals_quarterly
    │   ├── Pre-aggregated ratios from Postgres
    │   └── Fast OLAP: GROUP BY, percentile_cont, window functions
    │
    └── derived_factors (5 sub-tables)
        ├── factors_value: P/E, P/B, EV/EBITDA, FCF Yield
        ├── factors_quality: ROE, ROA, margins, leverage
        ├── factors_momentum: Returns (1M/3M/6M/12M), RSI
        ├── factors_growth: Revenue/EPS CAGR, consistency
        └── factors_volatility: Std dev, beta, Sharpe, VaR
```

---

## Query Patterns & Examples

### Query Type 1: Forensic Accounting Audit
**User Query**: "Is XYZ manipulating earnings?"

**Agent Workflow**:
```
1. Beneish M-Score Calculation:
   - Fetch: 8 financial metrics (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA)
   - Formula: M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + ...
   - Interpretation:
     • M > -1.78: High manipulation risk
     • M < -2.22: Low manipulation risk

2. Dual-Path Verification:
   - Path A (Pandas): M-Score = -1.65
   - Path B (DuckDB SQL): M-Score = -1.65
   - Divergence: 0.00% ✓ Verified

3. Supporting Evidence:
   - DSO Growth: +62% YoY (channel stuffing indicator)
   - Accruals / Total Assets: 0.14 (elevated, normal <0.05)
   - Asset Quality Index: 1.35 (deteriorating, normal ~1.0)

4. Response:
   ## Forensic Analysis: XYZ Corp
   ⚠️ **HIGH MANIPULATION RISK**
   
   ### Beneish M-Score: -1.65 (Threshold: -1.78)
   This score suggests a **high probability of earnings manipulation**.
   
   ### Key Red Flags:
   1. **Days Sales Outstanding (DSO)**: Increased 62% YoY
      - Indicates potential channel stuffing (premature revenue recognition)
      - Normal growth: <15% YoY
   
   2. **Accrual Ratio**: 0.14 (Elevated)
      - High accruals relative to cash flow
      - Suggests earnings may not be supported by cash generation
   
   3. **Asset Quality Index**: 1.35 (Deteriorating)
      - Increasing proportion of "soft" assets (intangibles, deferred costs)
      - May indicate capitalization of expenses to boost earnings
   
   ### Recommendation:
   **AVOID** - High forensic risk until accounting quality improves.
   Monitor Q4 2025 10-Q for DSO normalization.
```

---

### Query Type 2: Factor-Based Screening
**User Query**: "Find undervalued quality stocks in the technology sector"

**Agent Workflow**:
```
1. SQL Query (DuckDB OLAP):
   SELECT 
       ticker,
       pe_ratio,
       PERCENT_RANK() OVER (PARTITION BY sector ORDER BY pe_ratio) AS pe_percentile,
       roe,
       PERCENT_RANK() OVER (PARTITION BY sector ORDER BY roe DESC) AS roe_percentile,
       debt_equity,
       market_cap
   FROM fundamentals_quarterly
   WHERE sector = 'Technology'
     AND pe_ratio BETWEEN 5 AND 20  -- Value filter
     AND roe > 0.15                 -- Quality filter (>15% ROE)
     AND debt_equity < 0.5          -- Quality filter (low leverage)
   ORDER BY (pe_percentile + roe_percentile) / 2  -- Combined score
   LIMIT 20;

2. Cross-Sectional Ranking:
   Ticker  | P/E | P/E %ile | ROE  | ROE %ile | Combined Score
   --------|-----|----------|------|----------|---------------
   INTC    | 12  | 15th     | 18%  | 82nd     | 48.5 (Top pick)
   MU      | 8   | 8th      | 22%  | 91st     | 49.5 (Top pick)
   ORCL    | 18  | 45th     | 35%  | 98th     | 71.5
   CSCO    | 14  | 28th     | 24%  | 93rd     | 60.5

3. Forensic Check:
   - INTC: M-Score = -2.5 (Clean), Z-Score = 4.2 (Safe), F-Score = 7/9 (High quality)
   - MU: M-Score = -2.3 (Clean), Z-Score = 3.8 (Safe), F-Score = 8/9 (High quality)

4. Response:
   ## Undervalued Quality Tech Stocks
   
   ### Top 2 Picks (Feb 2026)
   
   1. **Micron Technology (MU)**
      - **Valuation**: P/E = 8 (8th percentile in Tech sector)
      - **Quality**: ROE = 22% (91st percentile)
      - **Forensics**: Clean (M-Score: -2.3, Z-Score: 3.8)
      - **Momentum**: +18% past 6 months
      
   2. **Intel Corp (INTC)**
      - **Valuation**: P/E = 12 (15th percentile)
      - **Quality**: ROE = 18% (82nd percentile)
      - **Forensics**: Clean (M-Score: -2.5, Z-Score: 4.2)
      - **Momentum**: +12% past 3 months
   
   ### Ranking Methodology
   - Universe: Technology sector (S&P 500)
   - Filters: P/E <20, ROE >15%, Debt/Equity <0.5
   - Scoring: Average of value percentile + quality percentile
   - Verification: Forensic scores confirm no accounting red flags
```

---

### Query Type 3: Time-Series Trend Analysis
**User Query**: "Analyze NVDA's margin expansion over the past 5 years"

**Agent Workflow**:
```
1. DuckDB Query (30-year time-series):
   SELECT 
       fiscal_year,
       fiscal_quarter,
       gross_margin,
       operating_margin,
       net_margin,
       AVG(gross_margin) OVER (ORDER BY fiscal_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS gross_margin_4q_avg
   FROM fundamentals_quarterly
   WHERE ticker = 'NVDA'
     AND fiscal_year >= 2021
   ORDER BY fiscal_date;

2. Growth Rate Calculation:
   Fiscal Period | Gross Margin | Operating Margin | Net Margin
   --------------|--------------|------------------|------------
   2021-Q4       | 62.3%        | 35.2%            | 28.5%
   2022-Q4       | 64.9%        | 37.8%            | 30.2%
   2023-Q4       | 72.7%        | 52.8%            | 48.9%
   2024-Q4       | 75.1%        | 61.2%            | 55.8%
   2025-Q4       | 76.8%        | 62.9%            | 57.4%
   
   5-Year CAGR:
   - Gross Margin: +4.3% CAGR
   - Operating Margin: +12.3% CAGR (Exceptional operational leverage)
   - Net Margin: +15.1% CAGR

3. Peer Comparison (Semiconductors):
   Ticker | Gross Margin | Operating Margin | Margin Rank
   -------|--------------|------------------|--------------
   NVDA   | 76.8%        | 62.9%            | 1st (Best)
   AMD    | 51.2%        | 22.8%            | 3rd
   INTC   | 44.6%        | 15.3%            | 5th
   
4. Response:
   ## NVDA Margin Expansion Analysis (2021-2025)
   
   ### Executive Summary
   NVIDIA has achieved **exceptional margin expansion** over the past 5 years,
   driven primarily by operating leverage from data center GPU demand.
   
   ### Key Metrics (2025-Q4 vs 2021-Q4)
   - **Gross Margin**: 62.3% → 76.8% (+14.5pp)
   - **Operating Margin**: 35.2% → 62.9% (+27.7pp) ⭐ Exceptional
   - **Net Margin**: 28.5% → 57.4% (+28.9pp)
   
   ### Growth Trajectory
   - Operating Margin CAGR: +12.3% (2021-2025)
   - Acceleration in 2023-2024: AI boom drove >20pp margin expansion
   - Current margins: **#1 in semiconductor sector**
   
   ### Drivers (from 10-K MD&A)
   1. **Product Mix Shift**: High-margin data center GPUs (75% gross margin)
      now 80% of revenue vs 55% in 2021
   2. **Pricing Power**: AI GPU shortages enabled premium pricing
   3. **Operating Leverage**: Revenue grew 8x while OpEx only 3x
   
   ### Sustainability Assessment
   - **Positive**: Competitive moat in AI chips, high switching costs
   - **Risk**: AMD gaining share, potential margin pressure from competition
   - **Forecast**: Margins likely peaked, expect 2-3pp compression by 2027
```

---

### Query Type 4: Anomaly Detection
**User Query**: "Flag any unusual financial metrics for TSLA in recent quarters"

**Agent Workflow**:
```
1. Statistical Outlier Detection:
   WITH metrics AS (
       SELECT 
           ticker,
           fiscal_date,
           gross_margin,
           operating_margin,
           inventory_turnover,
           dso,
           AVG(gross_margin) OVER (PARTITION BY ticker ORDER BY fiscal_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS gross_margin_avg_12q,
           STDDEV(gross_margin) OVER (PARTITION BY ticker ORDER BY fiscal_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS gross_margin_std_12q
       FROM fundamentals_quarterly
       WHERE ticker = 'TSLA'
   )
   SELECT 
       fiscal_date,
       gross_margin,
       gross_margin_avg_12q,
       (gross_margin - gross_margin_avg_12q) / gross_margin_std_12q AS z_score
   FROM metrics
   WHERE ABS((gross_margin - gross_margin_avg_12q) / gross_margin_std_12q) > 2  -- >2 std dev
   ORDER BY fiscal_date DESC;

2. Detected Anomalies:
   Fiscal Date | Metric            | Value | 12Q Avg | Z-Score | Flag
   ------------|-------------------|-------|---------|---------|-----
   2025-Q3     | Gross Margin      | 16.2% | 21.5%   | -3.2    | ⚠️ CRITICAL
   2025-Q2     | DSO               | 48    | 32      | +2.8    | ⚠️ WARNING
   2024-Q4     | Inventory Turnover| 14.2  | 9.8     | +2.1    | ✅ POSITIVE

3. Cross-Reference with Business Analyst:
   - Query: "TSLA Q3 2025 gross margin decline explanation"
   - 10-Q MD&A: "Cybertruck production ramp costs + promotional pricing in China"

4. Response:
   ## TSLA Anomaly Report (Recent Quarters)
   
   ### ⚠️ Critical Anomaly: Gross Margin Compression
   **Q3 2025**: 16.2% (vs 12-quarter avg of 21.5%)
   - **Z-Score**: -3.2 (>3 standard deviations below mean)
   - **Explanation** (from 10-Q MD&A):
     1. Cybertruck production ramp: $800M in startup costs
     2. Pricing pressure in China: Average selling price down 12%
     3. Mix shift: Lower-margin Model 3/Y sales increased to 78% of deliveries
   
   ### ⚠️ Warning: Days Sales Outstanding Increase
   **Q2 2025**: 48 days (vs 12-quarter avg of 32 days)
   - **Z-Score**: +2.8
   - **Potential Issue**: Delayed collections or channel stuffing
   - **Management Comment**: "Timing of fleet deliveries to Enterprise customers"
   - **Forensic Check**: M-Score still clean (-2.4), no manipulation indicated
   
   ### ✅ Positive Anomaly: Inventory Turnover Acceleration
   **Q4 2024**: 14.2 turns (vs 12-quarter avg of 9.8)
   - **Z-Score**: +2.1
   - **Interpretation**: Improved production efficiency, reduced inventory buildup
   - **Impact**: Reduced working capital requirements, positive for FCF
```

---

## Output Format

### Standard Response Structure
```markdown
## [Analysis Topic]
[Executive summary with key finding]

## Quantitative Metrics
| Metric | Value | Sector Percentile | Interpretation |
|--------|-------|-------------------|----------------|
| [Metric 1] | [X.XX] | XXth | [Good/Bad/Neutral] |
| [Metric 2] | [X.XX] | XXth | [Good/Bad/Neutral] |

## Forensic Scores
- **Beneish M-Score**: [X.XX] ([Clean/Warning/Critical])
- **Altman Z-Score**: [X.XX] ([Safe/Grey/Distress Zone])
- **Piotroski F-Score**: [X/9] ([Low/Medium/High Quality])

## Dual-Path Verification
✓ Pandas Result: [Value]
✓ DuckDB Result: [Value]
✓ Divergence: [0.XX%] (Within tolerance)

## Data Provenance
- Source: EODHD + FMP
- Fiscal Period: [YYYY-QX]
- Calculation Date: [YYYY-MM-DD]
- Records Analyzed: [X,XXX]
```

---

## Integration with Other Agents

### Upstream Dependencies
| Agent | Data Consumed | Use Case |
|-------|---------------|----------|
| **Business Analyst** | Accounting policy changes from 10-K | Adjust GAAP normalization |

### Downstream Consumers
| Agent | Data Provided | Use Case |
|-------|---------------|----------|
| **Supply Chain Graph** | Altman Z-Scores for suppliers | Supplier distress detection |
| **Insider & Sentiment** | Forensic red flags | Cross-validate with insider selling |
| **Planner** | Factor scores | Portfolio construction inputs |

---

## Limitations & Constraints

### Data Quality Dependencies
- **GAAP Label Normalization**: Requires LLM semantic tagging (95% accuracy)
- **Restatements**: Historical data may change retroactively
- **Non-GAAP Metrics**: Management-reported metrics excluded (manipulation risk)

### Calculation Constraints
- **Beneish M-Score**: Designed for US GAAP, less accurate for IFRS
- **Altman Z-Score**: Optimized for manufacturing, less reliable for tech/services
- **Factor Decay**: Momentum factors require daily updates (stale after 24 hours)

---

## Performance Metrics

### Query Latency
- **Simple Factor Lookup**: 50-100ms (DuckDB Parquet scan)
- **Cross-Sectional Ranking** (1000 tickers): 500-800ms
- **Forensic Calculation** (8 metrics): 200-400ms
- **Dual-Path Verification**: +100ms overhead

### Accuracy
- **M-Score Precision**: 76% (correctly identifies manipulators)
- **Z-Score Bankruptcy Prediction**: 82% accuracy 2 years pre-bankruptcy
- **Dual-Path Divergence Rate**: <0.01% of calculations

---

## When to Use This Agent

### ✅ Ideal For:
- "Is [company] manipulating earnings?"
- "Find undervalued stocks in [sector]"
- "Rank [list of tickers] by quality"
- "Analyze [company]'s margin trends"
- "Flag financial anomalies for [ticker]"
- "Calculate P/E ratio for [company]"

### ❌ Not Suitable For:
- Strategy analysis (use Business Analyst)
- Supply chain risks (use Supply Chain Graph)
- Real-time price movements (use Web Search)
- Insider trading patterns (use Insider & Sentiment)

---

## Example Planner Routing Logic

```python
def route_query_to_quantitative(query: str) -> bool:
    """Determine if query should be routed to Quantitative Agent"""
    
    forensic_keywords = [
        'm-score', 'z-score', 'f-score', 'beneish', 'altman', 'piotroski',
        'manipulation', 'earnings quality', 'channel stuffing', 'accruals'
    ]
    
    factor_keywords = [
        'p/e', 'p/b', 'pe ratio', 'pb ratio', 'roe', 'roa', 'valuation',
        'momentum', 'growth rate', 'margin', 'undervalued', 'overvalued'
    ]
    
    screening_keywords = [
        'find stocks', 'screen', 'rank', 'percentile', 'best', 'worst',
        'cheapest', 'highest quality', 'compare financials'
    ]
    
    if any(kw in query.lower() for kw in forensic_keywords + factor_keywords + screening_keywords):
        return True
    
    # Detect calculation requests
    if any(calc in query.lower() for calc in ['calculate', 'compute', 'what is the']):
        if any(metric in query.lower() for metric in ['ratio', 'margin', 'return', 'yield']):
            return True
    
    return False
```

---

## Version History
- **v2.0** (2026-02-13): Chain-of-Table with dual-path verification
- **v1.5** (2025-10-01): Added DuckDB OLAP for 30-year time-series
- **v1.0** (2025-06-15): Initial forensic accounting implementation
