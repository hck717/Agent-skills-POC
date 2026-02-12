# Data Folder Structure for Multi-Agent Equity Research System

## Overview
This document defines the data organization for 6 specialized agents with different RAG strategies and database backends.

---

## Folder Structure

```
data/
├── 01_business_analyst/          # Graph-Augmented CRAG
│   ├── qdrant/                   # Vector store data
│   │   ├── eodhd/
│   │   │   ├── 10k_sections/     # Business Overview, MD&A, Risk Factors
│   │   │   └── 10q_sections/
│   │   ├── fmp/
│   │   │   ├── 10k_full/         # Complete 10-K filings
│   │   │   ├── 10q_full/
│   │   │   ├── mda_sections/
│   │   │   └── earnings_transcripts/
│   │   └── metadata_schema.json  # {doc_type, section, filing_date, ticker, proposition_id, confidence_score}
│   │
│   └── neo4j/                    # Knowledge graph data
│       ├── import/
│       │   ├── nodes/
│       │   │   ├── concepts.csv
│       │   │   ├── strategies.csv
│       │   │   ├── risks.csv
│       │   │   ├── technologies.csv
│       │   │   ├── markets.csv
│       │   │   └── regulations.csv
│       │   └── relationships/
│       │       ├── depends_on.csv
│       │       ├── impacts.csv
│       │       └── affects.csv
│       └── schema.cypher         # Neo4j schema definitions
│
├── 02_quantitative_fundamental/  # Chain-of-Table with Code Verification
│   ├── postgres/                 # Structured financial statements
│   │   ├── eodhd/
│   │   │   ├── income_statements/     # 30+ years quarterly & annual
│   │   │   ├── balance_sheets/
│   │   │   └── cash_flows/
│   │   ├── fmp/
│   │   │   ├── detailed_financials/   # Line-item details
│   │   │   ├── financial_ratios/
│   │   │   ├── key_metrics/
│   │   │   └── growth_metrics/
│   │   ├── calculated/
│   │   │   ├── beneish_m_score.csv    # Earnings manipulation detection
│   │   │   ├── altman_z_score.csv     # Bankruptcy risk
│   │   │   ├── dso_trends.csv         # Days Sales Outstanding
│   │   │   └── accrual_ratios.csv
│   │   └── schema/
│   │       ├── tables.sql             # DDL for financial_statements, normalized_taxonomy, quality_flags, audit_trail
│   │       └── chain_of_table_views.sql
│   │
│   └── duckdb/                   # Time-series OLAP
│       ├── eodhd/
│       │   ├── prices/                # 30+ years, 70+ exchanges (partitioned by year-month)
│       │   ├── fundamentals/          # Ratios, margins (quarterly snapshots)
│       │   ├── technical_indicators/
│       │   └── corporate_actions/     # Splits, dividends
│       ├── fmp/
│       │   ├── bulk_prices/
│       │   ├── historical_ratios/
│       │   └── historical_metrics/
│       ├── derived_factors/           # Daily calculated factors
│       │   ├── value/                 # P/E, P/B, EV/EBITDA
│       │   ├── quality/               # ROE, ROA, margins
│       │   ├── momentum/              # 1M/3M/6M/12M returns
│       │   ├── volatility/            # Beta, std dev
│       │   ├── growth/                # CAGR
│       │   └── size/                  # Market cap classifications
│       └── schema/
│           └── tables.sql             # Partitioning strategy, indexes
│
├── 03_supply_chain_graph/        # GraphRAG with Network Analysis
│   └── neo4j/
│       ├── import/
│       │   ├── nodes/
│       │   │   ├── companies.csv
│       │   │   ├── executives.csv
│       │   │   ├── institutions.csv
│       │   │   ├── products.csv
│       │   │   └── geographies.csv
│       │   └── relationships/
│       │       ├── supplies_to.csv
│       │       ├── customer_of.csv
│       │       ├── holds.csv          # Institutional holdings
│       │       ├── competes_in.csv
│       │       └── exposed_to.csv
│       ├── eodhd/
│       │   ├── institutional_holders.csv
│       │   └── etf_constituents.csv
│       ├── fmp/
│       │   ├── 13f_filings/           # Institutional ownership
│       │   ├── customer_supplier/     # Extracted from 10-K
│       │   └── stock_peers.csv
│       ├── precomputed/
│       │   ├── pagerank.csv           # Centrality scores
│       │   ├── betweenness.csv
│       │   ├── louvain_communities.csv
│       │   └── degree_centrality.csv
│       └── schema.cypher
│
├── 04_macro_economic/            # Text-to-SQL + Time-Series RAG
│   ├── duckdb/                   # Macro time-series
│   │   ├── eodhd/
│   │   │   └── macro_indicators/     # 30+ indicators from 1960+
│   │   │       ├── gdp.parquet
│   │   │       ├── unemployment.parquet
│   │   │       ├── inflation_cpi.parquet
│   │   │       ├── real_interest_rate.parquet
│   │   │       ├── trade_balance.parquet
│   │   │       └── debt_to_gdp.parquet
│   │   ├── fmp/
│   │   │   ├── treasury_yields/
│   │   │   ├── economic_calendar/
│   │   │   ├── market_risk_premiums/
│   │   │   └── central_bank_rates/
│   │   ├── external/
│   │   │   └── global_macro_database/  # 73 variables, 240 countries, 1086-2029
│   │   │       ├── by_country/
│   │   │       └── by_indicator/
│   │   ├── derived/
│   │   │   ├── economic_cycles.parquet        # Expansion/contraction classification
│   │   │   ├── yield_curve_slopes.parquet
│   │   │   └── inflation_adjusted_rates.parquet
│   │   └── schema/
│   │       ├── tables.sql                     # Partitioning by country-year
│   │       └── views.sql
│   │
│   └── qdrant/                   # Central bank communications
│       ├── fomc/                        # Federal Reserve
│       │   ├── minutes/
│       │   └── statements/
│       ├── ecb/                         # European Central Bank
│       ├── boj/                         # Bank of Japan
│       ├── pboc/                        # People's Bank of China
│       └── metadata_schema.json         # {country, date, policy_decision, rate_change, tone_score}
│
├── 05_insider_sentiment/         # Temporal Contrastive RAG
│   ├── postgres_timescaledb/     # Time-series optimized
│   │   ├── eodhd/
│   │   │   ├── insider_transactions/   # Form 4 data
│   │   │   └── institutional_holdings/
│   │   ├── fmp/
│   │   │   ├── insider_trading/
│   │   │   ├── sentiment_scores/
│   │   │   └── earnings_dates/
│   │   ├── aggregates/
│   │   │   └── insider_quarterly.csv   # Pre-aggregated by quarter
│   │   └── schema/
│   │       └── hypertables.sql         # TimescaleDB hypertable definitions
│   │
│   └── qdrant/                   # Earnings call transcripts
│       ├── fmp/
│       │   ├── earnings_calls/         # Chunked by speaker turn
│       │   └── conference_calls/
│       └── metadata_schema.json        # {ticker, quarter, fiscal_year, speaker_role, transcript_section, timestamp, sentiment_score}
│
├── 06_web_search/                # HyDE + Step-Back + Reranking (No Persistence)
│   ├── api_configs/
│   │   ├── eodhd_endpoints.json        # Real-time quotes, economic calendar
│   │   ├── fmp_endpoints.json          # News API, press releases, analyst upgrades
│   │   └── external_apis.json          # News APIs, SEC EDGAR RSS
│   ├── cache/                          # Optional temporary cache (<24h)
│   │   └── .gitignore                  # Don't commit cached data
│   └── prompts/
│       ├── hyde_templates.txt          # Hypothetical document generation
│       ├── stepback_templates.txt      # Abstract reasoning prompts
│       └── reranking_criteria.json
│
└── shared/                       # Common resources across agents
    ├── tickers/
    │   ├── sp500.csv
    │   ├── nasdaq100.csv
    │   ├── dow30.csv
    │   └── custom_watchlist.csv
    ├── taxonomies/
    │   ├── gaap_mapping.json           # Financial statement taxonomy
    │   ├── industry_classifications.csv # GICS, SIC codes
    │   └── sec_section_mapping.json    # 10-K/10-Q section labels
    ├── embeddings/
    │   └── model_configs.json          # Embedding model settings for Qdrant
    └── README.md                       # Data sources documentation
```

---

## Database Mapping

### Qdrant (Vector Store)
- **Collections:**
  - `business_analyst_10k` - Proposition-chunked 10-K sections
  - `business_analyst_10q` - Proposition-chunked 10-Q sections
  - `business_analyst_transcripts` - Earnings call chunks
  - `macro_central_bank` - FOMC/ECB/BOJ communications
  - `insider_earnings_calls` - Transcript chunks with sentiment

### Neo4j (Knowledge Graphs)
- **Databases:**
  - `business_analyst` - Concept relationships, dependencies
  - `supply_chain` - Network graph with centrality metrics

### Postgres
- **Schemas:**
  - `quantitative` - Financial statements, normalized taxonomy, audit trail
  - `insider` - Transactions, holdings, aggregates (with TimescaleDB extension)

### DuckDB
- **Databases:**
  - `quantitative.duckdb` - Prices, fundamentals, derived factors (partitioned)
  - `macro.duckdb` - Economic indicators, rates, cycles (partitioned by country-year)

---

## Data Source Summary

### EODHD APIs
- Financial statements (30+ years)
- EOD prices (70+ exchanges)
- Macro indicators (1960+)
- Insider transactions (Form 4)
- Institutional holdings
- ETF constituents

### FMP APIs
- Detailed 10-K/10-Q line items
- Earnings call transcripts
- 13F filings
- Real-time news & analyst upgrades
- Economic calendar
- Treasury yields

### External Sources
- Global Macro Database (73 variables, 240 countries, 1086-2029)
- Central bank communications (FOMC, ECB, BOJ, PBoC)
- News APIs
- SEC EDGAR RSS

---

## Next Steps (Post-Structure Setup)

1. **Schema Creation**
   - Run DDL scripts in `schema/` folders
   - Create Qdrant collections with appropriate embeddings
   - Initialize Neo4j graph schemas

2. **Ingestion Scripts** (Future)
   - EODHD MCP server integration
   - FMP API batch downloaders
   - External data fetchers
   - Incremental update mechanisms

3. **Quality Validation**
   - Data completeness checks
   - Schema validation
   - Metadata integrity

4. **Agent Configuration**
   - Update each agent's config to point to respective data paths
   - Set up database connection strings
   - Configure embedding models for Qdrant

---

## Metadata Standards

### Universal Fields (All Sources)
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "cik": "0000320193",
  "source": "eodhd|fmp|external",
  "ingestion_timestamp": "2026-02-12T14:11:00Z",
  "data_version": "v1.0"
}
```

### Document Metadata (Qdrant)
```json
{
  "doc_type": "10-K|10-Q|8-K|earnings_call|fomc_minutes",
  "section": "Business Overview|MD&A|Risk Factors|...",
  "filing_date": "2025-10-31",
  "fiscal_year": 2025,
  "fiscal_quarter": "Q4",
  "proposition_id": "uuid",
  "confidence_score": 0.95,
  "page_number": 12,
  "char_start": 1234,
  "char_end": 5678
}
```

### Graph Metadata (Neo4j)
```cypher
// Node properties
{
  extracted_from_filing: "AAPL-10K-2025",
  mention_frequency: 5,
  temporal_validity: ["2025-01-01", "2025-12-31"],
  confidence: 0.87,
  source_page: 42
}
```

### Time-Series Metadata (DuckDB/Postgres)
```sql
-- Common columns
ticker VARCHAR,
date DATE,
fiscal_period VARCHAR,
source VARCHAR,
quality_flag VARCHAR, -- 'VERIFIED', 'ESTIMATED', 'RESTATED'
last_updated TIMESTAMP
```
