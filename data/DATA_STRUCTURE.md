# Data Structure Documentation - Part 1: Folder Structure & Overview

**Version**: 2.0  
**Last Updated**: February 13, 2026  
**System**: Six-Agent Equity Research Platform

***

## Overview

This document defines the complete data organization for a six-agent equity research system with hybrid database architecture. Each agent has dedicated data storage optimized for its specific RAG strategy and retrieval patterns.

***

## Root Folder Structure

```
data/
├── postgres_schemas/              # PostgreSQL DDL scripts (auto-init)
├── duckdb_storage/               # DuckDB databases + Parquet files
├── qdrant_collections/           # Qdrant collection metadata schemas
├── neo4j_import/                 # CSV files for Neo4j LOAD CSV
├── raw_downloads/                # API staging area (temporary)
├── web_search/                   # API configs (no persistence)
└── shared/                       # Common resources across agents
```

***

## 1. PostgreSQL Schema Directory

**Path**: `data/postgres_schemas/`  
**Purpose**: SQL DDL scripts executed automatically on PostgreSQL container initialization  
**Container Mount**: `/docker-entrypoint-initdb.d/` (auto-run in alphabetical order)

### Structure

```
postgres_schemas/
├── 01_init_databases.sql              # Create databases and enable extensions
├── 02_quantitative_schema.sql         # Financial statements, forensic scores
├── 03_insider_schema.sql              # Insider transactions, TimescaleDB setup
├── 04_shared_schema.sql               # Agent logs, metadata, audit trails
└── README.md                          # Schema documentation
```

### File Descriptions

#### `01_init_databases.sql`
```sql
-- Creates:
-- - Database: equity_research
-- - Extensions: timescaledb, pg_trgm (fuzzy search), uuid-ossp
-- - Roles: agent_readonly, agent_readwrite
```

#### `02_quantitative_schema.sql`
```sql
-- Schema: quantitative
-- Tables:
--   - financial_statements (normalized line items, 30+ years)
--   - normalized_taxonomy (GAAP label mapping)
--   - forensic_scores (Beneish M-score, Altman Z, Piotroski F)
--   - audit_trail (dual-path verification results)
```

#### `03_insider_schema.sql`
```sql
-- Schema: insider
-- Tables:
--   - insider_transactions (Form 4 data, TimescaleDB hypertable)
--   - insider_quarterly_agg (pre-computed net buy/sell by quarter)
--   - institutional_holdings (13F data)
--   - earnings_dates (fiscal calendar)
```

#### `04_shared_schema.sql`
```sql
-- Schema: shared
-- Tables:
--   - agent_execution_logs (orchestrator traces)
--   - data_quality_alerts (ingestion errors, schema violations)
--   - ticker_metadata (CIK, company name, sector, market cap)
```

***

## 2. DuckDB Storage Directory

**Path**: `data/duckdb_storage/`  
**Purpose**: File-based OLAP databases for 30+ years of time-series analytics  
**Access Method**: Python DuckDB library (no container needed)

### Structure

```
duckdb_storage/
├── quantitative.duckdb                # Prices, fundamentals, factors (40+ GB)
├── macro.duckdb                       # Macro indicators, FX rates (5+ GB)
├── parquet/                           # Partitioned Parquet files
│   ├── prices/                        # EOD prices (partitioned by year-month)
│   │   ├── 2020_01/
│   │   │   ├── AAPL.parquet
│   │   │   ├── MSFT.parquet
│   │   │   └── ...
│   │   ├── 2020_02/
│   │   └── ...
│   │
│   ├── fundamentals/                  # Quarterly snapshots
│   │   ├── 2020_Q1/
│   │   ├── 2020_Q2/
│   │   └── ...
│   │
│   ├── macro_indicators/              # Country-year partitioned
│   │   ├── USA/
│   │   │   ├── gdp.parquet
│   │   │   ├── unemployment.parquet
│   │   │   └── ...
│   │   ├── CHN/
│   │   └── ...
│   │
│   └── factors/                       # Daily derived factors
│       ├── value/
│       │   ├── pe_ratio.parquet
│       │   ├── pb_ratio.parquet
│       │   └── ...
│       ├── quality/
│       ├── momentum/
│       ├── growth/
│       └── volatility/
│
└── README.md                          # DuckDB schema documentation
```

### Database: `quantitative.duckdb`

**Tables**:

1. **`prices_eod`** - End-of-day prices (30+ years, 10,000+ tickers)
   - Partitioned by: `YEAR(date), MONTH(date)`
   - Columns: `ticker, date, open, high, low, close, volume, adj_close`
   - Indexes: Bloom filter on `ticker`, date range indexes

2. **`fundamentals_quarterly`** - Quarterly financial snapshots
   - Partitioned by: `fiscal_year, fiscal_quarter`
   - Columns: `ticker, date, fiscal_quarter, pe_ratio, pb_ratio, roe, roa, gross_margin, debt_equity, market_cap`
   - Source: Pre-aggregated from Postgres financial_statements

3. **`technical_indicators`** - Daily calculated indicators
   - Columns: `ticker, date, sma_50, sma_200, rsi_14, macd, bollinger_upper, bollinger_lower`

4. **`derived_factors`** - Factor model inputs (daily refresh)
   - Tables: `factors_value`, `factors_quality`, `factors_momentum`, `factors_growth`, `factors_volatility`
   - Cross-sectional rankings: Percentile scores within sector/market

### Database: `macro.duckdb`

**Tables**:

1. **`macro_indicators`** - Economic indicators (1960+)
   - Partitioned by: `country, year`
   - Columns: `country, date, indicator_name, value, source`
   - Indicators: GDP, unemployment, CPI, ISM PMI, trade balance, debt-to-GDP

2. **`fx_rates`** - Daily currency exchange rates
   - Columns: `date, currency_pair, rate, source`
   - Pairs: USD/CNY, EUR/USD, GBP/USD, USD/JPY, etc.

3. **`interest_rate_differentials`** - Central bank rate spreads
   - Columns: `date, pair, fed_rate, foreign_rate, differential, carry_trade_signal`

4. **`economic_cycles`** - Cycle phase classification
   - Columns: `country, date, phase, gdp_growth, unemployment_rate, yield_curve_slope, ism_pmi, confidence_score`
   - Phases: `Expansion`, `Peak`, `Contraction`, `Trough`

***

## 3. Qdrant Collections Directory

**Path**: `data/qdrant_collections/`  
**Purpose**: Vector collection metadata schemas (actual vectors stored in Docker volume)  
**Container Storage**: `qdrant_storage` volume → `/qdrant/storage/`

### Structure

```
qdrant_collections/
├── business_analyst_10k.json          # 10-K proposition embeddings
├── business_analyst_10q.json          # 10-Q proposition embeddings
├── business_analyst_transcripts.json  # Earnings call embeddings (legacy)
├── macro_central_bank.json            # FOMC/ECB/BOJ communications
├── insider_earnings_calls.json        # Earnings transcripts (temporal RAG)
└── README.md                          # Collection schema documentation
```

### Collection Schema: `business_analyst_10k.json`

```json
{
  "collection_name": "business_analyst_10k",
  "vector_size": 768,
  "distance": "Cosine",
  "on_disk": false,
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  },
  "payload_schema": {
    "ticker": {"type": "keyword", "indexed": true},
    "filing_date": {"type": "datetime", "indexed": true},
    "fiscal_year": {"type": "integer", "indexed": true},
    "section": {"type": "keyword", "indexed": true},
    "proposition_id": {"type": "uuid"},
    "text": {"type": "text"},
    "page_number": {"type": "integer"},
    "char_start": {"type": "integer"},
    "char_end": {"type": "integer"},
    "confidence_score": {"type": "float"},
    "source": {"type": "keyword"}
  },
  "sparse_index": {
    "type": "bm25",
    "fields": ["text"]
  }
}
```

**Purpose**: Stores proposition-chunked 10-K sections (Business Overview, MD&A, Risk Factors)

**Retrieval Strategy**:
- **Dense search**: Semantic similarity on proposition embeddings
- **Sparse search**: BM25 keyword matching on exact terms
- **Hybrid search**: Combines both with reciprocal rank fusion

### Collection Schema: `macro_central_bank.json`

```json
{
  "collection_name": "macro_central_bank",
  "vector_size": 768,
  "distance": "Cosine",
  "payload_schema": {
    "central_bank": {"type": "keyword", "indexed": true},
    "country": {"type": "keyword", "indexed": true},
    "date": {"type": "datetime", "indexed": true},
    "document_type": {"type": "keyword"},
    "policy_decision": {"type": "text"},
    "rate_change": {"type": "float"},
    "text": {"type": "text"},
    "tone_score": {"type": "float"},
    "hawkish_dovish": {"type": "keyword"}
  }
}
```

**Purpose**: Stores FOMC minutes, ECB statements, BOJ/PBoC policy announcements with sentiment

**Tone Scoring**:
- `1.0` = Hawkish (rate hikes likely)
- `0.0` = Neutral
- `-1.0` = Dovish (rate cuts likely)

### Collection Schema: `insider_earnings_calls.json`

```json
{
  "collection_name": "insider_earnings_calls",
  "vector_size": 768,
  "distance": "Cosine",
  "payload_schema": {
    "ticker": {"type": "keyword", "indexed": true},
    "quarter": {"type": "keyword", "indexed": true},
    "fiscal_year": {"type": "integer", "indexed": true},
    "speaker_role": {"type": "keyword", "indexed": true},
    "transcript_section": {"type": "keyword", "indexed": true},
    "text": {"type": "text"},
    "timestamp": {"type": "datetime"},
    "sentiment_score": {"type": "float"},
    "turn_number": {"type": "integer"}
  }
}
```

**Purpose**: Temporal contrastive RAG for narrative drift detection

**Temporal Query Pattern**:
```python
# Current quarter guidance
current = qdrant.search(
    collection="insider_earnings_calls",
    query_vector=embed("guidance outlook"),
    filter={"quarter": "2025-Q4", "transcript_section": "Guidance"},
    limit=5
)

# Prior quarter guidance
prior = qdrant.search(
    collection="insider_earnings_calls",
    query_vector=embed("guidance outlook"),
    filter={"quarter": "2025-Q3", "transcript_section": "Guidance"},
    limit=5
)

# Calculate cosine similarity between current and prior embeddings
drift_score = 1 - cosine_similarity(current_avg_vector, prior_avg_vector)
```

***

## 4. Neo4j Import Directory

**Path**: `data/neo4j_import/`  
**Purpose**: CSV files for `LOAD CSV` commands (mounted to Neo4j container)  
**Container Mount**: `/var/lib/neo4j/import/`

### Structure

```
neo4j_import/
├── business_analyst/              # Concept relationship graph
│   ├── nodes/
│   │   ├── concepts.csv
│   │   ├── strategies.csv
│   │   ├── risks.csv
│   │   ├── technologies.csv
│   │   ├── markets.csv
│   │   └── regulations.csv
│   ├── relationships/
│   │   ├── depends_on.csv
│   │   ├── impacts.csv
│   │   ├── affects.csv
│   │   └── mitigates.csv
│   └── schema.cypher
│
├── supply_chain/                  # Network graph
│   ├── nodes/
│   │   ├── companies.csv
│   │   ├── executives.csv
│   │   ├── institutions.csv
│   │   ├── products.csv
│   │   └── geographies.csv
│   ├── relationships/
│   │   ├── supplies_to.csv
│   │   ├── customer_of.csv
│   │   ├── holds.csv
│   │   ├── competes_in.csv
│   │   └── exposed_to.csv
│   ├── precomputed/
│   │   ├── pagerank.csv
│   │   ├── betweenness.csv
│   │   ├── louvain_communities.csv
│   │   └── degree_centrality.csv
│   └── schema.cypher
│
└── README.md
```

### Node CSV Schema: `business_analyst/nodes/concepts.csv`

```csv
id,name,type,ticker,filing_date,extracted_from_section,mention_frequency,confidence,temporal_validity_start,temporal_validity_end
c001,"AI Strategy",strategy,MSFT,2025-10-31,"Business Overview",12,0.92,2025-01-01,2025-12-31
c002,"Cloud Revenue Growth",concept,MSFT,2025-10-31,"MD&A",8,0.88,2025-01-01,2025-12-31
r001,"Competition from AWS",risk,MSFT,2025-10-31,"Risk Factors",15,0.95,2025-01-01,2025-12-31
t001,"Generative AI",technology,MSFT,2025-10-31,"Business Overview",20,0.97,2025-01-01,2025-12-31
```

### Relationship CSV Schema: `business_analyst/relationships/depends_on.csv`

```csv
source_id,target_id,relationship_type,strength,mention_count,filing_date,ticker
c001,t001,DEPENDS_ON,0.85,5,2025-10-31,MSFT
c002,c001,DRIVES,0.78,3,2025-10-31,MSFT
r001,c001,THREATENS,0.62,4,2025-10-31,MSFT
```

### Node CSV Schema: `supply_chain/nodes/companies.csv`

```csv
id,ticker,name,sector,market_cap,extracted_from_filing,mention_frequency,is_supplier,is_customer,is_competitor
co001,AAPL,"Apple Inc.",Technology,2800000000000,AAPL-10K-2025,null,false,false,false
co002,TSMC,"Taiwan Semiconductor",Semiconductors,580000000000,AAPL-10K-2025,8,true,false,false
co003,GOOG,"Alphabet Inc.",Technology,1950000000000,AAPL-10K-2025,3,false,false,true
```

### Relationship CSV Schema: `supply_chain/relationships/supplies_to.csv`

```csv
source_id,target_id,relationship_type,pct_of_cogs,pct_of_revenue,concentration_risk,filing_date
co002,co001,SUPPLIES_TO,0.35,null,HIGH,2025-10-31
co004,co001,SUPPLIES_TO,0.12,null,MEDIUM,2025-10-31
```

### Precomputed Metrics: `supply_chain/precomputed/pagerank.csv`

```csv
node_id,ticker,pagerank_score,centrality_category,last_computed
co002,TSMC,0.87,CRITICAL,2026-02-13
co001,AAPL,0.65,HIGH,2026-02-13
co005,FOXCONN,0.54,MEDIUM,2026-02-13
```

**PageRank Interpretation**:
- **0.8-1.0**: Critical chokepoint (system-wide risk if fails)
- **0.6-0.8**: High centrality (important supplier/customer)
- **0.4-0.6**: Moderate importance
- **0.0-0.4**: Low systemic impact

***

## 5. Raw Downloads Directory (Staging)

**Path**: `data/raw_downloads/`  
**Purpose**: Temporary staging area for API downloads before processing  
**Retention**: 7 days (automatic cleanup after successful DAG completion)

### Structure

```
raw_downloads/
├── eodhd/
│   ├── sec_filings/
│   │   ├── AAPL_10-K_2025-10-31.json
│   │   ├── AAPL_10-Q_2025-07-31.json
│   │   └── ...
│   ├── fundamentals/
│   │   ├── AAPL_income_quarterly.parquet
│   │   ├── AAPL_balance_quarterly.parquet
│   │   └── ...
│   ├── prices/
│   │   ├── AAPL_prices.parquet
│   │   └── ...
│   ├── macro/
│   │   ├── USA_GDP.parquet
│   │   ├── USA_UNEMPLOYMENT.parquet
│   │   └── ...
│   └── insider_transactions/
│       ├── AAPL_insider.csv
│       └── ...
│
├── fmp/
│   ├── sec_filings/
│   ├── transcripts/
│   ├── financials/
│   ├── news/
│   └── sentiment/
│
├── external/
│   ├── global_macro_database/
│   └── central_bank_docs/
│
└── .gitignore                     # Ignore all staging files
```

**Cleanup Policy**:
- Files deleted after 7 days OR after successful processing (whichever comes first)
- Implemented via Airflow `FileSensor` + `BashOperator` cleanup tasks

***

## 6. Web Search Directory (No Persistence)

**Path**: `data/web_search/`  
**Purpose**: API configurations and prompt templates (no data storage)

### Structure

```
web_search/
├── api_configs/
│   ├── eodhd_endpoints.json
│   ├── fmp_endpoints.json
│   └── external_apis.json
│
├── prompts/
│   ├── hyde_templates.txt
│   ├── stepback_templates.txt
│   └── reranking_criteria.json
│
├── cache/                         # Optional <24h cache
│   └── .gitignore                 # Don't commit cached data
│
└── README.md
```

### File: `api_configs/eodhd_endpoints.json`

```json
{
  "real_time_quotes": {
    "url": "https://eodhd.com/api/real-time/{ticker}",
    "method": "GET",
    "params": ["api_token", "fmt"]
  },
  "economic_calendar": {
    "url": "https://eodhd.com/api/economic-events",
    "method": "GET",
    "params": ["api_token", "from", "to"]
  },
  "news_api": {
    "url": "https://eodhd.com/api/news",
    "method": "GET",
    "params": ["s", "api_token", "limit"]
  }
}
```

### File: `prompts/hyde_templates.txt`

```
# HyDE Template 1: Stock Price Movement
Generate a hypothetical news article explaining why {ticker} stock moved {direction}:

Headline: "{ticker} Shares {direction_verb} {pct_change}% on {reason}"

Article: "{ticker} stock {direction_verb} {pct_change}% today following {catalyst}. Analysts noted {analysis}. The move was driven by {driver}. Trading volume was {volume_description}, with {technical_context}."

---

# HyDE Template 2: Earnings Results
Generate a hypothetical earnings report summary for {ticker}:

"{ticker} reported Q{quarter} {year} earnings that {beat_miss} expectations. Revenue of ${revenue}M {comparison} consensus of ${consensus}M. EPS of ${eps} {vs_consensus}. Key highlights included {highlights}. Management guidance {guidance_tone}, projecting {forward_guidance}."
```

***

## 7. Shared Resources Directory

**Path**: `data/shared/`  
**Purpose**: Common resources used across multiple agents

### Structure

```
shared/
├── tickers/
│   ├── sp500.csv
│   ├── nasdaq100.csv
│   ├── dow30.csv
│   └── custom_watchlist.csv
│
├── taxonomies/
│   ├── gaap_mapping.json
│   ├── industry_classifications.csv
│   └── sec_section_mapping.json
│
├── embeddings/
│   └── model_configs.json
│
└── README.md
```

### File: `tickers/custom_watchlist.csv`

```csv
ticker,company_name,sector,exchange,inclusion_date,priority
AAPL,"Apple Inc.",Technology,NASDAQ,2024-01-01,HIGH
MSFT,"Microsoft Corp.",Technology,NASDAQ,2024-01-01,HIGH
TSLA,"Tesla Inc.",Automotive,NASDAQ,2024-01-01,MEDIUM
NVDA,"NVIDIA Corp.",Semiconductors,NASDAQ,2024-01-01,HIGH
```

### File: `taxonomies/gaap_mapping.json`

```json
{
  "revenue_mappings": {
    "Net Sales": "us_gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "Rev": "us_gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues": "us_gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "Total Revenue": "us_gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
  },
  "expense_mappings": {
    "Cost of Goods Sold": "us_gaap:CostOfGoodsAndServicesSold",
    "COGS": "us_gaap:CostOfGoodsAndServicesSold",
    "Cost of Revenue": "us_gaap:CostOfRevenue"
  }
}
```

***

