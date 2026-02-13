# Airflow DAG Documentation - Data Ingestion Pipeline

**Version**: 2.0  
**Last Updated**: February 13, 2026  
**System**: Six-Agent Equity Research Platform  
**Orchestrator**: Apache Airflow 2.8+

***

## Table of Contents

1. [DAG Overview](#dag-overview)
2. [Sequential Execution Flow](#sequential-execution-flow)
3. [DAG 1: Foundation Ingestion](#dag-1-foundation-ingestion)
4. [DAG 2: Business Analyst Preparation](#dag-2-business-analyst-preparation)
5. [DAG 3: Quantitative Fundamental Preparation](#dag-3-quantitative-fundamental-preparation)
6. [DAG 4: Supply Chain Graph Build](#dag-4-supply-chain-graph-build)
7. [DAG 5: Macro Economic Preparation](#dag-5-macro-economic-preparation)
8. [DAG 6: Insider & Sentiment Preparation](#dag-6-insider--sentiment-preparation)
9. [DAG 7: Web Search Configuration](#dag-7-web-search-configuration)
10. [Error Handling & Retries](#error-handling--retries)
11. [Monitoring & Alerts](#monitoring--alerts)
12. [Manual Backfill Procedures](#manual-backfill-procedures)

***

## 1. DAG Overview

### Purpose

Automated sequential data ingestion from EODHD and FMP APIs to populate six specialized agent databases with historical (30+ years) and current market data.

### Architecture Principles

- **Sequential Execution**: DAGs run in dependency order to ensure data consistency
- **Idempotency**: All tasks can be re-run safely without duplicating data
- **Incremental Updates**: Only fetch new/changed data after initial backfill
- **Quality Gates**: Data validation at each stage with automated alerts
- **Fault Tolerance**: Automatic retries with exponential backoff

### Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Apache Airflow | 2.8+ | DAG orchestration |
| EODHD Python Client | 1.1.1+ | EODHD API wrapper |
| FMP Python SDK | Custom | FMP API client |
| Ollama Python | 0.1.6+ | Local LLM for NER/chunking |
| psycopg2 | 2.9+ | PostgreSQL driver |
| duckdb | 0.10+ | OLAP database |
| qdrant-client | 1.7+ | Vector database client |
| neo4j | 5.15+ | Graph database driver |

***

## 2. Sequential Execution Flow

### Daily Schedule (Hong Kong Time)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daily Execution Timeline                      │
└─────────────────────────────────────────────────────────────────┘

06:00 HKT  │  DAG 1: Foundation Ingestion (60-90 min)
           │  ├─ Fetch EODHD SEC filings
           │  ├─ Fetch FMP SEC filings & transcripts
           │  ├─ Fetch EODHD fundamentals (30+ years)
           │  ├─ Fetch EODHD EOD prices (30+ years)
           │  ├─ Fetch EODHD macro indicators (1960+)
           │  ├─ Fetch EODHD insider transactions
           │  └─ Fetch FMP detailed financials
           │
07:00 HKT  │  DAG 2: Business Analyst Prep (45-60 min)
           │  ├─ Proposition chunking (LLM-based)
           │  ├─ Embed to Qdrant (768-dim vectors)
           │  ├─ NER extraction (concepts, risks, strategies)
           │  └─ Build Neo4j concept graph
           │
08:00 HKT  │  DAG 3: Quantitative Fundamental Prep (30-45 min)
           │  ├─ Semantic GAAP label tagging
           │  ├─ Load to Postgres (normalized tables)
           │  ├─ Forensic calculations (M-score, Z-score, F-score)
           │  ├─ Dual-path verification (pandas vs DuckDB)
           │  ├─ Load to DuckDB (partitioned Parquet)
           │  └─ Factor analysis (Value, Quality, Momentum, Growth)
           │
09:00 HKT  │  DAG 4: Supply Chain Graph Build (30-45 min)
           │  ├─ Entity extraction from 10-K Item 1
           │  ├─ Fetch EODHD institutional holders
           │  ├─ Fetch FMP 13F filings
           │  ├─ Build Neo4j network graph
           │  └─ Compute centrality metrics (PageRank, Betweenness)
           │
10:00 HKT  │  DAG 5: Macro Economic Prep (20-30 min)
           │  ├─ Load macro indicators to DuckDB
           │  ├─ Economic cycle classification
           │  ├─ FX & interest rate differentials
           │  ├─ Fetch central bank communications
           │  └─ Embed to Qdrant with tone scoring
           │
11:00 HKT  │  DAG 6: Insider & Sentiment Prep (30-45 min)
           │  ├─ Load insider transactions to TimescaleDB
           │  ├─ Aggregate quarterly net buy/sell
           │  ├─ Chunk earnings transcripts by speaker
           │  ├─ Sentiment scoring (LLM-based)
           │  └─ Embed to Qdrant (temporal contrastive)
           │
On-Demand  │  DAG 7: Web Search Config (5 min)
           │  ├─ Validate API endpoints
           │  ├─ Update HyDE prompts
           │  └─ Update Step-Back prompts
```

### Dependency Graph

```
                    DAG 1 (Foundation)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     DAG 2           DAG 3            DAG 5
  (Business)        (Quant)          (Macro)
      │
      │ (needs entities)
      ▼
    DAG 4
(Supply Chain)
      │
      │ (needs transcripts from DAG 1)
      ▼
    DAG 6
  (Insider)
```

***

## 3. DAG 1: Foundation Ingestion

### Overview

**DAG ID**: `dag_01_foundation_ingestion`  
**Schedule**: `0 6 * * *` (Daily 6:00 AM HKT)  
**Duration**: 60-90 minutes (first run: 8-12 hours for full backfill)  
**Output**: `data/raw_downloads/` (staging area)

### Tasks

#### Task 1.1: `fetch_eodhd_sec_filings`

**Purpose**: Download 10-K/10-Q full text from EODHD MCP server

**API Endpoint**: EODHD Fundamental API → `Filings` section

**Logic**:
```python
for ticker in watchlist:
    filings = eodhd.get_fundamental_data(ticker, filter_='Filings')
    
    # 10-K filings
    for filing in filings.get('10-K', [])[-3:]:  # Last 3 years
        save_json({
            'ticker': ticker,
            'filing_type': '10-K',
            'filing_date': filing['date'],
            'fiscal_year': filing['fiscalYear'],
            'text': filing['text'],
            'sections': {
                'Item 1': extract_section(filing['text'], 'Item 1'),
                'Item 1A': extract_section(filing['text'], 'Item 1A'),
                'Item 7': extract_section(filing['text'], 'Item 7')
            }
        }, f"data/raw_downloads/eodhd/sec_filings/{ticker}_10-K_{filing['date']}.json")
    
    # 10-Q filings
    for filing in filings.get('10-Q', [])[-8:]:  # Last 8 quarters
        save_json({...}, f"data/raw_downloads/eodhd/sec_filings/{ticker}_10-Q_{filing['date']}.json")
```

**Output Structure**:
```
data/raw_downloads/eodhd/sec_filings/
├── AAPL_10-K_2025-10-31.json
├── AAPL_10-K_2024-10-31.json
├── AAPL_10-Q_2025-07-31.json
├── AAPL_10-Q_2025-04-30.json
└── ...
```

**Rate Limits**: 1000 requests/day (EODHD All-World plan)

**Error Handling**:
- HTTP 429 (Rate Limit): Wait 60s, retry
- HTTP 404 (No filings): Skip ticker, log warning
- JSON parse error: Save raw text, flag for manual review

***

#### Task 1.2: `fetch_fmp_sec_filings`

**Purpose**: Download 10-K/10-Q full text + MD&A sections from FMP

**API Endpoint**: `https://financialmodelingprep.com/api/v3/sec_filings/{ticker}`

**Logic**:
```python
for ticker in watchlist:
    filings = fmp.get_sec_filings(ticker, limit=10)
    
    for filing in filings:
        if filing['type'] in ['10-K', '10-Q']:
            # Download full text
            full_text = requests.get(filing['finalLink']).text
            
            # Extract MD&A (Item 7 for 10-K, Item 2 for 10-Q)
            mda_section = extract_mda(full_text, filing['type'])
            
            save_json({
                'ticker': ticker,
                'filing_type': filing['type'],
                'filing_date': filing['fillingDate'],
                'accepted_date': filing['acceptedDate'],
                'cik': filing['cik'],
                'full_text': full_text,
                'mda_section': mda_section,
                'final_link': filing['finalLink']
            }, f"data/raw_downloads/fmp/sec_filings/{ticker}_{filing['type']}_{filing['fillingDate']}.json")
```

**Output Structure**:
```
data/raw_downloads/fmp/sec_filings/
├── AAPL_10-K_2025-10-31.json
├── AAPL_10-Q_2025-07-31.json
└── ...
```

**Rate Limits**: 250 requests/day (FMP Professional plan)

***

#### Task 1.3: `fetch_fmp_earnings_transcripts`

**Purpose**: Download earnings call transcripts (last 8 quarters)

**API Endpoint**: `https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}`

**Logic**:
```python
for ticker in watchlist:
    transcripts = fmp.get_earning_call_transcript(ticker, limit=8)
    
    for transcript in transcripts:
        save_json({
            'ticker': ticker,
            'quarter': transcript['quarter'],
            'fiscal_year': transcript['year'],
            'date': transcript['date'],
            'content': transcript['content'],
            'participants': extract_participants(transcript['content'])
        }, f"data/raw_downloads/fmp/transcripts/{ticker}_{transcript['year']}_Q{transcript['quarter']}.json")
```

**Output Structure**:
```
data/raw_downloads/fmp/transcripts/
├── AAPL_2025_Q4.json
├── AAPL_2025_Q3.json
└── ...
```

**Participant Extraction**:
```
Content format:
"Operator: Good day, and welcome...
John Doe (CEO): Thank you for joining...
Jane Smith (CFO): Regarding margins..."

→ Extract: [
    {'name': 'John Doe', 'role': 'CEO'},
    {'name': 'Jane Smith', 'role': 'CFO'}
]
```

***

#### Task 1.4: `fetch_eodhd_fundamentals`

**Purpose**: Download income statements, balance sheets, cash flows (30+ years)

**API Endpoint**: EODHD Fundamental API → `Financials` section

**Logic**:
```python
for ticker in watchlist:
    fundamentals = eodhd.get_fundamental_data(ticker)
    
    # Income statements
    income_q = pd.DataFrame(fundamentals['Financials']['Income_Statement']['quarterly'])
    income_a = pd.DataFrame(fundamentals['Financials']['Income_Statement']['yearly'])
    
    # Add metadata
    income_q['ticker'] = ticker
    income_q['statement_type'] = 'income'
    income_q['frequency'] = 'quarterly'
    
    income_a['ticker'] = ticker
    income_a['statement_type'] = 'income'
    income_a['frequency'] = 'annual'
    
    # Save as Parquet (compressed)
    income_q.to_parquet(
        f"data/raw_downloads/eodhd/fundamentals/{ticker}_income_quarterly.parquet",
        compression='snappy',
        index=False
    )
    
    income_a.to_parquet(
        f"data/raw_downloads/eodhd/fundamentals/{ticker}_income_annual.parquet",
        compression='snappy',
        index=False
    )
    
    # Repeat for Balance_Sheet and Cash_Flow
```

**Output Structure**:
```
data/raw_downloads/eodhd/fundamentals/
├── AAPL_income_quarterly.parquet      # ~120 rows (30 years × 4 quarters)
├── AAPL_income_annual.parquet         # ~30 rows
├── AAPL_balance_quarterly.parquet
├── AAPL_balance_annual.parquet
├── AAPL_cashflow_quarterly.parquet
├── AAPL_cashflow_annual.parquet
└── ...
```

**Data Schema** (income_quarterly.parquet):
```
Columns:
- date: datetime (fiscal period end)
- totalRevenue: float64
- costOfRevenue: float64
- grossProfit: float64
- operatingIncome: float64
- netIncome: float64
- ebitda: float64
- eps: float64
- ticker: string
- statement_type: string
- frequency: string
```

***

#### Task 1.5: `fetch_eodhd_eod_prices`

**Purpose**: Download end-of-day prices (30+ years, all exchanges)

**API Endpoint**: EODHD EOD Data API

**Logic**:
```python
from datetime import datetime, timedelta

for ticker in watchlist:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*30)  # 30 years
    
    prices = eodhd.get_prices_eod(
        f"{ticker}.US",
        from_=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d')
    )
    
    df = pd.DataFrame(prices)
    df['ticker'] = ticker
    
    # Partition by year-month for efficient queries
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    df.to_parquet(
        f"data/raw_downloads/eodhd/prices/{ticker}_prices.parquet",
        compression='snappy',
        partition_cols=['year_month'],
        index=False
    )
```

**Output Structure**:
```
data/raw_downloads/eodhd/prices/
├── AAPL_prices.parquet/
│   ├── year_month=2020-01/
│   │   └── data.parquet
│   ├── year_month=2020-02/
│   │   └── data.parquet
│   └── ...
└── MSFT_prices.parquet/
    └── ...
```

**Data Schema** (prices):
```
Columns:
- date: datetime
- open: float64
- high: float64
- low: float64
- close: float64
- adjusted_close: float64
- volume: int64
- ticker: string
- year_month: period (partition key)
```

**Storage Size**: ~100-200 MB per ticker (30 years daily data)

***

#### Task 1.6: `fetch_eodhd_macro_indicators`

**Purpose**: Download 30+ macro indicators from 1960+

**API Endpoint**: EODHD Macroeconomics API

**Logic**:
```python
indicators = [
    'GDP', 'GDPC', 'UNEMPLOYMENT', 'INFLATION_CONSUMER_PRICES',
    'REAL_INTEREST_RATE', 'TRADE_BALANCE', 'DEBT_TO_GDP',
    'ISM_MANUFACTURING', 'CONSUMER_CONFIDENCE', 'RETAIL_SALES',
    'INDUSTRIAL_PRODUCTION', 'HOUSING_STARTS', 'PMI'
]

countries = ['USA', 'CHN', 'JPN', 'GBR', 'DEU', 'FRA']

for country in countries:
    for indicator in indicators:
        try:
            data = eodhd.get_macro_indicator(indicator, country)
            
            df = pd.DataFrame(data)
            df['country'] = country
            df['indicator'] = indicator
            df['ingestion_timestamp'] = datetime.now()
            
            df.to_parquet(
                f"data/raw_downloads/eodhd/macro/{country}_{indicator}.parquet",
                compression='snappy',
                index=False
            )
        except Exception as e:
            log.warning(f"Failed {country} {indicator}: {e}")
            continue
```

**Output Structure**:
```
data/raw_downloads/eodhd/macro/
├── USA_GDP.parquet                    # Quarterly since 1947
├── USA_UNEMPLOYMENT.parquet           # Monthly since 1948
├── USA_INFLATION_CONSUMER_PRICES.parquet
├── CHN_GDP.parquet
└── ...
```

**Data Schema** (macro indicators):
```
Columns:
- date: datetime
- value: float64
- country: string
- indicator: string
- ingestion_timestamp: datetime
```

***

#### Task 1.7: `fetch_eodhd_insider_transactions`

**Purpose**: Download Form 4 insider transactions

**API Endpoint**: EODHD Insider Transactions API

**Logic**:
```python
for ticker in watchlist:
    # Fetch last 2 years of transactions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    transactions = eodhd.get_insider_transactions(
        ticker,
        from_=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d'),
        limit=1000
    )
    
    df = pd.DataFrame(transactions)
    
    # Clean transaction types
    df['transaction_type'] = df['transactionCode'].map({
        'P': 'Buy',
        'S': 'Sell',
        'M': 'Option Exercise',
        'G': 'Gift',
        'A': 'Award',
        'D': 'Disposition'
    })
    
    df.to_csv(
        f"data/raw_downloads/eodhd/insider_transactions/{ticker}_insider.csv",
        index=False
    )
```

**Output Structure**:
```
data/raw_downloads/eodhd/insider_transactions/
├── AAPL_insider.csv
├── MSFT_insider.csv
└── ...
```

**Data Schema** (insider_transactions):
```
Columns:
- date: datetime (transaction date)
- owner: string (insider name)
- position: string (CEO, CFO, Director, 10% Owner)
- transactionCode: string (P, S, M, G, A, D)
- transaction_type: string (Buy, Sell, Option Exercise)
- sharesTraded: int64
- pricePerShare: float64
- valueOfShares: float64
- sharesOwned: int64 (after transaction)
- ticker: string
```

***

#### Task 1.8: `fetch_fmp_detailed_financials`

**Purpose**: Download detailed financial statement line items (granular)

**API Endpoint**: `https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{ticker}`

**Logic**:
```python
for ticker in watchlist:
    # Income statement (as-reported, not normalized)
    income = fmp.get_income_statement_as_reported(ticker, period='quarter', limit=40)
    
    # Balance sheet
    balance = fmp.get_balance_sheet_as_reported(ticker, period='quarter', limit=40)
    
    # Cash flow
    cashflow = fmp.get_cash_flow_as_reported(ticker, period='quarter', limit=40)
    
    # Save each with metadata
    for stmt_type, data in [('income', income), ('balance', balance), ('cashflow', cashflow)]:
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        df['statement_type'] = stmt_type
        df['source'] = 'fmp'
        
        df.to_parquet(
            f"data/raw_downloads/fmp/financials/{ticker}_{stmt_type}_detailed.parquet",
            compression='snappy',
            index=False
        )
```

**Output Structure**:
```
data/raw_downloads/fmp/financials/
├── AAPL_income_detailed.parquet       # 100+ columns (all line items)
├── AAPL_balance_detailed.parquet
├── AAPL_cashflow_detailed.parquet
└── ...
```

**Key Difference vs EODHD**:
- FMP: As-reported labels (e.g., "RevenueFromContractWithCustomerIncludingAssessedTax")
- EODHD: Normalized labels (e.g., "totalRevenue")
- Both used together for cross-validation and semantic tagging

***

#### Task 1.9: `fetch_fmp_institutional_holdings`

**Purpose**: Download 13F institutional ownership data

**API Endpoint**: `https://financialmodelingprep.com/api/v3/institutional-holder/{ticker}`

**Logic**:
```python
for ticker in watchlist:
    holdings = fmp.get_institutional_holders(ticker)
    
    df = pd.DataFrame(holdings)
    df['ticker'] = ticker
    df['report_date'] = datetime.now().strftime('%Y-%m-%d')
    
    df.to_csv(
        f"data/raw_downloads/fmp/institutional/{ticker}_13f.csv",
        index=False
    )
```

**Output Structure**:
```
data/raw_downloads/fmp/institutional/
├── AAPL_13f.csv
└── ...
```

**Data Schema**:
```
Columns:
- holder: string (institution name)
- shares: int64
- dateReported: datetime
- change: int64 (change from prior quarter)
- ticker: string
```

***

#### Task 1.10: `validate_foundation_data`

**Purpose**: Data quality gate before downstream DAGs

**Logic**:
```python
def validate_foundation_data():
    issues = []
    
    # Check 1: All tickers have SEC filings
    for ticker in watchlist:
        eodhd_filings = glob(f"data/raw_downloads/eodhd/sec_filings/{ticker}_10-K_*.json")
        fmp_filings = glob(f"data/raw_downloads/fmp/sec_filings/{ticker}_10-K_*.json")
        
        if len(eodhd_filings) == 0 and len(fmp_filings) == 0:
            issues.append(f"Missing 10-K filings for {ticker}")
    
    # Check 2: Price data completeness (expect ~7500 rows for 30 years)
    for ticker in watchlist:
        prices = pd.read_parquet(f"data/raw_downloads/eodhd/prices/{ticker}_prices.parquet")
        if len(prices) < 5000:  # Allow some gaps
            issues.append(f"Insufficient price data for {ticker}: {len(prices)} rows")
    
    # Check 3: Macro data freshness (latest date should be recent)
    usa_gdp = pd.read_parquet("data/raw_downloads/eodhd/macro/USA_GDP.parquet")
    latest_date = pd.to_datetime(usa_gdp['date']).max()
    if (datetime.now() - latest_date).days > 180:  # GDP is quarterly
        issues.append(f"Stale macro data: latest GDP date is {latest_date}")
    
    if issues:
        send_slack_alert(f"Foundation Ingestion Validation Failed:\n" + "\n".join(issues))
        raise AirflowException("Data quality gate failed")
    
    return "✅ All validation checks passed"
```

**Validation Checks**:
1. SEC filings exist for all watchlist tickers
2. Price data has >5000 rows per ticker (30 years ≈ 7500 trading days)
3. Macro data is not stale (latest GDP date within 6 months)
4. No JSON parse errors in staging files
5. File sizes reasonable (detect truncated downloads)

***

### DAG Configuration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

default_args = {
    'owner': 'equity_research',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 13),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30)
}

dag = DAG(
    'dag_01_foundation_ingestion',
    default_args=default_args,
    description='Fetch raw data from EODHD/FMP APIs',
    schedule_interval='0 6 * * *',  # 6 AM HKT daily
    catchup=False,  # Don't backfill missed runs
    max_active_runs=1,  # Only one instance at a time
    tags=['foundation', 'ingestion', 'eodhd', 'fmp']
)

# Task groups for organization
with TaskGroup('eodhd_tasks', dag=dag) as eodhd_group:
    task_1_1 = PythonOperator(task_id='fetch_eodhd_sec_filings', ...)
    task_1_4 = PythonOperator(task_id='fetch_eodhd_fundamentals', ...)
    task_1_5 = PythonOperator(task_id='fetch_eodhd_eod_prices', ...)
    task_1_6 = PythonOperator(task_id='fetch_eodhd_macro_indicators', ...)
    task_1_7 = PythonOperator(task_id='fetch_eodhd_insider_transactions', ...)

with TaskGroup('fmp_tasks', dag=dag) as fmp_group:
    task_1_2 = PythonOperator(task_id='fetch_fmp_sec_filings', ...)
    task_1_3 = PythonOperator(task_id='fetch_fmp_earnings_transcripts', ...)
    task_1_8 = PythonOperator(task_id='fetch_fmp_detailed_financials', ...)
    task_1_9 = PythonOperator(task_id='fetch_fmp_institutional_holdings', ...)

task_1_10 = PythonOperator(
    task_id='validate_foundation_data',
    python_callable=validate_foundation_data,
    dag=dag
)

# Execution order: EODHD and FMP tasks run in parallel, then validation
[eodhd_group, fmp_group] >> task_1_10
```

***

### Performance Optimization

**Parallelization**:
- EODHD and FMP tasks run in parallel (separate API rate limits)
- Within each group, tasks process tickers sequentially to respect rate limits

**Incremental Updates** (After Initial Backfill):
```python
def fetch_eodhd_sec_filings_incremental(**context):
    """Only fetch filings newer than last successful run"""
    last_run_date = get_last_successful_run_date('dag_01_foundation_ingestion')
    
    for ticker in watchlist:
        filings = eodhd.get_fundamental_data(ticker, filter_='Filings')
        
        # Filter for new filings only
        new_10k = [f for f in filings.get('10-K', []) 
                   if pd.to_datetime(f['date']) > last_run_date]
        
        new_10q = [f for f in filings.get('10-Q', []) 
                   if pd.to_datetime(f['date']) > last_run_date]
        
        # Process only new filings
        for filing in new_10k + new_10q:
            save_filing(...)
```

**Caching**:
- Macro data cached for 24 hours (changes infrequently)
- Price data: Only fetch last 7 days after initial backfill
- SEC filings: Check EDGAR RSS feed for new filings before full API calls

***

### Output Summary

**Total Data Volume** (Initial Backfill for 100 Tickers):
- SEC Filings: ~50 GB (300 10-Ks + 800 10-Qs)
- Prices: ~20 GB (30 years × 100 tickers)
- Fundamentals: ~5 GB (quarterly + annual × 30 years)
- Macro Indicators: ~2 GB (60+ years × 30 indicators × 6 countries)
- Insider Transactions: ~500 MB (2 years × 100 tickers)
- Earnings Transcripts: ~2 GB (8 quarters × 100 tickers)

**Total**: ~80 GB staged in `data/raw_downloads/`

**Retention**: Files deleted after 7 days OR after successful processing by downstream DAGs

***
