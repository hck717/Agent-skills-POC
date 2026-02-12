# Airflow DAGs for Multi-Agent Equity Research System

## Overview

This directory contains 13 Airflow DAGs that orchestrate data ingestion, transformation, and loading for a sophisticated equity research platform.

## DAG Architecture

### Daily DAGs
- **DAG 1**: Daily Market Data (6 PM ET) - EOD prices, technical indicators, factors
- **DAG 2**: News & Sentiment (3x daily) - Breaking news, SEC filings detection
- **DAG 11**: Data Quality Monitoring (11 PM HKT) - Cross-database validation
- **DAG 13**: Neo4j Auto-Ingestion (11 PM HKT) - CSV imports to knowledge graphs

### Weekly DAGs
- **DAG 3**: Fundamental Data (Sun 2 AM HKT) - Financial statements, forensic scores
- **DAG 9**: Macro Indicators (Mon 1 AM HKT) - GDP, rates, FX, economic calendar

### Monthly DAGs
- **DAG 6**: Insider Trading (1st, 3 AM HKT) - Form 4 transactions
- **DAG 7**: Institutional Holdings (15th) - 13F filings, ETF constituents
- **DAG 12**: Model Retraining (1st, 4 AM HKT) - CRAG evaluator, sentiment models

### Quarterly/Triggered DAGs
- **DAG 4**: SEC Filings (triggered) - 10-K/10-Q processing for RAG
- **DAG 5**: Earnings Transcripts (triggered) - Transcript chunking, sentiment analysis
- **DAG 8**: Supply Chain Graph (quarterly) - Network relationships, centrality metrics
- **DAG 10**: Central Bank Comms (triggered) - FOMC/ECB/BOJ policy statements

## Dependency Graph

```
DAG 1 (Daily Prices)
  ↓
DAG 3 (Weekly Fundamentals)
  ↓
DAG 4 (SEC Filings) ← triggered by DAG 2
  ↓
DAG 8 (Supply Chain)

DAG 2 (News) → triggers DAG 4, DAG 5

DAG 3 → DAG 5 (Earnings Transcripts)

DAG 9 (Macro) → DAG 10 (Central Bank)

DAG 6, DAG 7 (Insider/Institutional) → DAG 8 (Supply Chain)

DAG 11 (Quality) → monitors ALL

DAG 13 (Neo4j) ← runs after DAG 4, DAG 8
```

## Database Targets

| DAG | DuckDB | Postgres | Qdrant | Neo4j |
|-----|--------|----------|--------|-------|
| 1 | ✓ Prices, Factors | | | |
| 2 | | ✓ News | | |
| 3 | ✓ Fundamentals | ✓ Financials | | |
| 4 | | | ✓ 10-K/10-Q | ✓ Concepts |
| 5 | | ✓ Earnings Dates | ✓ Transcripts | |
| 6 | | ✓ Insider Trades | | |
| 7 | | | | ✓ Ownership |
| 8 | | | | ✓ Supply Chain |
| 9 | ✓ Macro | | | |
| 10 | | | ✓ Central Bank | |
| 13 | | | | ✓ All Graphs |

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
EODHD_API_KEY=your_eodhd_key
FMP_API_KEY=your_fmp_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Database Connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=equity_research
POSTGRES_USER=airflow
POSTGRES_PASSWORD=your_password

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_key

REDIS_HOST=localhost
REDIS_PORT=6379

# Notifications
ALERT_EMAIL=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Airflow Configuration

1. **Connections**: Create connections in Airflow UI
   - `postgres_equity_research`: PostgreSQL connection
   - `neo4j_connection`: Neo4j connection

2. **Variables**: Set in Airflow UI > Admin > Variables
   - `tickers_list`: JSON array of tickers to track
   - `data_quality_threshold`: Minimum data coverage (0.95)

## Running DAGs

### Local Docker Setup

```bash
# Navigate to project root
cd /Users/brianho/Agent-skills-POC

# Start Docker services
docker-compose up -d

# Access Airflow UI
open http://localhost:8080
# Default: airflow / airflow
```

### Manual Trigger

```bash
# Trigger a DAG via CLI
airflow dags trigger dag_01_daily_market_data_pipeline

# Trigger with config
airflow dags trigger dag_04_quarterly_sec_filings_pipeline --conf '{"ticker":"AAPL"}'
```

### Testing Individual Tasks

```bash
# Test a single task
airflow tasks test dag_01_daily_market_data_pipeline extract_eodhd_eod_prices 2026-02-12
```

## Data Quality Monitoring

**DAG 11** runs nightly and checks:
- DuckDB: Data freshness, ticker coverage, price consistency
- Postgres: NULL values, financial statement balance checks
- Qdrant: Index health, document staleness
- Neo4j: Orphaned nodes, centrality score completeness

Alerts sent via:
- Email (configured in `config.py`)
- Slack webhook (optional)

## Error Handling

All DAGs implement:
- **Retries**: 3 attempts with exponential backoff
- **Fallback APIs**: Switch EODHD ↔ FMP on failure
- **Idempotency**: UPSERT operations prevent duplicates
- **Sensors**: Wait for upstream DAG completion before starting

## Performance Optimization

- **Parallel Execution**: Independent tasks run concurrently
- **Incremental Loads**: Date filters fetch only new data
- **Bulk APIs**: EODHD bulk endpoints for large datasets
- **Redis Caching**: 1-hour TTL for redundant API calls
- **Partitioning**: DuckDB tables partitioned by year-month

## Development

### Adding a New DAG

1. Create file: `dag_XX_your_pipeline.py`
2. Import config: `import config`
3. Define task functions
4. Create DAG with `default_args = config.DEFAULT_ARGS.copy()`
5. Add to this README

### Testing

```bash
# Validate DAG syntax
python scripts/dags/dag_01_daily_market_data.py

# Check for import errors
airflow dags list

# Run backfill
airflow dags backfill dag_01_daily_market_data_pipeline \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

## Monitoring

### Airflow UI
- **Graph View**: Visualize task dependencies
- **Gantt Chart**: Task duration analysis
- **Task Logs**: Debug failures

### Metrics
- DAG run duration
- Task success rate
- Data quality score trends

## Troubleshooting

### Common Issues

**DAG not appearing in UI:**
```bash
# Check for syntax errors
python scripts/dags/dag_XX_pipeline.py

# Refresh DAGs
airflow dags list-import-errors
```

**Task failing with connection error:**
- Check database/API credentials in `.env`
- Verify network connectivity
- Check rate limits

**DuckDB locked:**
```bash
# Kill stale connections
pkill -f duckdb
```

**Neo4j import fails:**
- Check CSV file permissions
- Verify Neo4j import path configured correctly
- Ensure CSV headers match schema

## Next Steps

1. **Generate Remaining DAGs**: Run `python scripts/generate_remaining_dags.py`
2. **Setup Databases**: Run `docker-compose up` to start all services
3. **Configure Connections**: Add credentials in Airflow UI
4. **Enable DAGs**: Toggle DAGs on in Airflow UI
5. **Monitor**: Check DAG 11 quality reports

## Support

For issues or questions:
- Check Airflow logs: `docker-compose logs airflow-webserver`
- Review task logs in Airflow UI
- Consult config.py for settings
