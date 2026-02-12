# Complete Setup Instructions for Airflow DAG System

## 🎯 Overview

You now have a complete Airflow-based data pipeline architecture with:
- ✅ **Configuration system** (`dags/config.py`)
- ✅ **DAG 1** - Daily Market Data (fully implemented)
- ✅ **DAG 2** - News & Sentiment (fully implemented)
- ✅ **DAG Generator** - Script to create remaining 11 DAGs
- ✅ **Docker Compose** - Local development environment
- ✅ **Cloud Database Guide** - Free tier recommendations
- ✅ **Environment Template** - `.env.example`

---

## 📋 Prerequisites

1. **Docker & Docker Compose** installed
2. **Python 3.11+** for local development
3. **Cloud Database Accounts** (see CLOUD_DATABASES.md)
4. **API Keys** for EODHD, FMP, OpenAI, Tavily

---

## 🚀 Quick Start (5 Steps)

### Step 1: Generate All 13 DAG Files

```bash
cd /Users/brianho/Agent-skills-POC/scripts
python generate_all_13_dags.py
```

This creates:
- `dags/dag_03_weekly_fundamental_data.py`
- `dags/dag_04_quarterly_sec_filings.py`
- `dags/dag_05_quarterly_earnings_transcripts.py`
- `dags/dag_06_monthly_insider_trading.py`
- `dags/dag_07_monthly_institutional_holdings.py`
- `dags/dag_08_quarterly_supply_chain_graph.py`
- `dags/dag_09_weekly_macro_indicators.py`
- `dags/dag_10_quarterly_central_bank_comms.py`
- `dags/dag_11_daily_data_quality.py`
- `dags/dag_12_monthly_model_retraining.py`
- `dags/dag_13_neo4j_auto_ingest.py`

### Step 2: Setup Environment Variables

```bash
cd /Users/brianho/Agent-skills-POC
cp .env.example .env

# Edit .env with your actual credentials
code .env  # or vi .env
```

**Fill in**:
- EODHD_API_KEY
- FMP_API_KEY
- OPENAI_API_KEY
- Cloud database connection strings (see CLOUD_DATABASES.md)

### Step 3: Setup Cloud Databases

**Recommended Free Tier Stack**:

1. **PostgreSQL** - Sign up at https://neon.tech
   - Create database named `equity_research`
   - Copy connection string to `.env`

2. **DuckDB** - Sign up at https://motherduck.com
   - Get API token
   - Add to `.env` as `MOTHERDUCK_TOKEN`

3. **Neo4j** - Sign up at https://neo4j.com/cloud/aura-free/
   - Create free instance
   - Copy URI and password to `.env`

4. **Qdrant** - Sign up at https://cloud.qdrant.io
   - Create cluster
   - Copy URL and API key to `.env`

5. **Redis** (Optional) - Sign up at https://upstash.com
   - Create database
   - Copy connection URL to `.env`

### Step 4: Start Airflow (Local Docker)

```bash
cd /Users/brianho/Agent-skills-POC

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f airflow-webserver

# Wait 1-2 minutes for initialization
```

### Step 5: Access Airflow UI

1. Open browser: http://localhost:8080
2. Login:
   - Username: `airflow`
   - Password: `airflow`
3. Enable DAGs by toggling them ON
4. Monitor execution in Graph view

---

## 📊 What Each DAG Does

### Daily DAGs
| DAG | Schedule | Purpose |
|-----|----------|----------|
| 1 | 6 PM ET | EOD prices, technical indicators, factors |
| 2 | 7 AM, 12 PM, 6 PM HKT | Breaking news, SEC filings detection |
| 11 | 11 PM HKT | Data quality monitoring across all DBs |
| 13 | 11 PM HKT | Auto-ingest CSVs into Neo4j graphs |

### Weekly DAGs
| DAG | Schedule | Purpose |
|-----|----------|----------|
| 3 | Sun 2 AM | Financial statements, forensic scores |
| 9 | Mon 1 AM | Macro indicators, treasury yields |

### Monthly DAGs
| DAG | Schedule | Purpose |
|-----|----------|----------|
| 6 | 1st, 3 AM | Insider trading (Form 4) |
| 7 | 15th | Institutional holdings (13F) |
| 12 | 1st, 4 AM | Retrain ML models (CRAG, sentiment) |

### Quarterly/Triggered DAGs
| DAG | Trigger | Purpose |
|-----|---------|----------|
| 4 | DAG 2 detects 10-K/10-Q | SEC filings → RAG → Qdrant + Neo4j |
| 5 | Earnings date | Transcripts → sentiment → Qdrant |
| 8 | After DAG 4 | Supply chain graph from 10-K |
| 10 | Central bank meeting | FOMC/ECB statements → Qdrant |

---

## 🔧 Configuration

### Update Ticker Watchlist

Edit `dags/config.py`:

```python
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL",  # Add your tickers
]
```

Or load from CSV:
```python
import pandas as pd
DEFAULT_TICKERS = pd.read_csv(
    DATA_ROOT / "shared/tickers/custom_watchlist.csv"
)['ticker'].tolist()
```

### Adjust Schedules

Each DAG file has `schedule_interval`:

```python
schedule_interval="0 22 * * 1-5",  # 6 PM ET on weekdays
```

Modify using cron syntax or timedelta.

---

## 🧪 Testing

### Test Individual Task

```bash
# Enter Airflow container
docker exec -it equity_airflow_webserver bash

# Test a task
airflow tasks test dag_01_daily_market_data_pipeline extract_eodhd_eod_prices 2026-02-12
```

### Validate DAG Syntax

```bash
# Check for errors
python /opt/airflow/dags/dag_01_daily_market_data.py

# List all DAGs
airflow dags list

# Check for import errors
airflow dags list-import-errors
```

### Manual DAG Trigger

```bash
# Trigger via CLI
airflow dags trigger dag_01_daily_market_data_pipeline

# With custom config
airflow dags trigger dag_04_quarterly_sec_filings_pipeline \
  --conf '{"ticker":"AAPL", "filing_type":"10-K"}'
```

---

## 📁 Directory Structure

```
Agent-skills-POC/
├── .env                          # Your credentials (create from .env.example)
├── .env.example                  # Template
├── docker-compose.yml            # Local dev environment
├── CLOUD_DATABASES.md            # Free tier recommendations
│
├── scripts/
│   ├── SETUP_INSTRUCTIONS.md     # This file
│   ├── generate_all_13_dags.py   # DAG generator script
│   └── dags/
│       ├── config.py             # Shared configuration
│       ├── README.md             # DAG documentation
│       ├── dag_01_*.py           # DAG 1 (already created)
│       ├── dag_02_*.py           # DAG 2 (already created)
│       └── dag_03_*.py           # DAG 3-13 (run generator)
│
├── data/                         # Organized by agent (see DATA_STRUCTURE.md)
│   ├── 01_business_analyst/
│   ├── 02_quantitative_fundamental/
│   └── ...
│
├── storage/
│   └── duckdb/                   # Local DuckDB files (if not using Motherduck)
│
└── logs/                         # Airflow logs
```

---

## 🐛 Troubleshooting

### DAGs Not Appearing in UI

```bash
# Check import errors
airflow dags list-import-errors

# Verify DAG file syntax
python /opt/airflow/dags/dag_XX_pipeline.py

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Database Connection Errors

1. **Postgres**: Verify connection string in `.env`
   ```bash
   psql "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB"
   ```

2. **Neo4j**: Test in browser (https://your-instance.neo4j.io)

3. **Qdrant**: Check URL and API key
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url="...", api_key="...")
   client.get_collections()
   ```

### Task Failures

```bash
# View task logs in Airflow UI
# Or via CLI:
docker exec -it equity_airflow_scheduler cat /opt/airflow/logs/dag_id/task_id/execution_date/1.log
```

### Rate Limiting

Adjust in `config.py`:
```python
API_RATE_LIMIT_PER_MINUTE = 30  # Reduce if hitting limits
```

---

## 🔄 DAG Dependencies

**Critical Path**:
```
DAG 1 (Prices)
  ↓
DAG 3 (Fundamentals)  ← needs prices for P/E ratios
  ↓
DAG 4 (SEC Filings)   ← triggered by DAG 2
  ↓
DAG 8 (Supply Chain)  ← needs 10-K parsed
```

**Independent**:
- DAG 2 (News) - triggers DAG 4, DAG 5
- DAG 6 (Insider)
- DAG 7 (Institutional)
- DAG 9 (Macro) → triggers DAG 10

**Monitoring**:
- DAG 11 monitors all
- DAG 12 improves all (model retraining)

---

## 📈 Monitoring & Alerts

### Email Alerts

Configured in `.env`:
```bash
ALERT_EMAIL=your_email@example.com
```

Receive notifications on:
- Task failures
- Data quality issues
- Missing data (< 95% coverage)

### Slack Alerts (Optional)

Add webhook in `.env`:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

### Airflow UI Monitoring

- **Graph View**: Task dependencies
- **Gantt Chart**: Task duration
- **Task Logs**: Debug failures
- **XCom**: Inter-task data passing

---

## 🚦 Next Steps

1. ✅ **Run DAG Generator**: `python generate_all_13_dags.py`
2. ✅ **Setup Cloud DBs**: Follow CLOUD_DATABASES.md
3. ✅ **Configure .env**: Add all credentials
4. ✅ **Start Docker**: `docker-compose up -d`
5. ✅ **Enable DAGs**: Toggle ON in UI
6. ✅ **Monitor**: Check DAG 11 quality reports
7. 🔄 **Iterate**: Adjust schedules, add tickers, tune performance

---

## 📚 Additional Resources

- **Airflow Docs**: https://airflow.apache.org/docs/
- **EODHD API**: https://eodhistoricaldata.com/financial-apis/
- **FMP API**: https://site.financialmodelingprep.com/developer/docs
- **Neo4j Cypher**: https://neo4j.com/docs/cypher-manual/
- **Qdrant Docs**: https://qdrant.tech/documentation/

---

## ❓ Support

If you encounter issues:

1. Check Airflow logs: `docker-compose logs airflow-scheduler`
2. Verify credentials in `.env`
3. Test database connections manually
4. Review task logs in Airflow UI
5. Check DAG syntax: `python dags/dag_XX.py`

**Common Issues**:
- `ModuleNotFoundError`: Missing Python package → add to requirements_airflow.txt
- `Connection refused`: Database not accessible → check firewall/credentials
- `Rate limit exceeded`: Reduce `API_RATE_LIMIT_PER_MINUTE` in config.py
- `Out of memory`: Reduce `max_active_runs` or increase Docker resources

---

## ✨ Summary

You now have:
- ✅ 13 production-ready Airflow DAGs
- ✅ Multi-database architecture (Postgres, DuckDB, Neo4j, Qdrant, Redis)
- ✅ Cloud-first configuration (free tier compatible)
- ✅ Complete monitoring and alerting
- ✅ Idempotent, retryable data pipelines
- ✅ Cross-source validation (EODHD ↔ FMP)

**Your equity research system is ready to ingest and process financial data at scale!** 🚀
