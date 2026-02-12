# 🚀 Airflow Deployment Guide

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- `.env` file configured with all required environment variables
- PostgreSQL database URL (Neon or local)

### First Time Setup

```bash
# 1. Stop any running containers
docker-compose down

# 2. Build the custom Airflow image (includes all dependencies)
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Monitor startup (this may take 5-10 minutes on first run)
docker-compose logs -f airflow-webserver
```

### Access Airflow UI

- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`

### Verify DAGs Loaded

Once the webserver is running, check the DAGs page. You should see:
- `00_connection_test`
- `dag_02_daily_news_sentiment_pipeline`
- `dag_03_weekly_fundamental_data_pipeline`
- `dag_04_quarterly_sec_filings_pipeline`
- `dag_05_quarterly_earnings_transcripts_pipeline`
- `06_monthly_insider_trading`
- `dag_07_monthly_institutional_holdings_pipeline`
- `dag_09_weekly_macro_indicators_pipeline`
- `dag_11_daily_data_quality_monitoring`
- `dag_13_neo4j_auto_ingest_pipeline`
- And others...

---

## Environment Variables

Ensure your `.env` file contains:

```bash
# Database (Required)
POSTGRES_URL=postgresql://user:password@host:port/database

# Neo4j (Required)
NEO4J_URI=neo4j+s://your-neo4j-instance.com
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Qdrant (Required)
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your-api-key

# Financial Data APIs
EODHD_API_KEY=your-eodhd-key
FMP_API_KEY=your-fmp-key
TAVILY_API_KEY=your-tavily-key
```

---

## Fixes Implemented

### ✅ Timeout Issues Fixed
- **Increased webserver timeout**: 120s → 600s (10 minutes)
- Accommodates slow Neon database connection from Hong Kong
- Allows time for all 13 DAGs to serialize on startup

### ✅ Restart Loop Fixed
- **Pre-installed dependencies**: All packages built into Docker image
- No more runtime `pip install` on every restart
- Saves 30-60 seconds per restart

### ✅ Database Connection Optimized
- **Connection pooling**: Pool size 10, max overflow 20
- **Pool recycling**: 3600s to handle Neon connection limits
- **Reduced DAG parsing frequency**: 300s interval (from default 30s)

### ✅ Healthchecks Added
- Webserver: HTTP health endpoint check
- Scheduler: Process monitoring
- Better visibility into container health

---

## Troubleshooting

### Container Keeps Restarting

```bash
# Check logs
docker-compose logs airflow-webserver

# Common issues:
# 1. Database connection failed - verify POSTGRES_URL in .env
# 2. Still timing out - increase timeout further in docker-compose.yml
# 3. DAG syntax errors - check ./scripts/DAGs/ files
```

### DAGs Not Appearing

```bash
# 1. Check DAG files are in correct location
ls -la ./scripts/DAGs/

# 2. Check webserver logs for parsing errors
docker-compose logs airflow-webserver | grep -i error

# 3. Trigger manual DAG scan
docker exec -it airflow_webserver airflow dags list
```

### Database Connection Issues

```bash
# Test database connection
docker exec -it airflow_webserver bash
python -c "from airflow.settings import Session; Session().execute('SELECT 1').fetchone(); print('DB OK')"
```

### Slow Startup from Hong Kong

Your Neon database is in US East, causing latency:

**Option 1: Use Local PostgreSQL (Development)**
```bash
# Add to docker-compose.yml:
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:

# Update .env:
POSTGRES_URL=postgresql://airflow:airflow@postgres:5432/airflow
```

**Option 2: Keep Neon (Current Setup)**
- Current timeouts (600s) should work
- Expect 5-10 minute initial startup
- Subsequent restarts faster (~2-3 minutes)

---

## Useful Commands

```bash
# View logs
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose build
docker-compose up -d

# Clear everything and start fresh
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Access webserver shell
docker exec -it airflow_webserver bash

# List DAGs
docker exec -it airflow_webserver airflow dags list

# Trigger a DAG manually
docker exec -it airflow_webserver airflow dags trigger 00_connection_test

# Check database status
docker exec -it airflow_webserver airflow db check
```

---

## Performance Expectations

### First Startup (Fresh Build)
- Image build: 2-3 minutes
- Database migration: 3-5 minutes (with Neon latency)
- DAG serialization: 2-4 minutes (13 DAGs)
- **Total**: ~10-12 minutes

### Subsequent Startups
- No build needed
- Database already migrated
- DAG reserialization: 1-2 minutes
- **Total**: ~2-3 minutes

### Local PostgreSQL (If Switched)
- First startup: ~3-4 minutes
- Subsequent: ~30-60 seconds

---

## Testing Your DAGs

1. **Connection Test DAG** (Start Here)
   ```bash
   # Enable and trigger
   docker exec -it airflow_webserver airflow dags unpause 00_connection_test
   docker exec -it airflow_webserver airflow dags trigger 00_connection_test
   ```

2. **Monitor in UI**
   - Go to http://localhost:8080
   - Click on DAG name
   - View Graph or Grid view
   - Check task logs by clicking task boxes

3. **Enable Other DAGs Gradually**
   - Start with daily/simple DAGs first
   - Then weekly/monthly DAGs
   - Monitor resource usage

---

## Next Steps

1. ✅ **Containers running**: Check with `docker ps`
2. ✅ **Access UI**: http://localhost:8080
3. ✅ **Run connection test**: Trigger `00_connection_test` DAG
4. ✅ **Enable your DAGs**: Toggle ON in UI
5. ✅ **Monitor execution**: Watch logs and task status

---

## Support

If issues persist:
1. Check logs: `docker-compose logs -f`
2. Verify all env vars in `.env`
3. Test database connectivity
4. Review DAG syntax errors
5. Consider local PostgreSQL for development
