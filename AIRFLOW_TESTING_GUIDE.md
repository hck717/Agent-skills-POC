# 🧪 Airflow Testing Guide

## Prerequisites Checklist

Before starting, ensure you have:

- ✅ Docker Desktop installed and running
- ✅ `.env` file in project root with all API keys and database credentials filled
- ✅ Ports 8080 (Airflow) available on your machine
- ✅ Internet connection for cloud database access

---

## Step 1: Verify Your .env File

Make sure your `.env` file has all required variables:

```bash
cat .env
```

You should see:
```bash
# API Keys
EODHD_API_KEY
FMP_API_KEY=
TAVILY_API_KEY=

# Cloud Databases
POSTGRES_URL=postgresql://user:pass@cloud-host/db?sslmode=require
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_aura_pass
QDRANT_URL=https://cloud.qdrant.io:6333
QDRANT_API_KEY=your_key

# Ollama (local)
OLLAMA_URL=http://host.docker.internal:11434
```

**Important**: Replace placeholder values with your actual credentials!

---

## Step 2: Start Airflow Services

### Option A: Fresh Start (Recommended for First Time)

```bash
# Navigate to project directory
cd ~/Agent-skills-POC

# Pull latest code (if you just updated)
git pull origin main

# Remove old containers (if any)
docker-compose down -v

# Start services
docker-compose up -d
```

### Option B: Restart Existing Services

```bash
# Restart services
docker-compose restart

# Or rebuild if you changed docker-compose.yml
docker-compose up -d --build
```

---

## Step 3: Monitor Container Startup

### Check Container Status

```bash
# View running containers
docker-compose ps

# Should show:
# airflow_webserver   Up   0.0.0.0:8080->8080/tcp
# airflow-scheduler   Up
```

### Watch Logs (Critical for Debugging)

```bash
# Watch all logs
docker-compose logs -f

# Or just webserver logs
docker-compose logs -f airflow-webserver

# Or just scheduler logs
docker-compose logs -f airflow-scheduler
```

**What to Look For**:
- ✅ `pip install psycopg2-binary neo4j qdrant-client` completes successfully
- ✅ `airflow db init` runs without errors
- ✅ `Airflow webserver started` message appears
- ✅ No connection errors to databases

**Common Issues**:
- ❌ `ModuleNotFoundError: No module named 'psycopg2'` → Dependencies not installed (wait longer)
- ❌ `sqlalchemy.exc.OperationalError` → Check POSTGRES_URL in .env
- ❌ `Connection refused` → Check cloud database is accessible

---

## Step 4: Access Airflow Web UI

### Open Browser

1. Navigate to: [http://localhost:8080](http://localhost:8080)
2. **Default Credentials**:
   - Username: `admin`
   - Password: `admin`

**First Time Setup**: If login fails, create admin user:

```bash
docker exec -it airflow_webserver bash
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
exit
```

---

## Step 5: Run Connection Test DAG

### In Airflow UI:

1. **Find the DAG**: Look for `00_connection_test` in the DAG list
2. **Toggle On**: Click the toggle switch to activate the DAG (if paused)
3. **Trigger**: Click the "▶" (play) button on the right to trigger manually
4. **Monitor**: Click on the DAG name to see the Graph View

### Expected Behavior:

The DAG runs 4 tests in parallel:
- `test_postgres` → Tests Neon/Supabase connection
- `test_neo4j` → Tests Neo4j Aura connection
- `test_qdrant` → Tests Qdrant Cloud connection
- `test_api_keys` → Validates EODHD, FMP, Tavily API keys

Then runs:
- `print_summary` → Shows overall results

### Check Results:

**Option 1: UI Graph View**
- Green boxes ✅ = Success
- Red boxes ❌ = Failed
- Click any box → "Log" button to see details

**Option 2: Command Line Logs**

```bash
# View DAG run logs
docker-compose logs airflow-scheduler | grep "connection_test"

# Or check webserver logs
docker-compose logs airflow-webserver | grep -A 50 "CONNECTION TEST SUMMARY"
```

**Option 3: Exec into Container**

```bash
docker exec -it airflow_webserver bash
airflow dags test 00_connection_test 2026-02-12
```

---

## Step 6: Interpret Test Results

### Success Output Should Show:

```
======================================================================
CONNECTION TEST SUMMARY
======================================================================
Deployment Mode: CLOUD

✅ PostgreSQL: success
✅ Neo4j: success
✅ Qdrant: success (5 collections)

API Keys:
  ✅ EODHD: valid
  ✅ FMP: valid
  ✅ TAVILY: loaded
======================================================================
🎉 All connection tests completed!
======================================================================
```

### Troubleshooting Common Failures:

#### ❌ PostgreSQL Connection Failed

**Symptoms**: `psycopg2.OperationalError: connection refused`

**Solutions**:
1. Verify `POSTGRES_URL` format in `.env`:
   ```bash
   postgresql://user:password@host.neon.tech/dbname?sslmode=require
   ```
2. Check Neon dashboard - is database active?
3. Verify password has no special chars that need escaping
4. Test connection outside Docker:
   ```bash
   psql "postgresql://user:pass@host/db?sslmode=require"
   ```

#### ❌ Neo4j Connection Failed

**Symptoms**: `Neo4j connection failed: Unable to retrieve routing information`

**Solutions**:
1. Verify `NEO4J_URI` uses `neo4j+s://` (not `bolt://`)
2. Check Neo4j Aura console - is instance running?
3. Verify credentials match Aura console
4. Check firewall/network allows Neo4j ports

#### ❌ Qdrant Connection Failed

**Symptoms**: `Qdrant connection failed: Unauthorized` or timeout

**Solutions**:
1. Verify `QDRANT_URL` is correct: `https://xxx.cloud.qdrant.io:6333`
2. Check `QDRANT_API_KEY` is set in `.env`
3. Verify API key in Qdrant Cloud dashboard
4. Test with curl:
   ```bash
   curl -H "api-key: YOUR_KEY" https://xxx.cloud.qdrant.io:6333/collections
   ```

#### ⚠️ API Keys Invalid

**Symptoms**: `EODHD API returned status 403` or `FMP API invalid`

**Solutions**:
1. Verify API keys are active in provider dashboards
2. Check API rate limits not exceeded
3. Test manually:
   ```bash
   # EODHD
   curl "https://eodhistoricaldata.com/api/eod/AAPL.US?api_token=YOUR_KEY&fmt=json&limit=1"
   
   # FMP
   curl "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=YOUR_KEY"
   ```

---

## Step 7: Run a Data Pipeline DAG (Optional)

After connection tests pass, try a real data pipeline:

### Test Market Data Ingestion:

1. In Airflow UI, find DAG: `01_daily_market_data`
2. Toggle ON
3. Click "▶" to trigger
4. Monitor execution in Graph View

**What It Does**:
- Fetches OHLCV data from EODHD API
- Stores in PostgreSQL `market_prices` table
- Takes 2-5 minutes for default tickers

### Verify Data Loaded:

```bash
# Connect to Postgres
psql "postgresql://your_user:pass@host/db?sslmode=require"

# Check data
SELECT ticker, date, close FROM market_prices ORDER BY date DESC LIMIT 10;
```

---

## Step 8: Access Airflow Scheduler Logs

### Check DAG Parsing:

```bash
# View scheduler discovering DAGs
docker-compose logs airflow-scheduler | grep "DAG"

# Should see:
# "Loaded DAG <DAG: 00_connection_test>"
# "Loaded DAG <DAG: 01_daily_market_data>"
# etc.
```

### Check for Errors:

```bash
# Look for import errors
docker-compose logs airflow-scheduler | grep -i "error\|exception"

# Check task execution
docker-compose logs airflow-scheduler | grep "Task instance" | tail -20
```

---

## Useful Commands Cheat Sheet

### Container Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View status
docker-compose ps

# View logs (all)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
```

### Airflow CLI (Inside Container)

```bash
# Enter container
docker exec -it airflow_webserver bash

# List DAGs
airflow dags list

# Test DAG (no actual execution)
airflow dags test 00_connection_test 2026-02-12

# Trigger DAG
airflow dags trigger 00_connection_test

# List DAG runs
airflow dags list-runs -d 00_connection_test

# Check task status
airflow tasks states-for-dag-run 00_connection_test <run_id>
```

### Database Verification

```bash
# Postgres
psql "${POSTGRES_URL}"
\dt                    # List tables
SELECT * FROM connection_test;

# Neo4j (via browser)
# Open: http://aura-instance.databases.neo4j.io/browser
# Query: MATCH (n:ConnectionTest) RETURN n;

# Qdrant (via API)
curl -H "api-key: YOUR_KEY" https://xxx.cloud.qdrant.io:6333/collections
```

---

## Next Steps After Successful Testing

1. ✅ **Connection Test Passes** → Your infrastructure is ready!

2. **Configure Tickers**: Edit `scripts/DAGs/config.py`:
   ```python
   DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]  # Add your tickers
   ```

3. **Schedule Production DAGs**: In Airflow UI, toggle on:
   - `01_daily_market_data` - Daily at 6 PM HKT
   - `02_daily_news_sentiment` - Daily at 7 AM HKT
   - `03_weekly_fundamental_data` - Weekly on Mondays
   - etc.

4. **Monitor Data Quality**: Check DAG `11_daily_data_quality` runs automatically

5. **Set Up Alerts**: Configure email in `.env`:
   ```bash
   ALERT_EMAIL=your-email@example.com
   ```

---

## Troubleshooting Resources

### Check Configuration Loaded:

```bash
docker exec -it airflow_webserver python3 << EOF
import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import *
print(f"Postgres: {POSTGRES_HOST}")
print(f"Neo4j: {NEO4J_URI}")
print(f"Qdrant: {QDRANT_URL}")
print(f"Mode: {DEPLOYMENT_MODE}")
EOF
```

### View Environment Variables:

```bash
docker exec airflow_webserver env | grep -E "POSTGRES|NEO4J|QDRANT|API_KEY"
```

### Clean Restart:

```bash
# Stop everything
docker-compose down -v

# Remove logs
rm -rf logs/*

# Start fresh
docker-compose up -d

# Wait 60 seconds for initialization
sleep 60

# Check logs
docker-compose logs -f
```

---

## Performance Tips

- **Mac M1/M3**: Airflow runs well on Apple Silicon
- **RAM**: Allocate at least 8GB to Docker Desktop
- **First Start**: Takes 2-3 minutes for pip installs and DB init
- **DAG Refresh**: New DAGs appear in UI within 30 seconds
- **Parallel Tasks**: Increase `parallelism` in `airflow.cfg` for faster execution

---

## Getting Help

If issues persist:

1. **Check Full Logs**:
   ```bash
   docker-compose logs > airflow_logs.txt
   ```

2. **Verify Network**:
   ```bash
   docker network ls | grep equity
   docker network inspect equity_network
   ```

3. **Test Cloud Connectivity** (outside Docker):
   ```bash
   # Postgres
   nc -zv your-host.neon.tech 5432
   
   # Neo4j
   nc -zv xxx.databases.neo4j.io 7687
   
   # Qdrant
   nc -zv cloud.qdrant.io 6333
   ```

---

**🎉 Happy Testing! Your Airflow equity research data pipeline is ready to roll!**
