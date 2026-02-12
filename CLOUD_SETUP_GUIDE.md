# ☁️ Cloud Database Setup Guide - Step by Step

**Estimated Time: 30-40 minutes**

Follow this guide to set up all 5 cloud databases for your equity research system.

---

## 📋 Overview

You'll set up:
1. ✅ **Neon** (PostgreSQL) - 10 min
2. ✅ **Motherduck** (DuckDB) - 5 min
3. ✅ **Neo4j AuraDB** (Graph) - 10 min
4. ✅ **Qdrant Cloud** (Vectors) - 10 min
5. ✅ **Upstash Redis** (Cache) - 5 min [OPTIONAL]

---

## 🗄️ Database 1: Neon (PostgreSQL)

### Why Neon?
- Best free tier: 0.5 GB storage, serverless
- True PostgreSQL 16 compatible
- Instant scale to zero (saves costs)
- Built-in branching (like Git for databases)

### Step-by-Step Setup

#### 1. Sign Up
```bash
# Open browser
open https://neon.tech
```
- Click **"Sign Up"**
- Use GitHub or email to register
- Choose **Free Plan** (no credit card needed)

#### 2. Create Database
1. After login, click **"Create a project"**
2. Settings:
   - **Project name**: `equity-research-system`
   - **Postgres version**: 16
   - **Region**: Choose closest to Hong Kong (e.g., `AWS Singapore ap-southeast-1`)
3. Click **"Create project"**

#### 3. Create Database
1. In your new project, go to **"Databases"** tab
2. Click **"New Database"**
3. Enter name: `equity_research`
4. Click **"Create"**

#### 4. Get Connection String
1. Click **"Connection Details"** (top right)
2. Copy the connection string:
   ```
   postgresql://username:password@ep-xxx-xxx.us-west-2.aws.neon.tech/equity_research?sslmode=require
   ```

#### 5. Extract Credentials
From the connection string, extract:
```bash
# Example: postgresql://alex:AbC123xyz@ep-cool-sun-123456.us-west-2.aws.neon.tech:5432/equity_research

POSTGRES_HOST=ep-cool-sun-123456.us-west-2.aws.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=equity_research
POSTGRES_USER=alex
POSTGRES_PASSWORD=AbC123xyz
```

#### 6. Test Connection (Optional)
```bash
# Install psql if you don't have it
brew install postgresql

# Test connection
psql "postgresql://username:password@ep-xxx.neon.tech/equity_research?sslmode=require"

# If successful, you'll see:
# equity_research=>

# Type \q to quit
```

✅ **Neon Setup Complete!** Save credentials for later.

---

## 🦆 Database 2: Motherduck (DuckDB)

### Why Motherduck?
- Cloud-hosted DuckDB (native compatibility)
- 10 GB free storage
- Fast analytical queries
- S3/Parquet integration

### Step-by-Step Setup

#### 1. Sign Up
```bash
open https://motherduck.com
```
- Click **"Get Started"** or **"Sign Up"**
- Use GitHub or email
- Free tier includes 10 GB

#### 2. Get API Token
1. After login, go to **Settings** (gear icon, top right)
2. Click **"API Tokens"** or **"Access Tokens"**
3. Click **"Create New Token"**
4. Name it: `equity-research-airflow`
5. Copy the token (starts with `md_...` or similar)
   ```
   md_3x4mpl3t0k3n1234567890abcdef
   ```

#### 3. Test Connection (Optional)
```bash
# Install DuckDB CLI
brew install duckdb

# Test Motherduck connection
duckdb

D SELECT 1;
# Should return: 1

D .exit
```

#### 4. Set Environment Variable
```bash
MOTHERDUCK_TOKEN=md_your_token_here
DUCKDB_CONNECTION=md:?motherduck_token=${MOTHERDUCK_TOKEN}
```

✅ **Motherduck Setup Complete!**

---

## 🕸️ Database 3: Neo4j AuraDB (Graph Database)

### Why Neo4j AuraDB?
- Official Neo4j cloud (fully managed)
- Free tier: 200k nodes + 400k relationships
- Cypher query language
- Graph Data Science (GDS) library included

### Step-by-Step Setup

#### 1. Sign Up
```bash
open https://neo4j.com/cloud/aura-free/
```
- Click **"Start Free"**
- Create account with email
- Verify email

#### 2. Create Free Instance
1. After login, click **"New Instance"**
2. Choose **"AuraDB Free"**
3. Settings:
   - **Instance name**: `equity-research`
   - **Region**: Choose closest to Hong Kong (e.g., `GCP Asia Southeast 1 (Singapore)`)
4. Click **"Create"**

#### 3. Save Credentials
**IMPORTANT**: You'll see a dialog with credentials **only once**!

```bash
# Save these immediately:
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=A1b2C3d4E5f6G7h8  # Auto-generated
```

⚠️ **Download the credentials file** or copy to a safe place!

#### 4. Wait for Instance to Start
- Status will show **"Running"** (takes ~1-2 minutes)
- You'll see a green checkmark when ready

#### 5. Test Connection (Optional)
1. Click **"Open"** next to your instance
2. Or go to **"Query"** tab
3. Run test query:
   ```cypher
   CREATE (test:TestNode {name: 'Hello World'})
   RETURN test
   ```
4. Should see a node created
5. Delete test node:
   ```cypher
   MATCH (test:TestNode) DELETE test
   ```

#### 6. Enable APOC (Optional but Recommended)
1. Go to your instance settings
2. Find **"Plugins"** or **"Extensions"**
3. Enable **APOC** (Awesome Procedures on Cypher)
4. Enable **Graph Data Science (GDS)**

✅ **Neo4j AuraDB Setup Complete!**

---

## 🎯 Database 4: Qdrant Cloud (Vector Database)

### Why Qdrant?
- Best vector search performance
- 1 GB RAM cluster free
- Easy Python SDK
- Fast similarity search

### Step-by-Step Setup

#### 1. Sign Up
```bash
open https://cloud.qdrant.io
```
- Click **"Get Started"** or **"Sign Up"**
- Use GitHub or email
- Free tier: 1 GB cluster

#### 2. Create Cluster
1. After login, click **"Create Cluster"**
2. Settings:
   - **Cluster name**: `equity-research`
   - **Cloud provider**: AWS or GCP
   - **Region**: `ap-southeast-1` (Singapore) closest to HK
   - **Plan**: **Free** (1 GB)
3. Click **"Create"**

#### 3. Wait for Provisioning
- Takes 2-3 minutes
- Status will change to **"Running"**

#### 4. Get Connection Details
1. Click on your cluster name
2. You'll see:
   ```bash
   Cluster URL: https://xxxxx-xxxxx.cloud.qdrant.io
   Port: 6333 (HTTPS) / 6334 (gRPC)
   ```

#### 5. Create API Key
1. Go to **"API Keys"** tab
2. Click **"Create API Key"**
3. Name: `airflow-dags`
4. Copy the key:
   ```
   qdrant_api_key_1234567890abcdef...
   ```
   ⚠️ **Save immediately** - won't be shown again!

#### 6. Set Environment Variables
```bash
QDRANT_URL=https://xxxxx-xxxxx.cloud.qdrant.io
QDRANT_HOST=xxxxx-xxxxx.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=qdrant_api_key_1234567890abcdef...
```

#### 7. Test Connection (Optional)
```python
# In Python or terminal:
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://xxxxx.cloud.qdrant.io",
    api_key="your_api_key"
)

# Test
print(client.get_collections())
# Should return: CollectionsResponse(collections=[])
```

✅ **Qdrant Cloud Setup Complete!**

---

## 🔴 Database 5: Upstash Redis (Optional)

### Why Upstash?
- Serverless Redis (pay per request)
- 10k commands/day free
- Global replication
- Good for caching API responses

### Step-by-Step Setup

#### 1. Sign Up
```bash
open https://upstash.com
```
- Click **"Get Started"**
- Use GitHub or email
- Free tier: 10k requests/day

#### 2. Create Database
1. Click **"Create Database"**
2. Settings:
   - **Name**: `equity-research-cache`
   - **Type**: **Redis**
   - **Region**: Choose closest to Hong Kong
   - **Eviction**: LRU (Least Recently Used)
3. Click **"Create"**

#### 3. Get Connection Details
1. Click on your database
2. Go to **"Details"** tab
3. Copy connection details:
   ```bash
   REDIS_URL=redis://default:password@us1-xxx-xxx.upstash.io:6379
   
   # Or separately:
   REDIS_HOST=us1-xxx-xxx.upstash.io
   REDIS_PORT=6379
   REDIS_PASSWORD=your_password_here
   ```

#### 4. Test Connection (Optional)
```bash
# Install redis-cli
brew install redis

# Test
redis-cli -h us1-xxx-xxx.upstash.io -p 6379 -a your_password

# In redis-cli:
127.0.0.1:6379> PING
PONG

127.0.0.1:6379> SET test "Hello Redis"
OK

127.0.0.1:6379> GET test
"Hello Redis"

127.0.0.1:6379> quit
```

✅ **Upstash Redis Setup Complete!** (Optional)

---

## 📝 Update .env File

Now collect all your credentials and update the `.env` file:

```bash
cd /Users/brianho/Agent-skills-POC

# Edit .env
code .env  # or: nano .env
```

Paste your credentials:

```bash
# ============================================================================
# API KEYS
# ============================================================================
EODHD_API_KEY=your_eodhd_key
FMP_API_KEY=your_fmp_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# ============================================================================
# CLOUD DATABASES
# ============================================================================

# PostgreSQL (Neon)
POSTGRES_HOST=ep-cool-sun-123456.us-west-2.aws.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=equity_research
POSTGRES_USER=alex
POSTGRES_PASSWORD=AbC123xyz

# DuckDB (Motherduck)
MOTHERDUCK_TOKEN=md_3x4mpl3t0k3n1234567890abcdef
DUCKDB_CONNECTION=md:?motherduck_token=${MOTHERDUCK_TOKEN}

# Neo4j (AuraDB)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=A1b2C3d4E5f6G7h8

# Qdrant (Cloud)
QDRANT_URL=https://xxxxx-xxxxx.cloud.qdrant.io
QDRANT_HOST=xxxxx-xxxxx.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=qdrant_api_key_1234567890abcdef

# Redis (Upstash) - OPTIONAL
REDIS_URL=redis://default:password@us1-xxx-xxx.upstash.io:6379
REDIS_HOST=us1-xxx-xxx.upstash.io
REDIS_PORT=6379
REDIS_PASSWORD=your_password_here

# ============================================================================
# NOTIFICATIONS
# ============================================================================
ALERT_EMAIL=your_email@example.com
```

---

## 🧪 Test All Connections

Create a test script:

```bash
cd /Users/brianho/Agent-skills-POC
cat > test_connections.py << 'EOF'
#!/usr/bin/env python3
"""
Test all cloud database connections
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Cloud Database Connections...\n")
print("="*70)

# Test 1: PostgreSQL (Neon)
print("\n1. Testing PostgreSQL (Neon)...")
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )
    cur = conn.cursor()
    cur.execute('SELECT version()')
    version = cur.fetchone()[0]
    print(f"   ✅ Connected! Version: {version[:50]}...")
    cur.close()
    conn.close()
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: DuckDB (Motherduck)
print("\n2. Testing DuckDB (Motherduck)...")
try:
    import duckdb
    conn = duckdb.connect(os.getenv('DUCKDB_CONNECTION'))
    result = conn.execute('SELECT 1 as test').fetchone()
    print(f"   ✅ Connected! Test query result: {result[0]}")
    conn.close()
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Neo4j (AuraDB)
print("\n3. Testing Neo4j (AuraDB)...")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        test_value = result.single()['test']
        print(f"   ✅ Connected! Test query result: {test_value}")
    driver.close()
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Qdrant
print("\n4. Testing Qdrant Cloud...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    collections = client.get_collections()
    print(f"   ✅ Connected! Collections: {len(collections.collections)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 5: Redis (Optional)
if os.getenv('REDIS_HOST'):
    print("\n5. Testing Redis (Upstash)...")
    try:
        import redis
        r = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        r.ping()
        print(f"   ✅ Connected! Ping successful")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "="*70)
print("\nConnection tests complete!\n")
EOF

chmod +x test_connections.py
```

Run the test:

```bash
# Install required packages
pip install psycopg2-binary duckdb neo4j qdrant-client redis python-dotenv

# Run test
python test_connections.py
```

You should see:
```
✅ Connected! for all 5 databases
```

---

## 🐳 Update Docker Compose (Use Cloud DBs)

Since you're using cloud databases, you can simplify `docker-compose.yml` to run **only Airflow**:

```bash
cd /Users/brianho/Agent-skills-POC

# Backup original
cp docker-compose.yml docker-compose.yml.backup

# Create simplified version
cat > docker-compose-cloud.yml << 'EOF'
version: '3.8'

# Using cloud databases - only run Airflow services

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.8.1-python3.11
  env_file:
    - .env
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
  volumes:
    - ./scripts/dags:/opt/airflow/dags
    - .//opt/airflow/data
    - ./storage:/opt/airflow/storage
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
  user: "${AIRFLOW_UID:-50000}:0"

services:
  airflow-webserver:
    <<: *airflow-common
    container_name: airflow_webserver
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped

  airflow-scheduler:
    <<: *airflow-common
    container_name: airflow_scheduler
    command: scheduler
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8974/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped
EOF
```

---

## 🚀 Start Airflow

Now start Airflow:

```bash
cd /Users/brianho/Agent-skills-POC

# Initialize Airflow (first time only)
docker-compose -f docker-compose-cloud.yml run --rm airflow-webserver airflow db init

# Create admin user
docker-compose -f docker-compose-cloud.yml run --rm airflow-webserver \
  airflow users create \
  --username airflow \
  --password airflow \
  --firstname Air \
  --lastname Flow \
  --role Admin \
  --email admin@example.com

# Start services
docker-compose -f docker-compose-cloud.yml up -d

# Check logs
docker-compose -f docker-compose-cloud.yml logs -f
```

Wait 1-2 minutes for Airflow to start.

---

## 🌐 Access Airflow UI

```bash
open http://localhost:8080
```

**Login:**
- Username: `airflow`
- Password: `airflow`

You should see all 13 DAGs listed!

---

## ✅ Verify DAGs

1. **Check DAG List**
   - You should see 13 DAGs:
     - dag_01_daily_market_data_pipeline
     - dag_02_daily_news_sentiment_pipeline
     - ... (all 13)

2. **Enable DAGs**
   - Toggle each DAG **ON** (switch on left)
   - Start with DAG 1 (market data)

3. **Trigger Test Run**
   - Click on **dag_01_daily_market_data_pipeline**
   - Click **"Trigger DAG"** (play button, top right)
   - Go to **"Graph"** view
   - Watch tasks turn green (success)

4. **Check Logs**
   - Click any task square
   - Click **"Log"**
   - Should see execution details

---

## 🎉 Success Checklist

- [ ] Neon PostgreSQL: Connected ✅
- [ ] Motherduck DuckDB: Connected ✅
- [ ] Neo4j AuraDB: Connected ✅
- [ ] Qdrant Cloud: Connected ✅
- [ ] Redis Upstash: Connected ✅ (optional)
- [ ] `.env` file updated ✅
- [ ] Connection test passed ✅
- [ ] Airflow running ✅
- [ ] All 13 DAGs visible ✅
- [ ] DAG 1 test run successful ✅

---

## 🆘 Troubleshooting

### Connection Errors

**"Connection refused"**
- Check firewall/network settings
- Verify credentials in `.env`
- Check if service is running in cloud console

**"Authentication failed"**
- Double-check username/password
- For Neon: Ensure `?sslmode=require` in connection string
- For Neo4j: Use `neo4j+s://` (with SSL)

**"Host not found"**
- Verify host URL in `.env`
- Check region selection (should be Asia/Singapore)

### Airflow Issues

**"DAGs not appearing"**
```bash
# Check for import errors
docker exec airflow_scheduler airflow dags list-import-errors

# Restart scheduler
docker-compose -f docker-compose-cloud.yml restart airflow-scheduler
```

**"Task failed immediately"**
- Check task logs in Airflow UI
- Verify API keys in `.env`
- Check database connections

---

## 📊 Free Tier Limits

| Service | Storage | Queries | Cost After Free |
|---------|---------|---------|------------------|
| Neon | 0.5 GB | Unlimited | $19/mo (3 GB) |
| Motherduck | 10 GB | Unlimited | $25/mo (100 GB) |
| Neo4j AuraDB | 200k nodes | Unlimited | $65/mo (1M nodes) |
| Qdrant | 1 GB RAM | Unlimited | $25/mo (2 GB) |
| Upstash | 256 MB | 10k/day | $0.2 per 100k |

**Your setup should stay free for ~50-100 tickers with 5 years of data.**

---

## 🎓 Next Steps

1. ✅ **Run DAG 1** to ingest first batch of prices
2. ✅ **Run DAG 2** to fetch news
3. ✅ **Monitor DAG 11** for data quality
4. ✅ **Check databases** in cloud consoles to see data populated
5. 🔄 **Let DAGs run automatically** on their schedules

---

## 💡 Pro Tips

1. **Monitor Usage**
   - Check each cloud dashboard weekly
   - Set up usage alerts if available

2. **Backup Credentials**
   - Save `.env` to password manager
   - Never commit `.env` to Git

3. **Scale Gradually**
   - Start with 10-20 tickers
   - Add more as system stabilizes

4. **Use Branches**
   - Neon supports database branching
   - Create `dev` branch for testing

---

## 🎉 You're Done!

Your cloud databases are ready. Your Airflow DAGs will now:
- Fetch data daily/weekly/monthly
- Store in cloud databases
- Monitor quality automatically
- Alert you on failures

**Welcome to production! 🚀**
