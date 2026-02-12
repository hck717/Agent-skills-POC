# Cloud Database Recommendations for Multi-Agent Equity Research System

## Overview
Best free/low-cost cloud database services for production deployment.

---

## 1. PostgreSQL + TimescaleDB

### Recommended: **Neon** (Best Free Tier)
- **URL**: https://neon.tech
- **Free Tier**:
  - 0.5 GB storage
  - 1 compute unit
  - Unlimited projects
  - Serverless Postgres with autoscaling
- **Pros**:
  - True serverless (pay only when active)
  - Built-in branching (like Git for databases)
  - PostgreSQL 16 compatible
  - Easy TimescaleDB extension installation
- **Connection String Format**:
  ```
  postgresql://username:password@ep-xxx-xxx.us-east-2.aws.neon.tech/equity_research?sslmode=require
  ```

### Alternative: **Supabase** (Generous Free Tier)
- **URL**: https://supabase.com
- **Free Tier**:
  - 500 MB database space
  - 1 GB file storage
  - 2 GB bandwidth/month
  - Up to 50 MB file uploads
- **Pros**:
  - Built on PostgreSQL
  - Real-time subscriptions
  - Built-in authentication
  - RESTful API auto-generated
- **Connection String Format**:
  ```
  postgresql://postgres:password@db.xxx.supabase.co:5432/postgres
  ```

### Alternative: **ElephantSQL** (Simple & Reliable)
- **URL**: https://www.elephantsql.com
- **Free Tier**:
  - 20 MB storage
  - Shared CPU
  - Good for testing
- **Pros**:
  - Very simple setup
  - Direct PostgreSQL access
  - Multiple data centers

---

## 2. DuckDB (Embedded/File-Based)

### Recommended: **Motherduck** (Cloud-Hosted DuckDB)
- **URL**: https://motherduck.com
- **Free Tier**:
  - 10 GB storage
  - Unlimited queries
  - Collaborative analytics
- **Pros**:
  - True cloud DuckDB (not just file storage)
  - S3/Parquet integration
  - Shares DuckDB API compatibility
  - Fast analytical queries
- **Connection String Format**:
  ```python
  import duckdb
  conn = duckdb.connect('md:?motherduck_token=YOUR_TOKEN')
  ```

### Alternative: **Cloudflare R2 + Local DuckDB**
- **URL**: https://www.cloudflare.com/products/r2/
- **Free Tier**:
  - 10 GB storage/month
  - 1 million Class A operations
  - S3-compatible API
- **Setup**:
  - Store DuckDB files (.duckdb) on R2
  - Access via S3 protocol
  - Download locally for queries
- **Cost**: Essentially free for < 10 GB

### Alternative: **AWS S3 + DuckDB** (Pay-as-you-go)
- Store Parquet files on S3 Free Tier (5 GB)
- Query directly with DuckDB's S3 extension
- Example:
  ```python
  import duckdb
  conn = duckdb.connect()
  conn.execute("SELECT * FROM 's3://bucket/prices/*.parquet'")
  ```

---

## 3. Neo4j (Graph Database)

### Recommended: **Neo4j AuraDB Free** (Official Cloud)
- **URL**: https://neo4j.com/cloud/aura-free/
- **Free Tier**:
  - 200k nodes + 400k relationships
  - 50 MB storage
  - Always-on instance
  - Neo4j 5.x
- **Pros**:
  - Official Neo4j cloud service
  - Fully managed
  - Automatic backups
  - Cypher query interface
- **Connection String Format**:
  ```
  neo4j+s://xxx.databases.neo4j.io
  Username: neo4j
  Password: <generated>
  ```

### Alternative: **Railway** (Docker-based)
- **URL**: https://railway.app
- **Free Tier**:
  - $5 credit/month (enough for Neo4j)
  - Deploy from Docker image
  - Persistent storage
- **Setup**:
  - Deploy `neo4j:5.15-community` image
  - Enable APOC and GDS plugins

---

## 4. Qdrant (Vector Database)

### Recommended: **Qdrant Cloud Free Tier**
- **URL**: https://cloud.qdrant.io
- **Free Tier**:
  - 1 GB RAM cluster
  - 1M vectors (depending on dimensions)
  - Always-on
- **Pros**:
  - Official Qdrant cloud
  - High-performance vector search
  - Built-in clustering
- **Connection Details**:
  ```python
  from qdrant_client import QdrantClient
  client = QdrantClient(
      url="https://xxx.cloud.qdrant.io",
      api_key="YOUR_API_KEY"
  )
  ```

### Alternative: **Zilliz Cloud** (Milvus-based)
- **URL**: https://zilliz.com/cloud
- **Free Tier**:
  - 1 CU (compute unit)
  - 1M vectors
  - 90-day trial
- **Pros**:
  - Milvus compatible
  - GPU acceleration
  - Good for large-scale embeddings

### Alternative: **Pinecone** (Popular Choice)
- **URL**: https://www.pinecone.io
- **Free Tier**:
  - 1 pod (1 GB storage)
  - 100k vectors with 768 dimensions
  - 5M queries/month
- **Pros**:
  - Very easy to use
  - Great Python SDK
  - Built-in filtering

---

## 5. Redis (Caching)

### Recommended: **Upstash Redis** (Serverless)
- **URL**: https://upstash.com
- **Free Tier**:
  - 10k commands/day
  - 256 MB storage
  - Serverless (pay per request)
- **Pros**:
  - True serverless
  - Global replication
  - Redis 7 compatible
  - REST API + native protocol
- **Connection String Format**:
  ```
  redis://default:password@us1-xxx.upstash.io:6379
  ```

### Alternative: **Redis Cloud** (Official)
- **URL**: https://redis.com/try-free/
- **Free Tier**:
  - 30 MB storage
  - Shared cluster
  - Up to 30 connections
- **Pros**:
  - Official Redis cloud
  - RedisJSON, RedisSearch modules

---

## Recommended Configuration

### For Your System (Optimal Free Setup):

```bash
# PostgreSQL/TimescaleDB
POSTGRES_HOST=ep-xxx.neon.tech  # Neon
POSTGRES_PORT=5432
POSTGRES_DB=equity_research
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# DuckDB (Motherduck)
MOTHERDUCK_TOKEN=your_token
DUCKDB_CONNECTION=md:?motherduck_token=${MOTHERDUCK_TOKEN}

# Neo4j (AuraDB Free)
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=generated_password

# Qdrant (Cloud)
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=your_api_key

# Redis (Upstash)
REDIS_URL=redis://default:password@us1-xxx.upstash.io:6379
```

---

## Cost Estimates (If Scaling Beyond Free Tier)

| Service | Free Tier Limit | Paid Tier Start | Best For |
|---------|-----------------|-----------------|----------|
| Neon | 0.5 GB | $19/mo (3 GB) | Small projects |
| Motherduck | 10 GB | $25/mo (100 GB) | Analytics workloads |
| Neo4j Aura | 200k nodes | $65/mo (1M nodes) | Medium graphs |
| Qdrant Cloud | 1 GB RAM | $25/mo (2 GB) | < 5M vectors |
| Upstash Redis | 10k cmds/day | $0.2 per 100k | Caching |

**Total Free Tier Capacity**:
- PostgreSQL: 500 MB (Supabase) to 3 GB (Neon trial)
- DuckDB: 10 GB (Motherduck)
- Neo4j: 200k nodes
- Qdrant: ~500k-1M vectors (depending on dimensions)
- Redis: 10k commands/day

**Sufficient for**: ~50-100 tickers with 5 years of historical data.

---

## Setup Priority

1. **Start with Neon** (PostgreSQL) - Most critical for structured data
2. **Add Motherduck** (DuckDB) - For time-series analytics
3. **Add Neo4j AuraDB** - For knowledge graphs
4. **Add Qdrant Cloud** - For RAG embeddings
5. **Add Upstash Redis** - For caching (optional, can skip initially)

---

## Migration Notes

- All cloud services support standard connection strings
- Update `config.py` with cloud URLs instead of `localhost`
- Enable SSL (`sslmode=require` for Postgres, `neo4j+s://` for Neo4j)
- Store credentials in `.env` file (never commit!)
- Use `docker-compose.yml` for local development only
