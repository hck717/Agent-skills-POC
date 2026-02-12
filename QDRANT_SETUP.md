# Qdrant Vector Database Setup Guide

## 🚀 Quick Start (Local Development)

Your repository is now configured to use **local Qdrant** for development, which provides:
- ⚡ **Zero latency** (vs 300ms to Frankfurt)
- 💰 **Free** (no cloud costs)
- 🔒 **Private** (data stays on your machine)
- 🧪 **Perfect for testing** DAGs and agents

## 📋 Setup Instructions

### 1. Update Your `.env` File

```bash
# Edit your .env file
cd Agent-skills-POC

# Set Qdrant to local (comment out Frankfurt)
QDRANT_URL=http://qdrant:6333
# QDRANT_API_KEY=  # Not needed for local

# Comment out your Frankfurt cloud instance
# QDRANT_URL=https://30f03311-9184-4356-81e1-526d3185aa06.europe-west3-0.gcp.cloud.qdrant.io
# QDRANT_API_KEY=26e9f748-e202-4736-bcf1-4d108c30efda|y97SDFs7iET9Ss45VVm6jjVnZMA6iU8eJKe7Ji0c4Ytm8Q7w54AlBQ
```

### 2. Pull Latest Changes

```bash
git pull origin main
```

### 3. Start All Services

```bash
# Stop existing containers
docker-compose down

# Start everything (including new Qdrant service)
docker-compose up -d

# Watch the logs
docker-compose logs -f
```

### 4. Verify Qdrant is Running

```bash
# Check Qdrant health
curl http://localhost:6333/healthz
# Should return: {"title":"healthz","version":"..."}

# Check collections
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]}}

# Or check via Docker
docker ps | grep qdrant
# Should show: qdrant container running
```

### 5. Run Connection Test

Go to Airflow UI (http://localhost:8080) and run DAG `00_connection_test`.

You should see:
```
✅ PostgreSQL: success (Singapore ~40ms)
✅ Neo4j: success
✅ Qdrant: success (~1ms local) ⚡⚡⚡
✅ API Keys: valid
```

---

## 🌍 Switching Between Local and Cloud

### Local Qdrant (Development)

**Use when**: Testing DAGs, developing agents, experimenting

**`.env` config**:
```bash
QDRANT_URL=http://qdrant:6333
# QDRANT_API_KEY=  # Leave empty or remove
```

**Pros**:
- ⚡ Instant responses (0-1ms)
- 💰 Free
- 🔒 Private data
- 🧪 Easy reset/rebuild

**Cons**:
- 📦 Data only on your machine
- 🔄 Need to migrate when deploying

### Cloud Qdrant (Production)

**Use when**: Deploying to production, need shared access, team collaboration

**Option 1: Frankfurt (Free but slow from Asia)**
```bash
QDRANT_URL=https://your-cluster.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
```
- Latency: ~300ms from Hong Kong
- Cost: Free
- Best for: Testing cloud features

**Option 2: Singapore/Tokyo (Paid, fast from Asia)**
```bash
QDRANT_URL=https://your-cluster.asia-southeast1.gcp.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
```
- Latency: ~40-60ms from Hong Kong
- Cost: ~$25-50/month
- Best for: Production deployment

**After changing `.env`**:
```bash
docker-compose restart airflow-webserver airflow-scheduler
```

---

## 📊 Data Migration Guide

### Exporting from Cloud to Local

```python
# Python script to export collections
from qdrant_client import QdrantClient
import json

# Connect to cloud Qdrant
cloud_client = QdrantClient(
    url="https://your-cluster.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="your-api-key"
)

# Connect to local Qdrant
local_client = QdrantClient(url="http://localhost:6333")

# Get all collections
collections = cloud_client.get_collections().collections

for collection in collections:
    name = collection.name
    print(f"Migrating {name}...")
    
    # Get collection info
    info = cloud_client.get_collection(name)
    
    # Create collection locally
    local_client.recreate_collection(
        collection_name=name,
        vectors_config=info.config.params.vectors
    )
    
    # Copy all points (in batches)
    offset = None
    while True:
        records, offset = cloud_client.scroll(
            collection_name=name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        if not records:
            break
        
        local_client.upsert(
            collection_name=name,
            points=records
        )
        
        if offset is None:
            break
    
    print(f"✅ {name} migrated")
```

### Importing from Local to Cloud

Same script, just swap `cloud_client` and `local_client`.

---

## 🔧 Advanced Configuration

### Custom Qdrant Settings

Edit `docker-compose.yml` to customize Qdrant:

```yaml
qdrant:
  image: qdrant/qdrant:latest
  environment:
    # Increase gRPC message size for large vectors
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB: 128
    # Enable telemetry (optional)
    QDRANT__TELEMETRY_DISABLED: true
    # Set log level
    QDRANT__LOG_LEVEL: INFO
```

### Persistent Data Location

Your Qdrant data is stored in a Docker volume:
```bash
# View volume
docker volume ls | grep qdrant
# Should show: qdrant_data

# Inspect volume
docker volume inspect qdrant_data

# Backup volume
docker run --rm -v qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz -C /data .

# Restore volume
docker run --rm -v qdrant_data:/data -v $(pwd):/backup alpine tar xzf /backup/qdrant_backup.tar.gz -C /data
```

### Reset/Clear All Data

```bash
# Stop containers
docker-compose down

# Remove Qdrant volume
docker volume rm qdrant_data

# Restart (will create fresh volume)
docker-compose up -d
```

---

## 🐛 Troubleshooting

### Issue: Qdrant container won't start

```bash
# Check logs
docker-compose logs qdrant

# Common fix: Port already in use
sudo lsof -i :6333
# Kill the process or change port in docker-compose.yml
```

### Issue: Connection refused

```bash
# Verify Qdrant is healthy
docker-compose ps
# Look for: qdrant | Up (healthy)

# Check network
docker network inspect equity_network
# Qdrant should be listed

# Test from inside Airflow container
docker-compose exec airflow-webserver curl http://qdrant:6333/healthz
```

### Issue: "Collection not found" errors

```bash
# List existing collections
curl http://localhost:6333/collections

# Create missing collection manually
curl -X PUT http://localhost:6333/collections/business_analyst_10k_filings \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    }
  }'
```

### Issue: Slow performance

Local Qdrant should be <10ms. If slow:

```bash
# Check Docker resources
docker stats qdrant
# CPU should be low, memory depends on data size

# Increase Docker memory (Docker Desktop > Settings > Resources)
# Recommended: 4GB+ for Qdrant with large collections
```

---

## 📈 Performance Comparison

### Query Latency (from Hong Kong)

| Setup | Single Query | Batch (10 vectors) | Collection Scan |
|-------|--------------|---------------------|------------------|
| **Local Qdrant** | 1-5ms | 10-20ms | 100-500ms |
| **Singapore Cloud** | 40-60ms | 200-400ms | 2-5s |
| **Frankfurt Cloud** | 300-400ms | 2-3s | 15-30s |

### Startup Time Impact

| Setup | Connection Test | DAG 13 Full Run |
|-------|-----------------|------------------|
| **Local Qdrant** | ~3s | ~5-10min |
| **Frankfurt Cloud** | ~10s (timeout) | Would timeout |

---

## 🎯 Recommendations

### For Development (Now)
- ✅ Use **local Qdrant**
- ✅ Keep cloud Qdrant config in `.env` but commented out
- ✅ Test all DAGs locally first
- ✅ Export data before switching to cloud

### For Production (Later)
- If staying in Hong Kong: Deploy to **AWS/GCP Singapore**
- If moving to cloud: Use **Qdrant Singapore** (paid tier)
- If self-hosting: Run Qdrant on same server as Airflow

### For Team Collaboration
- Use **cloud Qdrant** for shared collections
- Use **local Qdrant** for individual testing
- Keep data migration scripts ready

---

## 🔗 Useful Links

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Cloud Regions](https://qdrant.tech/documentation/cloud/regions/)
- [Python Client Guide](https://qdrant.tech/documentation/frameworks/python-client/)
- [Collection Management](https://qdrant.tech/documentation/concepts/collections/)

---

## ✅ Summary

You're now set up with:
- ⚡ **Local Qdrant** for zero-latency development
- 🔄 **Easy switching** between local and cloud
- 📦 **Persistent storage** via Docker volumes
- 🚀 **Ready for 5 new agents** development

All connection tests should now pass! 🎉
