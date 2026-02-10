# Testing GraphRAG v28.0

This directory contains testing and diagnostic scripts for Ultimate GraphRAG v28.0.

## Quick Start

### 1. Run Quick Test (Recommended First)

```bash
cd skills/business_analyst_graphrag
python quick_test.py
```

This will:
- Check if data exists
- Verify vector store is populated
- Run a simple test query
- Guide you through any issues

### 2. Run Diagnostics (If Issues)

```bash
python diagnose_graphrag.py
```

This will check:
- Data files in ./data/AAPL/
- Vector store chunk count
- Neo4j graph statistics
- Configuration settings
- Provide specific recommendations

### 3. Run Full Test Suite

```bash
python test_graphrag_v28.py
```

This runs 5 comprehensive tests:
1. Simple query (single-entity)
2. Medium complexity (section-target)
3. Abstract query (HyDE strategy)
4. Multi-hop query (graph reasoning)
5. Complex query (decomposition)

## Common Issues & Solutions

### Issue 1: Low Confidence Scores (< 0.75)

**Symptoms:**
```
📊 Multi-factor confidence: 0.60
   Semantic: 0.18 | Authority: 1.00
```

**Solutions:**
```python
# Lower threshold temporarily
agent.confidence_threshold = 0.55

# Or adjust multi-factor weights in code
# Edit graph_agent_graphrag_v28.py _score_confidence_multifactor()
```

### Issue 2: No Entities Validated

**Symptoms:**
```
✅ Validated: 0/9 entities (threshold=0.6)
```

**Solutions:**
```python
# Lower entity threshold
agent.entity_confidence_threshold = 0.3

# Re-run ingestion
agent.ingest_data()
```

### Issue 3: Empty Vector Store

**Symptoms:**
```
Total chunks: 0
```

**Solutions:**
```bash
# Check data exists
ls -la ../../data/AAPL/

# Run ingestion
python -c "from graph_agent_graphrag_v28 import UltimateGraphRAGBusinessAnalyst; agent = UltimateGraphRAGBusinessAnalyst(); agent.ingest_data()"
```

### Issue 4: Neo4j Connection Failed

**Symptoms:**
```
❌ Error connecting to Neo4j
```

**Solutions:**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Start Neo4j if needed
docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Test connection
curl http://localhost:7474
```

## Test Scripts Overview

### quick_test.py
- **Purpose:** Fast end-to-end validation
- **Duration:** 30-60 seconds
- **Use when:** First time setup or quick verification

### diagnose_graphrag.py
- **Purpose:** Comprehensive system check
- **Duration:** 10-20 seconds
- **Use when:** Troubleshooting issues

### test_graphrag_v28.py
- **Purpose:** Full feature testing
- **Duration:** 3-5 minutes
- **Use when:** Validating all v28.0 features

## Manual Testing

### From Python REPL

```python
# From project root (Agent-skills-POC/)
python

>>> from skills.business_analyst_graphrag.graph_agent_graphrag_v28 import UltimateGraphRAGBusinessAnalyst
>>> agent = UltimateGraphRAGBusinessAnalyst(
...     data_path="./data",
...     db_path="./storage/chroma_db"
... )

# Lower thresholds if needed
>>> agent.confidence_threshold = 0.55
>>> agent.entity_confidence_threshold = 0.3

# Run test query
>>> result = agent.analyze("What is Apple's revenue?")
>>> print(result[:300])
```

### Check Vector Store

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./storage/chroma_db/AAPL",
    embedding_function=embeddings,
    collection_name="AAPL"
)

print(f"Chunks: {vectorstore._collection.count()}")
results = vectorstore.similarity_search("revenue", k=3)
for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
```

### Check Neo4j Graph

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    nodes = session.run("MATCH (n) RETURN count(n)").single()[0]
    rels = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
    print(f"Nodes: {nodes}, Relationships: {rels}")

driver.close()
```

## Performance Benchmarks

Expected performance on test queries:

| Query Type | Expected Time | Expected Accuracy |
|------------|---------------|-------------------|
| Simple | 12-14s | 98%+ |
| Medium | 18-22s | 97%+ |
| Abstract (HyDE) | 20-25s | 95%+ |
| Multi-hop | 25-35s | 97%+ |
| Complex | 30-40s | 99%+ |

## Adjusting Configuration

### In Code

```python
# In graph_agent_graphrag_v28.py

class UltimateGraphRAGBusinessAnalyst:
    def __init__(self, ...):
        # Adjust these values
        self.confidence_threshold = 0.75  # Lower to 0.55-0.65 if needed
        self.entity_confidence_threshold = 0.6  # Lower to 0.3-0.4 if needed
        self.max_correction_attempts = 3  # Reduce to 2 for faster queries
```

### At Runtime

```python
agent = UltimateGraphRAGBusinessAnalyst()
agent.confidence_threshold = 0.55
agent.entity_confidence_threshold = 0.3
agent.max_correction_attempts = 2
```

## Getting Help

If tests fail:

1. Run `python diagnose_graphrag.py` first
2. Check the recommendations section
3. Review logs for specific error messages
4. Try lowering thresholds as suggested
5. Ensure data files exist in ./data/AAPL/
6. Verify Neo4j is running: `docker ps | grep neo4j`

## Next Steps

After successful testing:

1. Integrate with your orchestrator
2. Add your own test queries
3. Adjust thresholds based on your data
4. Monitor performance metrics
5. Customize query strategies if needed
