# Business Analyst (CRAG Deep Reader)

Goal: produce high-accuracy, citation-controlled business analysis (strategy, operating model, revenue drivers, opportunities, risks) from **system-authenticated sources** (Neo4j graph + local filings) with corrective retrieval when context is weak.

## What this skill does

`BusinessAnalystCRAG` is a graph-augmented corrective RAG (CRAG) specialist.

- Retrieval: Hybrid search across Neo4j (graph context) and Qdrant (vector store) when configured.
- Evaluation: A CRAG evaluator grades whether retrieved context is CORRECT / AMBIGUOUS / INCORRECT.
- Correction: If INCORRECT and a web tool is configured, it can fall back to web search; otherwise it will clearly state that web search is disabled.
- Generation: Produces a structured answer with strict `--- SOURCE: ... ---` markers after every paragraph/bullet.

## System-authenticated sources

This agent treats the following as **system-authenticated sources**:

- Neo4j local graph context (Docker) seeded from your local `./data/<TICKER>/...` files.
- Local SEC/annual-report documents stored under `./data/<TICKER>/`.

When the agent cites these, the orchestrator can label them as “System Authenticated Source”.

## Inputs / Outputs

### Inputs

- `query` (str): the user’s question.
- `ticker` (str, optional): defaults to `AAPL` if not supplied; the orchestrator should always pass the detected ticker.
- `prior_analysis` (str, optional): reserved for multi-agent flows.

### Output

Markdown with the following sections:

- `## Operating model (2026)`
- `## Revenue drivers`
- `## Opportunities (2026)`
- `## Risks (2026)`
- `## Trade-offs / contradictions`

After every paragraph or bullet, it appends exactly one source tag:

- `--- SOURCE: 2025_AnnualReport.docx ---`
- `--- SOURCE: Knowledge Graph ---`

## Setup

### Required (local)

- Neo4j running (Docker recommended).
- Seeded graph for the target ticker.

### Optional (cloud)

- Qdrant Cloud for embeddings (set `QDRANT_URL`, `QDRANT_API_KEY`).

### Environment variables

- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_USER` (default `neo4j`)
- `NEO4J_PASSWORD` (default `password`)
- `QDRANT_URL` (optional)
- `QDRANT_API_KEY` (optional)
- `DATA_PATH` (default `./data`)

## Quick usage

```python
from skills.business_analyst_crag import BusinessAnalystCRAG

agent = BusinessAnalystCRAG(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_key=os.getenv("QDRANT_API_KEY"),
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
    neo4j_pass=os.getenv("NEO4J_PASSWORD", "password"),
)

print(agent.analyze("Analyze Microsoft's cloud strategy", ticker="MSFT"))
```

## Limitations

- If your graph seeding extracts only a few entities, the analysis will be constrained to what is present; consider improving ingestion/chunking.
- Web fallback requires a configured web-search client (not enabled by default).
