# Skill Card: Business Analyst (CRAG)

## Name
Business Analyst (CRAG Deep Reader)

## Goal
Produce institutional-style business analysis grounded in system-authenticated context (10-K / annual report text + Neo4j graph facts) with corrective retrieval when context quality is weak.

## When to use
- Business model / operating model analysis
- Strategy + opportunities + risks
- Segment and revenue driver mapping
- Management discussion / risk factor synthesis

## Inputs
- `query: str`
- `ticker: str` (recommended)
- `prior_analysis: str` (optional)

## Outputs
Markdown sections (Operating model, Revenue drivers, Opportunities, Risks, Trade-offs) with **one** `--- SOURCE: ... ---` tag after every paragraph/bullet.

## Source policy
- Local Neo4j facts and local documents are “System Authenticated Source”.
- Web is only used when configured and only when CRAG evaluator flags context as INCORRECT.

## Dependencies
- `neo4j` python driver
- Optional: `qdrant_client`
- `ollama`

## Failure modes
- Empty graph/doc context → INCORRECT → web fallback (if available) else explicit insufficiency message.
- Poor ingestion → shallow entities → generic-looking analysis (still properly sourced).
