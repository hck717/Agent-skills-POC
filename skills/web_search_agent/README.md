# 🌐 Web Search Agent (The “News Desk”)

Goal: Find **unknown unknowns** fast—market-moving news, regulatory surprises, competitor actions, and analyst narrative shifts that are not in filings/graphs.

## Architecture

This agent implements:

- Step-Back Prompting (Query transformation)
- HyDE-style expansion (Hypothetical document → intent-rich query)
- Corrective filtering with re-ranking (optional Cohere Rerank; fallback to local LLM rerank)

### 1) Query Transformation (Step-Back)

User query: “Why is AAPL down today?”

Step-back: “What are the major recent news events affecting large-cap tech stocks or Apple specifically?”

This broadens context to catch macro/sector catalysts, not just ticker keywords.

### 2) Expansion (HyDE)

The LLM generates a **hypothetical** (fake) news brief describing what a good answer might look like.

We then extract 1–2 intent-rich search queries from that brief (company, catalysts, timeframe, event type), and run real web searches using Tavily.

Benefit: Finds articles that match **intent**, not just exact keywords.

### 3) Filtering (Corrective)

The agent merges + deduplicates results, then reranks to reduce noise:

- Preferred: Cohere Rerank (if `COHERE_API_KEY` is set)
- Fallback: Local LLM reranker (fast scoring prompt)

It then synthesizes a short “news desk” style update with strict source markers.

## Setup

### Dependencies

```bash
pip install tavily-python ollama
```

### Environment variables

- `TAVILY_API_KEY` (required)
- `COHERE_API_KEY` (optional, reranking)
- `OLLAMA_BASE_URL` (optional, default `http://localhost:11434`)

## Output format (required)

The orchestrator expects this exact format so it can convert into numbered references:

```text
…analysis paragraph…
--- SOURCE: Title (https://example.com) ---
```

If the LLM forgets citations, the agent injects them automatically.
