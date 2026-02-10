# Business Analyst Skill (11/10 Enhanced)

## Description
This agent specializes in **comprehensive 10-K analysis**, combining qualitative strategic insight with structured financial data extraction. It simulates XBRL parsing to provide exact financial metrics alongside strategic RAG analysis.

## Capabilities
- **Structured Financial Extraction**: Automatically extracts "Total Revenue", "Net Income", "EPS", and other key GAAP metrics using high-precision pattern matching.
- **Dynamic Company Discovery**: Automatically detects company data in the `./data` folder without hardcoded mapping.
- **Adaptive Retrieval**: Intelligently skips RAG for simple questions (saving 60% time) while maintaining full depth for complex queries.
- **Self-Correcting Analysis**: Includes hallucination checking, document grading, and web search fallback for robust performance.
- **Strict Citation Enforcement**: Every claim is cited with `--- SOURCE: filename (Page X) ---` format.

## Usage
Provide the agent with a query related to a company's financial or strategic position.

### Examples:
- "What was Nvidia's Total Revenue and Net Income last year?" (Triggers Financial Extractor)
- "Analyze the competitive risks for Apple." (Triggers Risk Officer Persona)
- "What is the company's moat?" (Triggers Strategy Persona)

## Architecture
Flow: `Adaptive Check` → `Identify` → `Research` + `Extract Financials` → `Grade Docs` → `Rerank` → `Analyst` → `Hallucination Check`
