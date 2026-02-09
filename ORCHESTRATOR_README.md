# Multi-Agent Orchestration System

## Overview

This orchestration system coordinates multiple specialist agents for comprehensive equity research analysis. It uses **Perplexity API** to power two critical meta-agents:

1. **Planner Agent** - Analyzes queries and determines which specialist agents to deploy
2. **Synthesis Agent** - Combines outputs from multiple specialists into a coherent final report

## Architecture

```
User Query
    ↓
[Planner Agent] → Selects specialist agents & assigns tasks
    ↓
[Specialist Agents Execute in Parallel]
  • Business Analyst
  • Quantitative Analyst  
  • Market Analyst
  • Industry Analyst
  • ESG Analyst
  • Macro Analyst
    ↓
[Synthesis Agent] → Combines insights into final report
    ↓
Final Research Report
```

## The 6 Specialist Agents

### 1. Business Analyst ✅ (Implemented)
- **Focus**: 10-K filings, financial statements, business models, competitive positioning
- **Capabilities**: Financial analysis, competitive intelligence, risk assessment
- **Implementation**: `skills/business_analyst/graph_agent.py`

### 2. Quantitative Analyst ⏳ (Planned)
- **Focus**: Financial ratios, valuation metrics, growth rates, statistical modeling
- **Capabilities**: DCF valuation, ratio analysis, trend forecasting, comparative metrics

### 3. Market Analyst ⏳ (Planned)
- **Focus**: Market sentiment, technical indicators, price movements, trading volumes
- **Capabilities**: Sentiment analysis, technical analysis, market trends

### 4. Industry Analyst ⏳ (Planned)
- **Focus**: Sector insights, industry trends, regulatory landscape, peer benchmarking
- **Capabilities**: Industry dynamics, regulatory analysis, peer comparison

### 5. ESG Analyst ⏳ (Planned)
- **Focus**: Environmental, social, governance factors, sustainability initiatives
- **Capabilities**: ESG scoring, sustainability assessment, governance evaluation

### 6. Macro Analyst ⏳ (Planned)
- **Focus**: Macroeconomic factors, interest rates, currency impacts, geopolitical risks
- **Capabilities**: Economic indicators, rate sensitivity, FX exposure analysis

## Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Set API Keys

```bash
# Perplexity API (required for Planner & Synthesis agents)
export PERPLEXITY_API_KEY="your-perplexity-api-key"

# EODHD API (for market data in specialist agents)
export EODHD_API_KEY="your-eodhd-api-key"
```

### 3. Ensure Ollama is Running (for specialist agents)

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Usage

### Basic Usage

```bash
python orchestrator.py
```

### Programmatic Usage

```python
from orchestrator import EquityResearchOrchestrator
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent

# Initialize orchestrator
orchestrator = EquityResearchOrchestrator()

# Register your specialist agents
business_analyst = BusinessAnalystGraphAgent(
    data_path="./data", 
    db_path="./storage/chroma_db"
)
orchestrator.register_specialist("business_analyst", business_analyst)

# Execute research
query = "What are the key competitive risks for Apple in 2024?"
report = orchestrator.research(query)

print(report)
```

## How It Works

### Phase 1: Planning

The Planner Agent analyzes your query and:
- Identifies which type of analysis is needed
- Selects 1-4 most relevant specialist agents
- Assigns specific tasks to each agent
- Prioritizes execution order

**Example Query**: "Compare Apple and Microsoft's profit margins and assess competitive risks"

**Planner Output**:
```json
{
  "reasoning": "This query requires financial metrics analysis and competitive intelligence. Business analyst can extract margin data from 10-Ks, while quantitative analyst can perform comparative calculations.",
  "selected_agents": [
    {
      "agent_name": "business_analyst",
      "task_description": "Extract net profit margin, operating margin, and gross margin for Apple and Microsoft from latest 10-K filings",
      "priority": 1
    },
    {
      "agent_name": "quantitative_analyst",
      "task_description": "Calculate margin trends over 3 years and perform comparative analysis between Apple and Microsoft",
      "priority": 2
    }
  ]
}
```

### Phase 2: Execution

Each selected specialist agent executes its assigned task:
- **Business Analyst** uses RAG to search 10-K documents
- **Quantitative Analyst** performs calculations and statistical analysis
- **Market Analyst** fetches real-time market data
- (Other specialists execute their domain-specific workflows)

### Phase 3: Synthesis

The Synthesis Agent:
- Receives outputs from all specialist agents
- Identifies common themes and contradictions
- Combines quantitative and qualitative insights
- Structures a comprehensive final report
- Maintains citations and data provenance

**Output Structure**:
```markdown
## Executive Summary
[High-level findings]

## Key Insights
[Synthesized analysis]

## Quantitative Highlights  
[Metrics and ratios]

## Risk Factors
[Identified risks]

## Conclusion
[Balanced assessment]
```

## Example Queries

```python
# Competitive Analysis
"How does Tesla's market share compare to traditional automakers?"

# Risk Assessment
"What are the top 5 regulatory risks facing Meta in 2024?"

# Financial Analysis  
"Calculate NVIDIA's revenue CAGR and compare to AMD"

# ESG Analysis
"Evaluate Apple's carbon neutrality commitments and progress"

# Market Sentiment
"What is the current analyst sentiment on Microsoft's AI strategy?"

# Macro Impact
"How would a Fed rate cut affect bank stocks' valuations?"
```

## Extending the System

### Adding a New Specialist Agent

1. **Define the agent in PlannerAgent.SPECIALIST_AGENTS**:

```python
"credit_analyst": {
    "description": "Analyzes credit quality, debt levels, and default risk",
    "capabilities": ["Debt analysis", "Credit rating", "Default probability"]
}
```

2. **Implement the specialist agent class**:

```python
class CreditAnalyst:
    def __init__(self):
        # Initialize models, data sources, etc.
        pass
    
    def analyze(self, query: str) -> str:
        # Implement analysis logic
        return "Credit analysis report..."
```

3. **Register with orchestrator**:

```python
credit_analyst = CreditAnalyst()
orchestrator.register_specialist("credit_analyst", credit_analyst)
```

## Configuration

### Perplexity Model Selection

You can customize the Perplexity model used:

```python
client = PerplexityClient(api_key="your-key")
# Use different model
response = client.chat(
    messages=[{"role": "user", "content": "..."}],
    model="llama-3.1-sonar-small-128k-online"  # Faster, cheaper
)
```

Available models:
- `llama-3.1-sonar-small-128k-online` - Fast, cost-effective
- `llama-3.1-sonar-large-128k-online` - More capable (default)
- `llama-3.1-sonar-huge-128k-online` - Maximum performance

## Advantages Over Single-Agent Systems

| Feature | Single Agent | Multi-Agent Orchestration |
|---------|-------------|---------------------------|
| **Specialization** | Generalist approach | Domain experts for each task |
| **Scalability** | Limited by single model context | Parallel execution across agents |
| **Accuracy** | Single perspective | Cross-validated insights |
| **Flexibility** | Fixed workflow | Dynamic agent selection |
| **Maintainability** | Monolithic system | Modular, independent agents |

## Performance Considerations

- **Planner Agent**: ~5-10 seconds (Perplexity API call)
- **Specialist Agents**: Varies by complexity
  - Business Analyst: ~15-30 seconds (local Ollama + RAG)
  - Market Analyst: ~3-5 seconds (API calls)
- **Synthesis Agent**: ~10-15 seconds (Perplexity API call)

**Total**: ~30-60 seconds for comprehensive multi-agent analysis

## Troubleshooting

### Issue: "PERPLEXITY_API_KEY not found"
**Solution**: Ensure you've set the environment variable:
```bash
export PERPLEXITY_API_KEY="your-api-key"
```

### Issue: Planner selects wrong agents
**Solution**: Provide more specific queries or refine agent descriptions in `SPECIALIST_AGENTS`

### Issue: Synthesis output is incomplete
**Solution**: Check that specialist agents return complete outputs with proper formatting

## Future Enhancements

- [ ] Implement remaining 5 specialist agents
- [ ] Add agent memory for multi-turn conversations
- [ ] Implement parallel agent execution
- [ ] Add confidence scoring for agent outputs
- [ ] Create agent skill marketplace
- [ ] Add human-in-the-loop approval gates
- [ ] Implement cost tracking and optimization

## License

MIT License - feel free to extend and customize for your equity research needs.
