# ReAct Framework for Multi-Agent Orchestration

## Overview

This system implements the **ReAct (Reasoning + Acting)** paradigm for intelligent multi-agent orchestration. Unlike traditional one-shot planning, ReAct enables iterative, adaptive decision-making through a continuous loop of reasoning and action.

## What is ReAct?

ReAct is a framework that combines **reasoning** (chain-of-thought) with **actions** (tool/agent calls) in an interleaved manner. The orchestrator:

1. **Thinks** - Reasons about the current state and decides what to do next
2. **Acts** - Executes the decided action (call specialist agent, synthesize, etc.)
3. **Observes** - Analyzes the results of the action
4. **Repeats** - Goes back to step 1 with updated context

This continues until the task is complete or max iterations reached.

## Architecture Comparison

### Traditional Planner-Executor (Old)

```
User Query
    ↓
[Planner] ───────> ONE-TIME planning
    ↓
[Execute All Agents in Parallel]
    ↓
[Synthesize]
    ↓
Final Report
```

**Limitations:**
- Cannot adapt if first plan was suboptimal
- No feedback loop
- Over-orchestrates (may call unnecessary agents)
- Cannot self-correct

### ReAct Framework (New)

```
User Query
    ↓
╭────────────────────────────────╮
│  ITERATION LOOP (max 5)      │
│                              │
│  1. [Thought]                │
│     → Reason about state     │
│     → Decide next action    │
│                              │
│  2. [Action]                 │
│     → Call specialist agent │
│     → OR synthesize         │
│     → OR finish             │
│                              │
│  3. [Observation]            │
│     → Analyze results       │
│     → Update context        │
│                              │
│  4. Loop back to step 1      │
╰────────────────────────────────╯
    ↓
[Final Synthesis]
    ↓
Final Report + ReAct Trace
```

**Advantages:**
- Adaptive planning based on intermediate results
- Can call additional agents if initial info insufficient
- Can stop early if sufficient info gathered
- Self-correcting - can change strategy mid-execution
- Transparent reasoning trace

## Implementation Details

### Core Classes

#### 1. `ReActTrace`
Maintains complete history of the reasoning process:
```python
@dataclass
class ReActTrace:
    thoughts: List[Thought]        # All reasoning steps
    actions: List[Action]          # All actions taken
    observations: List[Observation] # All results observed
```

#### 2. `Action` Types
```python
class ActionType(Enum):
    CALL_SPECIALIST = "call_specialist"  # Call a specific agent
    SYNTHESIZE = "synthesize"            # Combine results
    FINISH = "finish"                    # Stop iteration, proceed to synthesis
    REFINE_QUERY = "refine_query"        # Adjust query (future use)
```

#### 3. `ReActOrchestrator`
Main orchestration engine with:
- `_reason()` - LLM-powered reasoning step
- `_execute_action()` - Execute decided action
- `_synthesize()` - Final synthesis with full context
- `research()` - Main ReAct loop

### ReAct Loop Flow

```python
for iteration in range(1, max_iterations + 1):
    # Step 1: Reason
    action = self._reason(user_query, iteration)
    
    # Step 2: Act
    observation = self._execute_action(action)
    
    # Step 3: Observe & Record
    self.trace.add_observation(observation)
    
    # Step 4: Check if done
    if action.action_type in [FINISH, SYNTHESIZE]:
        break
```

### Reasoning Prompt Structure

The reasoning prompt includes:
1. **User Query** - Original question
2. **Current Iteration** - Which iteration we're on (1-5)
3. **Available Agents** - Specialist agent specs with keywords
4. **Agent Availability** - Which are implemented vs planned
5. **Previous History** - All previous thoughts, actions, observations
6. **Action Options** - What the orchestrator can do next

Example reasoning:
```json
{
  "thought": "User asks about competitive risks AND margins. I need both qualitative (10-K risks) and quantitative (margin calculations). No previous actions yet.",
  "action": "call_specialist",
  "agent_name": "business_analyst",
  "task_description": "Extract competitive risk factors from Apple's 10-K filing",
  "reasoning": "Start with qualitative analysis, then call quantitative analyst next iteration"
}
```

## Example ReAct Execution

### Query: "What are Apple's competitive risks and profit margins?"

#### **Iteration 1**

**Thought:**
> "Query requires both qualitative risk assessment and quantitative margin analysis. Start with Business Analyst to extract risks from 10-K."

**Action:**
> `call_specialist` → `business_analyst`
> Task: "Extract competitive risk factors from Apple's 10-K filing"

**Observation:**
> Business Analyst returns 5 key competitive risks with citations (page numbers, sources)

---

#### **Iteration 2**

**Thought:**
> "Good, we have competitive risks. Now need profit margin data. Quantitative Analyst can calculate margins if we have financial data."

**Action:**
> `call_specialist` → `quantitative_analyst`
> Task: "Calculate Apple's net profit margin, operating margin, and gross margin for last 3 years"

**Observation:**
> Quantitative Analyst returns margin trends with year-over-year analysis

---

#### **Iteration 3**

**Thought:**
> "We now have both competitive risks (qualitative) and margin data (quantitative). This is sufficient to answer the user's query comprehensively."

**Action:**
> `finish`

**Observation:**
> "Orchestration complete - proceeding to synthesis"

---

#### **Final Synthesis**

Combines both specialist outputs into structured report:
- Executive Summary
- Competitive Risk Analysis (from Business Analyst)
- Financial Performance (from Quantitative Analyst)
- Integrated Insights
- Conclusion

**Result:** User gets comprehensive answer in 3 iterations instead of blindly calling all 6 agents.

## Key Features

### 1. Dynamic Agent Selection

**Problem:** Traditional planner commits to all agents upfront.

**ReAct Solution:** Decides iteratively based on what's been gathered.

```python
# Iteration 1: Call Business Analyst
# Observation: Got risks but no financial data

# Iteration 2: Realizes need Quantitative Analyst
# Observation: Got margins

# Iteration 3: Decides sufficient info → finish
```

### 2. Self-Correction

**Example:**
```python
# Iteration 1: Calls Industry Analyst for peer comparison
# Observation: "No specific company data available"

# Iteration 2: Reasoning detects failure
# Action: Switches to Business Analyst for company-specific analysis
```

### 3. Early Stopping

**Efficiency Gain:**
- Traditional: Always calls N agents (predetermined)
- ReAct: Stops when sufficient info gathered

**Example:**
```
Simple query: "What does Apple do?"

Iteration 1: Business Analyst provides business overview
Iteration 2: Reasoning says "sufficient" → finish

Saved: 4 unnecessary agent calls
```

### 4. Transparent Reasoning

**ReAct Trace** shows complete decision-making process:
```
=== Iteration 1 ===
Thought: Need financial data from 10-K
Action: call_specialist → business_analyst
Observation: Extracted revenue and profit data...

=== Iteration 2 ===
Thought: Have raw data, need calculations
Action: call_specialist → quantitative_analyst
Observation: Calculated margins and growth rates...

=== Iteration 3 ===
Thought: Sufficient information gathered
Action: finish
Observation: Orchestration complete
```

## Configuration

### Max Iterations

Control depth of reasoning:

```python
orchestrator = ReActOrchestrator(max_iterations=5)
```

**Recommendations:**
- Simple queries: 2-3 iterations
- Complex queries: 4-5 iterations
- Deep research: 6-8 iterations

**Trade-off:**
- Higher iterations = more adaptive but slower
- Lower iterations = faster but less flexible

### Temperature

Control reasoning creativity:

```python
self.client.chat(messages, temperature=0.1)  # Reasoning (deterministic)
self.client.chat(messages, temperature=0.3)  # Synthesis (creative)
```

## Usage

### Basic Usage

```python
from orchestrator_react import ReActOrchestrator

# Initialize
orchestrator = ReActOrchestrator(max_iterations=5)

# Register specialists
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
business_analyst = BusinessAnalystGraphAgent()
orchestrator.register_specialist("business_analyst", business_analyst)

# Research with ReAct
report = orchestrator.research(
    "Analyze Tesla's competitive position and growth trajectory"
)

print(report)

# View reasoning trace
print(orchestrator.get_trace_summary())
```

### Running the System

```bash
# Set API key
export PERPLEXITY_API_KEY="your-key"

# Run ReAct orchestrator
python main_orchestrated.py
```

### Commands

- **Normal query** - Ask any research question
- **`trace`** - Show ReAct trace from last query
- **`ingest`** - Process new documents
- **`quit`** - Exit

## Performance Comparison

| Metric | Traditional Planner | ReAct Framework |
|--------|--------------------|-----------------|
| **Planning** | One-shot | Iterative |
| **Adaptability** | None | High |
| **Efficiency** | Calls all planned agents | Stops early if sufficient |
| **Self-correction** | No | Yes |
| **Transparency** | Limited | Full trace |
| **Typical iterations** | 1 planning step | 2-4 reasoning steps |
| **Total time** | 40-60s (fixed) | 30-70s (variable) |

## Advantages for Equity Research

### 1. Query Complexity Adaptation

**Simple Query:** "What does Apple do?"
- ReAct: 1-2 iterations (Business Analyst only)
- Traditional: Calls 3+ agents unnecessarily

**Complex Query:** "Compare Apple and Microsoft across financials, competitive position, and ESG"
- ReAct: 4-5 iterations (strategically calling multiple agents)
- Traditional: May miss nuances or call wrong agents

### 2. Data Availability Handling

**Scenario:** Query about a company not in database

**ReAct:**
```
Iteration 1: Calls Business Analyst
Observation: "No documents found for Company X"
Iteration 2: Reasoning switches to Market Analyst (real-time data)
Observation: Gets basic company info from market APIs
Iteration 3: Finish with available data
```

**Traditional:** Fails after planned agents return nothing.

### 3. Incremental Refinement

**Query:** "Analyze Apple's risks"

**ReAct:**
```
Iteration 1: Business Analyst extracts general risks
Observation: Risks mention "supply chain" and "regulation"
Iteration 2: Reasoning decides to call Industry Analyst for sector-specific context
Observation: Gets semiconductor supply chain dynamics
Iteration 3: Calls Macro Analyst for geopolitical context
Observation: Gets China-US trade risk assessment
Iteration 4: Finish with comprehensive risk analysis
```

## Best Practices

### 1. Design Clear Agent Descriptions

The reasoning step depends on accurate agent specs in `SPECIALIST_AGENTS.md`:

```python
"business_analyst": {
    "keywords": ["10-K", "filing", "risk", "competitive"]  # Clear triggers
}
```

### 2. Monitor Iteration Count

Track average iterations per query type:
- If always hitting max_iterations → increase limit
- If always finishing in 1-2 → queries too simple or agents too powerful

### 3. Review ReAct Traces

Regularly inspect traces to identify:
- Poor reasoning decisions
- Missing agent capabilities
- Opportunities for optimization

### 4. Balance Iteration Depth

```python
# For production (speed priority)
orchestrator = ReActOrchestrator(max_iterations=3)

# For research (thoroughness priority)
orchestrator = ReActOrchestrator(max_iterations=8)
```

## Future Enhancements

- [ ] **Parallel agent execution** within iterations
- [ ] **Memory system** for multi-turn conversations
- [ ] **Cost tracking** per iteration
- [ ] **Confidence scoring** for early stopping
- [ ] **Agent performance feedback** loop
- [ ] **Human-in-the-loop** approval gates
- [ ] **Query refinement** action type
- [ ] **Automatic iteration tuning** based on query complexity

## References

- **ReAct Paper:** [Yao et al. 2023 - ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **LangGraph:** For stateful agent workflows
- **Agent Systems:** Tool-using LLMs and orchestration patterns

---

**Built for:** Transaction Banking & Equity Research  
**Framework:** ReAct + Multi-Agent Orchestration  
**Last Updated:** February 9, 2026
