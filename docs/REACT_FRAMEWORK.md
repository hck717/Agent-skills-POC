# ReAct Framework for Multi-Agent Orchestration

[View comprehensive guide in repo]

This document explains the ReAct (Reasoning + Acting) implementation for iterative agent orchestration.

## Quick Reference

### ReAct Loop
```
1. Think 🧠 → What should I do next?
2. Act ⚡ → Execute specialist agent
3. Observe 👁️ → Analyze results  
4. Repeat 🔁 → Until done or max iterations
```

### Usage
```python
from orchestrator.react import ReActOrchestrator

orchestrator = ReActOrchestrator(max_iterations=5)
report = orchestrator.research("Your query")
print(orchestrator.get_trace_summary())
```

See full implementation details in `orchestrator/react.py`

---

For complete documentation, refer to the main repo.
