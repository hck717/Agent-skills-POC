# Orchestrator Documentation

## Overview

The orchestrator coordinates multiple specialist agents to answer complex queries.

## Implementation

- **ReAct Framework:** `orchestrator/react.py` (Recommended)
- **Legacy Planner:** `orchestrator/legacy.py`

## Usage

```python
from orchestrator.react import ReActOrchestrator

orchestrator = ReActOrchestrator(max_iterations=5)
report = orchestrator.research("Your query")
```

## Architecture

See docs/REACT_FRAMEWORK.md for detailed architecture.
