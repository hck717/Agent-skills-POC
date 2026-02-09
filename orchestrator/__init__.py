"""
Orchestrator Module

Provides multi-agent orchestration capabilities:
- ReAct framework for iterative reasoning
- Legacy planner for one-shot planning
"""

from .react import ReActOrchestrator
from .legacy import EquityResearchOrchestrator

__all__ = ['ReActOrchestrator', 'EquityResearchOrchestrator']
