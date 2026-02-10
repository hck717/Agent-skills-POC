"""
Business Analyst Skill Package

Provides two RAG implementations:
1. Standard RAG (graph_agent.py) - Original hybrid search + reranking
2. Self-RAG (graph_agent_selfrag.py) - Enhanced with adaptive retrieval, document grading, and hallucination checking
"""

# Standard RAG Agent
try:
    from .graph_agent import BusinessAnalystGraphAgent
except ImportError:
    BusinessAnalystGraphAgent = None

# Self-RAG Enhanced Agent
try:
    from .graph_agent_selfrag import SelfRAGBusinessAnalyst
except ImportError:
    SelfRAGBusinessAnalyst = None

# Self-RAG Components
try:
    from .semantic_chunker import SemanticChunker
except ImportError:
    SemanticChunker = None

try:
    from .document_grader import DocumentGrader
except ImportError:
    DocumentGrader = None

try:
    from .hallucination_checker import HallucinationChecker
except ImportError:
    HallucinationChecker = None

try:
    from .adaptive_retrieval import AdaptiveRetrieval
except ImportError:
    AdaptiveRetrieval = None

__all__ = [
    'BusinessAnalystGraphAgent',
    'SelfRAGBusinessAnalyst',
    'SemanticChunker',
    'DocumentGrader',
    'HallucinationChecker',
    'AdaptiveRetrieval'
]

__version__ = '25.0.0'
__author__ = 'hck717'
