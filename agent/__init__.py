"""
Self-Learning Coding Agent
Core modules for task understanding, retrieval, coding, evaluation, and memory.
"""

from .config import AgentConfig, get_config
from .llm import LLMClient
from .evaluator import Evaluator, EvaluationResult
from .task_understanding import TaskUnderstanding, TaskSpecification
from .retriever import ContextRetriever, RetrievedContext
from .coding_agent import CodingAgent, CodingAttemptResult, AttemptLog
from .memory import MemoryStore, MemoryEntry
from .orchestrator import Orchestrator, OrchestratorResult, OrchestratorEvent, StepStatus

__all__ = [
    "AgentConfig",
    "get_config",
    "LLMClient",
    "Evaluator",
    "EvaluationResult",
    "TaskUnderstanding",
    "TaskSpecification",
    "ContextRetriever",
    "RetrievedContext",
    "CodingAgent",
    "CodingAttemptResult",
    "AttemptLog",
    "MemoryStore",
    "MemoryEntry",
    "Orchestrator",
    "OrchestratorResult",
    "OrchestratorEvent",
    "StepStatus",
]
