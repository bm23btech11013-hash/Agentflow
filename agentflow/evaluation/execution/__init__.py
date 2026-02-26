"""
Execution result container for agent evaluation.

ExecutionResult holds tool calls, trajectory and response
built by AgentEvaluator._execution_from_collector() after ainvoke.
"""

from .result import ExecutionResult, NodeResponseData


__all__ = [
    "ExecutionResult",
    "NodeResponseData",
]
