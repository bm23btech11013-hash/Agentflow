"""
Evaluation criteria package.

All criteria accept ExecutionResult as the first argument to evaluate().
ExecutionResult is built by AgentEvaluator._execution_from_collector()
using the TrajectoryCollector wired in at graph compile time.

Example:
    ```python
    from agentflow.evaluation.criteria import (
        TrajectoryMatchCriterion,
        ResponseMatchCriterion,
        LLMJudgeCriterion,
        RubricBasedCriterion,
        HallucinationCriterion,
        SafetyCriterion,
        FactualAccuracyCriterion,
    )
    ```
"""

from .base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from .factual_accuracy import FactualAccuracyCriterion
from .hallucination import HallucinationCriterion
from .llm_judge import LLMJudgeCriterion
from .llm_utils import LLMCallerMixin
from .response import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    ResponseMatchCriterion,
    RougeMatchCriterion,
)
from .rubric import RubricBasedCriterion
from .safety import SafetyCriterion
from .simulation_goals import SimulationGoalsCriterion
from .trajectory import NodeOrderMatchCriterion, ToolNameMatchCriterion, TrajectoryMatchCriterion


__all__ = [
    # Base classes
    "BaseCriterion",
    "SyncCriterion",
    "CompositeCriterion",
    "WeightedCriterion",
    "LLMCallerMixin",
    # Trajectory
    "TrajectoryMatchCriterion",
    "NodeOrderMatchCriterion",
    "ToolNameMatchCriterion",
    # Response
    "ResponseMatchCriterion",
    "RougeMatchCriterion",
    "ExactMatchCriterion",
    "ContainsKeywordsCriterion",
    # LLM-based
    "SimulationGoalsCriterion",
    "LLMJudgeCriterion",
    "RubricBasedCriterion",
    "HallucinationCriterion",
    "SafetyCriterion",
    "FactualAccuracyCriterion",
]
