"""
Agent Evaluation Module for Agentflow.

This module provides comprehensive evaluation capabilities for agent graphs,
including trajectory analysis, response quality assessment, and LLM-as-judge
evaluation patterns.

Main Components:
    - EvalSet, EvalCase:             Data models for test cases
    - EvalConfig, EvalPresets:       Configuration for evaluation criteria
    - AgentEvaluator:                Runs evaluation over EvalSet or single EvalCase
    - TrajectoryCollector:           Captures tool calls, node visits, and LLM outputs
    - ExecutionResult:               Holds tool calls, trajectory, final response
    - BaseCriterion + subclasses:    All evaluation criteria
    - EvalReport, EvalSummary:       Result and reporting models
    - Reporters:                     Console, JSON, HTML report generators

Example:
    ```python
    from agentflow.evaluation import AgentEvaluator, EvalConfig, CriterionConfig
    from agentflow.evaluation.dataset import EvalCase, ToolCall

    case = EvalCase.single_turn(
        eval_id="test_1",
        user_query="What is the weather in London?",
        expected_response="The weather in London is sunny",
        expected_tools=[ToolCall(name="get_weather")],
    )

    evaluator = AgentEvaluator(
        graph,
        collector=collector,
        config=EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig(threshold=1.0),
            }
        ),
    )
    result = await evaluator.evaluate_case(case)
    assert result.passed
    ```
"""

# --- Dataset ---
# --- Collectors (event-based trajectory capture via callback_manager) ---
from agentflow.evaluation.collectors.trajectory_collector import (
    EventCollector,
    PublisherCallback,
    TrajectoryCollector,
    make_trajectory_callback,
)

# --- Config ---
from agentflow.evaluation.config.eval_config import (
    CriterionConfig,
    EvalConfig,
    MatchType,
    ReporterConfig,
    Rubric,
    UserSimulatorConfig,
)
from agentflow.evaluation.config.presets import EvalPresets

# --- Criteria: base ---
from agentflow.evaluation.criteria.base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from agentflow.evaluation.criteria.factual_accuracy import FactualAccuracyCriterion

# --- Criteria: advanced ---
from agentflow.evaluation.criteria.hallucination import HallucinationCriterion
from agentflow.evaluation.criteria.llm_judge import LLMJudgeCriterion
from agentflow.evaluation.criteria.llm_utils import LLMCallerMixin

# --- Criteria: response ---
from agentflow.evaluation.criteria.response import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    ResponseMatchCriterion,
    RougeMatchCriterion,
)
from agentflow.evaluation.criteria.rubric import RubricBasedCriterion
from agentflow.evaluation.criteria.safety import SafetyCriterion

# --- Criteria: LLM-as-judge ---
from agentflow.evaluation.criteria.simulation_goals import SimulationGoalsCriterion

# --- Criteria: trajectory ---
from agentflow.evaluation.criteria.trajectory import (
    NodeOrderMatchCriterion,
    ToolNameMatchCriterion,
    TrajectoryMatchCriterion,
)
from agentflow.evaluation.dataset.builder import EvalSetBuilder
from agentflow.evaluation.dataset.eval_set import (
    EvalCase,
    EvalSet,
    Invocation,
    MessageContent,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)

# --- Results ---
from agentflow.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
    EvalSummary,
)

# --- Evaluator ---
from agentflow.evaluation.evaluator import AgentEvaluator, EvaluationRunner

# --- Execution ---
from agentflow.evaluation.execution.result import ExecutionResult, NodeResponseData
from agentflow.evaluation.quick_eval import QuickEval

# --- Reporters ---
from agentflow.evaluation.reporters.base import BaseReporter
from agentflow.evaluation.reporters.console import Colors, ConsoleReporter, print_report
from agentflow.evaluation.reporters.html import HTMLReporter
from agentflow.evaluation.reporters.json import JSONReporter, JUnitXMLReporter
from agentflow.evaluation.reporters.manager import ReporterManager, ReporterOutput

# --- Simulators ---
from agentflow.evaluation.simulators import (
    BatchSimulator,
    ConversationScenario,
    SimulationResult,
    UserSimulator,
)

# --- Testing helpers ---
from agentflow.evaluation.testing import (
    EvalFixtures,
    EvalPlugin,
    EvalTestCase,
    assert_criterion_passed,
    assert_eval_passed,
    create_eval_app,
    create_simple_eval_set,
    eval_test,
    parametrize_eval_cases,
    run_eval,
)


__all__ = [
    # --- Dataset ---
    "EvalSet",
    "EvalCase",
    "EvalSetBuilder",
    "Invocation",
    "MessageContent",
    "SessionInput",
    "StepType",
    "ToolCall",
    "TrajectoryStep",
    # --- Execution ---
    "ExecutionResult",
    "NodeResponseData",
    # --- Collectors ---
    "TrajectoryCollector",
    "EventCollector",
    "PublisherCallback",
    "make_trajectory_callback",
    # --- Criteria: base ---
    "BaseCriterion",
    "SyncCriterion",
    "CompositeCriterion",
    "WeightedCriterion",
    "LLMCallerMixin",
    # --- Criteria: trajectory ---
    "TrajectoryMatchCriterion",
    "ToolNameMatchCriterion",
    "NodeOrderMatchCriterion",
    # --- Criteria: response ---
    "ResponseMatchCriterion",
    "RougeMatchCriterion",
    "ExactMatchCriterion",
    "ContainsKeywordsCriterion",
    # --- Criteria: LLM-as-judge ---
    "SimulationGoalsCriterion",
    "LLMJudgeCriterion",
    "RubricBasedCriterion",
    # --- Criteria: advanced ---
    "HallucinationCriterion",
    "SafetyCriterion",
    "FactualAccuracyCriterion",
    # --- Config ---
    "EvalConfig",
    "CriterionConfig",
    "MatchType",
    "Rubric",
    "UserSimulatorConfig",
    "ReporterConfig",
    "EvalPresets",
    # --- Results ---
    "CriterionResult",
    "EvalCaseResult",
    "EvalReport",
    "EvalSummary",
    # --- Reporters ---
    "BaseReporter",
    "ConsoleReporter",
    "Colors",
    "HTMLReporter",
    "JSONReporter",
    "JUnitXMLReporter",
    "ReporterManager",
    "ReporterOutput",
    "print_report",
    # --- Evaluator ---
    "AgentEvaluator",
    "EvaluationRunner",
    "QuickEval",
    # --- Simulators ---
    "UserSimulator",
    "BatchSimulator",
    "ConversationScenario",
    "SimulationResult",
    # --- Testing ---
    "run_eval",
    "create_eval_app",
    "create_simple_eval_set",
    "EvalPlugin",
    "EvalTestCase",
    "EvalFixtures",
    "eval_test",
    "assert_eval_passed",
    "assert_criterion_passed",
    "parametrize_eval_cases",
]
