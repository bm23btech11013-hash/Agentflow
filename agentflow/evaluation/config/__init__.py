"""
Evaluation configuration package.

Provides configuration models and presets for agent evaluation.

Example:
    ```python
    # All old imports still work unchanged
    from agentflow.evaluation.config import EvalConfig, CriterionConfig
    from agentflow.evaluation.config import EvalPresets, MatchType, Rubric

    # Use a preset
    config = EvalPresets.tool_usage(strict=True)

    # Or build custom config
    config = EvalConfig(
        criteria={
            "trajectory_match": CriterionConfig.trajectory(threshold=1.0),
        }
    )
    ```
"""

from .eval_config import (
    DEFAULT_JUDGE_MODEL,
    CriterionConfig,
    EvalConfig,
    MatchType,
    ReporterConfig,
    Rubric,
    UserSimulatorConfig,
)
from .presets import EvalPresets


__all__ = [
    # Core config models
    "EvalConfig",
    "CriterionConfig",
    "MatchType",
    "Rubric",
    "UserSimulatorConfig",
    "ReporterConfig",
    # Presets
    "EvalPresets",
    "DEFAULT_JUDGE_MODEL",
]
