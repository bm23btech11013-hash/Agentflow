"""Preset evaluation configurations for common scenarios."""

from agentflow.evaluation.eval_config import CriterionConfig, EvalConfig, MatchType


class EvalPresets:
    """Preset evaluation configurations for common use cases.

    Provides ready-to-use configurations for:
    - Response quality checking
    - Tool usage validation
    - Conversation flow testing
    - Combined scenarios

    Example:
        ```python
        # Use a preset
        config = EvalPresets.response_quality()

        # Combine presets
        config = EvalPresets.combine(
            EvalPresets.response_quality(),
            EvalPresets.tool_usage(),
        )
        ```
    """

    @classmethod
    def response_quality(
        cls,
        threshold: float = 0.7,
        use_llm_judge: bool = True,
    ) -> EvalConfig:
        """Preset for checking response quality.

        Focuses on whether the agent's responses are relevant and accurate.

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            use_llm_judge: Whether to use LLM-as-judge for evaluation

        Returns:
            EvalConfig with response quality criteria
        """
        criteria = {
            "response_match": CriterionConfig(
                threshold=threshold,
                enabled=True,
            ),
        }

        if use_llm_judge:
            criteria["llm_judge"] = CriterionConfig(
                threshold=threshold,
                judge_model="gpt-4o-mini",
                num_samples=1,
                enabled=True,
            )

        return EvalConfig(criteria=criteria)

    @classmethod
    def tool_usage(
        cls,
        threshold: float = 1.0,
        strict: bool = True,
        check_args: bool = True,
    ) -> EvalConfig:
        """Preset for validating tool usage.

        Checks whether the agent calls the right tools with correct arguments.

        Args:
            threshold: Minimum score to pass
            strict: Whether to require exact tool matches (EXACT vs IN_ORDER)
            check_args: Whether to validate tool arguments

        Returns:
            EvalConfig with tool usage criteria
        """
        match_type = MatchType.EXACT if strict else MatchType.IN_ORDER

        return EvalConfig(
            criteria={
                "trajectory_match": CriterionConfig(
                    threshold=threshold,
                    match_type=match_type,
                    check_args=check_args,
                    enabled=True,
                ),
            }
        )

    @classmethod
    def conversation_flow(
        cls,
        threshold: float = 0.8,
    ) -> EvalConfig:
        """Preset for testing conversation flow.

        Validates both responses and tool usage in multi-turn conversations.

        Args:
            threshold: Minimum score to pass

        Returns:
            EvalConfig for conversation testing
        """
        return EvalConfig(
            criteria={
                "response_match": CriterionConfig(
                    threshold=threshold,
                    enabled=True,
                ),
                "trajectory_match": CriterionConfig(
                    threshold=threshold,
                    match_type=MatchType.IN_ORDER,
                    check_args=False,
                    enabled=True,
                ),
            }
        )

    @classmethod
    def quick_check(cls) -> EvalConfig:
        """Minimal preset for quick sanity checks.

        Uses relaxed thresholds for rapid iteration.

        Returns:
            EvalConfig with lenient criteria
        """
        return EvalConfig(
            criteria={
                "response_match": CriterionConfig(
                    threshold=0.5,
                    enabled=True,
                ),
            }
        )

    @classmethod
    def comprehensive(
        cls,
        threshold: float = 0.8,
        use_llm_judge: bool = True,
    ) -> EvalConfig:
        """Comprehensive evaluation with all criteria.

        Args:
            threshold: Minimum score to pass
            use_llm_judge: Whether to include LLM judge

        Returns:
            EvalConfig with all criteria enabled
        """
        criteria = {
            "response_match": CriterionConfig(
                threshold=threshold,
                enabled=True,
            ),
            "trajectory_match": CriterionConfig(
                threshold=threshold,
                match_type=MatchType.IN_ORDER,
                check_args=True,
                enabled=True,
            ),
        }

        if use_llm_judge:
            criteria["llm_judge"] = CriterionConfig(
                threshold=threshold,
                judge_model="gpt-4o-mini",
                num_samples=3,
                enabled=True,
            )

        return EvalConfig(criteria=criteria)

    @classmethod
    def combine(cls, *configs: EvalConfig) -> EvalConfig:
        """Combine multiple preset configurations.

        Args:
            *configs: Multiple EvalConfig instances to merge

        Returns:
            Combined EvalConfig with all criteria
        """
        combined_criteria = {}

        for config in configs:
            combined_criteria.update(config.criteria)

        return EvalConfig(criteria=combined_criteria)

    @classmethod
    def custom(
        cls,
        response_threshold: float | None = None,
        tool_threshold: float | None = None,
        llm_judge_threshold: float | None = None,
        tool_match_type: MatchType = MatchType.IN_ORDER,
        check_tool_args: bool = True,
    ) -> EvalConfig:
        """Create custom configuration from individual parameters.

        Args:
            response_threshold: Enable response matching with this threshold
            tool_threshold: Enable tool matching with this threshold
            llm_judge_threshold: Enable LLM judge with this threshold
            tool_match_type: How to match tool trajectories
            check_tool_args: Whether to check tool arguments

        Returns:
            Custom EvalConfig
        """
        criteria = {}

        if response_threshold is not None:
            criteria["response_match"] = CriterionConfig(
                threshold=response_threshold,
                enabled=True,
            )

        if tool_threshold is not None:
            criteria["trajectory_match"] = CriterionConfig(
                threshold=tool_threshold,
                match_type=tool_match_type,
                check_args=check_tool_args,
                enabled=True,
            )

        if llm_judge_threshold is not None:
            criteria["llm_judge"] = CriterionConfig(
                threshold=llm_judge_threshold,
                judge_model="gpt-4o-mini",
                num_samples=3,
                enabled=True,
            )

        return EvalConfig(criteria=criteria)
