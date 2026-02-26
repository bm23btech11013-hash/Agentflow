"""Conversation goals criterion (UserSimulator only).

Evaluates whether goals were achieved across a full multi-turn
conversation transcript produced by ``UserSimulator``.

.. important::

    This criterion is designed **exclusively** for use with
    ``UserSimulator``.  The simulator sets ``actual_response`` to the
    complete conversation transcript (all turns).  In the normal
    ``AgentEvaluator`` flow, ``actual_response`` contains only the
    agent's final response, so this criterion would not see prior turns.

The LLM judge receives the full transcript and a list of goals, then
checks whether each goal was addressed at any point during the
conversation — not just in the final message.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentflow.evaluation.criteria.base import BaseCriterion
from agentflow.evaluation.criteria.llm_utils import LLMCallerMixin
from agentflow.evaluation.eval_result import CriterionResult


if TYPE_CHECKING:
    from agentflow.evaluation.dataset.eval_set import EvalCase
    from agentflow.evaluation.execution.result import ExecutionResult

logger = logging.getLogger("agentflow.evaluation")

CONVERSATION_GOALS_PROMPT = (
    """You are evaluating whether an AI assistant achieved specific goals """
    """across a multi-turn conversation.

CONVERSATION TRANSCRIPT:
{conversation}

GOALS TO ACHIEVE:
{goals}

For each goal, determine if it was achieved at any point during the conversation -
not just in the final message.

Respond with a JSON object:
{{
    "score": <float 0.0 to 1.0, where 1.0 = all goals achieved, 0.0 = none achieved>,
    "achieved_goals": ["<exact goal text>", ...],
    "unachieved_goals": ["<exact goal text>", ...],
    "reasoning": "<brief explanation covering each goal>"
}}

Score = number_of_achieved_goals / total_goals.
"""
)


class SimulationGoalsCriterion(LLMCallerMixin, BaseCriterion):
    """Evaluate whether goals were achieved in a UserSimulator conversation.

    **UserSimulator-only** — this criterion expects ``actual_response`` to
    contain the full multi-turn conversation transcript as set by
    ``UserSimulator._evaluate_simulation()``.  It will **not** work
    correctly with the normal ``AgentEvaluator`` flow, where
    ``actual_response`` is only the agent's final response.

    The LLM judge receives the complete transcript and a semicolon-separated
    list of goals (from the ``EvalCase`` expected response), then checks
    whether each goal was addressed at any point during the conversation.

    Score: ``achieved_goals / total_goals``  (0.0 - 1.0)

    Example::

        from agentflow.evaluation import SimulationGoalsCriterion, CriterionConfig, UserSimulator

        judge = SimulationGoalsCriterion(config=CriterionConfig(threshold=0.7))
        simulator = UserSimulator(
            model="gemini/gemini-2.5-flash",
            criteria=[judge],
        )
        result = await simulator.run(graph, scenario)
        print(result.criterion_scores["simulation_goals"])
    """

    name = "simulation_goals"
    description = "LLM-based goal achievement evaluation for UserSimulator conversations"

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        try:
            conversation_transcript = actual.actual_response
            goals_text = self._extract_goals(expected)

            if not goals_text:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No goals defined — skipping evaluation"},
                )

            prompt = CONVERSATION_GOALS_PROMPT.format(
                conversation=conversation_transcript,
                goals=goals_text,
            )

            result_dict = await self._call_llm_json(prompt)
            score = float(result_dict.get("score", 0.0))

            return CriterionResult.success(
                criterion=self.name,
                score=score,
                threshold=self.threshold,
                details={
                    "achieved_goals": result_dict.get("achieved_goals", []),
                    "unachieved_goals": result_dict.get("unachieved_goals", []),
                    "reasoning": result_dict.get("reasoning", ""),
                    "reason": result_dict.get("reasoning", ""),
                    "judge_model": self.config.judge_model,
                },
            )

        except Exception as e:
            logger.error("SimulationGoalsCriterion evaluation failed: %s", e)
            return CriterionResult.failure(criterion=self.name, error=str(e))

    def _extract_goals(self, expected: EvalCase) -> str:
        """Extract semicolon-separated goals string from EvalCase."""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                return invocation.expected_final_response.get_text()
        return ""
