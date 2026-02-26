"""
LLM-as-Judge criterion for semantic response matching.
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


SEMANTIC_MATCH_PROMPT = """
You are an expert evaluator. Your task is to determine if the ACTUAL response is
semantically equivalent to the EXPECTED response.

QUESTION: {question}

EXPECTED RESPONSE: {expected}

ACTUAL RESPONSE: {actual}

Evaluate if the ACTUAL response conveys the same meaning and information as the
EXPECTED response. Minor differences in wording are acceptable as long as the core
meaning is preserved.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Score guidelines:
- 1.0: Perfect semantic match, all key information preserved
- 0.8-0.9: Minor differences but core meaning intact
- 0.5-0.7: Partial match, some key information missing or different
- 0.2-0.4: Significant differences but some overlap
- 0.0-0.1: Completely different or contradictory
"""


class LLMJudgeCriterion(LLMCallerMixin, BaseCriterion):
    """Use an LLM to judge semantic equivalence between actual and expected responses.

    Supports majority voting across multiple samples for reliable scores.
    """

    name = "final_response_match_v2"
    description = "LLM-based semantic response matching"

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        try:
            question = self._extract_question(expected)
            expected_response = self._extract_expected_response(expected)

            actual_response = actual.actual_response

            if not expected_response:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No expected response to compare"},
                )

            scores, reasonings = await self._run_samples(
                question=question,
                expected=expected_response,
                actual=actual_response,
            )

            final_score = sum(scores) / len(scores) if scores else 0.0

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "samples": len(scores),
                    "individual_scores": scores,
                    "reasonings": reasonings,
                    "judge_model": self.config.judge_model,
                    "reason": reasonings[-1] if reasonings else "",
                },
            )

        except Exception as e:
            logger.error("LLM judge evaluation failed: %s", e)
            return CriterionResult.failure(criterion=self.name, error=str(e))

    async def _run_samples(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> tuple[list[float], list[str]]:
        scores: list[float] = []
        reasonings: list[str] = []

        prompt = SEMANTIC_MATCH_PROMPT.format(
            question=question,
            expected=expected,
            actual=actual,
        )

        for _ in range(self.config.num_samples):
            try:
                score, reasoning = await self._call_llm_score(prompt)
                scores.append(score)
                reasonings.append(reasoning)
            except Exception as e:
                logger.warning("LLM sample failed: %s", e)

        return scores, reasonings

    def _extract_question(self, expected: EvalCase) -> str:
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_expected_response(self, expected: EvalCase) -> str:
        # Use the last invocation's expected response to match the
        # actual_response which comes from the final conversation turn.
        result = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                result = invocation.expected_final_response.get_text()
        return result
