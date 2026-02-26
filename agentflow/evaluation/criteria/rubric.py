"""
Rubric-based evaluation criterion.
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


RUBRIC_PROMPT = """
You are an expert evaluator. Your task is to evaluate the quality of an AI
assistant's response.

QUESTION: {question}

RESPONSE: {response}

{rubric_section}

Evaluate the response and provide a score.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}
"""


class RubricBasedCriterion(LLMCallerMixin, BaseCriterion):
    """Evaluate response quality against custom rubrics.

    Uses an LLM to grade responses against user-defined rubrics,
    allowing for flexible and domain-specific evaluation.
    """

    name = "rubric_based_final_response_quality_v1"
    description = "LLM-based rubric evaluation"

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        try:
            rubrics = self.config.rubrics
            if not rubrics:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No rubrics configured"},
                )

            question = self._extract_question(expected)

            actual_response = actual.actual_response

            rubric_lines = ["Evaluate the response against these criteria:"]
            for rubric in rubrics:
                rubric_lines.append(f"- {rubric.rubric_id}: {rubric.content}")
            rubric_section = "\n".join(rubric_lines)

            prompt = RUBRIC_PROMPT.format(
                question=question,
                response=actual_response,
                rubric_section=rubric_section,
            )

            scores: list[float] = []
            reasonings: list[str] = []

            for _ in range(self.config.num_samples):
                try:
                    score, reasoning = await self._call_llm_score(prompt)
                    scores.append(score)
                    reasonings.append(reasoning)
                except Exception as e:
                    logger.warning("Rubric sample failed: %s", e)

            final_score = sum(scores) / len(scores) if scores else 0.0

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "rubrics": [r.rubric_id for r in rubrics],
                    "scores": scores,
                    "reasonings": reasonings,
                },
            )

        except Exception as e:
            logger.error("Rubric evaluation failed: %s", e)
            return CriterionResult.failure(criterion=self.name, error=str(e))

    def _extract_question(self, expected: EvalCase) -> str:
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""
