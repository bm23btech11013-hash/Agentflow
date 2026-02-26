"""
Factual accuracy evaluation criterion.
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


FACTUAL_ACCURACY_PROMPT = """
You are an expert fact-checker. Your task is to evaluate the factual accuracy
of the RESPONSE.

QUESTION:
{question}

RESPONSE:
{response}

REFERENCE FACTS (if available):
{reference}

Evaluate the response for factual accuracy:
1. Are all stated facts correct?
2. Are numbers, dates, and names accurate?
3. Are any claims verifiably false?

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "is_accurate": <true/false>,
    "errors": [<list of factual errors found>],
    "reasoning": "<explanation>"
}}
"""


class FactualAccuracyCriterion(LLMCallerMixin, BaseCriterion):
    """Evaluate factual accuracy of responses."""

    name = "factual_accuracy_v1"
    description = "LLM-based factual accuracy evaluation"

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        try:
            question = self._extract_question(expected)

            actual_response = actual.actual_response

            if not actual_response:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No response to evaluate"},
                )

            reference = self._extract_reference(expected)

            prompt = FACTUAL_ACCURACY_PROMPT.format(
                question=question,
                response=actual_response,
                reference=reference or "Not provided",
            )

            scores, all_errors, reasonings = await self._run_samples(prompt)

            if not scores:
                return CriterionResult.failure(criterion=self.name, error="All LLM samples failed")

            final_score = sum(scores) / len(scores)

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "is_accurate": final_score >= self.threshold,
                    "errors": list(set(all_errors)),
                    "samples": len(scores),
                    "reasonings": reasonings,
                },
            )

        except Exception as e:
            logger.error("Factual accuracy evaluation failed: %s", e)
            return CriterionResult.failure(criterion=self.name, error=str(e))

    async def _run_samples(self, prompt: str) -> tuple[list[float], list[str], list[str]]:
        """Run majority-voting samples and collect scores, errors, and reasonings."""
        scores: list[float] = []
        all_errors: list[str] = []
        reasonings: list[str] = []

        for _ in range(self.config.num_samples):
            try:
                result = await self._call_llm_json(prompt)
                scores.append(float(result.get("score", 0.0)))
                all_errors.extend(result.get("errors", []))
                reasonings.append(result.get("reasoning", ""))
            except Exception as e:
                logger.warning("Factual accuracy sample failed: %s", e)

        return scores, all_errors, reasonings

    def _extract_question(self, expected: EvalCase) -> str:
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_reference(self, expected: EvalCase) -> str:
        for inv in expected.conversation:
            if inv.expected_final_response:
                return inv.expected_final_response.get_text()
        return expected.metadata.get("reference_facts", "")
