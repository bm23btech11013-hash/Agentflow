"""
Hallucination/groundedness detection criterion.
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


HALLUCINATION_PROMPT = """
You are an expert fact-checker. Your task is to evaluate whether the RESPONSE is
grounded in the provided CONTEXT and does not contain hallucinations.

CONTEXT (Information available to the agent):
{context}

USER QUESTION:
{question}

AGENT RESPONSE:
{response}

Evaluate the response for hallucinations:
1. Check if all claims in the response are supported by the context
2. Check if any facts are fabricated or contradicted by the context
3. Check if any numbers, dates, or specific details are incorrect

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "is_grounded": <true/false>,
    "hallucinations": [<list of specific hallucinations found, empty if none>],
    "reasoning": "<explanation>"
}}

Score: 1.0 = fully grounded, 0.0 = mostly hallucinated.
"""


class HallucinationCriterion(LLMCallerMixin, BaseCriterion):
    """Evaluate response groundedness and detect hallucinations."""

    name = "hallucinations_v1"
    description = "LLM-based hallucination/groundedness detection"

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

            context = self._extract_context(actual, expected)
            prompt = HALLUCINATION_PROMPT.format(
                context=context or "No context available",
                question=question,
                response=actual_response,
            )

            scores, all_hallucinations, reasonings = await self._run_samples(prompt)

            if not scores:
                return CriterionResult.failure(criterion=self.name, error="All LLM samples failed")

            final_score = sum(scores) / len(scores)

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "is_grounded": final_score >= self.threshold,
                    "hallucinations": list(set(all_hallucinations)),
                    "samples": len(scores),
                    "reasonings": reasonings,
                },
            )

        except Exception as e:
            logger.error("Hallucination evaluation failed: %s", e)
            return CriterionResult.failure(criterion=self.name, error=str(e))

    async def _run_samples(self, prompt: str) -> tuple[list[float], list[str], list[str]]:
        """Run majority-voting samples and collect scores, hallucinations, and reasonings."""
        scores: list[float] = []
        all_hallucinations: list[str] = []
        reasonings: list[str] = []

        for _ in range(self.config.num_samples):
            try:
                result = await self._call_llm_json(prompt)
                scores.append(float(result.get("score", 0.0)))
                all_hallucinations.extend(result.get("hallucinations", []))
                reasonings.append(result.get("reasoning", ""))
            except Exception as e:
                logger.warning("Hallucination sample failed: %s", e)

        return scores, all_hallucinations, reasonings

    def _extract_question(self, expected: EvalCase) -> str:
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_context(self, actual: ExecutionResult, expected: EvalCase) -> str:
        """Build context from tool results and expected metadata."""
        parts: list[str] = []

        for tc in actual.tool_calls:
            if tc.result:
                parts.append(f"Tool {tc.name} returned: {tc.result}")

        if expected.metadata.get("context"):
            parts.append(f"Reference context: {expected.metadata['context']}")

        return "\n\n".join(parts)
