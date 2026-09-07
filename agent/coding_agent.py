"""
agent/coding_agent.py

Generative Coding Agent and Self-Correction Engine.
Generates initial code implementations from retrieved task context,
evaluates them in the sandbox, and iteratively patches errors (AssertionErrors,
SyntaxErrors, timeouts) until the solution passes or the retry budget is exhausted.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .config import AgentConfig, get_config
from .llm import LLMClient
from .evaluator import Evaluator, EvaluationResult
from .retriever import RetrievedContext

logger = logging.getLogger(__name__)


@dataclass
class AttemptLog:
    """Audit log of a single coding or self-correction attempt."""
    attempt: int
    code: str
    eval_result: EvaluationResult
    error_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "code": self.code,
            "passed": self.eval_result.passed,
            "timed_out": self.eval_result.timed_out,
            "error_summary": self.error_summary,
            "execution_time": self.eval_result.execution_time,
        }


@dataclass
class CodingAttemptResult:
    """Final outcome of an autonomous coding and self-correction session."""
    success: bool
    final_code: str
    total_attempts: int
    attempts: List[AttemptLog] = field(default_factory=list)
    final_eval_result: Optional[EvaluationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "final_code": self.final_code,
            "attempts": [a.to_dict() for a in self.attempts],
        }


class CodingAgent:
    """
    Code generator with iterative self-correction capabilities.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config)
        self.evaluator = evaluator or Evaluator(self.config, self.llm_client)

    def generate_initial_code(self, context: RetrievedContext) -> str:
        """
        Generates the first zero-shot code implementation from the retrieved context.
        """
        system_prompt = (
            "You are an expert Python software engineer. "
            "Write clean, correct, efficient, and robust Python code that implements the specified contract. "
            "Include any required standard library imports at the top. "
            "Return ONLY executable Python code inside a markdown code block (```python ... ```). "
            "Do NOT include conversational commentary or explanations."
        )

        user_prompt = (
            f"{context.formatted_prompt}\n\n"
            "Implement the target function completely according to the specifications and docstring above."
        )

        raw_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens or 16384,
        )

        code = self._extract_clean_code(raw_response)
        if not code or not code.strip():
            logger.warning(f"Initial code generation for `{context.task_spec.entry_point}` returned empty. Retrying with direct prompt...")
            fallback_user_prompt = (
                f"Implement the following Python task:\n\n"
                f"{context.task_spec.raw_prompt}\n\n"
                f"Write the complete Python function in a ```python ... ``` code block."
            )
            raw_response = self.llm_client.generate(
                prompt=fallback_user_prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 16384,
            )
            code = self._extract_clean_code(raw_response)

        return code

    def self_correct(
        self,
        code: str,
        context: RetrievedContext,
        eval_result: EvaluationResult,
        attempt: int,
    ) -> str:
        """
        Generates a targeted bug fix or patch based on sandbox execution feedback.
        """
        system_prompt = (
            "You are an expert Python debugger and algorithmic optimizer. "
            "Your task is to fix the provided code based on the execution failure diagnostics. "
            "Diagnose the root cause, fix the algorithmic or syntax flaw, and return the complete "
            "corrected code in a markdown block (```python ... ```). "
            "Do NOT write markdown text outside the code block."
        )

        # Build diagnostic feedback
        diagnostic_lines: List[str] = []
        if eval_result.timed_out:
            diagnostic_lines.append(
                "🚨 TIMEOUT ERROR: The code was killed because it exceeded the execution timeout limit! "
                "This indicates an INFINITE LOOP (e.g. while condition never updating, missing index increment) "
                "or non-terminating recursion. Fix the loop termination conditions immediately!"
            )
        elif eval_result.error_message and "assertionerror" in eval_result.error_message.lower():
            diagnostic_lines.append(
                f"❌ ASSERTION FAILED: The code failed a test case.\n"
                f"Error details: {eval_result.error_message}"
            )
        elif eval_result.stderr:
            diagnostic_lines.append(
                f"❌ RUNTIME EXCEPTION:\n{eval_result.stderr.strip()}"
            )
        else:
            diagnostic_lines.append(
                f"❌ EXECUTION FAILED with return code {eval_result.return_code}. Error: {eval_result.error_message}"
            )

        if eval_result.stdout:
            diagnostic_lines.append(f"Standard output captured:\n{eval_result.stdout.strip()}")

        diagnostics_text = "\n".join(diagnostic_lines)

        user_prompt = (
            f"### Problem Specification\n{context.formatted_prompt}\n\n"
            f"### Previous Implementation (Attempt {attempt - 1})\n"
            f"```python\n{code}\n```\n\n"
            f"### Failure Diagnostics\n"
            f"{diagnostics_text}\n\n"
            f"Please carefully analyze why the previous code failed. "
            f"Fix the bug, handle edge cases, and output the complete corrected Python code."
        )

        raw_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
        )

        cleaned_code = self._extract_clean_code(raw_response)
        if cleaned_code and cleaned_code.strip():
            return cleaned_code
        return code

    def solve(
        self,
        context: RetrievedContext,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> CodingAttemptResult:
        """
        Executes the autonomous Generate -> Evaluate -> (Self-Correct Loop).
        """
        budget = max_retries if max_retries is not None else self.config.max_retries
        target_entry_point = entry_point or context.task_spec.entry_point

        # Ensure test suite is consistent across all retries if not provided
        if not test_code or not test_code.strip():
            if context.task_spec.raw_prompt:
                try:
                    test_code = self.evaluator.generate_tests(
                        prompt=context.task_spec.raw_prompt,
                        entry_point=target_entry_point,
                    )
                except Exception as e:
                    logger.warning(f"Could not pre-generate unit tests: {e}")

        attempts: List[AttemptLog] = []

        def _resolve_best_code(candidate: str) -> str:
            if candidate and candidate.strip():
                return candidate.strip()
            for att in reversed(attempts):
                if att.code and att.code.strip():
                    return att.code.strip()
            return ""

        # Attempt 1: Generate initial code
        logger.info(f"Generating initial code for `{target_entry_point}`...")
        current_code = self.generate_initial_code(context)

        # Evaluate attempt 1
        eval_result = self.evaluator.evaluate(
            code=current_code,
            test_code=test_code,
            prompt=context.task_spec.raw_prompt,
            entry_point=target_entry_point,
            raw_prompt=context.task_spec.raw_prompt,
        )

        attempts.append(AttemptLog(
            attempt=1,
            code=current_code,
            eval_result=eval_result,
            error_summary=eval_result.error_message,
        ))

        if eval_result.passed:
            logger.info(f"✅ Solution for `{target_entry_point}` passed on attempt 1!")
            return CodingAttemptResult(
                success=True,
                final_code=_resolve_best_code(current_code),
                total_attempts=1,
                attempts=attempts,
                final_eval_result=eval_result,
            )

        # Self-correction loop for subsequent attempts
        for attempt_num in range(2, budget + 2):
            logger.warning(
                f"Attempt {attempt_num - 1} failed ({eval_result.error_message}). "
                f"Starting self-correction (Attempt {attempt_num}/{budget + 1})..."
            )

            new_code = self.self_correct(
                code=current_code,
                context=context,
                eval_result=eval_result,
                attempt=attempt_num,
            )
            if new_code and new_code.strip():
                current_code = new_code

            eval_result = self.evaluator.evaluate(
                code=current_code,
                test_code=test_code,
                prompt=context.task_spec.raw_prompt,
                entry_point=target_entry_point,
                raw_prompt=context.task_spec.raw_prompt,
            )

            attempts.append(AttemptLog(
                attempt=attempt_num,
                code=current_code,
                eval_result=eval_result,
                error_summary=eval_result.error_message,
            ))

            if eval_result.passed:
                logger.info(f"✅ Solution for `{target_entry_point}` passed after self-correction on attempt {attempt_num}!")
                return CodingAttemptResult(
                    success=True,
                    final_code=_resolve_best_code(current_code),
                    total_attempts=attempt_num,
                    attempts=attempts,
                    final_eval_result=eval_result,
                )

        logger.error(f"❌ Failed to solve `{target_entry_point}` after {len(attempts)} attempts.")
        return CodingAttemptResult(
            success=False,
            final_code=_resolve_best_code(current_code),
            total_attempts=len(attempts),
            attempts=attempts,
            final_eval_result=eval_result,
        )

    @staticmethod
    def _extract_clean_code(text: str) -> str:
        """
        Extracts pure Python code from model responses, handling markdown code fences.
        """
        if not text:
            return ""

        # Search for ```python ... ```, ```py ... ```, or ``` ... ```
        pattern = r"```(?:python[0-9]?|py)?\s*\n?(.*?)\s*```"
        matches: list[str] = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        valid_matches = [m.strip() for m in matches if m.strip()]
        if valid_matches:
            # Pick the largest non-empty code block
            return max(valid_matches, key=len)

        # Check for unclosed markdown code fence if LLM output was truncated
        unclosed = re.search(r"```(?:python[0-9]?|py)?\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
        if unclosed:
            content = unclosed.group(1).strip()
            content = re.sub(r"```.*$", "", content, flags=re.DOTALL).strip()
            if content:
                return content

        return text.strip()
