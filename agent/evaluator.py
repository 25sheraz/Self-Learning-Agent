"""
agent/evaluator.py

Isolated sandbox execution engine and test evaluator for the Self-Learning Coding Agent.
Executes candidate code within a subprocess with strict timeout limits and captures
assertion failures, exceptions, timeouts, and standard outputs.
"""

import sys
import os
import re
import time
import tempfile
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional

from .config import AgentConfig, get_config
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Represents the execution outcome of code tested in the sandbox."""
    passed: bool
    timed_out: bool = False
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0


class Evaluator:
    """
    Manages safe subprocess execution and test evaluation.
    Supports HumanEval check(candidate) suites and dynamic LLM test generation fallback.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client
        self.timeout = self.config.sandbox_timeout

    def run_in_sandbox(
        self,
        full_code: str,
        timeout: Optional[float] = None,
    ) -> EvaluationResult:
        """
        Executes a Python script in an isolated subprocess with timeout enforcement.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        start_time = time.perf_counter()

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
                temp_path = temp_file.name
                temp_file.write(full_code)

            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
            )
            execution_time = time.perf_counter() - start_time
            passed = (proc.returncode == 0)
            error_msg = None
            if not passed:
                error_msg = self._extract_error_summary(proc.stderr or proc.stdout)

            return EvaluationResult(
                passed=passed,
                timed_out=False,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                error_message=error_msg,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired as e:
            execution_time = time.perf_counter() - start_time
            stdout_str = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", "replace") if e.stdout else "")
            stderr_str = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", "replace") if e.stderr else "")
            return EvaluationResult(
                passed=False,
                timed_out=True,
                return_code=None,
                stdout=stdout_str,
                stderr=stderr_str,
                error_message=f"Execution timed out after {effective_timeout:.1f} seconds",
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Unexpected sandbox runner error: {e}")
            return EvaluationResult(
                passed=False,
                timed_out=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=f"Sandbox execution error: {e}",
                execution_time=execution_time,
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def evaluate_code(
        self,
        code: str,
        test_code: str,
        entry_point: Optional[str] = None,
        timeout: Optional[float] = None,
        raw_prompt: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Combines solution code with test code and runs it in the sandbox.
        Handles the HumanEval check(candidate) calling convention if entry_point is given.

        raw_prompt: when provided, its code content is prepended so helper functions
        defined in the original problem (e.g. `poly` in HumanEval/32) are available
        both inside the generated solution and inside the test harness.
        """
        if not code or not code.strip():
            return EvaluationResult(
                passed=False,
                timed_out=False,
                return_code=-1,
                stdout="",
                stderr="No solution code provided.",
                error_message="Solution code is empty.",
                execution_time=0.0,
            )

        script_parts: list[str] = []

        # Prepend any helper code from the original prompt (e.g. `poly` for HumanEval/32).
        # We only include lines that look like Python definitions/imports, not docstrings.
        if raw_prompt and raw_prompt.strip():
            helper_lines = []
            for line in raw_prompt.splitlines():
                stripped = line.rstrip()
                # Keep import statements and function/class definitions from the prompt
                if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
                    helper_lines.append(stripped)
                elif helper_lines:
                    # Keep body lines after a def/class header
                    helper_lines.append(stripped)
            helper_code = "\n".join(helper_lines).strip()
            if helper_code:
                script_parts.append(helper_code)

        script_parts.append(code.strip())
        script_parts.append(test_code.strip())

        # If test suite defines `check(candidate)` and hasn't called it yet, invoke it.
        if entry_point and "def check(" in test_code:
            invocation = f"check({entry_point})"
            if invocation not in test_code:
                script_parts.append(invocation)

        # If test suite defines test functions (e.g. def test_...():) and doesn't call them, invoke them.
        test_funcs = re.findall(r"def (test_[a-zA-Z0-9_]+)\s*\(", test_code)
        for fn in test_funcs:
            if f"{fn}(" not in test_code:
                script_parts.append(f"{fn}()")

        full_script = "\n\n".join(script_parts) + "\n"
        return self.run_in_sandbox(full_script, timeout=timeout)

    def generate_tests(
        self,
        prompt: str,
        entry_point: Optional[str] = None,
    ) -> str:
        """
        Uses LLM to dynamically generate unit test assertions when no tests exist.
        """
        if self.llm_client is None:
            self.llm_client = LLMClient(self.config)

        if not self.llm_client.is_configured:
            raise RuntimeError(
                "LLMClient is not configured with an API key. "
                "Cannot generate fallback tests dynamically."
            )

        target = f"`{entry_point}`" if entry_point else "the solution function"
        system_prompt = (
            "You are an expert Python test engineer. "
            "Your task is to write strict unit tests using standard `assert` statements. "
            "Return ONLY executable Python code with assertions. Do NOT include markdown fences, "
            "explanations, or comments."
        )
        user_prompt = (
            f"Given the following problem description:\n\n"
            f"{prompt}\n\n"
            f"Generate 3 to 5 comprehensive assert statements testing edge cases, boundary values, "
            f"and typical cases for {target}. "
            f"Do not redefine the function. Only write the assert statements."
        )

        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
        )

        return self._extract_code_block(response)

    def evaluate(
        self,
        code: str,
        test_code: Optional[str] = None,
        prompt: Optional[str] = None,
        entry_point: Optional[str] = None,
        timeout: Optional[float] = None,
        raw_prompt: Optional[str] = None,
    ) -> EvaluationResult:
        """
        High-level evaluation entry point.
        Uses existing test_code if present; otherwise generates tests via LLM if prompt is provided.
        raw_prompt: original problem prompt, forwarded to evaluate_code so helper functions
        defined in it (e.g. `poly`) are available in the sandbox.
        """
        if test_code and test_code.strip():
            return self.evaluate_code(
                code=code,
                test_code=test_code,
                entry_point=entry_point,
                timeout=timeout,
                raw_prompt=raw_prompt,
            )

        if prompt and prompt.strip():
            logger.info("No test code provided. Generating tests via LLM fallback...")
            generated_test_code = self.generate_tests(prompt=prompt, entry_point=entry_point)
            return self.evaluate_code(
                code=code,
                test_code=generated_test_code,
                entry_point=entry_point,
                timeout=timeout,
                raw_prompt=raw_prompt,
            )

        # If neither tests nor prompt are supplied, simply execute the code in sandbox
        return self.run_in_sandbox(code, timeout=timeout)

    @staticmethod
    def _extract_error_summary(stderr: str) -> str:
        """
        Extracts the last meaningful exception or traceback line from stderr.
        """
        lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        if not lines:
            return "Process exited with non-zero status but produced no error output."
        return lines[-1]

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """
        Extracts raw python code from an LLM response if wrapped in markdown code blocks.
        """
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(matches).strip()
        return text.strip()
