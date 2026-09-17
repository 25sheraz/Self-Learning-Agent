"""
agent/orchestrator.py

Master Orchestrator for the Self-Learning Coding Agent.
Coordinates the end-to-end autonomous coding lifecycle:
  1. Task Understanding: Parse intent, arguments, types, constraints, and entry points.
  2. Context & Memory Retrieval: Gather standard library imports and inject past lessons.
  3. Coding & Self-Correction: Generate code, run in isolated sandbox, iteratively patch errors.
  4. Self-Learning Memory Store: Distill actionable reflections and persist them to prevent repeat mistakes.
  5. Delivery: Deliver validated solution, execution telemetry, and distilled lessons.
"""

from __future__ import annotations

import time
import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable

from .config import AgentConfig, get_config
from .llm import LLMClient
from .evaluator import Evaluator, EvaluationResult
from .task_understanding import TaskUnderstanding, TaskSpecification
from .retriever import ContextRetriever, RetrievedContext
from .coding_agent import CodingAgent, CodingAttemptResult, AttemptLog
from .memory import MemoryStore, MemoryEntry

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Lifecycle stages emitted during orchestration."""
    START = "start"
    UNDERSTANDING = "understanding"
    RETRIEVING = "retrieving"
    CODING = "coding"
    EVALUATING = "evaluating"
    SELF_CORRECTING = "self_correcting"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestratorEvent:
    """Event payload emitted to step listeners / UI callbacks."""
    step: StepStatus
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class OrchestratorResult:
    """Complete audit and outcome of an orchestration session."""
    task_id: Optional[str]
    entry_point: str
    success: bool
    final_code: str
    total_attempts: int
    attempts: List[AttemptLog] = field(default_factory=list)
    task_spec: Optional[TaskSpecification] = None
    retrieved_context: Optional[RetrievedContext] = None
    new_reflection: Optional[MemoryEntry] = None
    retrieved_lessons: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "entry_point": self.entry_point,
            "success": self.success,
            "final_code": self.final_code,
            "total_attempts": self.total_attempts,
            "attempts": [a.to_dict() for a in self.attempts],
            "task_spec": self.task_spec.to_dict() if self.task_spec else None,
            "retrieved_context": self.retrieved_context.to_dict() if self.retrieved_context else None,
            "new_reflection": self.new_reflection.to_dict() if self.new_reflection else None,
            "retrieved_lessons": self.retrieved_lessons,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "explanation": self.explanation,
        }


class Orchestrator:
    """
    Central Coordinator connecting Task Understanding, Memory & Context Retrieval,
    Coding Agent, Sandbox Evaluator, and the Self-Learning Memory Store.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        evaluator: Optional[Evaluator] = None,
        task_understanding: Optional[TaskUnderstanding] = None,
        retriever: Optional[ContextRetriever] = None,
        coding_agent: Optional[CodingAgent] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config)
        self.evaluator = evaluator or Evaluator(self.config, self.llm_client)
        self.task_understanding = task_understanding or TaskUnderstanding(self.config, self.llm_client)
        self.memory_store = memory_store or MemoryStore(memory_file=self.config.memory_file)
        self.retriever = retriever or ContextRetriever(self.config, memory_store=self.memory_store)
        self.coding_agent = coding_agent or CodingAgent(self.config, self.llm_client, self.evaluator)

    def run(
        self,
        task_input: Union[str, Dict[str, Any]],
        entry_point: Optional[str] = None,
        test_code: Optional[str] = None,
        task_id: Optional[str] = None,
        max_retries: Optional[int] = None,
        on_event: Optional[Callable[[OrchestratorEvent], None]] = None,
    ) -> OrchestratorResult:
        """
        Executes the autonomous Plan -> Retrieve -> Code -> Evaluate -> Fix -> Learn loop.

        Args:
            task_input: Either raw prompt string or a benchmark task dictionary (containing prompt, entry_point, test).
            entry_point: Optional entry point override.
            test_code: Optional test suite code. If None, uses test_code from dict or generates via LLM/doctests.
            task_id: Optional identifier for tracking (e.g. HumanEval/0).
            max_retries: Override maximum self-correction retry budget.
            on_event: Optional callback to receive live step status notifications.

        Returns:
            OrchestratorResult containing the final code, attempt history, evaluation details, and reflections.
        """
        start_time = time.perf_counter()

        def _emit(step: StepStatus, message: str, data: Optional[Dict[str, Any]] = None) -> None:
            event = OrchestratorEvent(step=step, message=message, data=data or {})
            if on_event:
                try:
                    on_event(event)
                except Exception as e:
                    logger.warning(f"Error in on_event listener: {e}")

        # 0. Unpack task input
        if isinstance(task_input, dict):
            prompt = str(task_input.get("prompt", ""))
            entry_point = entry_point or task_input.get("entry_point")
            test_code = test_code or task_input.get("test")
            task_id = task_id or task_input.get("task_id")
        else:
            prompt = task_input

        if not prompt or not prompt.strip():
            error_msg = "Task input prompt cannot be empty."
            _emit(StepStatus.FAILED, error_msg)
            return OrchestratorResult(
                task_id=task_id,
                entry_point=entry_point or "unknown",
                success=False,
                final_code="",
                total_attempts=0,
                error_message=error_msg,
                execution_time=time.perf_counter() - start_time,
            )

        _emit(StepStatus.START, f"Starting orchestration session for {task_id or 'custom task'}.", {
            "task_id": task_id,
            "entry_point": entry_point,
        })

        # 0.5 Bug Fix 4 — Prompt Enrichment:
        # Vague/short prompts (no `def`, no docstring, < 15 words) produce ambiguous
        # TaskSpecifications with empty argument lists, causing wrong signatures to be
        # baked into both the test-generation and code-generation prompts.
        # Expand the prompt into a full spec BEFORE task understanding so every downstream
        # stage has concrete type/signature/example information to work with.
        prompt = self._enrich_vague_prompt(prompt=prompt, on_emit=_emit)

        # 1. Task Understanding
        _emit(StepStatus.UNDERSTANDING, "Analyzing prompt and extracting task specification...", {
            "prompt_length": len(prompt),
        })
        try:
            task_spec = self.task_understanding.parse(prompt=prompt, entry_point=entry_point)
            target_entry_point = task_spec.entry_point or entry_point or "solution"
        except Exception as e:
            logger.error(f"Task understanding failed: {e}")
            task_spec = TaskSpecification(
                raw_prompt=prompt,
                entry_point=entry_point or "solution",
                goal_summary=prompt[:100],
            )
            target_entry_point = task_spec.entry_point

        # 2. Context & Memory Retrieval
        _emit(StepStatus.RETRIEVING, f"Retrieving dependencies and historical lessons for `{target_entry_point}`...", {
            "entry_point": target_entry_point,
        })
        try:
            context = self.retriever.retrieve(task_spec)
            retrieved_lessons = list(context.past_reflections)
        except Exception as e:
            logger.warning(f"Context retrieval error: {e}. Proceeding with raw prompt.")
            context = RetrievedContext(
                task_spec=task_spec,
                formatted_prompt=task_spec.raw_prompt,
            )
            retrieved_lessons = []

        if retrieved_lessons:
            _emit(StepStatus.RETRIEVING, f"Injected {len(retrieved_lessons)} historical lesson(s) to avoid past mistakes.", {
                "lessons": retrieved_lessons,
            })

        # 3. Coding Agent & Self-Correction Loop
        # Derive test assertions from docstrings if available and test_code is not provided
        if (not test_code or not test_code.strip()) and task_spec.examples:
            test_lines = []
            for ex in task_spec.examples:
                call = ex.get("call", "").rstrip(";")
                expected = ex.get("expected", "").strip()
                if call and expected:
                    test_lines.append(f"assert {call} == {expected}")
                elif call:
                    test_lines.append(call)
            if test_lines:
                test_code = "\n".join(test_lines)
                logger.info(f"Derived {len(test_lines)} test assertions from docstring examples.")

        _emit(StepStatus.CODING, f"Coding agent generating solution for `{target_entry_point}`...", {
            "entry_point": target_entry_point,
            "max_retries": max_retries or self.config.max_retries,
        })

        try:
            coding_result: CodingAttemptResult = self.coding_agent.solve(
                context=context,
                test_code=test_code,
                entry_point=target_entry_point,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.error(f"Coding agent execution failed: {e}")
            total_time = time.perf_counter() - start_time
            _emit(StepStatus.FAILED, f"Coding agent crashed: {e}")
            return OrchestratorResult(
                task_id=task_id,
                entry_point=target_entry_point,
                success=False,
                final_code="",
                total_attempts=0,
                task_spec=task_spec,
                retrieved_context=context,
                retrieved_lessons=retrieved_lessons,
                execution_time=total_time,
                error_message=f"Coding agent error: {e}",
            )

        # 4. Self-Learning Memory Store Reflection
        _emit(StepStatus.LEARNING, "Evaluating attempt session for self-learning opportunities...", {
            "total_attempts": coding_result.total_attempts,
            "success": coding_result.success,
        })

        new_reflection: Optional[MemoryEntry] = None
        try:
            new_reflection = self.memory_store.record_attempt(
                task_spec=task_spec,
                attempt_result=coding_result,
                llm_client=self.llm_client,
            )
            if new_reflection:
                _emit(StepStatus.LEARNING, f"💡 Distilled and saved new reflection: {new_reflection.lesson}", {
                    "reflection": new_reflection.to_dict(),
                })
        except Exception as e:
            logger.warning(f"Memory recording failed: {e}")

        # 5. Finalize and Deliver
        total_time = time.perf_counter() - start_time

        error_message = None
        if not coding_result.success:
            if coding_result.final_eval_result:
                error_message = (
                    coding_result.final_eval_result.error_message
                    or coding_result.final_eval_result.stderr
                    or "Solution failed evaluation tests."
                )
            else:
                error_message = "Failed to produce a passing solution within retry budget."

            _emit(StepStatus.FAILED, f"❌ Session finished with failure after {coding_result.total_attempts} attempts.", {
                "error_message": error_message,
            })
        else:
            _emit(StepStatus.COMPLETED, f"✅ Solution passed in {coding_result.total_attempts} attempt(s)!", {
                "total_attempts": coding_result.total_attempts,
                "execution_time": total_time,
            })

        return OrchestratorResult(
            task_id=task_id,
            entry_point=target_entry_point,
            success=coding_result.success,
            final_code=coding_result.final_code,
            total_attempts=coding_result.total_attempts,
            attempts=coding_result.attempts,
            task_spec=task_spec,
            retrieved_context=context,
            new_reflection=new_reflection,
            retrieved_lessons=retrieved_lessons,
            execution_time=total_time,
            error_message=error_message,
            explanation=f"Solved `{target_entry_point}` in {coding_result.total_attempts} attempt(s)." if coding_result.success else None,
        )

    def _enrich_vague_prompt(
        self,
        prompt: str,
        on_emit: Optional[Callable] = None,
    ) -> str:
        """
        Bug Fix 4 — Prompt Enrichment.

        Detects vague or short prompts (fewer than 15 words, no `def`, no docstring markers)
        and calls the LLM to expand them into a complete Python function specification with:
          - A properly typed function signature
          - A clear docstring
          - At least 3 doctest examples (>>> call / expected output)

        This prevents `TaskUnderstanding` from producing an empty-argument `TaskSpecification`,
        which causes the retriever to emit a wrong zero-arg signature and the coding LLM to
        generate an incompatible function stub that fails all test assertions.
        """
        word_count = len(prompt.strip().split())
        has_def = "def " in prompt
        has_docstring = '"""' in prompt or "'''" in prompt or ">>>" in prompt

        # Rich prompt — no enrichment needed
        if word_count > 20 or has_def or has_docstring:
            return prompt

        if on_emit:
            on_emit(
                StepStatus.UNDERSTANDING,
                f"⚠️  Prompt is very short ({word_count} word(s)). "
                "Enriching into a full Python function specification via LLM...",
            )

        try:
            system_prompt = (
                "You are a Python specification writer. "
                "Given a brief function name or description, produce a complete Python function stub "
                "with: a properly typed signature, a detailed docstring, and at least 3 >>> doctest "
                "examples showing input → output. "
                "Return ONLY the stub (body = `pass`) inside a ```python ... ``` code block. "
                "Do NOT implement the body."
            )
            user_prompt = (
                f"Write a complete Python function specification for: {prompt.strip()}\n\n"
                "Include:\n"
                "- Typed parameters (use Python type hints)\n"
                "- Return type annotation\n"
                "- Clear docstring explaining what the function does\n"
                "- At least 3 doctest examples with expected output\n"
                "- `pass` as the function body\n"
            )
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
            )
            pattern = r"```(?:python[0-9]?|py)?\s*\n?(.*?)\s*```"
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            valid: list[str] = [m.strip() for m in matches if m.strip()]
            if valid:
                enriched = max(valid, key=len)
                logger.info(
                    f"Prompt enriched: '{prompt.strip()[:50]}' "
                    f"→ {len(enriched)}-char spec with signature and examples."
                )
                if on_emit:
                    on_emit(
                        StepStatus.UNDERSTANDING,
                        f"✅ Prompt expanded into full function spec ({len(enriched)} chars). "
                        "Proceeding with enriched specification.",
                    )
                return enriched
        except Exception as exc:
            logger.warning(f"Prompt enrichment failed: {exc}. Using original prompt.")

        return prompt

    def run_task(
        self,
        task_dict: Dict[str, Any],
        max_retries: Optional[int] = None,
        on_event: Optional[Callable[[OrchestratorEvent], None]] = None,
    ) -> OrchestratorResult:
        """Convenience method to execute a benchmark task dictionary directly."""
        return self.run(
            task_input=task_dict,
            max_retries=max_retries,
            on_event=on_event,
        )
