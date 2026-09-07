"""
agent/memory.py

Self-Learning Memory Store for the Coding Agent.
Captures failure reflections and self-correction lessons, persists them in a JSON store,
and retrieves actionable rules to inject into future coding tasks.
"""

from __future__ import annotations

import os
import json
import uuid
import re
import logging
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Set, TYPE_CHECKING

from .config import AgentConfig, get_config
from .task_understanding import TaskSpecification

if TYPE_CHECKING:
    from .coding_agent import CodingAttemptResult, AttemptLog
    from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A learned reflection or rule captured from self-correction experience."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: Optional[str] = None
    task_tag: str = ""
    error_type: str = ""
    failed_pattern: str = ""
    fix_pattern: str = ""
    lesson: str = ""
    reflection: str = ""  # Mirror of lesson for backwards compatibility
    keywords: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success_after_fix: bool = True
    usage_count: int = 0

    def __post_init__(self):
        # Synchronize reflection and lesson
        if not self.reflection and self.lesson:
            self.reflection = self.lesson
        elif not self.lesson and self.reflection:
            self.lesson = self.reflection

    def to_dict(self) -> Dict[str, Any]:
        """Converts memory entry to a serializable dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_tag": self.task_tag,
            "error_type": self.error_type,
            "failed_pattern": self.failed_pattern,
            "fix_pattern": self.fix_pattern,
            "lesson": self.lesson,
            "reflection": self.reflection,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
            "success_after_fix": self.success_after_fix,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Reconstructs MemoryEntry from a dictionary."""
        lesson = data.get("lesson") or data.get("reflection") or ""
        reflection = data.get("reflection") or lesson
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            task_id=data.get("task_id"),
            task_tag=data.get("task_tag", ""),
            error_type=data.get("error_type", ""),
            failed_pattern=data.get("failed_pattern", ""),
            fix_pattern=data.get("fix_pattern", ""),
            lesson=lesson,
            reflection=reflection,
            keywords=data.get("keywords", []),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            success_after_fix=data.get("success_after_fix", True),
            usage_count=data.get("usage_count", 0),
        )


class MemoryStore:
    """
    Persistent self-learning memory repository.
    Stores lessons distilled from agent errors and fixes, and retrieves them for new tasks.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were",
        "it", "this", "that", "be", "have", "has", "do", "does", "def"
    }

    def __init__(
        self,
        memory_file: Optional[str] = None,
        max_entries: int = 500,
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or get_config()
        self.memory_file = memory_file or self.config.memory_file
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []

        self.load()

    def add(self, entry: MemoryEntry) -> bool:
        """
        Adds a memory entry to the store with deduplication.
        Returns True if added, False if duplicate.
        """
        # Deduplication check: same tag, error_type, and identical/near-identical lesson
        normalized_lesson = entry.lesson.strip().lower()
        for existing in self.entries:
            if (
                existing.task_tag.lower() == entry.task_tag.lower()
                and existing.error_type.lower() == entry.error_type.lower()
                and (
                    existing.lesson.strip().lower() == normalized_lesson
                    or (normalized_lesson in existing.lesson.strip().lower())
                )
            ):
                logger.debug(f"Skipping duplicate memory entry for `{entry.task_tag}`")
                return False

        self.entries.append(entry)

        # Capacity pruning: keep most recent entries if limit exceeded
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        self.save()
        logger.info(f"Learned new lesson stored in memory: [{entry.error_type}] {entry.lesson[:80]}...")
        return True

    def search(
        self,
        query: Union[str, TaskSpecification],
        top_k: int = 3,
        min_score: float = 1.0,
    ) -> List[MemoryEntry]:
        """
        Searches memory entries relevant to a task specification or query string.
        """
        if not self.entries:
            return []

        query_tokens, query_entry_point, query_error_type = self._extract_query_features(query)

        scored_entries: List[tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            score = 0.0

            # 1. Exact entry_point or task_tag match
            entry_tag = entry.task_tag.lower()
            if query_entry_point and (query_entry_point == entry_tag or query_entry_point in entry_tag):
                score += 10.0

            # 2. Error type match
            if query_error_type and entry.error_type and (query_error_type.lower() in entry.error_type.lower()):
                score += 5.0

            # 3. Keyword / Token overlap
            entry_tokens = {k.lower() for k in entry.keywords}.union(
                self._tokenize(entry.lesson),
                self._tokenize(entry.failed_pattern),
            )

            common_tokens = query_tokens.intersection(entry_tokens)
            score += len(common_tokens) * 2.0

            # Favor lessons proven successful after self-correction
            if entry.success_after_fix:
                score += 0.5

            if score >= min_score:
                scored_entries.append((score, entry))

        # Sort descending by score, then recency
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        selected = [entry for _, entry in scored_entries[:top_k]]
        for entry in selected:
            entry.usage_count += 1

        return selected

    def get_lessons(
        self,
        query: Union[str, TaskSpecification],
        top_k: int = 3,
    ) -> List[str]:
        """
        Returns relevant lesson strings formatted for prompt injection.
        """
        relevant_entries = self.search(query, top_k=top_k)
        lessons: List[str] = []
        for entry in relevant_entries:
            lesson_text = entry.lesson or entry.reflection
            if lesson_text and lesson_text not in lessons:
                lessons.append(lesson_text)
        return lessons

    def create_reflection(
        self,
        task_spec: TaskSpecification,
        failed_attempt: AttemptLog,
        successful_attempt: Optional[AttemptLog] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> MemoryEntry:
        """
        Extracts a reflection and actionable lesson from a failed attempt and its fix.
        Uses LLM distillation if available; falls back to heuristic generation.
        """
        error_type = self._deduce_error_type(failed_attempt)
        failed_snippet = self._summarize_failure(failed_attempt)
        fix_snippet = "Applied targeted correction to resolve error." if successful_attempt else "Unresolved."

        lesson = ""
        # Try LLM distillation if client provided
        if llm_client is not None:
            try:
                lesson = self._distill_reflection_with_llm(
                    llm_client=llm_client,
                    task_spec=task_spec,
                    failed_attempt=failed_attempt,
                    successful_attempt=successful_attempt,
                    error_type=error_type,
                )
            except Exception as e:
                logger.warning(f"LLM reflection distillation failed: {e}. Falling back to heuristic.")

        # Fallback to heuristic distillation if LLM not used or failed
        if not lesson:
            lesson = self._distill_heuristic_reflection(
                task_spec=task_spec,
                failed_attempt=failed_attempt,
                error_type=error_type,
            )

        keywords = self._extract_keywords(task_spec, error_type)

        return MemoryEntry(
            task_id=task_spec.entry_point,
            task_tag=task_spec.entry_point,
            error_type=error_type,
            failed_pattern=failed_snippet,
            fix_pattern=fix_snippet,
            lesson=lesson,
            reflection=lesson,
            keywords=keywords,
            success_after_fix=(successful_attempt is not None),
        )

    def record_attempt(
        self,
        task_spec: TaskSpecification,
        attempt_result: CodingAttemptResult,
        llm_client: Optional[LLMClient] = None,
    ) -> Optional[MemoryEntry]:
        """
        Analyzes a completed CodingAttemptResult. If self-correction occurred,
        distills and stores the lesson in memory.
        """
        # Case 1: Agent recovered from failures (prime learning event)
        if attempt_result.total_attempts > 1 and attempt_result.success:
            first_failure = next((a for a in attempt_result.attempts if not a.eval_result.passed), None)
            successful_attempt = attempt_result.attempts[-1]

            if first_failure:
                entry = self.create_reflection(
                    task_spec=task_spec,
                    failed_attempt=first_failure,
                    successful_attempt=successful_attempt,
                    llm_client=llm_client,
                )
                self.add(entry)
                return entry

        # Case 2: Exhausted retries without passing
        elif not attempt_result.success and attempt_result.attempts:
            last_failure = attempt_result.attempts[-1]
            entry = self.create_reflection(
                task_spec=task_spec,
                failed_attempt=last_failure,
                successful_attempt=None,
                llm_client=llm_client,
            )
            self.add(entry)
            return entry

        return None

    def save(self) -> None:
        """Atomically saves memory records to JSON file."""
        if not self.memory_file:
            return

        target_dir = os.path.dirname(os.path.abspath(self.memory_file))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        data = [entry.to_dict() for entry in self.entries]

        temp_path: Optional[str] = None
        # Write to temporary file in same directory first to ensure atomic replacement
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=target_dir or ".",
                prefix="mem_temp_",
                suffix=".json",
                text=True,
            )
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic file swap
            os.replace(temp_path, self.memory_file)
        except Exception as e:
            logger.error(f"Failed to persist memory to {self.memory_file}: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def load(self) -> None:
        """Loads memory records from JSON file."""
        self.entries = []
        if not self.memory_file or not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, dict):
                        self.entries.append(MemoryEntry.from_dict(item))
            logger.debug(f"Loaded {len(self.entries)} entries from memory store: {self.memory_file}")
        except Exception as e:
            logger.warning(f"Error reading memory file {self.memory_file}: {e}. Starting with fresh memory.")

    def clear(self) -> None:
        """Clears in-memory entries and resets persistent file."""
        self.entries = []
        self.save()

    # --- Internal Helpers ---

    def _tokenize(self, text: str) -> Set[str]:
        """Extracts normalized, non-stopword tokens from text."""
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def _extract_query_features(self, query: Union[str, TaskSpecification]) -> tuple[Set[str], str, str]:
        """Extracts tokens, entry_point, and error hints from query."""
        if isinstance(query, TaskSpecification):
            entry_point = query.entry_point.lower()
            combined_text = f"{query.entry_point} {query.goal_summary} {query.docstring} {' '.join([a['name'] for a in query.arguments])}"
            tokens = self._tokenize(combined_text)
            return tokens, entry_point, ""
        else:
            text = query.lower()
            tokens = self._tokenize(text)
            # Check for error name indicators in string query
            error_hint = ""
            for err in ["assertionerror", "indexerror", "typeerror", "valueerror", "zerodivisionerror", "timeouterror", "syntaxerror"]:
                if err in text:
                    error_hint = err
                    break
            return tokens, "", error_hint

    def _deduce_error_type(self, failed_attempt: AttemptLog) -> str:
        """Identifies standard Python exception type from attempt diagnostics."""
        res = failed_attempt.eval_result
        if res.timed_out:
            return "TimeoutError"
        if res.error_type:
            return res.error_type

        # Look in stderr and exception summary
        full_err = f"{failed_attempt.error_summary or ''} {res.stderr or ''}"
        match = re.search(r"([A-Z][a-zA-Z0-9_]*Error|[A-Z][a-zA-Z0-9_]*Exception):", full_err)
        if match:
            return match.group(1)

        return "AssertionError" if "assertion" in full_err.lower() else "RuntimeError"

    def _summarize_failure(self, failed_attempt: AttemptLog) -> str:
        """Extracts a short diagnostic snippet of what failed."""
        if failed_attempt.error_summary:
            return failed_attempt.error_summary.strip()
        if failed_attempt.eval_result.stderr:
            lines = [l.strip() for l in failed_attempt.eval_result.stderr.splitlines() if l.strip()]
            return lines[-1] if lines else "Test execution failed."
        return "Failed tests."

    def _extract_keywords(self, task_spec: TaskSpecification, error_type: str) -> List[str]:
        """Gathers keywords for indexing."""
        kw: Set[str] = {task_spec.entry_point.lower(), error_type.lower()}
        for arg in task_spec.arguments:
            kw.add(arg["name"].lower())
        kw.update(self._tokenize(task_spec.goal_summary))
        return sorted(list(kw))

    def _distill_heuristic_reflection(
        self,
        task_spec: TaskSpecification,
        failed_attempt: AttemptLog,
        error_type: str,
    ) -> str:
        """Creates a rule heuristic when LLM reflection is offline."""
        summary = self._summarize_failure(failed_attempt)

        if error_type == "TimeoutError":
            return f"For `{task_spec.entry_point}`: watch out for infinite loops or O(N^2) complexity; optimize recursion/iteration bounds."
        elif error_type == "ZeroDivisionError":
            return f"For `{task_spec.entry_point}`: guard against division by zero on empty inputs or zero denominators."
        elif error_type == "IndexError":
            return f"For `{task_spec.entry_point}`: verify boundary conditions and check list length before indexing elements."
        elif error_type == "TypeError":
            return f"For `{task_spec.entry_point}`: check type compatibility between operands and handle None or varied input types."
        elif error_type == "AssertionError":
            if "==" in summary:
                return f"For `{task_spec.entry_point}`: verify edge cases and return formatting to satisfy expected assertions: {summary[:90]}."
            return f"For `{task_spec.entry_point}`: handle boundary conditions carefully to satisfy test assertions."
        else:
            return f"For `{task_spec.entry_point}`: ensure robustness against {error_type} under edge case inputs."

    def _distill_reflection_with_llm(
        self,
        llm_client: LLMClient,
        task_spec: TaskSpecification,
        failed_attempt: AttemptLog,
        successful_attempt: Optional[AttemptLog],
        error_type: str,
    ) -> str:
        """Prompts the LLM to distill a concise, generalizable rule."""
        system_prompt = (
            "You are an expert AI code analyst. Your job is to extract a concise, "
            "actionable lesson (1-2 sentences maximum) from a code failure and fix so "
            "the agent avoids repeating the mistake in future tasks."
        )

        user_prompt = (
            f"### Target Function: `{task_spec.entry_point}`\n"
            f"Goal: {task_spec.goal_summary}\n\n"
            f"### Bug Diagnostics ({error_type}):\n"
            f"{self._summarize_failure(failed_attempt)}\n\n"
            f"### Flawed Implementation:\n"
            f"```python\n{failed_attempt.code}\n```\n\n"
        )

        if successful_attempt:
            user_prompt += (
                f"### Fixed Implementation (Passed):\n"
                f"```python\n{successful_attempt.code}\n```\n\n"
            )

        user_prompt += (
            "Formulate a 1-2 sentence actionable guideline starting with 'When implementing...' "
            "or 'Remember to...'. Do not output explanation or code blocks, only the guideline."
        )

        response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
        )

        clean_rule = response.strip().replace("\n", " ")
        # Clean quotes
        clean_rule = re.sub(r'^["\']|["\']$', '', clean_rule).strip()
        return clean_rule
