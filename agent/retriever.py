"""
agent/retriever.py

Context and Memory Retriever for the Self-Learning Coding Agent.
Discovers required imports, finds relevant workspace context,
and retrieves past reflections/mistakes from memory to enrich coding prompts.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, TYPE_CHECKING

from .config import AgentConfig, get_config
from .task_understanding import TaskSpecification

if TYPE_CHECKING:
    from .memory import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Enriched context assembled before code generation."""
    task_spec: TaskSpecification
    recommended_imports: List[str] = field(default_factory=list)
    workspace_snippets: List[Dict[str, str]] = field(default_factory=list)
    past_reflections: List[str] = field(default_factory=list)
    formatted_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts retrieved context to a dictionary."""
        return {
            "task_spec": self.task_spec.to_dict(),
            "recommended_imports": self.recommended_imports,
            "workspace_snippets": self.workspace_snippets,
            "past_reflections": self.past_reflections,
            "formatted_prompt": self.formatted_prompt,
        }


class ContextRetriever:
    """
    Assembles contextual knowledge for the Coding Agent:
    - Standard library imports based on type annotations and keywords.
    - Relevant workspace file context.
    - Historical error reflections from the self-learning memory store.
    """

    TYPING_SYMBOLS = {
        "List", "Dict", "Tuple", "Set", "Optional", "Union", "Any",
        "Callable", "Iterable", "Sequence", "Mapping", "Literal", "Generator"
    }

    STDLIB_INDICATORS = {
        "math": ["math", "sqrt", "floor", "ceil", "log", "pi", "gcd", "factorial"],
        "re": ["regex", "re.", "match", "findall", "sub", "pattern"],
        "collections": ["counter", "defaultdict", "deque", "namedtuple", "ordereddict"],
        "itertools": ["permutations", "combinations", "product", "accumulate", "groupby", "chain"],
        "heapq": ["heap", "heappush", "heappop", "heapify", "nlargest", "nsmallest"],
        "bisect": ["bisect", "bisect_left", "bisect_right", "insort"],
        "copy": ["deepcopy", "copy.deepcopy", "shallow copy"],
        "datetime": ["datetime", "timedelta", "timezone", "timestamp"],
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory_file: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.config = config or get_config()
        self.memory_file = memory_file or self.config.memory_file
        if memory_store is not None:
            self.memory_store = memory_store
        elif self.memory_file and os.path.exists(self.memory_file):
            from .memory import MemoryStore
            self.memory_store = MemoryStore(memory_file=self.memory_file, config=self.config)
        else:
            self.memory_store = None

    def retrieve(
        self,
        task_spec: TaskSpecification,
        workspace_dir: Optional[str] = None,
    ) -> RetrievedContext:
        """
        Retrieves all relevant dependencies, past memory reflections,
        and constructs the enriched prompt for the coding agent.
        """
        imports = self._deduce_imports(task_spec)
        past_lessons = self._retrieve_past_lessons(task_spec)
        workspace_snippets = self._retrieve_workspace_snippets(task_spec, workspace_dir)

        formatted_prompt = self._build_formatted_prompt(
            task_spec=task_spec,
            imports=imports,
            past_lessons=past_lessons,
            workspace_snippets=workspace_snippets,
        )

        return RetrievedContext(
            task_spec=task_spec,
            recommended_imports=imports,
            workspace_snippets=workspace_snippets,
            past_reflections=past_lessons,
            formatted_prompt=formatted_prompt,
        )

    def _deduce_imports(self, task_spec: TaskSpecification) -> List[str]:
        """
        Detects required typing imports and standard library modules.
        """
        discovered_imports: List[str] = []
        found_typing: Set[str] = set()

        # Check types in arguments and return type
        all_type_strings = [arg["type"] for arg in task_spec.arguments]
        if task_spec.return_type:
            all_type_strings.append(task_spec.return_type)

        combined_types = " ".join(all_type_strings)
        for sym in self.TYPING_SYMBOLS:
            if sym in combined_types or sym in task_spec.raw_prompt:
                found_typing.add(sym)

        if found_typing:
            sorted_typing = ", ".join(sorted(found_typing))
            discovered_imports.append(f"from typing import {sorted_typing}")

        # Check stdlib indicators in prompt and docstring
        search_text = (task_spec.raw_prompt + " " + task_spec.docstring).lower()

        if any(w in search_text for w in self.STDLIB_INDICATORS["math"]):
            discovered_imports.append("import math")
        if any(w in search_text for w in self.STDLIB_INDICATORS["re"]):
            discovered_imports.append("import re")
        if any(w in search_text for w in self.STDLIB_INDICATORS["collections"]):
            discovered_imports.append("from collections import defaultdict, Counter, deque")
        if any(w in search_text for w in self.STDLIB_INDICATORS["itertools"]):
            discovered_imports.append("import itertools")
        if any(w in search_text for w in self.STDLIB_INDICATORS["heapq"]):
            discovered_imports.append("import heapq")
        if any(w in search_text for w in self.STDLIB_INDICATORS["bisect"]):
            discovered_imports.append("import bisect")

        return discovered_imports

    def _retrieve_past_lessons(self, task_spec: TaskSpecification) -> List[str]:
        """
        Searches historical memory file or store for past reflections relevant to this task.
        """
        if self.memory_store is not None:
            lessons = self.memory_store.get_lessons(task_spec, top_k=5)
            if lessons:
                return lessons

        if not self.memory_file or not os.path.exists(self.memory_file):
            return []

        lessons: List[str] = []
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                memory_data = json.load(f)

            if isinstance(memory_data, list):
                keywords = set(
                    [task_spec.entry_point.lower()] +
                    [arg["name"].lower() for arg in task_spec.arguments] +
                    task_spec.goal_summary.lower().split()
                )
                for item in memory_data:
                    if isinstance(item, dict):
                        reflection = item.get("reflection") or item.get("lesson")
                        task_tag = item.get("task_tag", "").lower()
                        error_type = item.get("error_type", "").lower()
                        if reflection and (
                            any(kw in task_tag for kw in keywords if len(kw) > 3) or
                            any(kw in error_type for kw in keywords if len(kw) > 3)
                        ):
                            if reflection not in lessons:
                                lessons.append(reflection)
        except Exception as e:
            logger.debug(f"Failed to read memory file {self.memory_file}: {e}")

        return lessons

    def _retrieve_workspace_snippets(
        self,
        task_spec: TaskSpecification,
        workspace_dir: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Gathers workspace snippets if specific files or modules are referenced.
        """
        # Kept lightweight and non-blocking
        return []

    def _build_formatted_prompt(
        self,
        task_spec: TaskSpecification,
        imports: List[str],
        past_lessons: List[str],
        workspace_snippets: List[Dict[str, str]],
    ) -> str:
        """
        Constructs a clean, structured context string for the Coding Agent.
        """
        sections: List[str] = []

        # 1. Goal and Function Contract
        sections.append(f"### Goal\n{task_spec.goal_summary or 'Implement the requested function.'}")

        # Bug Fix 3: Only emit an explicit signature when we have *known* argument info.
        # For vague prompts (empty arguments, no `def` in raw text), skipping the signature
        # prevents the coding LLM from being locked into a wrong zero-argument stub
        # (e.g. `def completely():` with 0 args when the real function needs multiple params).
        has_known_signature = (
            bool(task_spec.arguments)
            or bool(task_spec.return_type)
            or "def " in task_spec.raw_prompt
        )
        if has_known_signature:
            args_repr = ", ".join([f"{a['name']}: {a['type']}" for a in task_spec.arguments])
            ret_repr = f" -> {task_spec.return_type}" if task_spec.return_type else ""
            signature_line = f"def {task_spec.entry_point}({args_repr}){ret_repr}:"
            sections.append(f"### Target Function Signature\n```python\n{signature_line}\n```")

        # 2. Docstring & Instructions
        if task_spec.docstring:
            sections.append(f"### Specifications & Docstring\n{task_spec.docstring}")

        # 3. Examples
        if task_spec.examples:
            ex_lines = []
            for ex in task_spec.examples:
                ex_lines.append(f"- Input: `{ex['call']}` -> Expected: `{ex['expected']}`")
            sections.append("### Verified Examples\n" + "\n".join(ex_lines))

        # 4. Constraints
        if task_spec.constraints:
            c_lines = [f"- {c}" for c in task_spec.constraints]
            sections.append("### Identified Constraints & Edge Cases\n" + "\n".join(c_lines))

        # 5. Recommended Imports
        if imports:
            sections.append("### Recommended Imports\n```python\n" + "\n".join(imports) + "\n```")

        # 6. Past Learned Lessons (Crucial for Self-Learning!)
        if past_lessons:
            lesson_lines = [f"⚠️ Lesson from past experience: {lesson}" for lesson in past_lessons]
            sections.append("### Historical Mistakes to Avoid\n" + "\n".join(lesson_lines))

        # 7. Full Original Prompt — preserves helper functions (e.g. `poly` in HumanEval/32)
        #    that the structured sections above would otherwise silently drop.
        if task_spec.raw_prompt and task_spec.raw_prompt.strip():
            sections.append(
                f"### Full Original Problem (include ALL functions defined here)\n"
                f"```python\n{task_spec.raw_prompt.strip()}\n```"
            )

        return "\n\n".join(sections)
