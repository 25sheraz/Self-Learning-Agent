"""
agent/task_understanding.py

Parser and extractor for coding task specifications.
Uses Python's native AST and doctest parsing to reliably extract:
- Target function entry point
- Argument names and type annotations
- Return types
- Docstrings, natural language goals, and doctest examples
- Constraints and edge case clues
Provides fallback heuristic/LLM parsing for unstructured prompts.
"""

import ast
import doctest
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .config import AgentConfig, get_config
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TaskSpecification:
    """Structured representation of a programming task contract."""
    raw_prompt: str
    entry_point: str
    docstring: str = ""
    arguments: List[Dict[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    goal_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts task specification to a dictionary."""
        return {
            "entry_point": self.entry_point,
            "arguments": self.arguments,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "examples": self.examples,
            "constraints": self.constraints,
            "goal_summary": self.goal_summary,
        }


class TaskUnderstanding:
    """
    Analyzes user prompts and benchmark tasks into structured specifications.
    Prioritizes deterministic AST analysis over LLM token consumption.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client

    def parse(
        self,
        prompt: str,
        entry_point: Optional[str] = None,
    ) -> TaskSpecification:
        """
        Parses a code prompt or user instruction into a TaskSpecification.
        """
        prompt_clean = prompt.strip()

        # 1. Attempt AST-based parsing if prompt contains Python code
        spec = self._parse_ast(prompt_clean, entry_point)
        if spec is not None:
            return spec

        # 2. Fallback for natural language / unstructured requests
        return self._parse_fallback(prompt_clean, entry_point)

    def _parse_ast(
        self,
        prompt: str,
        entry_point: Optional[str] = None,
    ) -> Optional[TaskSpecification]:
        """
        Tries to parse the prompt as Python code using AST.
        """
        try:
            tree = ast.parse(prompt)
        except SyntaxError:
            # Not valid standalone Python code; might be an incomplete def or markdown
            # Try stripping markdown fences if present
            code_block = self._extract_code_from_markdown(prompt)
            if code_block != prompt:
                try:
                    tree = ast.parse(code_block)
                except SyntaxError:
                    return None
            else:
                return None

        # Collect all function definitions
        func_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_nodes:
            return None

        # Find the target function definition matching entry_point if provided
        target_fn = None
        if entry_point:
            for fn in func_nodes:
                if fn.name == entry_point:
                    target_fn = fn
                    break

        # If not matched or entry_point not specified, pick the last defined function
        if target_fn is None:
            target_fn = func_nodes[-1]

        target_name = target_fn.name
        docstring = ast.get_docstring(target_fn) or ""

        # Extract arguments and types
        args_list: List[Dict[str, str]] = []
        for arg in target_fn.args.args:
            if arg.arg in ("self", "cls"):
                continue
            ann_str = ast.unparse(arg.annotation) if arg.annotation is not None else "Any"
            args_list.append({"name": arg.arg, "type": ann_str})

        # Extract return type
        return_type_str: Optional[str] = None
        if target_fn.returns is not None:
            return_type_str = ast.unparse(target_fn.returns)

        # Extract doctest examples
        examples = self._extract_doctests(docstring)

        # Extract constraints and edge cases
        constraints = self._extract_constraints(docstring)

        # Summarize goal (first line or sentence of docstring)
        goal_summary = self._extract_goal_summary(docstring, target_name)

        return TaskSpecification(
            raw_prompt=prompt,
            entry_point=target_name,
            docstring=docstring,
            arguments=args_list,
            return_type=return_type_str,
            examples=examples,
            constraints=constraints,
            goal_summary=goal_summary,
        )

    def _parse_fallback(
        self,
        prompt: str,
        entry_point: Optional[str] = None,
    ) -> TaskSpecification:
        """
        Parses unstructured natural language prompts into a basic TaskSpecification.
        """
        detected_name = entry_point or self._heuristic_entry_point(prompt)
        constraints = self._extract_constraints(prompt)

        # If LLM client is available and entry_point is ambiguous, ask LLM
        if (not detected_name or detected_name == "solve") and self.llm_client and self.llm_client.is_configured:
            try:
                llm_spec = self._parse_with_llm(prompt)
                if llm_spec:
                    return llm_spec
            except Exception as e:
                logger.warning(f"LLM fallback parsing failed: {e}. Using heuristics.")

        return TaskSpecification(
            raw_prompt=prompt,
            entry_point=detected_name or "solution",
            docstring=prompt,
            arguments=[],
            return_type=None,
            examples=[],
            constraints=constraints,
            goal_summary=prompt.split("\n")[0].strip(),
        )

    def _parse_with_llm(self, prompt: str) -> Optional[TaskSpecification]:
        """
        Lightweight LLM call to extract entry point and signatures from plain English.
        """
        if not self.llm_client:
            return None

        sys_msg = (
            "You are a code analyzer. Given a natural language request, extract: "
            "1. entry_point (function name, python identifier)\n"
            "2. arguments in format: name: type\n"
            "3. return_type: expected type\n"
            "Respond in format:\n"
            "ENTRY_POINT: <name>\n"
            "ARGS: <arg1: type, arg2: type>\n"
            "RETURN: <type>"
        )

        response = self.llm_client.generate(prompt=prompt, system_prompt=sys_msg, temperature=self.config.temperature)
        lines = response.splitlines()

        entry_point = "solution"
        args: List[Dict[str, str]] = []
        return_type: Optional[str] = None

        for line in lines:
            if line.startswith("ENTRY_POINT:"):
                entry_point = line.replace("ENTRY_POINT:", "").strip()
            elif line.startswith("ARGS:"):
                raw_args = line.replace("ARGS:", "").strip()
                if raw_args and raw_args.lower() != "none":
                    for part in raw_args.split(","):
                        if ":" in part:
                            pname, ptype = part.split(":", 1)
                            args.append({"name": pname.strip(), "type": ptype.strip()})
                        else:
                            args.append({"name": part.strip(), "type": "Any"})
            elif line.startswith("RETURN:"):
                return_type = line.replace("RETURN:", "").strip()

        return TaskSpecification(
            raw_prompt=prompt,
            entry_point=entry_point,
            docstring=prompt,
            arguments=args,
            return_type=return_type,
            examples=[],
            constraints=self._extract_constraints(prompt),
            goal_summary=prompt.splitlines()[0].strip() if prompt else "",
        )

    @staticmethod
    def _extract_doctests(docstring: str) -> List[Dict[str, str]]:
        """
        Extracts test case pairs from docstring doctests (>>> ...).
        """
        if ">>>" not in docstring:
            return []

        examples: List[Dict[str, str]] = []
        try:
            parser = doctest.DocTestParser()
            for ex in parser.get_examples(docstring):
                call = ex.source.strip()
                expected = ex.want.strip()
                if call:
                    examples.append({"call": call, "expected": expected})
        except Exception as e:
            logger.debug(f"Doctest extraction encountered an issue: {e}")

        return examples

    @staticmethod
    def _extract_constraints(text: str) -> List[str]:
        """
        Extracts phrases or sentences that indicate boundary conditions or constraints.
        """
        constraint_indicators = [
            "empty",
            "assume",
            "must",
            "return none if",
            "return false if",
            "return true if",
            "between",
            "positive",
            "negative",
            "sorted",
            "duplicate",
            "order of",
            "time complexity",
            "space complexity",
            "case sensitive",
            "ignore",
        ]
        constraints: List[str] = []
        # Split text into sentences
        sentences = re.split(r"[.\n]", text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or s_clean.startswith(">>>"):
                continue
            lower_s = s_clean.lower()
            if any(ind in lower_s for ind in constraint_indicators):
                if len(s_clean) > 8 and s_clean not in constraints:
                    constraints.append(s_clean)

        return constraints

    @staticmethod
    def _extract_goal_summary(docstring: str, function_name: str) -> str:
        """
        Extracts a one-line goal description from the docstring.
        """
        if not docstring:
            return f"Implement {function_name}."

        lines = [line.strip() for line in docstring.splitlines() if line.strip() and not line.strip().startswith(">>>")]
        if lines:
            return lines[0]
        return f"Implement {function_name}."

    @staticmethod
    def _heuristic_entry_point(text: str) -> str:
        """
        Uses heuristics to find a target function name from natural language.
        """
        # First priority: check for explicit identifiers in backticks: `name`
        # Words that look like identifiers but are actually common English words
        _COMMON_WORDS = frozenset({
            "python", "function", "the", "a", "an", "given", "code",
            "def", "method", "class", "true", "false", "none", "str", "int",
            "list", "dict", "completely", "correctly", "properly", "fully",
            "according", "above", "below", "following", "specified", "target",
            "implement", "implementation", "return", "returns", "only",
            "executable", "complete", "specification", "contract",
        })
        backtick_matches = re.findall(r"[`']([a-zA-Z_]\w*)[`']", text)
        for candidate in backtick_matches:
            if candidate.lower() not in _COMMON_WORDS:
                return candidate

        # Second priority: look for phrases like function (called|named)? <name>
        patterns = [
            r"(?:function|method|def)(?:\s+(?:called|named))?\s+[`']?([a-zA-Z_]\w*)[`']?",
            r"[`']([a-zA-Z_]\w*)\s*\([^`']*?\)[`']",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1)
                if candidate.lower() not in (
                    "python", "function", "the", "a", "an", "given", "code",
                    "called", "named", "to", "that", "for",
                    "completely", "correctly", "properly", "fully", "according",
                    "above", "below", "following", "specified", "target",
                    "implement", "implementation", "return", "returns", "only",
                    "executable", "complete", "specification", "contract",
                ):
                    return candidate
        return "solve"

    @staticmethod
    def _extract_code_from_markdown(text: str) -> str:
        """
        Strips markdown code blocks if the prompt is wrapped in them.
        """
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(matches).strip()
        return text.strip()
