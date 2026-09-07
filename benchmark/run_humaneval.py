"""
benchmark/run_humaneval.py

HumanEval Benchmark Evaluation Harness for Self-Learning Coding Agent.
Evaluates agent code generation, test-driven self-correction recovery,
and the impact of persistent memory reflections across benchmark tasks.

Metrics Tracked:
  - Pass@1: Zero-shot pass rate (passed on attempt 1).
  - Pass@k / Self-Correction Recovery: Tasks that failed initial attempt but passed after repair.
  - Overall Pass Rate: Total tasks passed out of total evaluated.
  - Memory Learning Impact: Reflections created vs retrieved.
  - Latency & Attempt Distribution.

Usage:
    # Run first 5 problems
    python benchmark/run_humaneval.py --limit 5

    # Run specific range
    python benchmark/run_humaneval.py --range 0-10

    # Run specific tasks
    python benchmark/run_humaneval.py --tasks 0,1,2

    # Run with isolated memory or without memory
    python benchmark/run_humaneval.py --limit 5 --no-memory
    python benchmark/run_humaneval.py --limit 5 --isolated-memory
"""

from __future__ import annotations

import os
import sys
import io
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from agent.config import get_config, AgentConfig
from agent.orchestrator import Orchestrator, OrchestratorResult, StepStatus
from agent.memory import MemoryStore

# Force UTF-8 on standard streams to prevent Windows cp1252 charmap encoding errors
if sys.platform == "win32":
    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)


# ==============================================================================
# 1. Benchmark Data Structures
# ==============================================================================

@dataclass
class BenchmarkTaskResult:
    """Outcome record of a single benchmark task evaluation."""
    task_id: str
    entry_point: str
    success: bool
    pass_at_1: bool
    recovered: bool
    total_attempts: int
    execution_time: float
    error_message: Optional[str] = None
    retrieved_lessons_count: int = 0
    new_reflection_created: bool = False
    final_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "entry_point": self.entry_point,
            "success": self.success,
            "pass_at_1": self.pass_at_1,
            "recovered": self.recovered,
            "total_attempts": self.total_attempts,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "retrieved_lessons_count": self.retrieved_lessons_count,
            "new_reflection_created": self.new_reflection_created,
            "final_code": self.final_code,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark evaluation metrics."""
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    pass_at_1: int = 0
    recovered: int = 0
    overall_pass_rate: float = 0.0
    pass_at_1_rate: float = 0.0
    recovery_rate: float = 0.0
    total_attempts: int = 0
    avg_attempts_per_task: float = 0.0
    total_time: float = 0.0
    avg_time_per_task: float = 0.0
    reflections_retrieved_total: int = 0
    reflections_created_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "pass_at_1": self.pass_at_1,
            "recovered": self.recovered,
            "overall_pass_rate_pct": round(self.overall_pass_rate, 2),
            "pass_at_1_rate_pct": round(self.pass_at_1_rate, 2),
            "recovery_rate_pct": round(self.recovery_rate, 2),
            "total_attempts": self.total_attempts,
            "avg_attempts_per_task": round(self.avg_attempts_per_task, 2),
            "total_time_seconds": round(self.total_time, 2),
            "avg_time_per_task_seconds": round(self.avg_time_per_task, 2),
            "reflections_retrieved_total": self.reflections_retrieved_total,
            "reflections_created_total": self.reflections_created_total,
        }


def compute_benchmark_metrics(
    results: List[BenchmarkTaskResult],
    total_elapsed: float = 0.0,
) -> BenchmarkMetrics:
    """Computes aggregated metrics from individual task benchmark results."""
    total = len(results)
    if total == 0:
        return BenchmarkMetrics()

    passed = sum(1 for r in results if r.success)
    failed = total - passed
    pass_at_1 = sum(1 for r in results if r.pass_at_1)
    recovered = sum(1 for r in results if r.recovered)
    total_attempts = sum(r.total_attempts for r in results)
    calc_time = total_elapsed if total_elapsed > 0 else sum(r.execution_time for r in results)

    # Recovery rate is among tasks that failed initial attempt (total - pass_at_1)
    initial_failures = total - pass_at_1
    recovery_rate = (recovered / initial_failures * 100.0) if initial_failures > 0 else 0.0

    retrieved_total = sum(r.retrieved_lessons_count for r in results)
    created_total = sum(1 for r in results if r.new_reflection_created)

    return BenchmarkMetrics(
        total_tasks=total,
        passed=passed,
        failed=failed,
        pass_at_1=pass_at_1,
        recovered=recovered,
        overall_pass_rate=(passed / total * 100.0),
        pass_at_1_rate=(pass_at_1 / total * 100.0),
        recovery_rate=recovery_rate,
        total_attempts=total_attempts,
        avg_attempts_per_task=(total_attempts / total),
        total_time=calc_time,
        avg_time_per_task=(calc_time / total),
        reflections_retrieved_total=retrieved_total,
        reflections_created_total=created_total,
    )


# ==============================================================================
# 2. Problem Loading and Filtering Helpers
# ==============================================================================

def load_all_benchmark_problems(base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads all HumanEval benchmark problems from selected_problems.jsonl
    or by combining agent_tasks.jsonl and eval_data.jsonl.
    """
    base = base_dir or WORKSPACE_ROOT
    selected_path = os.path.join(base, "selected_problems.jsonl")

    if os.path.exists(selected_path):
        problems = []
        with open(selected_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        return _sort_problems(problems)

    # Fallback to merging agent_tasks.jsonl and eval_data.jsonl
    agent_path = os.path.join(base, "agent_tasks.jsonl")
    eval_path = os.path.join(base, "eval_data.jsonl")

    if os.path.exists(agent_path) and os.path.exists(eval_path):
        agent_map: Dict[str, Dict[str, Any]] = {}
        with open(agent_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    agent_map[item["task_id"]] = item

        combined = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_item = json.loads(line)
                    tid = eval_item["task_id"]
                    if tid in agent_map:
                        merged = dict(agent_map[tid])
                        merged.update(eval_item)
                        combined.append(merged)
        return _sort_problems(combined)

    return []


def _extract_task_num(task_id: str) -> int:
    """Helper to safely parse numeric index from 'HumanEval/X'."""
    try:
        parts = task_id.split("/")
        return int(parts[-1])
    except (ValueError, IndexError):
        return 999999


def _sort_problems(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts problem dictionaries by numeric task index."""
    return sorted(problems, key=lambda p: _extract_task_num(p.get("task_id", "")))


def filter_benchmark_problems(
    problems: List[Dict[str, Any]],
    limit: Optional[int] = None,
    task_range: Optional[Tuple[int, int]] = None,
    task_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filters problem list by task IDs, numeric range, or limit."""
    filtered = list(problems)

    # Filter 1: Specific task IDs
    if task_ids:
        normalized_ids = set()
        for tid in task_ids:
            clean_id = tid.strip()
            if clean_id.isdigit():
                clean_id = f"HumanEval/{clean_id}"
            normalized_ids.add(clean_id)
        filtered = [p for p in filtered if p.get("task_id") in normalized_ids]

    # Filter 2: Numeric range (e.g. 0 to 10)
    if task_range is not None:
        start_idx, end_idx = task_range
        filtered = [
            p for p in filtered
            if start_idx <= _extract_task_num(p.get("task_id", "")) <= end_idx
        ]

    # Filter 3: Slice limit
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    return filtered


def parse_range_arg(range_str: str) -> Tuple[int, int]:
    """Parses range strings such as '0-10' or '5..15'."""
    sep = "-" if "-" in range_str else ".."
    parts = range_str.split(sep)
    if len(parts) != 2:
        raise ValueError(f"Invalid range format '{range_str}'. Use START-END (e.g. 0-10).")
    start = int(parts[0].strip())
    end = int(parts[1].strip())
    if start > end:
        raise ValueError(f"Range start ({start}) cannot exceed range end ({end}).")
    return start, end


# ==============================================================================
# 3. Benchmark Runner Engine
# ==============================================================================

class HumanEvalBenchmarkRunner:
    """Executes benchmark tasks through the Orchestrator and calculates performance."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        orchestrator: Optional[Orchestrator] = None,
        memory_store: Optional[MemoryStore] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config or get_config()
        self.memory_store = memory_store or MemoryStore(memory_file=self.config.memory_file)
        self.orchestrator = orchestrator or Orchestrator(
            config=self.config,
            memory_store=self.memory_store,
        )
        self.output_dir = output_dir or os.path.join(WORKSPACE_ROOT, "benchmark", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_task(self, problem: Dict[str, Any], max_retries: Optional[int] = None) -> BenchmarkTaskResult:
        """Executes a single benchmark task and converts result into a BenchmarkTaskResult."""
        task_id = problem.get("task_id", "unknown")
        entry_point = problem.get("entry_point", "unknown")

        start_time = time.perf_counter()
        orch_res: OrchestratorResult = self.orchestrator.run(
            task_input=problem,
            task_id=task_id,
            entry_point=entry_point,
            max_retries=max_retries,
        )
        duration = time.perf_counter() - start_time

        # Analyze pass@1 vs self-correction recovery
        pass_at_1 = False
        recovered = False
        if orch_res.success:
            if orch_res.total_attempts == 1:
                pass_at_1 = True
            elif orch_res.total_attempts > 1:
                recovered = True

        return BenchmarkTaskResult(
            task_id=task_id,
            entry_point=entry_point,
            success=orch_res.success,
            pass_at_1=pass_at_1,
            recovered=recovered,
            total_attempts=orch_res.total_attempts,
            execution_time=duration,
            error_message=orch_res.error_message,
            retrieved_lessons_count=len(orch_res.retrieved_lessons),
            new_reflection_created=orch_res.new_reflection is not None,
            final_code=orch_res.final_code,
        )

    def run_benchmark(
        self,
        problems: List[Dict[str, Any]],
        max_retries: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[BenchmarkTaskResult], BenchmarkMetrics]:
        """Runs the entire benchmark suite over the provided list of problems."""
        results: List[BenchmarkTaskResult] = []
        total = len(problems)

        if total == 0:
            console.print("[yellow]No benchmark problems match the specified criteria.[/yellow]")
            return [], BenchmarkMetrics()

        if verbose:
            console.print(Panel(
                f"[bold cyan]HumanEval Benchmark Runner[/bold cyan]\n"
                f"Evaluating [bold green]{total}[/bold green] tasks with max retries: [bold]{max_retries or self.config.max_retries}[/bold]\n"
                f"Memory Store: [dim]{self.memory_store.memory_file}[/dim]",
                box=box.ROUNDED,
            ))

        start_all = time.perf_counter()

        # Execute tasks sequentially with live feedback
        for idx, problem in enumerate(problems, start=1):
            tid = problem.get("task_id", f"Task {idx}")
            ep = problem.get("entry_point", "")
            if verbose:
                console.print(f"[{idx}/{total}] Running [cyan]{tid}[/cyan] (`{ep}`)...", end=" ")

            task_res = self.run_task(problem, max_retries=max_retries)
            results.append(task_res)

            if verbose:
                if task_res.pass_at_1:
                    status_text = "[bold green]PASS@1[/bold green]"
                elif task_res.recovered:
                    status_text = f"[bold yellow]PASS (Recovered on attempt {task_res.total_attempts})[/bold yellow]"
                else:
                    status_text = f"[bold red]FAIL ({task_res.total_attempts} attempts)[/bold red]"

                lessons_str = f" [dim]({task_res.retrieved_lessons_count} lessons used)[/dim]" if task_res.retrieved_lessons_count else ""
                refl_str = " [magenta]+New Reflection[/magenta]" if task_res.new_reflection_created else ""
                console.print(f"{status_text} in {task_res.execution_time:.2f}s{lessons_str}{refl_str}")

        total_elapsed = time.perf_counter() - start_all
        metrics = compute_benchmark_metrics(results, total_elapsed=total_elapsed)

        return results, metrics

    def display_metrics_table(self, metrics: BenchmarkMetrics):
        """Renders stylized benchmark results table using Rich."""
        table = Table(title="🏆 HumanEval Benchmark Results Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan bold", justify="left")
        table.add_column("Value", style="green bold", justify="right")
        table.add_column("Details", style="dim", justify="left")

        table.add_row("Total Problems Evaluated", str(metrics.total_tasks), "Tasks evaluated")
        table.add_row("Overall Passed", f"{metrics.passed} ({metrics.overall_pass_rate:.1f}%)", "Total passed tasks")
        table.add_row("Pass@1 (Zero-Shot)", f"{metrics.pass_at_1} ({metrics.pass_at_1_rate:.1f}%)", "Passed without retries")
        table.add_row("Pass@k (Self-Corrected)", f"{metrics.recovered}", "Recovered after test failure")
        table.add_row("Self-Correction Recovery Rate", f"{metrics.recovery_rate:.1f}%", "% of initially failed tasks rescued")
        table.add_row("Failed Tasks", f"{metrics.failed} ({(metrics.failed / max(1, metrics.total_tasks) * 100):.1f}%)", "Exhausted retry budget")
        table.add_row("Average Attempts per Task", f"{metrics.avg_attempts_per_task:.2f}", "Total attempts / tasks")
        table.add_row("Total Benchmark Runtime", f"{metrics.total_time:.2f}s", f"Avg: {metrics.avg_time_per_task:.2f}s/task")
        table.add_row("Memory Lessons Retrieved", str(metrics.reflections_retrieved_total), "Historical tips injected")
        table.add_row("New Reflections Stored", str(metrics.reflections_created_total), "New lessons learned")

        console.print("\n")
        console.print(table)
        console.print("\n")

    def save_results(
        self,
        results: List[BenchmarkTaskResult],
        metrics: BenchmarkMetrics,
        output_file: Optional[str] = None,
        samples_file: Optional[str] = None,
    ) -> str:
        """Saves benchmark results to JSON and exports optional HumanEval samples."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file:
            final_path = output_file
        else:
            final_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")

        payload = {
            "timestamp": timestamp,
            "metrics": metrics.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Optional: Export HumanEval samples.jsonl format
        if samples_file:
            with open(samples_file, "w", encoding="utf-8") as f:
                for r in results:
                    sample_entry = {
                        "task_id": r.task_id,
                        "completion": r.final_code,
                    }
                    f.write(json.dumps(sample_entry) + "\n")

        return final_path


# ==============================================================================
# 4. CLI Argument Parser and Entry Point
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for HumanEval benchmark harness."""
    parser = argparse.ArgumentParser(
        description="Autonomous Self-Learning Coding Agent — HumanEval Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Problem Selection
    parser.add_argument("--limit", "-n", type=int, default=None, help="Evaluate first N problems.")
    parser.add_argument("--range", type=str, default=None, help="Evaluate numeric range (e.g. '0-10' or '5-25').")
    parser.add_argument("--tasks", "-t", type=str, default=None, help="Comma-separated task IDs (e.g. '0,1,2' or 'HumanEval/0').")
    parser.add_argument("--all", action="store_true", help="Evaluate all 164 HumanEval problems.")

    # Execution Options
    parser.add_argument("--retries", "-r", type=int, default=None, help="Maximum self-correction retries per task.")
    parser.add_argument("--timeout", type=float, default=None, help="Sandbox execution timeout in seconds.")

    # Memory Options
    parser.add_argument("--memory-file", type=str, default=None, help="Path to custom memory JSON file.")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory reflections retrieval & learning.")
    parser.add_argument("--isolated-memory", action="store_true", help="Use a temporary isolated memory file.")

    # Output Options
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write benchmark JSON report.")
    parser.add_argument("--samples-out", type=str, default=None, help="Path to write standard HumanEval samples.jsonl.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-task progress output.")

    return parser


def main() -> int:
    """Main CLI execution flow."""
    parser = build_parser()
    args = parser.parse_args()

    config = get_config()
    if args.retries is not None:
        config.max_retries = args.retries
    if args.timeout is not None:
        config.sandbox_timeout = args.timeout

    # Handle memory configurations
    if args.no_memory:
        # Dummy in-memory store that won't save
        memory_store = MemoryStore(memory_file="")
        memory_store.entries = []
    elif args.isolated_memory:
        iso_path = os.path.join(WORKSPACE_ROOT, "data", f"temp_memory_{int(time.time())}.json")
        memory_store = MemoryStore(memory_file=iso_path)
    elif args.memory_file:
        memory_store = MemoryStore(memory_file=args.memory_file)
    else:
        memory_store = MemoryStore(memory_file=config.memory_file)

    # Load dataset
    all_problems = load_all_benchmark_problems()
    if not all_problems:
        console.print("[red]Error: Could not load HumanEval dataset files. Run `python load_humaneval.py` first.[/red]")
        return 1

    # Filter tasks
    task_range = parse_range_arg(args.range) if args.range else None
    task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    # If neither limit, range, tasks, nor --all specified, default to --limit 5 to avoid running all 164 by accident
    limit = args.limit
    if not args.all and limit is None and task_range is None and task_ids is None:
        console.print("[yellow]Note: No task selection specified. Defaulting to first 5 problems (--limit 5). Use `--all` for all 164.[/yellow]\n")
        limit = 5

    selected = filter_benchmark_problems(
        all_problems,
        limit=limit,
        task_range=task_range,
        task_ids=task_ids,
    )

    runner = HumanEvalBenchmarkRunner(
        config=config,
        memory_store=memory_store,
    )

    results, metrics = runner.run_benchmark(
        selected,
        max_retries=args.retries,
        verbose=not args.quiet,
    )

    runner.display_metrics_table(metrics)

    output_file = runner.save_results(
        results,
        metrics,
        output_file=args.output,
        samples_file=args.samples_out,
    )
    console.print(f"[bold green]✓ Benchmark results saved to:[/bold green] [cyan]{output_file}[/cyan]")

    # Clean up isolated memory file if created
    if args.isolated_memory and os.path.exists(memory_store.memory_file):
        try:
            os.remove(memory_store.memory_file)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
