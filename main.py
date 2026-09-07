"""
main.py

Interactive CLI and Command-Line Interface for the Self-Learning Coding Agent.
Provides interactive prompts, benchmark task execution, memory inspection,
and real-time live telemetry for the autonomous coding lifecycle.
"""

from __future__ import annotations

import os
import sys
import io
import json
import argparse
from typing import Optional, Dict, Any

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import box

from agent.config import get_config, AgentConfig
from agent.orchestrator import Orchestrator, OrchestratorResult, OrchestratorEvent, StepStatus
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


def load_benchmark_problem(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads a problem from selected_problems.jsonl or agent_tasks.jsonl + eval_data.jsonl.
    """
    normalized_id = task_id.strip()
    if normalized_id.isdigit():
        normalized_id = f"HumanEval/{normalized_id}"

    # Priority 1: Check selected_problems.jsonl
    if os.path.exists("selected_problems.jsonl"):
        try:
            with open("selected_problems.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("task_id") == normalized_id:
                            return data
        except Exception as e:
            console.print(f"[yellow]Warning reading selected_problems.jsonl: {e}[/yellow]")

    # Priority 2: Combine agent_tasks.jsonl and eval_data.jsonl
    task_data = None
    if os.path.exists("agent_tasks.jsonl"):
        try:
            with open("agent_tasks.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("task_id") == normalized_id:
                            task_data = data
                            break
        except Exception:
            pass

    if task_data and os.path.exists("eval_data.jsonl"):
        try:
            with open("eval_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("task_id") == normalized_id:
                            task_data["test"] = data.get("test")
                            break
        except Exception:
            pass

    return task_data


def display_banner(config: AgentConfig, memory_store: MemoryStore) -> None:
    """Renders a visual header with system statistics."""
    lessons_count = len(memory_store.entries)
    banner_content = (
        f"[bold cyan]Self-Learning Coding Agent[/bold cyan] [dim]|[/dim] "
        f"[green]Model:[/green] [white]{config.model}[/white] [dim]|[/dim] "
        f"[green]Memory:[/green] [white]{lessons_count} lesson(s)[/white] [dim]|[/dim] "
        f"[green]Retries:[/green] [white]{config.max_retries}[/white] [dim]|[/dim] "
        f"[green]Sandbox Timeout:[/green] [white]{config.sandbox_timeout}s[/white]"
    )
    console.print(Panel(banner_content, border_style="cyan", box=box.ROUNDED))


def display_memory_stats(memory_store: MemoryStore) -> None:
    """Displays learned reflections stored in memory."""
    entries = memory_store.entries
    if not entries:
        console.print(Panel("[italic yellow]No memory reflections recorded yet.[/italic yellow]", title="Memory Store"))
        return

    table = Table(title=f"Self-Learning Memory Store ({len(entries)} entries)", box=box.ROUNDED, expand=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Task / Target", style="cyan", width=22)
    table.add_column("Error Type", style="red", width=18)
    table.add_column("Actionable Lesson / Rule", style="green")
    table.add_column("Used", style="magenta", justify="right", width=6)

    for idx, entry in enumerate(entries, start=1):
        lesson_text = entry.lesson or entry.reflection or "N/A"
        table.add_row(
            str(idx),
            entry.task_tag or entry.task_id or "general",
            entry.error_type or "Unknown",
            lesson_text,
            str(entry.usage_count),
        )

    console.print(table)


def create_event_listener():
    """Creates a real-time event logger for the terminal."""
    icons = {
        StepStatus.START: "🚀",
        StepStatus.UNDERSTANDING: "🧠",
        StepStatus.RETRIEVING: "🔍",
        StepStatus.CODING: "💻",
        StepStatus.EVALUATING: "🧪",
        StepStatus.SELF_CORRECTING: "🛠️",
        StepStatus.LEARNING: "📚",
        StepStatus.COMPLETED: "✅",
        StepStatus.FAILED: "❌",
    }

    colors = {
        StepStatus.START: "cyan",
        StepStatus.UNDERSTANDING: "magenta",
        StepStatus.RETRIEVING: "blue",
        StepStatus.CODING: "yellow",
        StepStatus.EVALUATING: "cyan",
        StepStatus.SELF_CORRECTING: "yellow",
        StepStatus.LEARNING: "green",
        StepStatus.COMPLETED: "bold green",
        StepStatus.FAILED: "bold red",
    }

    def on_event(event: OrchestratorEvent) -> None:
        icon = icons.get(event.step, "🔹")
        color = colors.get(event.step, "white")
        console.print(f"[{color}]{icon} [{event.step.value.upper()}][/{color}] {event.message}")

    return on_event


def display_result(result: OrchestratorResult) -> None:
    """Renders the outcome of an orchestration run."""
    console.print()

    # 1. Status Panel
    status_style = "bold green" if result.success else "bold red"
    status_icon = "✅ SUCCESS" if result.success else "❌ FAILED"
    time_str = f"{result.execution_time:.2f}s"
    
    status_text = (
        f"[bold]Target:[/bold] [cyan]{result.entry_point}[/cyan]  "
        f"[bold]Status:[/bold] [{status_style}]{status_icon}[/{status_style}]  "
        f"[bold]Attempts:[/bold] [yellow]{result.total_attempts}[/yellow]  "
        f"[bold]Time:[/bold] [white]{time_str}[/white]"
    )
    if result.task_id:
        status_text = f"[bold]Task ID:[/bold] [blue]{result.task_id}[/blue] | " + status_text

    console.print(Panel(status_text, title="Execution Summary", border_style="cyan", box=box.ROUNDED))

    # 2. Injected Memory Lessons
    if result.retrieved_lessons:
        rules_text = "\n".join([f"• {rule}" for rule in result.retrieved_lessons])
        console.print(Panel(
            rules_text,
            title="🛡️ Historical Memory Rules Injected",
            border_style="blue",
            box=box.ROUNDED,
        ))

    # 3. Final Code
    code_to_show = (result.final_code or "").strip()
    if not code_to_show and result.attempts:
        for att in reversed(result.attempts):
            if att.code and att.code.strip():
                code_to_show = att.code.strip()
                break

    if code_to_show:
        syntax = Syntax(
            code_to_show,
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(Panel(
            syntax,
            title=f"Generated Solution (`{result.entry_point}`)",
            border_style="green" if result.success else "red",
            box=box.ROUNDED,
        ))
    else:
        console.print(Panel(
            "[dim italic]No code was extracted for this solution.[/dim italic]",
            title=f"Generated Solution (`{result.entry_point}`)",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    # 4. Attempt History Table
    if result.attempts:
        attempt_table = Table(title="Iteration Audit Log", box=box.SIMPLE, expand=True)
        attempt_table.add_column("Attempt", justify="center", width=8)
        attempt_table.add_column("Outcome", justify="center", width=12)
        attempt_table.add_column("Duration", justify="right", width=10)
        attempt_table.add_column("Diagnostic / Error Summary", style="dim")

        for att in result.attempts:
            outcome = "[green]PASS[/green]" if att.eval_result.passed else (
                "[red]TIMEOUT[/red]" if att.eval_result.timed_out else "[red]FAIL[/red]"
            )
            duration = f"{att.eval_result.execution_time:.2f}s" if att.eval_result.execution_time else "N/A"
            diag = att.error_summary or (att.eval_result.error_message or "Clean execution")
            attempt_table.add_row(str(att.attempt), outcome, duration, diag[:120])

        console.print(attempt_table)

    # 5. New Distilled Reflection
    if result.new_reflection:
        ref_text = (
            f"[bold]Task:[/bold] {result.new_reflection.task_tag}\n"
            f"[bold]Error Type:[/bold] [red]{result.new_reflection.error_type}[/red]\n"
            f"[bold]Lesson Learned:[/bold] [green]{result.new_reflection.lesson}[/green]"
        )
        console.print(Panel(
            ref_text,
            title="💡 Self-Learning: New Reflection Stored in Memory",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    # 6. Error details if failed
    if not result.success and result.error_message:
        console.print(Panel(
            f"[red]{result.error_message}[/red]",
            title="Failure Diagnostics",
            border_style="red",
            box=box.ROUNDED,
        ))


def run_interactive_mode(orchestrator: Orchestrator, config: AgentConfig) -> None:
    """Runs the interactive user menu."""
    while True:
        display_banner(config, orchestrator.memory_store)
        console.print("\n[bold]Select an action:[/bold]")
        console.print("  [1] 📝 Enter a custom Python coding prompt")
        console.print("  [2] ⚡ Run a HumanEval benchmark problem by ID")
        console.print("  [3] 🧠 View Self-Learning Memory rules")
        console.print("  [4] 🧹 Clear Memory Store")
        console.print("  [5] 🚪 Exit\n")

        choice = Prompt.ask("Enter choice", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            console.print("\n[bold cyan]Enter your coding task prompt (type 'END' on a new line or press Enter twice when finished):[/bold cyan]")
            lines = []
            empty_count = 0
            while True:
                try:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    if not line.strip():
                        empty_count += 1
                        if empty_count >= 2 and lines:
                            while lines and not lines[-1].strip():
                                lines.pop()
                            break
                    else:
                        empty_count = 0
                    lines.append(line)
                except EOFError:
                    break
            prompt_text = "\n".join(lines).strip()
            if not prompt_text:
                console.print("[yellow]Prompt was empty. Returning to menu.[/yellow]\n")
                continue

            console.print("\n[dim]Running autonomous coding workflow...[/dim]\n")
            result = orchestrator.run(
                task_input=prompt_text,
                on_event=create_event_listener(),
            )
            display_result(result)

        elif choice == "2":
            task_id = Prompt.ask("Enter HumanEval Task ID (e.g. '0' or 'HumanEval/0')").strip()
            problem = load_benchmark_problem(task_id)
            if not problem:
                console.print(f"[red]Error: Problem `{task_id}` not found in benchmark datasets.[/red]\n")
                continue

            console.print(f"\n[bold green]Loaded {problem.get('task_id')}: `{problem.get('entry_point')}`[/bold green]")
            result = orchestrator.run(
                task_input=problem,
                on_event=create_event_listener(),
            )
            display_result(result)

        elif choice == "3":
            display_memory_stats(orchestrator.memory_store)

        elif choice == "4":
            if Confirm.ask("[bold red]Are you sure you want to clear all memory reflections?[/bold red]"):
                orchestrator.memory_store.entries = []
                orchestrator.memory_store.save()
                console.print("[green]Memory store cleared successfully.[/green]\n")

        elif choice == "5":
            console.print("[cyan]Goodbye! Keep coding.[/cyan]")
            break

        console.print("\n" + "─" * 60 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Builds command line arguments parser."""
    parser = argparse.ArgumentParser(
        description="Self-Learning Coding Agent: Autonomous code generation, sandbox evaluation, and self-learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", type=str, help="Direct prompt or function specification to solve.")
    parser.add_argument("-t", "--task", type=str, help="HumanEval task ID to run (e.g. HumanEval/0 or 0).")
    parser.add_argument("-f", "--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--test", type=str, help="Custom unit test code or assertions.")
    parser.add_argument("--test-file", type=str, help="Path to a file containing unit tests.")
    parser.add_argument("-r", "--retries", type=int, help="Maximum number of self-correction attempts.")
    parser.add_argument("--memory-file", type=str, help="Path to memory storage JSON file.")
    parser.add_argument("--memory-stats", action="store_true", help="Display memory reflections and exit.")
    parser.add_argument("--clear-memory", action="store_true", help="Clear all stored memory reflections and exit.")
    parser.add_argument("--no-interactive", action="store_true", help="Do not enter interactive mode if no prompt is given.")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    config = get_config()
    if args.retries is not None:
        config.max_retries = args.retries
    if args.memory_file:
        config.memory_file = args.memory_file

    memory_store = MemoryStore(memory_file=config.memory_file)
    orchestrator = Orchestrator(config=config, memory_store=memory_store)

    # Command: Clear memory
    if args.clear_memory:
        memory_store.entries = []
        memory_store.save()
        console.print("[green]Memory store cleared successfully.[/green]")
        return 0

    # Command: Memory stats
    if args.memory_stats:
        display_memory_stats(memory_store)
        return 0

    # Command: Prompt from file
    prompt_text = args.prompt
    if args.file:
        if not os.path.exists(args.file):
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            return 1
        with open(args.file, "r", encoding="utf-8") as f:
            prompt_text = f.read()

    # Command: Test code from file
    test_code = args.test
    if args.test_file:
        if not os.path.exists(args.test_file):
            console.print(f"[red]Error: Test file not found: {args.test_file}[/red]")
            return 1
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_code = f.read()

    # Command: Run benchmark task
    if args.task:
        problem = load_benchmark_problem(args.task)
        if not problem:
            console.print(f"[red]Error: Problem `{args.task}` not found in benchmark datasets.[/red]")
            return 1
        if test_code:
            problem["test"] = test_code

        result = orchestrator.run(
            task_input=problem,
            max_retries=args.retries,
            on_event=create_event_listener(),
        )
        display_result(result)
        return 0 if result.success else 1

    # Command: Run direct prompt
    if prompt_text:
        result = orchestrator.run(
            task_input=prompt_text,
            test_code=test_code,
            max_retries=args.retries,
            on_event=create_event_listener(),
        )
        display_result(result)
        return 0 if result.success else 1

    # If no task or prompt provided, launch interactive mode
    if args.no_interactive:
        parser.print_help()
        return 0

    try:
        run_interactive_mode(orchestrator, config)
        return 0
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]Session terminated by user.[/cyan]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
