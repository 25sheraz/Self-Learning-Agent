# Self-Learning Coding Agent — Project Roadmap & Status

This document tracks the step-by-step development of the **Self-Learning Coding Agent** based on the architecture defined in [`workflow`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/workflow).

---

## 🗺️ Architecture Overview

```
                     USER
                       │
                       ↓
           ┌───────────────────────┐
           │     User Request      │
           │ (or Benchmark Task)   │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │  Task Understanding   │
           │ (Parse Intent & Types)│
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │    Code Retriever     │
           │  (Context & Memory)   │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │     ORCHESTRATOR      │
           │  (Plan-Execute-Loop)  │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │     CODING AGENT      │
           │ (Generate / Refine)   │
           └───────────┬───────────┘
                       ↓
                 Generated Code
                       │
                       ↓
           ┌───────────────────────┐
           │       EVALUATOR       │
           │ (Isolated Sandbox)    │
           └───────────┬───────────┘
                 PASS  │  FAIL
          ┌────────────┴────────────┐
          ↓                         ↓
   ┌──────────────┐          ┌──────────────┐
   │ Final Answer │          │ Coding Agent │
   │ + Explanation│          │  fixes code  │
   └──────────────┘          └──────┬───────┘
                                    │
                                    ↓
                             EVALUATE AGAIN
                             + Store Lesson
                             in Memory
```

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Progress |
| :---: | :--- | :---: | :---: |
| **Phase 0** | **Environment & Dataset Preparation** | ✅ Completed | 100% |
| **Phase 1** | **Foundation, Config & LLM Client (Kimi)** | ✅ Completed | 100% |
| **Phase 2** | **Evaluator & Sandbox Test Runner** | ✅ Completed | 100% |
| **Phase 3** | **Task Understanding & Context Retriever** | ✅ Completed | 100% |
| **Phase 4** | **Coding Agent & Self-Correction Engine** | ✅ Completed | 100% |
| **Phase 5** | **Self-Learning Memory Store** | ✅ Completed | 100% |
| **Phase 6** | **Master Orchestrator & CLI Interface** | ✅ Completed | 100% |
| **Phase 7** | **HumanEval Benchmark Harness** | ✅ Completed | 100% |

---

## 📋 Detailed Phase Breakdown

### Phase 0: Environment & Dataset Preparation ✅
- [x] Create Python virtual environment (`.venv`).
- [x] Clone and install OpenAI `human-eval` benchmark in editable mode.
- [x] Build and run [`load_humaneval.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/load_humaneval.py).
- [x] Generate isolated datasets without test leakage:
  - [`agent_tasks.jsonl`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent_tasks.jsonl): 164 agent tasks (`task_id`, `prompt`, `entry_point`).
  - [`eval_data.jsonl`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/eval_data.jsonl): 164 private evaluation test suites.
  - [`selected_problems.jsonl`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/selected_problems.jsonl): 164 full reference problems.

---

### Phase 1: Foundation, Config & LLM Client (Kimi) ✅
- [x] Create [`requirements.txt`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/requirements.txt) with core dependencies (`openai`, `python-dotenv`, `pydantic`, `pytest`, `rich`).
- [x] Create [`.env.example`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/.env.example) configured for Moonshot AI (Kimi API).
- [x] Create [`agent/config.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/config.py) with credential checks, timeouts, and model parameters.
- [x] Create [`agent/llm.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/llm.py) with OpenAI-compatible Kimi client wrapper.
- [x] Create [`tests/test_config_llm.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_config_llm.py) with automated test cases.

---

### Phase 2: Evaluator & Sandbox Test Runner ✅
- [x] Build [`agent/evaluator.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/evaluator.py):
  - Subprocess execution sandbox with strict execution timeout (prevent infinite loops).
  - Captures `stdout`, `stderr`, runtime exceptions, and assertion tracebacks.
  - Test runner for existing unit tests (HumanEval format).
  - LLM test generator fallback when user provides no unit tests.
- [x] Add unit tests in [`tests/test_evaluator.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_evaluator.py) to verify pass/fail detection.

---

### Phase 3: Task Understanding & Context Retriever ✅
- [x] Build [`agent/task_understanding.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/task_understanding.py):
  - Extracts entry point function, docstrings, input/output types, and edge case constraints.
- [x] Build [`agent/retriever.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/retriever.py):
  - Gathers workspace files, module dependencies, and queries memory for relevant past lessons.
- [x] Add unit tests in [`tests/test_task_understanding_retriever.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_task_understanding_retriever.py).

---

### Phase 4: Coding Agent & Self-Correction Engine ✅
- [x] Build [`agent/coding_agent.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/coding_agent.py):
  - Zero-shot code generator from prompt and context.
  - Patch & self-correction generator: ingests failed test logs + tracebacks to fix code without rewriting everything from scratch.
  - Retry budget controller (up to `max_retries`).
- [x] Add unit tests in [`tests/test_coding_agent.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_coding_agent.py).

---

### Phase 5: Self-Learning Memory Store ✅
- [x] Build [`agent/memory.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/memory.py):
  - Persistent JSON reflection database (`data/memory.json`).
  - Logs `(task_type, error_type, failed_pattern, fix_reflection)`.
  - Injects relevant historical rules into future prompts so the agent does not repeat mistakes.
- [x] Add unit tests in [`tests/test_memory.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_memory.py).

---

### Phase 6: Master Orchestrator & CLI Interface ✅
- [x] Build [`agent/orchestrator.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/agent/orchestrator.py):
  - Coordinates the full lifecycle: `Understand -> Retrieve -> Code -> Evaluate -> (Self-Correct Loop) -> Deliver`.
- [x] Build [`main.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/main.py):
  - User-friendly interactive command line tool to submit requests and watch agent thoughts live.
- [x] Add automated test suite in [`tests/test_orchestrator.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_orchestrator.py) with 11 comprehensive tests.

---

### Phase 7: HumanEval Benchmark Harness ✅
- [x] Build [`benchmark/run_humaneval.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/benchmark/run_humaneval.py):
  - Batch evaluation across problems (`--limit`, `--range`, `--tasks`, `--all`).
  - Calculates metrics:
    - Pass@1 (zero-shot)
    - Pass@k with Self-Correction (recovery rate)
    - Memory learning impact (reflections created vs retrieved)
    - Attempts, execution latency, and error breakdown.
  - Generates JSON reports and standard HumanEval `samples.jsonl` files.
- [x] Add automated test suite in [`tests/test_benchmark.py`](file:///c:/Users/Sheraz/Desktop/Self-Learning-Agent/tests/test_benchmark.py) with 10 comprehensive tests covering dataset parsing, metrics calculations, mock execution, and serialization.

---

## 📁 Target Project Structure

```
Self-Learning-Agent/
├── .venv/                         # Virtual environment
├── human-eval/                    # HumanEval submodule
├── agent_tasks.jsonl              # 164 problems (agent-facing)
├── eval_data.jsonl                # 164 problems (test suites)
├── selected_problems.jsonl        # 164 problems (full data)
├── load_humaneval.py              # Dataset loader
├── workflow                       # Original architecture blueprint
├── ROADMAP.md                     # This file (phase status & tracker)
├── requirements.txt               # Dependencies
├── .env.example                   # Config template
├── agent/
│   ├── __init__.py
│   ├── config.py                  # Settings & environment variables
│   ├── llm.py                     # LLM client wrapper
│   ├── task_understanding.py      # Prompt & signature parser
│   ├── retriever.py               # Context & memory retriever
│   ├── coding_agent.py            # Code generator & patcher
│   ├── evaluator.py               # Sandbox runner & test generator
│   ├── memory.py                  # Self-learning reflection store
│   └── orchestrator.py            # Main workflow coordinator
├── benchmark/
│   ├── __init__.py
│   └── run_humaneval.py           # Benchmark runner
├── tests/
│   ├── test_coding_agent.py
│   ├── test_config_llm.py
│   ├── test_evaluator.py
│   ├── test_memory.py
│   ├── test_orchestrator.py
│   └── test_task_understanding_retriever.py
└── main.py                        # Interactive CLI
```
