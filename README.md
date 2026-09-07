<div align="center">

<h1>🤖 Self-Learning Coding Agent</h1>

<p>
  <strong>An autonomous AI coding agent that generates, evaluates, and self-corrects Python code — and <em>gets smarter with every mistake.</em></strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Kimi%20K2.7--Code-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HumanEval-164%20Problems-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Self--Learning-Memory%20Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

<p>
  <a href="#-architecture">Architecture</a> •
  <a href="#-features">Features</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-benchmark">Benchmark</a>
</p>

---

</div>

## 🧠 What Is This?

The **Self-Learning Coding Agent** is a fully autonomous Python code-generation system built on the [Kimi K2.7-Code](https://platform.moonshot.ai) model via the Moonshot AI API. It doesn't just write code — it **runs it, evaluates it, fixes its own mistakes**, and **stores actionable lessons** so it never makes the same error twice.

Think of it as a coding agent with a memory: every failure becomes a lesson that improves every future attempt.

```
📝 Task → 🧠 Understand → 🔍 Retrieve Context → 💻 Generate Code
                                                         ↓
                                                  🧪 Sandbox Test
                                                   ✅ PASS  ❌ FAIL
                                                     ↓         ↓
                                              📦 Deliver   🛠️ Self-Correct
                                                               ↓
                                                      📚 Store Lesson
                                                      🔁 Retry → Test
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Autonomous Code Generation** | Zero-shot Python code generation from natural language prompts |
| 🧪 **Isolated Sandbox Evaluation** | Subprocess-based execution with strict timeout — no infinite loops |
| 🔁 **Self-Correction Engine** | Ingests failure tracebacks and patches code without full rewrites |
| 📚 **Persistent Self-Learning Memory** | Stores `(error_type, failed_pattern, fix_lesson)` as reusable rules |
| 🔍 **Smart Context Retriever** | Pulls relevant workspace context + past memory lessons per task |
| 🧩 **Task Understanding Parser** | Extracts entry points, docstrings, I/O types, and edge cases |
| ⚡ **HumanEval Benchmark Harness** | Batch evaluates across 164 OpenAI HumanEval problems |
| 🖥️ **Rich Interactive CLI** | Beautiful real-time terminal UI powered by [Rich](https://github.com/Textualize/rich) |
| 📊 **Detailed Iteration Audit Log** | Full attempt history with pass/fail outcomes, durations, and diagnostics |

---

## 🏗️ Architecture

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
           │  Task Understanding   │  ← agent/task_understanding.py
           │ (Parse Intent & Types)│
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │    Code Retriever     │  ← agent/retriever.py
           │  (Context & Memory)   │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │     ORCHESTRATOR      │  ← agent/orchestrator.py
           │  (Plan-Execute-Loop)  │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │     CODING AGENT      │  ← agent/coding_agent.py
           │ (Generate / Refine)   │
           └───────────┬───────────┘
                       ↓
                 Generated Code
                       │
                       ↓
           ┌───────────────────────┐
           │       EVALUATOR       │  ← agent/evaluator.py
           │ (Isolated Sandbox)    │
           └───────────┬───────────┘
                 PASS  │  FAIL
        ┌─────────────┴─────────────┐
        ↓                           ↓
 ┌──────────────┐           ┌──────────────┐
 │ Final Answer │           │  Self-Correct │  ← agent/coding_agent.py
 │+ Explanation │           │  + Fix Code  │
 └──────────────┘           └──────┬───────┘
                                   ↓
                           EVALUATE AGAIN
                           + Store Lesson  ← agent/memory.py
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Self-Learning-Agent.git
cd Self-Learning-Agent
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example config
cp .env.example .env
```

Edit `.env` and add your **Moonshot AI (Kimi) API key**:

```env
KIMI_API_KEY=sk-your-api-key-here
KIMI_MODEL=kimi-k2.7-code
KIMI_BASE_URL=https://api.moonshot.ai/v1

MAX_RETRIES=4
SANDBOX_TIMEOUT=15.0
TEMPERATURE=1.0
MAX_TOKENS=16384
```

> **Get your API key** → [platform.moonshot.ai](https://platform.moonshot.ai)

### 5. Launch

```bash
python main.py
```

---

## 🖥️ Usage

### Interactive Mode (Default)

Launch with no arguments to enter the rich interactive CLI:

```bash
python main.py
```

You'll see a menu with the following options:

```
  [1] 📝 Enter a custom Python coding prompt
  [2] ⚡ Run a HumanEval benchmark problem by ID
  [3] 🧠 View Self-Learning Memory rules
  [4] 🧹 Clear Memory Store
  [5] 🚪 Exit
```

---

### Command-Line Mode

#### Solve a custom prompt directly
```bash
python main.py --prompt "Write a function that returns the nth Fibonacci number"
```

#### Solve from a file
```bash
python main.py --file my_problem.txt
```

#### Run a specific HumanEval benchmark problem
```bash
# By numeric ID
python main.py --task 0

# By full task ID
python main.py --task HumanEval/42
```

#### Run with custom unit tests
```bash
python main.py --prompt "Write a function add(a, b)" --test "assert add(1, 2) == 3"
```

#### Inspect memory store
```bash
python main.py --memory-stats
```

#### Clear all stored lessons
```bash
python main.py --clear-memory
```

#### Full CLI reference
```bash
python main.py --help
```

| Flag | Description |
|---|---|
| `-p`, `--prompt` | Direct coding prompt to solve |
| `-t`, `--task` | HumanEval task ID (e.g. `0` or `HumanEval/0`) |
| `-f`, `--file` | Path to a prompt file |
| `--test` | Custom unit test assertions |
| `--test-file` | Path to a unit test file |
| `-r`, `--retries` | Override max self-correction attempts |
| `--memory-file` | Custom memory storage JSON path |
| `--memory-stats` | Display all stored lessons and exit |
| `--clear-memory` | Wipe all stored memory reflections |
| `--no-interactive` | Disable interactive mode if no prompt given |

---

## 📁 Project Structure

```
Self-Learning-Agent/
│
├── agent/                         # 🧠 Core agent modules
│   ├── config.py                  # Environment config & validation
│   ├── llm.py                     # Moonshot AI (Kimi) LLM client
│   ├── task_understanding.py      # Prompt parser & signature extractor
│   ├── retriever.py               # Context & memory retriever
│   ├── coding_agent.py            # Code generator & self-correction patcher
│   ├── evaluator.py               # Isolated sandbox runner & test generator
│   ├── memory.py                  # Persistent self-learning memory store
│   └── orchestrator.py            # Master workflow coordinator
│
├── benchmark/
│   └── run_humaneval.py           # HumanEval batch benchmark harness
│
├── main.py                        # 🖥️ Interactive CLI entry point
├── agent_memory.json              # 💾 Persisted memory reflections (auto-generated)
├── agent_tasks.jsonl              # HumanEval agent task prompts (164 problems)
├── requirements.txt               # Python dependencies
├── .env.example                   # Config template
└── ROADMAP.md                     # Development phase tracker
```

---

## ⚙️ Configuration

All configuration is loaded from your `.env` file via `agent/config.py`.

| Variable | Default | Description |
|---|---|---|
| `KIMI_API_KEY` | *(required)* | Your Moonshot AI API key |
| `KIMI_MODEL` | `kimi-k2.7-code` | Model to use for code generation |
| `KIMI_BASE_URL` | `https://api.moonshot.ai/v1` | API base URL |
| `MAX_RETRIES` | `4` | Max self-correction attempts per task |
| `SANDBOX_TIMEOUT` | `15.0` | Sandbox execution timeout (seconds) |
| `TEMPERATURE` | `1.0` | LLM sampling temperature |
| `MAX_TOKENS` | `16384` | Maximum tokens per LLM response |

---

## 📊 Benchmark

Run the full HumanEval benchmark suite:

```bash
# Run first 10 problems
python benchmark/run_humaneval.py --limit 10

# Run a specific range
python benchmark/run_humaneval.py --range 0 50

# Run specific task IDs
python benchmark/run_humaneval.py --tasks HumanEval/0 HumanEval/5 HumanEval/42

# Run all 164 problems
python benchmark/run_humaneval.py --all
```

The benchmark harness reports:

- **Pass@1** — Zero-shot success rate
- **Pass@k with Self-Correction** — Recovery rate after retries
- **Memory Learning Impact** — Lessons created vs. lessons retrieved and reused
- **Error Breakdown** — Categorized failure diagnostics
- **Execution Latency** — Per-problem timing metrics

Results are saved as JSON reports and `samples.jsonl` files compatible with the official HumanEval evaluation script.

---

## 🧠 How Self-Learning Works

Every time the agent **fails** and then **succeeds** (or exhausts retries), it distills a structured reflection:

```json
{
  "task_tag": "string_manipulation",
  "error_type": "IndexError",
  "failed_pattern": "Accessed list index without bounds check",
  "lesson": "Always validate list length before indexing. Use len(lst) > 0 or try/except.",
  "usage_count": 3
}
```

This lesson is **persistently stored** in `agent_memory.json` and **automatically injected** into future prompts whenever the agent tackles a similar task — preventing the same class of mistake from recurring.

Over time, the agent builds a growing knowledge base of hard-won coding rules.

---

## 🔄 Lifecycle Events

The agent emits real-time status events visible in the terminal:

| Event | Meaning |
|---|---|
| 🚀 `START` | Task received, pipeline initializing |
| 🧠 `UNDERSTANDING` | Parsing prompt, extracting intent & types |
| 🔍 `RETRIEVING` | Fetching context and relevant memory lessons |
| 💻 `CODING` | LLM generating Python solution |
| 🧪 `EVALUATING` | Running code in isolated sandbox |
| 🛠️ `SELF_CORRECTING` | Failure detected — patching code with error context |
| 📚 `LEARNING` | Distilling new reflection into memory store |
| ✅ `COMPLETED` | Task solved successfully |
| ❌ `FAILED` | All retry attempts exhausted |

---

## 📦 Dependencies

```
openai          # OpenAI-compatible API client (used with Kimi)
python-dotenv   # Environment variable loader
pydantic        # Data validation & settings models
rich            # Beautiful terminal output & interactive UI
pytest          # Testing framework
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Development Status

All 7 phases are complete:

| Phase | Description | Status |
|:---:|:---|:---:|
| 0 | Environment & Dataset Preparation | ✅ |
| 1 | Foundation, Config & LLM Client | ✅ |
| 2 | Evaluator & Sandbox Test Runner | ✅ |
| 3 | Task Understanding & Context Retriever | ✅ |
| 4 | Coding Agent & Self-Correction Engine | ✅ |
| 5 | Self-Learning Memory Store | ✅ |
| 6 | Master Orchestrator & CLI Interface | ✅ |
| 7 | HumanEval Benchmark Harness | ✅ |

See [ROADMAP.md](ROADMAP.md) for the full detailed phase breakdown.

---

## 📄 License

This project is licensed under the **MIT License** — use it, fork it, build on it.

---

<div align="center">

**Built with ❤️ — an agent that learns, adapts, and gets better with every run.**

</div>
