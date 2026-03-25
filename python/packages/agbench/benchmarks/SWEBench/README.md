# SWE-bench

[SWE-bench](https://www.swebench.com/) is a benchmark for evaluating language models on real-world software engineering tasks drawn from GitHub issues and pull requests.

Each task presents:
- A GitHub repository at a specific commit (before the fix)
- An issue description (problem statement)
- A set of tests that should **pass after** the fix (`FAIL_TO_PASS`)
- A set of tests that should **continue to pass** (`PASS_TO_PASS`)

The agent must edit the repository source code to resolve the issue, after which the tests are run to verify correctness.

This benchmark uses [SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (300 tasks) by default.

---

## Setup

### 1. Install dependencies

```bash
pip install datasets
```

### 2. Download dataset and generate task files

```bash
cd Scripts
python init_tasks.py          # SWE-bench Lite (default)
python init_tasks.py --full   # Full SWE-bench (~2300 tasks)
```

This creates JSONL files in the `Tasks/` directory.

### 3. Configure models

Edit `config.yaml` to set your model provider and API key, or export:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Running

```bash
# Run a single JSONL file
agbench run Tasks/swebench_lite_test__MagenticOne.jsonl

# Run with subsampling (e.g. first 10 tasks)
agbench run Tasks/swebench_lite_test__MagenticOne.jsonl --subsample 10
```

---

## Tabulating Results

```bash
agbench tabulate Results/swebench_lite_test__MagenticOne/ --custom-tabulate Scripts/custom_tabulate.py
```

---

## Templates

### MagenticOne

Uses a MagenticOne multi-agent team consisting of:
- **Coder**: Analyzes the issue and writes code fixes
- **Executor**: Runs shell commands and Python scripts
- **FileSurfer**: Browses and reads files in the repository

The agent clones the repo at the base commit, makes edits, and then the framework runs the test suite to determine pass/fail.

---

## Evaluation

A task is marked **passed** if:
1. All `FAIL_TO_PASS` tests pass after the agent's changes.
2. All `PASS_TO_PASS` tests still pass (no regressions).

Results are written to `Results/` and can be summarized with `agbench tabulate`.
