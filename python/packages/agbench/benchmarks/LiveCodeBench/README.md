# LiveCodeBench Benchmark

This scenario implements [LiveCodeBench](https://livecodebench.github.io/) as
an agentic coding loop, modeled after the `HumanEval/Templates/AgentChat`
template.

## How it works

Each problem is solved by a two-agent `RoundRobinGroupChat`:

- **`coder`** — `MagenticOneCoderAgent`. Receives the problem statement
  (and starter code, when provided) and emits a single ```python``` block
  containing the complete solution.
- **`executor`** — `CustomCodeExecutorAgent` (subclass of `CodeExecutorAgent`).
  Wraps the coder's code in a sandbox harness that:
  1. Runs the **public** test cases against the solution.
  2. If all public tests pass, runs the **private** test cases (the full
     hidden test suite) and reports the final verdict, then emits
     `TERMINATE`.
  3. If any public test fails, prints the failures and the loop cycles back
     to the coder for another attempt.

The team runs for up to `MAX_TURNS = 10` messages (i.e. up to 5 coder→executor
cycles) before giving up.

The harness handles both LiveCodeBench test types:

- `stdin` — solution is `exec`-ed with `sys.stdin` redirected to the test
  input; stdout is compared to the expected output.
- `functional` — `Solution().<func_name>(*args)` is invoked and the return
  value is compared to the expected output (function name comes from
  LiveCodeBench `metadata.func_name`).

## Running the tasks

```bash
cd benchmarks/LiveCodeBench
```

Update `config.yaml` to point to your model, then download and expand the
dataset:

```bash
python Scripts/init_tasks.py                # default: code_generation_lite, release_v1
# or pick a different release / subset:
python Scripts/init_tasks.py --release release_v2 --subset code_generation_lite
```

This populates `Tasks/livecodebench_AgentChat.jsonl`. Run it with:

```bash
agbench run Tasks/livecodebench_AgentChat.jsonl
```

And tabulate:

```bash
agbench tabulate Results/livecodebench_AgentChat
```

## Outcome markers in the console log

The custom executor prints these sentinels (greppable for analysis):

- `PUBLIC TESTS: ALL N PASSED`
- `PUBLIC TESTS: K/N FAILED`
- `PRIVATE TESTS: ALL N PASSED`
- `PRIVATE TESTS: K/N FAILED`
- `ALL TESTS PASSED !#!#`  — public + private both passed (final success)
- `PRIVATE TESTS FAILED !#!#` — public passed but hidden tests failed
- `PUBLIC TESTS FAILED - TRY AGAIN !#!#` — coder loop continues

## References

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang,
Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica.
**LiveCodeBench: Holistic and Contamination Free Evaluation of Large
Language Models for Code.** [https://livecodebench.github.io/](https://livecodebench.github.io/)
