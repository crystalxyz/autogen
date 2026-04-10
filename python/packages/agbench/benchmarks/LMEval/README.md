# LMEval benchmark

This benchmark plugs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
into agbench so that standard academic benchmarks (GSM8K, MMLU, ARC,
HellaSwag, TruthfulQA, ...) can be run against the same SGLang servers
managed by `scripts/run_bench.py`, with results written to the same
`outputs/<run>/Results/<task>/result.json` schema used by the other
agbench benchmarks.

## Layout

```
benchmarks/LMEval/
├── README.md
├── ENV.yaml                       # forwards OPENAI_API_KEY
├── config.yaml                    # model_config template (base_url injected by run_bench.py)
├── Tasks/
│   ├── lm_eval_default.jsonl         # direct: one record per lm-eval task
│   ├── lm_eval_smoke.jsonl           # direct: small subset, limit=4
│   ├── lm_eval_agent_default.jsonl   # agent-wrapped: gsm8k/triviaqa/drop
│   └── lm_eval_agent_smoke.jsonl     # agent-wrapped: gsm8k limit=4
├── Templates/
│   ├── LMEvalRunner/                 # DIRECT mode (no agent)
│   │   ├── scenario.py               # calls lm_eval.simple_evaluate()
│   │   ├── requirements.txt          # lm_eval[api]
│   │   ├── task_name.txt / fewshot.txt / limit.txt
│   └── LMEvalAgent/                  # AGENT mode (multi-turn coder+executor)
│       ├── scenario.py               # subclasses lm_eval.api.model.LM
│       ├── requirements.txt
│       └── task_name.txt / fewshot.txt / limit.txt / max_turns.txt
└── Scripts/
    ├── init_tasks.py              # (re)generates the JSONL files
    └── custom_tabulate.py         # custom timer regex for "LMEval execution time"
```

## Two modes

### 1. Direct (`Templates/LMEvalRunner`)

Pure LLM, no agent. `scenario.py` calls
`lm_eval.simple_evaluate(model="local-chat-completions", ...)` which
issues raw chat-completion requests to the SGLang server. Use this to
measure the bare model on standardized benchmarks.

### 2. Agent-wrapped (`Templates/LMEvalAgent`)

Single-agent, multi-turn loop modeled on HumanEval's AgentChat template
(`benchmarks/HumanEval/Templates/AgentChat/scenario.py`):

- A fresh `MagenticOneCoderAgent` + `CodeExecutorAgent`
  (`LocalCommandLineCodeExecutor`) team is created per lm-eval request.
- The team runs as a `RoundRobinGroupChat` up to `max_turns` solver↔executor
  turns, allowing the coder to reason, run Python to compute or verify,
  and revise based on tool output.
- `scenario.py` subclasses `lm_eval.api.model.LM` with an
  `AutogenAgentLM` whose `generate_until` pumps each lm-eval request
  through the team, then extracts the final answer from the last
  coder message (looks for a `FINAL_ANSWER:` line, falls back to the
  full message).
- `simple_evaluate(model=<the LM instance>, ...)` wires it into lm-eval's
  standard task grading / metric pipeline.
- `loglikelihood` and `loglikelihood_rolling` raise `NotImplementedError`;
  use generation-only tasks (gsm8k, triviaqa, drop).
- `max_turns` is per-task, substituted from `max_turns.txt`.

## How it works (both modes)

1. `run_bench.py` starts an SGLang server and rewrites `config.yaml` so
   that `model_config.config.base_url` points at it.
2. For each row in the scenario JSONL, agbench expands the selected
   template directory into a task dir and substitutes the placeholder
   files.
3. `scenario.py` loads `config.yaml` and either (a) hands `base_url`/`model`
   to `local-chat-completions`, or (b) builds an autogen
   `ChatCompletionClient` from the same `model_config` block that
   HumanEval uses.
4. The full lm-eval result is dumped to `lm_eval_result.json`; the
   per-metric summary is emitted on stdout as
   `[lmeval_metric] task=<name> metric=<key> value=<v> stderr=<s>`.
5. Standard agbench markers (`ALL TESTS PASSED !#!#`,
   `LMEval execution time: ... seconds`) are printed so the default
   scorer and the LMEval custom tabulator produce a normal `result.json`.

## Serial execution (no batching)

Every layer is pinned to "one request at a time":

| Layer | Knob | Value | Where |
|---|---|---|---|
| agbench worker pool | `num_concurrent` | `1` | run_bench.py config (`config-lmeval-4b.yaml`) |
| lm-eval async pool | `num_concurrent` | `1` | `scenario.py` via `model_args` |
| lm-eval batching | `batch_size` | `1` | `scenario.py` (also enforced by `local-chat-completions`) |

To empirically verify, set `log_requests: true` in the server config and
confirm the SGLang log shows non-overlapping requests during a run.

## Task compatibility

lm-eval's `local-chat-completions` backend can only drive **generation**
tasks. Multiple-choice / loglikelihood tasks (MMLU, HellaSwag,
ARC-Challenge, TruthfulQA-MC2) need token logprobs that chat completions
APIs do not expose in a usable form. The default JSONL lists them
anyway; if a task errors out, either drop it from your JSONL or add a
sibling `LMEvalRunnerCompletions` template that uses
`model="local-completions"` against `{base_url}/completions` (which
SGLang does expose with logprobs).

Known-good generation tasks to start with: `gsm8k`, `triviaqa`, `drop`.

## Running

```bash
cd python/packages/agbench
python scripts/run_bench.py --config scripts/configs/config-lmeval-4b.yaml
```

Outputs land under `outputs/lmeval_qwen3_4b_smoke/Results/<task>/`.
