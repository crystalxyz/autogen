# RULER Benchmark

RULER (Really Understanding Long-context Evaluation and Reasoning) is a synthetic
benchmark for evaluating long-context capabilities of LLMs.

Reference: https://github.com/hsiehjackson/RULER

## Task Types

1. **NIAH Single** — Find a single key-value pair hidden in distractor text
2. **NIAH Multi-Key** — Find values for multiple keys hidden in text
3. **NIAH Multi-Value** — Find all values associated with one key
4. **NIAH Multi-Query** — Answer multiple queries about hidden key-value pairs
5. **Variable Tracking** — Track variable assignment chains (X = Y, Y = Z, ...)
6. **Common Words** — Find words common to multiple lists embedded in text
7. **Frequent Words** — Find the most frequently occurring word in noisy text

## Setup

Tasks are generated synthetically at configurable context lengths.

```bash
cd Scripts
python init_tasks.py
```

## Running

```bash
cd ..
agbench run Tasks/ruler_niah_single_4k_AgentChat.jsonl
```
