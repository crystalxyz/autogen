# LongBench Benchmark

LongBench is the first bilingual, multitask benchmark for long context understanding.
It includes 21 tasks across 6 categories: single-doc QA, multi-doc QA, summarization,
few-shot learning, synthetic retrieval, and code completion.

Reference: https://github.com/THUDM/LongBench

## Setup

```bash
pip install datasets rouge-score
```

## Running

1. Initialize the tasks (downloads from HuggingFace):

```bash
cd Scripts
python init_tasks.py
```

2. Run a specific task subset:

```bash
cd ..
agbench run Tasks/longbench_hotpotqa_AgentChat.jsonl
```

3. Tabulate results:

```bash
agbench tabulate Results/longbench_hotpotqa_AgentChat
```
