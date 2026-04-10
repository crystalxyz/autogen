# DS-1000 Benchmark

DS-1000 is a benchmark of 1000 data science coding problems spanning 7 popular
Python libraries: NumPy, Pandas, TensorFlow, PyTorch, SciPy, Scikit-learn, and
Matplotlib.

Reference: https://ds1000-code-gen.github.io/

## Setup

Install the `datasets` package if you haven't already:

```bash
pip install datasets
```

## Running

1. Initialize the tasks (downloads the DS-1000 dataset from HuggingFace):

```bash
cd Scripts
python init_tasks.py
```

2. Run the benchmark:

```bash
cd ..
agbench run Tasks/ds1000_AgentChat.jsonl
```

3. Tabulate the results:

```bash
agbench tabulate Results/ds1000_AgentChat
```
