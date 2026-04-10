# CodeXGlue Code-to-Text Benchmark

This scenario implements the code-to-text (code summarization) task from the
[CodeXGlue](https://github.com/microsoft/CodeXGLUE) benchmark. Given a code
snippet, the agent produces a natural language summary. Results are evaluated
using smoothed BLEU-4 against the reference docstring.

## Metrics

Evaluation uses the MOSES-style smoothed BLEU-4 from the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
CodeXGlue implementation. Scores are on a 0–100 scale.

- **Tokenization**: MOSES/NIST-style normalization (punctuation splitting, lowercasing, XML unescaping) via `splitPuncts` for prediction/reference text.
- **Smoothing**: Add-1 smoothing on n-gram counts for n > 1 (unigram counts are unsmoothed).
- **Brevity penalty**: `min(0, 1 - (reflen + 1) / (testlen + 1))` applied in log-space.
- **Per-task scores**: Each scenario computes and prints its BLEU-4 in real time as it completes. The score is saved in `result.json` under the `bleu_score` key.
- **Corpus score**: `custom_tabulate.py` computes the average of per-sentence smoothed BLEU-4 scores across all completed tasks (matching `smoothed_bleu_4` from `Scripts/bleu.py`).

## Running the tasks

Navigate to the benchmark directory:

```bash
cd benchmarks/CodeXGlue_Code2Text
```

Update `config.yaml` to point to your model host. The default configuration points to `gpt-4o`.

Initialize the tasks (downloads dataset from HuggingFace):

```bash
pip install datasets
python Scripts/init_tasks.py --max-tasks 200 --lang python
```

Run the benchmark:

```bash
agbench run Tasks/code2text_python_AgentChat.jsonl
```

View results:

```bash
python Scripts/custom_tabulate.py Results/code2text_python_AgentChat
```

## Options

- `--max-tasks N` — limit to N tasks (default: 200)
- `--lang LANG` — language config: python, java, go, php, javascript, ruby (default: python)
- `--split SPLIT` — dataset split: train, validation, test (default: test)

## References

**CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation**
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco,
Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al.
[arXiv:2102.04664](https://arxiv.org/abs/2102.04664)
