# TriviaQA Benchmark

This scenario implements the [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) question answering benchmark using a ReAct agent. Before you begin, make sure you have followed instructions in `../README.md` to prepare your environment.

## About TriviaQA

TriviaQA is a large-scale reading comprehension dataset containing over 650K question-answer-evidence triples. Questions are authored by trivia enthusiasts and evidence documents are gathered from Wikipedia and web search results.

Key characteristics:
- **Evidence-based QA**: Questions paired with Wikipedia pages and web search results as evidence
- **Multiple accepted answers**: Each question has a canonical answer plus aliases (e.g., "NYC", "New York City")
- **Diverse question types**: Factoid questions spanning history, science, geography, entertainment, etc.
- **Reading comprehension**: Requires finding the answer within provided evidence documents

Dataset: https://huggingface.co/datasets/trivia_qa

## ReAct Agent Approach

This benchmark uses a single ReAct agent that:
- Cannot see evidence documents directly — must use tools to search and read them
- Uses `list_paragraphs`, `search_paragraphs`, `lookup_paragraph`, and `lookup_in_paragraph` tools
- Gathers evidence before submitting a final answer
- Evaluation checks the answer against all accepted aliases using F1 and Exact Match

## Setup

Navigate to TriviaQA:

```bash
cd benchmarks/TriviaQA
```

Initialize the tasks:

```bash
python Scripts/init_tasks.py
```

This will download the TriviaQA dataset from Hugging Face and create task files.

The resulting folder structure should look like this:

```
.
./Downloads
./Scripts
./Templates
./Templates/ReAct
./Tasks
```

## Running TriviaQA

To run a specific subset:

```bash
# Run validation set (sanity check with 100 tasks)
agbench run Tasks/triviaqa_validation_rc__ReAct_sanity100.jsonl

# Run full validation set
agbench run Tasks/triviaqa_validation_rc__ReAct.jsonl

# Run with specific number of tasks
agbench run Tasks/triviaqa_validation_rc__ReAct.jsonl -s 50
```

To see a summary of the results:

```bash
agbench tabulate Results/triviaqa_validation_rc__ReAct/ --scorer Scripts/custom_tabulate.py
```

## Dataset Configuration

- **rc** (reading comprehension): Each question comes with Wikipedia entity pages as evidence documents. Evidence pages are truncated to ~2000 characters each, with up to 10 pages per question.

## Evaluation Metrics

TriviaQA uses:
- **Exact Match (EM)**: Normalized prediction matches any accepted alias exactly
- **F1 Score**: Token-level F1 between prediction and best-matching alias

The ReAct template considers an answer correct if EM is true OR best F1 >= 0.5, evaluated across all accepted answer aliases.

## References

**TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension**
Mandar Joshi, Eunsol Choi, Daniel Weld, Luke Zettlemoyer
https://arxiv.org/abs/1705.03551
