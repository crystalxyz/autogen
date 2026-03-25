# HotpotQA Benchmark

This scenario implements the [HotpotQA](https://hotpotqa.github.io/) multi-hop question answering benchmark using Multi-Agent Debate (MAD). Before you begin, make sure you have followed instructions in `../README.md` to prepare your environment.

## About HotpotQA

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The dataset contains ~113k Wikipedia-based question-answer pairs.

Key characteristics:
- **Multi-hop reasoning**: Questions require reasoning over multiple documents
- **Diverse question types**: Comparison and bridge questions
- **Supporting facts**: Ground truth evidence for explainability
- **Two settings**: Distractor (10 paragraphs) and Fullwiki (entire Wikipedia)

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa

## Multi-Agent Debate (MAD) Approach

This benchmark uses the Multi-Agent Debate framework:
- Multiple debater agents independently reason about the question
- Debaters share and critique each other's answers over multiple rounds
- A judge selects the best answer or synthesizes a final answer
- Evaluation uses HotpotQA's F1 and Exact Match metrics

## Setup

Navigate to HotpotQA:

```bash
cd benchmarks/HotpotQA
```

Initialize the tasks:

```bash
python Scripts/init_tasks.py
```

This will download the HotpotQA dataset from Hugging Face and create task files.

The resulting folder structure should look like this:

```
.
./Downloads
./Downloads/hotpotqa
./Scripts
./Templates
./Templates/MAD
./Tasks
```

## Running HotpotQA

To run a specific subset of HotpotQA:

```bash
# Run validation set with distractor setting
agbench run Tasks/hotpotqa_validation_distractor__MAD.jsonl

# Run with specific number of tasks
agbench run Tasks/hotpotqa_validation_distractor__MAD.jsonl -s 100
```

To see a summary of the results:

```bash
agbench tabulate Results/hotpotqa_validation_distractor__MAD/
```

## Dataset Configurations

- **distractor**: Each question comes with 10 paragraphs (2 gold + 8 distractors)
- **fullwiki**: Open-domain setting requiring retrieval from full Wikipedia

## Evaluation Metrics

HotpotQA uses:
- **Exact Match (EM)**: Percentage of predictions matching ground truth exactly
- **F1 Score**: Token-level F1 between prediction and ground truth

The MAD template considers an answer correct if EM is true OR F1 >= 0.5.

## References

**HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning
https://arxiv.org/abs/1809.09600
