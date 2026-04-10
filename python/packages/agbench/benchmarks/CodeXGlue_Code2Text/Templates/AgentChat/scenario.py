import asyncio
import json
import math
import re
import sys
import time
import xml.sax.saxutils

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient


# ---------------------------------------------------------------------------
# MOSES-style BLEU (from lm-eval / CodeXGlue)
# ---------------------------------------------------------------------------
nonorm = 0
preserve_case = False
eff_ref_len = "shortest"

_normalize1 = [
    (re.compile(r"<skipped>"), ""),
    (re.compile(r"-\n"), ""),
    (re.compile(r"\n"), " "),
]

_normalize2 = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def _normalize(s):
    if nonorm:
        return s.split()
    if not isinstance(s, str):
        s = " ".join(s)
    for pattern, replace in _normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {"&quot;": '"'})
    s = " %s " % s
    if not preserve_case:
        s = s.lower()
    for pattern, replace in _normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def _count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _cook_refs(refs, n=4):
    refs = [_normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = _count_ngrams(ref, n)
        for ngram, count in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def _cook_test(test, item, n=4):
    (reflens, refmaxcounts) = item
    test = _normalize(test)
    result = {}
    result["testlen"] = len(test)
    result["reflen"] = min(reflens)
    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]
    result["correct"] = [0] * n
    counts = _count_ngrams(test, n)
    for ngram, count in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)
    return result


def _score_cooked(allcomps, n=4, smooth=1):
    totalcomps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in allcomps:
        for key in ["testlen", "reflen"]:
            totalcomps[key] += comps[key]
        for key in ["guess", "correct"]:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    for k in range(n):
        correct = totalcomps["correct"][k]
        guess = totalcomps["guess"][k]
        addsmooth = 1 if (smooth == 1 and k > 0) else 0
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(
            guess + addsmooth + sys.float_info.min
        )
    logbleu /= float(n)
    brevPenalty = min(0, 1 - float(totalcomps["reflen"] + 1) / (totalcomps["testlen"] + 1))
    logbleu += brevPenalty
    return math.exp(logbleu) * 100.0


def _splitPuncts(line):
    return " ".join(re.findall(r"[\w]+|[^\s\w]", line))


def compute_bleu(reference, prediction):
    """Compute smoothed BLEU-4 for a single ref/pred pair (0-100 scale)."""
    gold = [_splitPuncts(reference.strip().lower())]
    hypothesis = _splitPuncts(prediction.strip().lower())
    refs = _cook_refs(gold)
    test = _cook_test(hypothesis, refs)
    return _score_cooked([test])


# ---------------------------------------------------------------------------

CODE = """__CODE__"""

REFERENCE = """__REFERENCE__"""


async def main() -> None:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    client = ChatCompletionClient.load_component(config["model_config"])

    start_time = time.time()

    task = f"""You are an expert code summarizer. Given the following Python function, write a concise natural language summary describing what the function does. Output ONLY the summary text, nothing else. Do not include any code, markdown formatting, or extra explanation.

```python
{CODE}
```
"""

    agent = AssistantAgent(
        name="summarizer",
        model_client=client,
        system_message="You are a code summarization expert. You produce concise, accurate natural language descriptions of code. Output only the summary text with no formatting.",
    )

    termination = MaxMessageTermination(max_messages=2)

    team = RoundRobinGroupChat(
        [agent],
        max_turns=1,
        termination_condition=termination,
    )

    stream = team.run_stream(task=task)
    result = await Console(stream)

    # Extract the prediction from the agent's response
    prediction = ""
    if result and result.messages:
        for msg in reversed(result.messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.source == "summarizer":
                prediction = msg.content.strip()
                break

    # Compute BLEU score immediately
    bleu_score = 0.0
    if prediction:
        bleu_score = compute_bleu(REFERENCE, prediction)

    # Save result for tabulation
    with open("result.json", "wt") as fh:
        json.dump({
            "prediction": prediction,
            "reference": REFERENCE,
            "bleu_score": bleu_score,
        }, fh, indent=2)

    end_time = time.time()
    print(f"\nPrediction: {prediction}")
    print(f"\nReference: {REFERENCE}")
    print(f"\nBLEU-4: {bleu_score:.2f}")
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
