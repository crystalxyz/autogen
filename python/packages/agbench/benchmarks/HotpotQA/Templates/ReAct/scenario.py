"""ReAct agent template for HotpotQA.

The agent is given only the question. It uses tools to search and read the
context paragraphs (the 10 distractor paragraphs in the distractor setting),
reasons in a ReAct loop, then submits a final answer via the `finish` tool.
"""

import asyncio
import re
import string
import time
from collections import Counter
from typing import List, Tuple

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient


# ---------------------------------------------------------------------------
# Prompt parsing: split the templated __PROMPT__ into question + paragraphs
# ---------------------------------------------------------------------------


def parse_prompt(raw_prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Parse the substituted prompt into (question, [(title, body), ...]).

    The init_tasks.py format is:
        Context:
        [1] Title One
        body...

        [2] Title Two
        body...

        Question: ...

    For fullwiki tasks the Context section is absent.
    """
    paragraphs: List[Tuple[str, str]] = []
    question = raw_prompt.strip()

    if "Question:" in raw_prompt:
        context_part, _, question_part = raw_prompt.partition("Question:")
        question = question_part.strip()
        context_part = context_part.strip()
        if context_part.startswith("Context:"):
            context_part = context_part[len("Context:") :].strip()

        # Split on blank-line boundaries followed by [N] header.
        chunks = re.split(r"\n\s*\n(?=\[\d+\])", context_part)
        for chunk in chunks:
            m = re.match(r"\[(\d+)\]\s*(.*?)\n(.*)", chunk, re.DOTALL)
            if m:
                title = m.group(2).strip()
                body = m.group(3).strip()
                paragraphs.append((title, body))

    return question, paragraphs


# ---------------------------------------------------------------------------
# HotpotQA scoring (token-level F1 + exact match)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# Tool factory: build ReAct tools bound to this task's paragraphs.
# ---------------------------------------------------------------------------


def build_tools(paragraphs: List[Tuple[str, str]]):
    """Create tool callables closed over the per-task context paragraphs.

    Tools exposed to the agent:
        list_paragraphs()              -> indices + titles
        search_paragraphs(keyword)     -> indices whose title or body contains keyword
        lookup_paragraph(index)        -> full paragraph text
        lookup_in_paragraph(index, kw) -> sentences in that paragraph containing kw
        finish(answer)                 -> submit final answer (also printed)
    """

    def list_paragraphs() -> str:
        """List all available context paragraphs by index and title."""
        if not paragraphs:
            return "No context paragraphs are available for this task."
        return "\n".join(f"[{i + 1}] {title}" for i, (title, _) in enumerate(paragraphs))

    def search_paragraphs(keyword: str) -> str:
        """Search context paragraphs for a keyword.

        Returns the indices and titles of paragraphs whose title or body
        contains the keyword (case-insensitive substring match).
        """
        kw = keyword.lower().strip()
        if not kw:
            return "Empty keyword."
        hits = []
        for i, (title, body) in enumerate(paragraphs):
            if kw in title.lower() or kw in body.lower():
                hits.append(f"[{i + 1}] {title}")
        if not hits:
            return f"No paragraphs match keyword '{keyword}'."
        return "\n".join(hits)

    def lookup_paragraph(index: int) -> str:
        """Return the full text of context paragraph #index (1-indexed)."""
        if index < 1 or index > len(paragraphs):
            return f"Invalid index {index}. Valid range: 1..{len(paragraphs)}."
        title, body = paragraphs[index - 1]
        return f"[{index}] {title}\n{body}"

    def lookup_in_paragraph(index: int, keyword: str) -> str:
        """Return sentences in paragraph #index (1-indexed) containing keyword."""
        if index < 1 or index > len(paragraphs):
            return f"Invalid index {index}. Valid range: 1..{len(paragraphs)}."
        _, body = paragraphs[index - 1]
        kw = keyword.lower().strip()
        sentences = re.split(r"(?<=[.!?])\s+", body)
        hits = [s for s in sentences if kw and kw in s.lower()]
        if not hits:
            return f"No sentence in paragraph [{index}] contains '{keyword}'."
        return "\n".join(hits)

    # def finish(answer: str) -> str:
    #     """Submit the final answer to the question. Use a short span when possible."""
    #     return f"FINAL_ANSWER_RECORDED: {answer}"

    return [list_paragraphs, search_paragraphs, lookup_paragraph, lookup_in_paragraph]


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------


SYSTEM_MESSAGE = """You are a ReAct agent answering HotpotQA multi-hop questions.

You CANNOT see the context paragraphs directly. You can ONLY learn about
them by calling tools. You MUST gather evidence from the paragraphs before
answering — do NOT rely on prior knowledge, even if you think you know the
answer. Most questions are designed so that the right answer is in the
provided paragraphs and is NOT what general knowledge would suggest.

Tools (call them as function calls, never describe them in prose):
- list_paragraphs(): list all paragraph indices and titles
- search_paragraphs(keyword): find paragraphs whose title or body contains a keyword
- lookup_paragraph(index): read a full paragraph by its 1-indexed number
- lookup_in_paragraph(index, keyword): read sentences in a paragraph containing a keyword

Required workflow:
1. First call list_paragraphs() to see what is available.
2. Use search_paragraphs / lookup_paragraph to read the paragraphs that
   look relevant to the question. Multi-hop questions usually need
   evidence from 2+ paragraphs — keep looking until you have read every
   paragraph that could be relevant.
3. Only AFTER you have read the supporting paragraphs, write your final
   answer on its own line in EXACTLY this format:

       FINAL_ANSWER_RECORDED: <answer>

Answer format rules:
- Answers must be a SHORT span copied/derived from the paragraphs:
  a name, an entity, a number, a year, or "yes" / "no".
- Do NOT write a full sentence, do NOT add explanations after the answer.
- Do NOT emit FINAL_ANSWER_RECORDED until you have actually called the
  lookup tools and seen paragraph contents in tool results.
"""


def extract_final_answer(messages) -> str:
    """Extract the answer from a `FINAL_ANSWER_RECORDED: <answer>` line.

    Scans text messages in order and returns the answer from the LAST
    occurrence (so a re-stated answer at the end wins).
    """
    answer = ""
    pattern = re.compile(r"FINAL_ANSWER_RECORDED:\s*(.+)")
    for msg in messages:
        text = getattr(msg, "content", None)
        if isinstance(text, str):
            for m in pattern.finditer(text):
                answer = m.group(1).strip()
    return answer


async def main() -> None:
    start_time = time.time()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("prompt.txt", "r") as f:
        raw_prompt = f.read().strip()
    with open("expected_answer.txt", "r") as f:
        expected_answer = f.read().strip()

    question, paragraphs = parse_prompt(raw_prompt)

    model_client = ChatCompletionClient.load_component(config["model_config"])

    tools = build_tools(paragraphs)

    agent = AssistantAgent(
        name="react_agent",
        model_client=model_client,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=False,
    )

    # Termination fires as soon as the finish() tool emits its sentinel string,
    # so the agent doesn't need to also write a separate TERMINATE token.
    termination = TextMentionTermination(text="FINAL_ANSWER_RECORDED") | MaxMessageTermination(max_messages=30)
    team = RoundRobinGroupChat([agent], termination_condition=termination, max_turns=20)

    task = (
        f"Question: {question}\n\n"
        f"There are {len(paragraphs)} context paragraphs available ONLY via tools. "
        f"Start by calling list_paragraphs(), then read the relevant paragraphs with "
        f"lookup_paragraph / search_paragraphs / lookup_in_paragraph. Once you have "
        f"the supporting evidence, submit your answer using the sentinel format "
        f"described in the system message. "
        f"Do NOT answer from prior knowledge — the expected answer is grounded in the paragraphs."
    )

    stream = team.run_stream(task=task)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Per-turn token usage summary (matches HumanEval AgentChat template).
    turn = 0
    for msg in task_result.messages:
        if getattr(msg, "models_usage", None) is not None:
            turn += 1
            print(
                f"[token_usage] turn={turn} source={msg.source}"
                f" prompt_tokens={msg.models_usage.prompt_tokens}"
                f" completion_tokens={msg.models_usage.completion_tokens}"
                f" reasoning_tokens={msg.models_usage.reasoning_tokens}"
            )

    # Score against ground truth.
    prediction = extract_final_answer(task_result.messages)
    f1 = f1_score(prediction, expected_answer)
    em = exact_match(prediction, expected_answer)
    passed = em or f1 >= 0.5

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Question:   {question}")
    print(f"Prediction: {prediction}")
    print(f"Expected:   {expected_answer}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Exact Match: {em}")

    if passed:
        print("\nCORRECT!")
        print("ALL TESTS PASSED !#!#")
    else:
        print("\nINCORRECT")


if __name__ == "__main__":
    asyncio.run(main())
