"""ReAct agent template for HotpotQA — fullwiki / open-domain setup.

Unlike the distractor ReAct template, the agent does NOT receive any context
paragraphs in the prompt. It must retrieve evidence from live Wikipedia via
search/lookup tools (mirroring the original ReAct paper, Yao et al. 2023).

Tools exposed (close to the paper):
    search(entity)        -> page summary if `entity` matches a Wikipedia page,
                             otherwise the top similar titles to try.
    lookup(keyword)       -> next sentence in the *current* page that contains
                             the keyword (Ctrl-F style; advances on each call).
    get_page(title)       -> full first section of a Wikipedia page (sets it as
                             the current page for `lookup`).
"""

import asyncio
import re
import string
import time
from collections import Counter
from typing import List, Optional

import yaml
import wikipedia
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient


# Wikipedia is generally English for HotpotQA.
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)


# ---------------------------------------------------------------------------
# Prompt parsing: fullwiki prompts only contain the question.
# ---------------------------------------------------------------------------


def parse_prompt(raw_prompt: str) -> str:
    text = raw_prompt.strip()
    if text.startswith("Question:"):
        text = text[len("Question:") :].strip()
    return text


# ---------------------------------------------------------------------------
# HotpotQA scoring (token-level F1 + exact match) — same as distractor template.
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
# Wikipedia tool factory.
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _fetch_page_summary(title: str, sentences: int = 5) -> str:
    """Get the lead summary of a Wikipedia page (raises on failure)."""
    return wikipedia.summary(title, sentences=sentences, auto_suggest=False, redirect=True)


def build_tools():
    """Create Wikipedia tool callables.

    Tool surface (ReAct paper style, two tools, no hidden state):
        search(entity)            -> intro paragraph of the matching Wikipedia
                                     page (with redirects), or a list of
                                     candidate titles on miss / disambiguation.
        lookup(page_title, keyword) -> next sentence in `page_title` containing
                                     `keyword`; repeated calls walk successive
                                     matches (cursor is keyed by page+keyword).
    """
    # Per-task caches.
    pages: dict = {}             # canonical_title -> full page.content
    cursors: dict = {}           # (canonical_title, keyword) -> next sentence idx

    def _load_page(title: str):
        """Resolve `title` to a canonical page and cache its content.

        Returns (canonical_title, error_str). On success error_str is None.
        """
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
        except wikipedia.DisambiguationError as e:
            opts = "; ".join(e.options[:8])
            return None, f"'{title}' is ambiguous on Wikipedia. Candidates: {opts}"
        except wikipedia.PageError:
            try:
                hits = wikipedia.search(title, results=8)
            except Exception:
                hits = []
            if not hits:
                return None, f"No Wikipedia page found for '{title}'."
            return None, f"No exact page for '{title}'. Candidates: " + "; ".join(hits)
        except Exception as e:
            return None, f"Wikipedia error for '{title}': {e}"
        canonical = page.title
        pages[canonical] = page.content
        return canonical, None

    def search(entity: str) -> str:
        """Fetch a Wikipedia page by entity name.

        Returns the intro paragraph of the matching page, prefixed with the
        canonical title. If the entity is ambiguous or does not match a page,
        returns a list of candidate titles — then call `search` again with
        one of those titles.

        After a successful call, you can use `lookup(page_title, keyword)` on
        the canonical title shown in the response to pull specific sentences
        out of the full page.
        """
        entity = (entity or "").strip()
        if not entity:
            return "Empty search query."
        canonical, err = _load_page(entity)
        if err is not None:
            return err
        # Return intro paragraph + sentence count so agent knows how much is
        # available via lookup().
        try:
            intro = _fetch_page_summary(canonical, sentences=5)
        except Exception:
            intro = pages[canonical][:1500]
        num_sentences = len(_split_sentences(pages[canonical]))
        return (
            f"[Page: {canonical}] ({num_sentences} sentences total, "
            f"use lookup(page_title=\"{canonical}\", keyword=...) for details)\n"
            f"{intro}"
        )

    def lookup(page_title: str, keyword: str) -> str:
        """Find the next sentence in `page_title` containing `keyword`.

        Mirrors the ReAct paper's Ctrl-F semantics. Repeated calls with the
        SAME (page_title, keyword) walk through successive matches; passing a
        new keyword or a new page resets the cursor for that pair.

        `page_title` must be a Wikipedia page title (use the canonical title
        returned by `search`). If the page has not been retrieved yet, it is
        fetched automatically.
        """
        page_title = (page_title or "").strip()
        keyword = (keyword or "").strip()
        if not page_title:
            return "Empty page_title."
        if not keyword:
            return "Empty keyword."

        # Load the page if not already cached.
        if page_title not in pages:
            canonical, err = _load_page(page_title)
            if err is not None:
                return err
            page_title = canonical  # prefer canonical for cursor key

        kw = keyword.lower()
        sentences = _split_sentences(pages[page_title])
        cursor_key = (page_title, kw)
        start = cursors.get(cursor_key, 0)
        total_matches = sum(1 for s in sentences if kw in s.lower())

        for i in range(start, len(sentences)):
            if kw in sentences[i].lower():
                cursors[cursor_key] = i + 1
                match_num = sum(1 for s in sentences[: i + 1] if kw in s.lower())
                return (
                    f"[Page: {page_title}] (match {match_num}/{total_matches}) "
                    f"{sentences[i]}"
                )

        # No more matches from the current cursor. Reset so a future call
        # starts over.
        cursors[cursor_key] = 0
        if total_matches == 0:
            return f"No sentence in '{page_title}' contains '{keyword}'."
        return (
            f"No further matches for '{keyword}' in '{page_title}' "
            f"(already walked all {total_matches} matches; cursor reset)."
        )

    return [search, lookup]


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------


SYSTEM_MESSAGE = """You are a ReAct agent answering open-domain HotpotQA questions.

You have NO context paragraphs in the prompt. To find evidence you MUST use
Wikipedia tools. Do not rely on prior knowledge — verify every fact via the
tools, because the questions are designed to require multi-hop retrieval.

Tools (call them as function calls, never describe them in prose):
- search(entity): Fetch the Wikipedia page for `entity` and return its intro
  paragraph. The response starts with `[Page: <canonical_title>]` — use that
  canonical title as the `page_title` for later lookup() calls. If the entity
  is ambiguous or not found, the response lists candidate titles; call
  search() again with one of those titles.
- lookup(page_title, keyword): Find the next sentence in `page_title` that
  contains `keyword`. Repeated calls with the SAME (page_title, keyword) walk
  through successive matches; the response shows "(match k/N)". When N is
  reached the cursor resets. `page_title` must be the canonical title shown
  in a prior search() response (or a title you're confident exists).

Required workflow:
1. Identify the entities needed to answer the question.
2. Call search(entity) for the most central one. If you get a candidates
   list, call search() again with the best candidate.
3. Use lookup(page_title, keyword) to pull specific sentences from the page
   you just retrieved. Use the canonical title from the [Page: ...] header.
4. Multi-hop questions usually need 2+ pages — call search() for each entity
   and lookup() on each page as needed.
5. Once you have enough evidence, write your final answer on its own line in
   EXACTLY this format:

       FINAL_ANSWER_RECORDED: <answer>

Answer format rules:
- Answers must be a SHORT span: a name, an entity, a number, a year, or
  "yes" / "no".
- Do NOT write a full sentence, do NOT add explanations after the answer.
- Do NOT emit the sentinel until you have actually retrieved evidence from
  Wikipedia tool results.
"""


def extract_final_answer(messages) -> str:
    """Extract the answer from the LAST `FINAL_ANSWER_RECORDED: <answer>` line."""
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

    question = parse_prompt(raw_prompt)

    model_client = ChatCompletionClient.load_component(config["model_config"])

    tools = build_tools()

    agent = AssistantAgent(
        name="react_agent",
        model_client=model_client,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=False,
    )

    termination = TextMentionTermination(text="FINAL_ANSWER_RECORDED") | MaxMessageTermination(max_messages=40)
    team = RoundRobinGroupChat([agent], termination_condition=termination, max_turns=25)

    task = (
        f"Question: {question}\n\n"
        f"You have NO context paragraphs in this message. Use the Wikipedia "
        f"tools (search / lookup) to retrieve evidence, then submit your "
        f"answer using the sentinel format described in the system message. "
        f"Do NOT answer from prior knowledge."
    )

    stream = team.run_stream(task=task)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Per-turn token usage summary.
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
