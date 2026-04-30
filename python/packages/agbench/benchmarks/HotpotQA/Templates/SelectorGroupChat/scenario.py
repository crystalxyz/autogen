import time
_script_start = time.time()

import asyncio
import logging
import os
import re
import string
import yaml
import warnings
from collections import Counter
from typing import Sequence

# Limit onnxruntime to 1 thread to avoid pthread_setaffinity_np errors
# under Slurm cgroup CPU restrictions. Must be done before any onnxruntime import.
import onnxruntime as _ort
_ort_so = _ort.SessionOptions()
_ort_so.intra_op_num_threads = 1
_ort_so.inter_op_num_threads = 1
_ort_orig_init = _ort.InferenceSession.__init__
def _ort_patched_init(self, *args, **kwargs):
    if "sess_options" not in kwargs and len(args) < 2:
        kwargs["sess_options"] = _ort_so
    _ort_orig_init(self, *args, **kwargs)
_ort.InferenceSession.__init__ = _ort_patched_init

from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.utils import content_to_str
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage, BaseAgentEvent, BaseChatMessage, MultiModalMessage, StopMessage
from autogen_core.models import LLMMessage, UserMessage

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.DEBUG)
# Silence the per-call tiktoken fallback warning ("Model X not found.
# Using cl100k_base encoding.") — cl100k_base IS the intended fallback for
# any non-OpenAI model and the warning fires once per create() call.
logging.getLogger("autogen_core.trace").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# HotpotQA scoring (token-level F1 + exact match) — for the local log only;
# authoritative scoring is done by Scripts/custom_tabulate.py on FINAL ANSWER.
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
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def parse_question(raw_prompt: str) -> str:
    text = raw_prompt.strip()
    if text.startswith("Question:"):
        text = text[len("Question:"):].strip()
    return text


async def main() -> None:

    # Read EVERYTHING from the instance dir BEFORE chdir-ing into the clean
    # workspace. The instance dir contains expected_answer.txt, prompt.txt,
    # and config.yaml — none of which the agents should be able to see, since
    # a                + shell-executor combination could trivially `cat` the ground
    # truth and bypass the benchmark.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("prompt.txt", "rt") as fh:
        raw_prompt = fh.read().strip()
    with open("expected_answer.txt", "rt") as fh:
        expected_answer = fh.read().strip()

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])

    # Unified latency instrumentation.
    # - orchestrator: per-LLM-call [LATENCY] with label=selector/termination/finalizer
    # - assistant + web_surfer: one [LATENCY] per agent TURN (e2e, including any
    #   tool execution like browser actions), with aggregated token counts from
    #   all LLM calls made during that turn.
    orchestrator_call_label = {"value": "selector"}
    agent_accum = {
        "assistant":  {"pt": 0, "ct": 0, "calls": 0},
        "web_surfer": {"pt": 0, "ct": 0, "calls": 0},
    }

    def _instrument_orch(client, label_ref):
        _orig = client.create
        async def _timed(*args, **kwargs):
            label = label_ref["value"]
            t0 = time.perf_counter()
            result = None
            try:
                result = await _orig(*args, **kwargs)
                return result
            finally:
                dt = time.perf_counter() - t0
                usage = getattr(result, "usage", None) if result is not None else None
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                print(
                    f"[LATENCY] role=orchestrator label={label} duration_s={dt:.3f} "
                    f"prompt_tokens={pt} completion_tokens={ct}",
                    flush=True,
                )
                content = getattr(result, "content", None) if result is not None else None
                if isinstance(content, str):
                    print(f"[ORCH {label}] {content}", flush=True)
        client.create = _timed

    def _instrument_client_accum(client, role):
        """Wrap client.create to accumulate token counts for the active agent turn."""
        _orig = client.create
        async def _timed(*args, **kwargs):
            result = await _orig(*args, **kwargs)
            usage = getattr(result, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            agent_accum[role]["pt"] += pt
            agent_accum[role]["ct"] += ct
            agent_accum[role]["calls"] += 1
            return result
        client.create = _timed

    def _instrument_agent(agent, role):
        """Time each full agent turn (on_messages_stream) as one [LATENCY] entry.

        SelectorGroupChat invokes agents via on_messages_stream (an async
        generator), so we wrap that rather than on_messages.
        """
        _orig = agent.on_messages_stream

        async def _timed(*args, **kwargs):
            agent_accum[role] = {"pt": 0, "ct": 0, "calls": 0}
            t0 = time.perf_counter()
            try:
                async for item in _orig(*args, **kwargs):
                    yield item
            finally:
                dt = time.perf_counter() - t0
                a = agent_accum[role]
                print(
                    f"[LATENCY] role={role} label={role} duration_s={dt:.3f} "
                    f"prompt_tokens={a['pt']} completion_tokens={a['ct']} llm_calls={a['calls']}",
                    flush=True,
                )

        agent.on_messages_stream = _timed

    _instrument_orch(orchestrator_client, orchestrator_call_label)
    _instrument_client_accum(coder_client, role="assistant")
    _instrument_client_accum(web_surfer_client, role="web_surfer")

    question = parse_question(raw_prompt)

    # Set up the team — no FileSurfer. All external information must come from
    # the web (no HotpotQA-provided context in the prompt).
    coder = MagenticOneCoderAgent(
        "Assistant",
        model_client=coder_client,
    )

    executor = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())

    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client=web_surfer_client,
        downloads_folder=os.getcwd(),
        debug_dir="logs",
        to_save_screenshots=False,
        start_page="https://en.wikipedia.org/wiki/Main_Page",
    )

    # Emit one [LATENCY] per agent TURN (e2e) for assistant and web_surfer.
    _instrument_agent(coder, role="assistant")
    _instrument_agent(web_surfer, role="web_surfer")

    # Replace viewport-only text extraction with the full rendered body.
    # WebSurfer was designed around vision models inspecting a screenshot;
    # with a text-only LLM the screenshot bytes are dropped and the LLM is
    # left with ~30-40% of the article (one viewport). We override the
    # state-description builder to return `document.body.innerText` of the
    # whole page, capped at a reasonable size so we don't blow the context.
    import types

    _PAGE_TEXT_CHAR_LIMIT = 12000

    async def _full_page_state_description(_self):
        page = _self._page
        try:
            page_title = await page.title()
        except Exception:
            page_title = page.url
        try:
            full_text = await page.evaluate("() => document.body.innerText")
        except Exception:
            full_text = await _self._playwright_controller.get_visible_text(page)
        if full_text and len(full_text) > _PAGE_TEXT_CHAR_LIMIT:
            full_text = (
                full_text[:_PAGE_TEXT_CHAR_LIMIT]
                + f"\n\n[...page truncated at {_PAGE_TEXT_CHAR_LIMIT} chars]"
            )
        return (
            f"web browser is open to the page [{page_title}]({page.url}).\n"
            f"The following is the full text content of the page:\n\n{full_text}"
        )

    web_surfer._get_state_description = types.MethodType(
        _full_page_state_description, web_surfer
    )

    # Wikipedia-only browsing.
    #
    # (1) URL rewrite: every Bing search (and any other non-Wikipedia URL the
    #     agent tries to visit) is redirected to Wikipedia's Special:Search.
    # (2) Network whitelist: a Playwright route handler aborts every request
    #     whose host isn't under wikipedia.org / wikimedia.org, so clicks on
    #     external links, tracking pixels, AI-widget XHRs, etc. all fail.
    #
    # Rationale: HotpotQA was constructed from Wikipedia, so every gold
    # answer has a canonical Wikipedia page. Restricting the corpus removes
    # SERP AI summaries, anti-bot walls, and off-topic noise in one shot.
    from urllib.parse import urlparse, parse_qs, quote_plus as _qp

    _WIKI_HOSTS = ("wikipedia.org", "wikimedia.org")

    def _is_wiki(host: str) -> bool:
        return any(host == h or host.endswith("." + h) for h in _WIKI_HOSTS)

    _orig_visit = web_surfer._playwright_controller.visit_page
    _route_state = {"installed": False}

    async def _install_route(page):
        async def _abort_external(route):
            host = urlparse(route.request.url).netloc
            if _is_wiki(host):
                await route.continue_()
            else:
                await route.abort()
        await page.route("**/*", _abort_external)

    async def _visit_wiki_only(page, url, *vargs, **vkwargs):
        if not _route_state["installed"]:
            try:
                await _install_route(page)
                _route_state["installed"] = True
            except Exception:
                pass
        parsed = urlparse(url)
        if parsed.netloc.endswith("bing.com") and parsed.path == "/search":
            q = parse_qs(parsed.query).get("q", [""])[0]
            url = f"https://en.wikipedia.org/w/index.php?search={_qp(q)}&fulltext=1"
        elif not _is_wiki(parsed.netloc):
            q = (parsed.netloc + parsed.path).strip("/")
            url = f"https://en.wikipedia.org/w/index.php?search={_qp(q)}&fulltext=1"
        return await _orig_visit(page, url, *vargs, **vkwargs)

    web_surfer._playwright_controller.visit_page = _visit_wiki_only

    task = (
        f"Question: {question}\n\n"
        "You are answering an open-domain HotpotQA question. No context "
        "paragraphs are provided. All evidence MUST be gathered by the "
        "WebSurfer agent from Wikipedia — the browser is restricted to "
        "wikipedia.org / wikimedia.org, so do NOT try other search engines "
        "or external sites. Do NOT rely on prior knowledge, and do NOT "
        "answer from memory. Expect multi-hop retrieval: you will typically "
        "need WebSurfer to consult 2+ Wikipedia pages before you have enough "
        "evidence. Note that WebSurfer can click any search result to open "
        "the full Wikipedia article and read its entire body — do not rely "
        "on search-result snippets alone; click into the relevant article "
        "and read it. The Assistant should plan and synthesize: if the "
        "available evidence is partial, ambiguous, or does not directly "
        "answer the question, the Assistant MUST explicitly say so and "
        "request a specific additional Wikipedia lookup (name the page or "
        "query). Only commit to a final answer when the evidence is "
        "unambiguous. ComputerTerminal may run code if useful, but facts "
        "must trace back to WebSurfer output."
    )

    # Stopping conditions: whichever fires first.
    #   (1) Hard cap: max_turns=20 speaker dispatches — deterministic,
    #       unambiguous, independent of how many internal messages each
    #       speaker emits per turn.
    #   (2) LLM-as-judge: LLMTermination asks the orchestrator client
    #       "is the task solved?" after each speaker turn and stops early
    #       with TERMINATE. Kept because many tasks are answerable well
    #       before turn 20.
    MAX_TURNS = 20
    llm_termination = LLMTermination(
        prompt=f"""Consider the following task:
{task.strip()}

Does the above conversation suggest that the task has been solved?
Reply with one word only: either "TERMINATE" or "CONTINUE".
Do not include any other words in your response.
Do not answer from prior knowledge — answers must be supported by evidence from Wikipedia retrieved during this conversation.
""",
        model_client=orchestrator_client,
        call_label=orchestrator_call_label,
    )

    team = SelectorGroupChat(
        [coder, executor, web_surfer],
        model_client=orchestrator_client,
        termination_condition=llm_termination,
        max_turns=MAX_TURNS,
        allow_repeated_speaker=True,
    )

    print(f"[TIMING] Import + setup overhead: {time.time() - _script_start:.2f}s", flush=True)
    task_start = time.time()
    stream = team.run_stream(task=task.strip())
    result = await Console(stream, output_stats=True)
    print(f"\n[TIMING] Team run completed in {time.time() - task_start:.2f}s", flush=True)

    # One more inference to format the final answer.
    final_context: Sequence[LLMMessage] = []
    for message in result.messages:
        if isinstance(message, TextMessage):
            final_context.append(UserMessage(content=message.content, source=message.source))
        elif isinstance(message, MultiModalMessage):
            if orchestrator_client.model_info["vision"]:
                final_context.append(UserMessage(content=message.content, source=message.source))
            else:
                final_context.append(UserMessage(content=content_to_str(message.content), source=message.source))
    final_context.append(UserMessage(
        content=f"""We have completed the following task:
{question}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a SHORT span — a name, entity, number, year, or "yes"/"no".
Do NOT write a full sentence; do NOT add explanations after the answer.
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
#""".strip(),
        source="user"))

    final_start = time.time()
    orchestrator_call_label["value"] = "finalizer"
    response = await orchestrator_client.create(final_context)
    orchestrator_call_label["value"] = "selector"
    print(response.content, flush=True)
    print(f"\n[TIMING] Final answer inference: {time.time() - final_start:.2f}s", flush=True)
    print(f"[TIMING] Total scenario time: {time.time() - task_start:.2f}s", flush=True)

    # Local HotpotQA-style scoring against the FINAL ANSWER line.
    prediction = ""
    if isinstance(response.content, str):
        matches = re.findall(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response.content, re.IGNORECASE | re.DOTALL)
        if matches:
            prediction = matches[-1].strip()
    f1 = f1_score(prediction, expected_answer)
    em = exact_match(prediction, expected_answer)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Question:    {question}")
    print(f"Prediction:  {prediction}")
    print(f"Expected:    {expected_answer}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Exact Match: {em}")


class LLMTermination(TerminationCondition):
    """Terminate the conversation if an LLM determines the task is complete."""

    def __init__(self, prompt: str, model_client: ChatCompletionClient, termination_phrase: str = "TERMINATE", call_label: dict | None = None) -> None:
        self._prompt = prompt
        self._model_client = model_client
        self._termination_phrase = termination_phrase
        self._terminated = False
        self._context: Sequence[LLMMessage] = []
        self._call_label = call_label

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")

        for message in messages:
            if isinstance(message, TextMessage):
                self._context.append(UserMessage(content=message.content, source=message.source))
            elif isinstance(message, MultiModalMessage):
                if self._model_client.model_info["vision"]:
                    self._context.append(UserMessage(content=message.content, source=message.source))
                else:
                    self._context.append(UserMessage(content=content_to_str(message.content), source=message.source))

        if len(self._context) == 0:
            return None

        # Require at least one non-user (agent) message before allowing termination.
        # Otherwise thinking models terminate immediately on seeing only the question,
        # answering from prior knowledge instead of gathering Wikipedia evidence.
        if not any(m.source != "user" for m in self._context):
            return None

        if self._call_label is not None:
            self._call_label["value"] = "termination"
        try:
            response = await self._model_client.create(self._context + [UserMessage(content=self._prompt, source="user")])
        finally:
            if self._call_label is not None:
                self._call_label["value"] = "selector"

        if isinstance(response.content, str):
            tokens = [t.strip("*_.,!:`\"'") for t in response.content.split()]
            decision = next(
                (t.upper() for t in reversed(tokens) if t.upper() in ("TERMINATE", "CONTINUE")),
                None,
            )
            if decision == self._termination_phrase:
                self._terminated = True
                return StopMessage(content=response.content, source="LLMTermination")
        return None

    async def reset(self) -> None:
        self._terminated = False
        self._context = []


if __name__ == "__main__":
    asyncio.run(main())
