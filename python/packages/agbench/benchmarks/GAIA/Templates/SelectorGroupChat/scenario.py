import time
_script_start = time.time()

import asyncio
import logging
import os
import yaml
import warnings
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
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage, BaseAgentEvent, BaseChatMessage, HandoffMessage, MultiModalMessage, StopMessage
from autogen_core.models import LLMMessage, UserMessage, AssistantMessage

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

# Enable trace logging so selector decisions are printed
logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.DEBUG)
# Silence the per-call tiktoken fallback warning ("Model X not found.
# Using cl100k_base encoding.") — cl100k_base IS the intended fallback for
# any non-OpenAI model and the warning fires once per create() call.
logging.getLogger("autogen_core.trace").setLevel(logging.ERROR)

async def main() -> None:

    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])

    # Unified latency instrumentation for all model clients.
    # Emits one line per create() call in a parseable format:
    #   [LATENCY] role=<role> label=<label> duration_s=<f> prompt_tokens=<int|None> completion_tokens=<int|None>
    # For the orchestrator the label is flipped between selector/termination/finalizer
    # via orchestrator_call_label; agent clients use a fixed label equal to their role.
    orchestrator_call_label = {"value": "selector"}

    def _instrument(client, role, label_ref=None):
        _orig_create = client.create

        async def _timed_create(*args, **kwargs):
            label = label_ref["value"] if label_ref is not None else role
            t0 = time.perf_counter()
            result = None
            try:
                result = await _orig_create(*args, **kwargs)
                return result
            finally:
                dt = time.perf_counter() - t0
                usage = getattr(result, "usage", None) if result is not None else None
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                print(
                    f"[LATENCY] role={role} label={label} duration_s={dt:.3f} "
                    f"prompt_tokens={pt} completion_tokens={ct}",
                    flush=True,
                )

        client.create = _timed_create

    _instrument(orchestrator_client, role="orchestrator", label_ref=orchestrator_call_label)
    _instrument(coder_client, role="coder")
    _instrument(web_surfer_client, role="web_surfer")
    _instrument(file_surfer_client, role="file_surfer")
    
    # Read the prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read().strip()
    filename = "__FILE_NAME__".strip()

    # Set up the team
    coder = MagenticOneCoderAgent(
        "Assistant",
        model_client = coder_client,
    )

    executor = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())

    file_surfer = FileSurfer(
        name="FileSurfer",
        model_client = file_surfer_client,
    )
                
    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client = web_surfer_client,
        downloads_folder=os.getcwd(),
        debug_dir="logs",
        to_save_screenshots=True,
    )

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Termination conditions
    max_messages_termination = MaxMessageTermination(max_messages=20)
    llm_termination = LLMTermination(
        prompt=f"""Consider the following task:
{task.strip()}

Does the above conversation suggest that the task has been solved?
If so, reply "TERMINATE", otherwise reply "CONTINUE"
""",
        model_client=orchestrator_client,
        call_label=orchestrator_call_label,
    )

    termination = max_messages_termination | llm_termination

    # Create the team
    team = SelectorGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        termination_condition=termination,
    )

    # Run the task
    print(f"[TIMING] Import + setup overhead: {time.time() - _script_start:.2f}s", flush=True)
    task_start = time.time()
    stream = team.run_stream(task=task.strip())
    result = await Console(stream, output_stats=True)
    print(f"\n[TIMING] Team run completed in {time.time() - task_start:.2f}s", flush=True)

    # Do one more inference to format the results
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
{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
#""".strip(),
        source="user"))

    # Call the model to evaluate
    final_start = time.time()
    orchestrator_call_label["value"] = "finalizer"
    response = await orchestrator_client.create(final_context)
    orchestrator_call_label["value"] = "selector"
    print(response.content, flush=True)
    print(f"\n[TIMING] Final answer inference: {time.time() - final_start:.2f}s", flush=True)
    print(f"[TIMING] Total scenario time: {time.time() - task_start:.2f}s", flush=True)


class LLMTermination(TerminationCondition):
    """Terminate the conversation if an LLM determines the task is complete.

    Args:
        prompt: The prompt to evaluate in the llm
        model_client: The LLM model_client to use
        termination_phrase: The phrase to look for in the LLM output to trigger termination
    """

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

        # Build the context
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

        # Call the model to evaluate
        if self._call_label is not None:
            self._call_label["value"] = "termination"
        try:
            response = await self._model_client.create(self._context + [UserMessage(content=self._prompt, source="user")])
        finally:
            if self._call_label is not None:
                self._call_label["value"] = "selector"

        # Check for termination
        if isinstance(message.content, str) and self._termination_phrase in response.content:
            self._terminated = True
            return StopMessage(content=message.content, source="LLMTermination")
        return None

    async def reset(self) -> None:
        self._terminated = False
        self._context = []


if __name__ == "__main__":
    asyncio.run(main())
