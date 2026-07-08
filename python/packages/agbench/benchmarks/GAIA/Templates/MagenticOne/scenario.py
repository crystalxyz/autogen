import asyncio
import os
import time
import yaml
import warnings
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

async def main() -> None:

    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])

    # Unified latency/token instrumentation for all model clients (same format as
    # the SelectorGroupChat template). Emits one parseable line per create() call:
    #   [LATENCY] role=<role> label=<label> duration_s=<f> prompt_tokens=<int|None> completion_tokens=<int|None>
    # Each orchestrator call marks a round boundary; downstream scripts parse these lines.
    def _instrument(client, role):
        _orig_create = client.create

        async def _timed_create(*args, **kwargs):
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
                    f"[LATENCY] role={role} label={role} duration_s={dt:.3f} "
                    f"prompt_tokens={pt} completion_tokens={ct}",
                    flush=True,
                )

        client.create = _timed_create

    _instrument(orchestrator_client, role="orchestrator")
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

    team = MagenticOneGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        max_turns=20,
        final_answer_prompt= f""",
We have completed the following task:

{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()
    )

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Run the task with per-round latency/token accounting.
    task_start = time.time()
    stream = team.run_stream(task=task.strip())

    round_idx = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_time = time.time()
    async for message in stream:
        now = time.time()
        latency = now - last_time
        # A TaskResult signals the end of the run.
        if isinstance(message, TaskResult):
            print(f"\n{'=' * 60}", flush=True)
            print(f"[DONE] Total rounds: {round_idx}", flush=True)
            print(
                f"[Total Tokens] prompt={total_prompt_tokens} completion={total_completion_tokens}",
                flush=True,
            )
            print(f"[TIMING] Total scenario time: {time.time() - task_start:.2f}s", flush=True)
            break

        round_idx += 1
        source = getattr(message, "source", type(message).__name__)
        usage = getattr(message, "models_usage", None)
        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ct = getattr(usage, "completion_tokens", None) if usage else None
        if pt is not None:
            total_prompt_tokens += pt
        if ct is not None:
            total_completion_tokens += ct
        print(f"\n{'=' * 60}", flush=True)
        print(f"[Round {round_idx}] source={source}", flush=True)
        print(f"[Latency] {latency:.2f}s", flush=True)
        print(f"[Tokens] prompt={pt} completion={ct}", flush=True)
        print(f"[Content] {str(getattr(message, 'content', ''))}", flush=True)
        last_time = now

if __name__ == "__main__":
    asyncio.run(main())
