import asyncio
import time

import yaml
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext


# Maximum number of (coder, executor) message pairs. With max_turns=6
# the team can take up to 3 coder->executor cycles before giving up.
MAX_TURNS = 6


async def main() -> None:
    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Model context
    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    start_time = time.time()

    # Coder
    coder_agent = MagenticOneCoderAgent(
        name="coder",
        model_client=model_client,
    )
    coder_agent._model_context = model_context  # type: ignore

    # Executor: runs public tests, then private tests if public passes.
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder"],
    )

    # Termination: executor signals TERMINATE only after the public tests pass
    # (then it has also already run the private tests for the final verdict).
    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    agent_team = RoundRobinGroupChat(
        [coder_agent, executor],
        max_turns=MAX_TURNS,
        termination_condition=termination,
    )

    # Read the problem prompt + starter code.
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()
    try:
        with open("starter_code.txt", "rt") as fh:
            starter_code = fh.read().strip()
    except FileNotFoundError:
        starter_code = ""

    if starter_code:
        starter_block = f"\n\nUse the following starter code (you must keep the signature):\n\n```python\n{starter_code}\n```\n"
    else:
        starter_block = ""

    task = f"""Solve the following programming problem. Provide your answer as a single Markdown ```python code block containing the complete solution.

The sandbox will run the public tests against your solution. If any public test fails, you will be told which tests failed and you can revise the code. Once all public tests pass, the sandbox will additionally run the hidden full test suite to verify correctness.

Problem:
{prompt}
{starter_block}"""

    stream = agent_team.run_stream(task=task)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Per-turn token usage summary.
    turn = 0
    for msg in task_result.messages:
        if msg.models_usage is not None:
            turn += 1
            print(
                f"[token_usage] turn={turn} source={msg.source}"
                f" prompt_tokens={msg.models_usage.prompt_tokens}"
                f" completion_tokens={msg.models_usage.completion_tokens}"
                f" reasoning_tokens={msg.models_usage.reasoning_tokens}"
            )


asyncio.run(main())
