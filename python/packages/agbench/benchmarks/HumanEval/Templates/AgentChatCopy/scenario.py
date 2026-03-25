import asyncio
import os
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


def make_model_context(model_client: ChatCompletionClient) -> ChatCompletionContext:
    if model_client.model_info["family"] == ModelFamily.R1:
        return ReasoningModelContext()
    return UnboundedChatCompletionContext()


async def main() -> None:
    # Load model configuration with two model configs.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Turn 1 model: qwen3-0.6b nothink (fast, cheap first attempt)
    client_turn1 = ChatCompletionClient.load_component(config["model_config_turn1"])
    # Turn 2 model: qwen3-4b think (stronger, reasoning-enabled second attempt)
    client_turn2 = ChatCompletionClient.load_component(config["model_config_turn2"])

    start_time = time.time()

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    # Turn 1 coder: qwen3-0.6b nothink
    coder1 = MagenticOneCoderAgent(
        name="coder_small",
        model_client=client_turn1,
    )
    coder1._model_context = make_model_context(client_turn1)  # type: ignore

    # Turn 1 executor
    executor1 = CustomCodeExecutorAgent(
        name="executor_1",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder_small"],
    )

    # Turn 2 coder: qwen3-4b think
    coder2 = MagenticOneCoderAgent(
        name="coder_large",
        model_client=client_turn2,
    )
    coder2._model_context = make_model_context(client_turn2)  # type: ignore

    # Turn 2 executor
    executor2 = CustomCodeExecutorAgent(
        name="executor_2",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder_large"],
    )

    # Terminate early if turn 1 already passes, or after turn 2
    termination = TextMentionTermination(text="TERMINATE", sources=["executor_1", "executor_2"])

    # Single team with shared conversation thread:
    # coder_small -> executor_1 -> coder_large -> executor_2
    # coder_large sees the full history: task, coder_small's code, executor_1's test results
    agent_team = RoundRobinGroupChat(
        [coder1, executor1, coder2, executor2],
        max_turns=4,
        termination_condition=termination,
    )

    # Run the team and stream messages to the console.
    stream = agent_team.run_stream(task=task)
    await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
