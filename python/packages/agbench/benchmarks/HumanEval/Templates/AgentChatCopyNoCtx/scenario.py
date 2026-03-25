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


def make_model_context(model_client: ChatCompletionClient) -> ChatCompletionContext:
    if model_client.model_info["family"] == ModelFamily.R1:
        return ReasoningModelContext()
    return UnboundedChatCompletionContext()


async def run_turn(
    turn_name: str,
    model_client: ChatCompletionClient,
    task: str,
) -> bool:
    """Run a single coder+executor turn. Returns True if tests passed (TERMINATE seen)."""
    coder = MagenticOneCoderAgent(
        name=f"coder_{turn_name}",
        model_client=model_client,
    )
    coder._model_context = make_model_context(model_client)  # type: ignore

    executor = CustomCodeExecutorAgent(
        name=f"executor_{turn_name}",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=[f"coder_{turn_name}"],
    )

    termination = TextMentionTermination(text="TERMINATE", sources=[f"executor_{turn_name}"])

    team = RoundRobinGroupChat(
        [coder, executor],
        max_turns=2,
        termination_condition=termination,
    )

    stream = team.run_stream(task=task)
    result = await Console(stream)

    return result.stop_reason is not None and "Text 'TERMINATE' mentioned" in result.stop_reason


async def main() -> None:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load up to 3 turn configs; pre-create clients before timing starts
    turn_clients = [
        ("turn1", ChatCompletionClient.load_component(config["model_config_turn1"])),
        ("turn2", ChatCompletionClient.load_component(config["model_config_turn2"])),
    ]
    if "model_config_turn3" in config:
        turn_clients.append(("turn3", ChatCompletionClient.load_component(config["model_config_turn3"])))

    start_time = time.time()

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    # Run turns sequentially; stop early if tests pass
    for turn_name, client in turn_clients:
        passed = await run_turn(turn_name, client, task)
        if passed:
            break

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
