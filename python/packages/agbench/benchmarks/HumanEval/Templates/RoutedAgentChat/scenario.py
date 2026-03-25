import asyncio
import time

import yaml
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily, UserMessage
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext


ROUTER_PROMPT = """You are a model router. Given a coding task, decide which model is sufficient to solve it.

Your options (from cheapest to most capable):
1. "0.6b" - A small 0.6B parameter model. Good for trivial tasks like simple string manipulation, basic math, straightforward list operations.
2. "1.7b" - A medium 1.7B parameter model. Good for moderate tasks requiring some logic, standard algorithms, or multiple steps.
3. "4b" - A larger 4B parameter model. Good for complex tasks requiring advanced algorithms, tricky edge cases, or careful reasoning.

Respond with ONLY one of: 0.6b, 1.7b, 4b

Here is the task:
{task}
"""


async def route_task(router_client: ChatCompletionClient, task: str) -> str:
    """Use the router model to pick which model should handle this task."""
    prompt = ROUTER_PROMPT.format(task=task)
    result = await router_client.create([UserMessage(content=prompt, source="user")])
    choice = result.content.strip().lower()

    # Parse the response - look for one of the valid options
    for option in ["0.6b", "1.7b", "4b"]:
        if option in choice:
            return option

    # Default to 4b if we can't parse the response
    print(f"WARNING: Could not parse router response '{choice}', defaulting to 4b")
    return "4b"


async def main() -> None:
    # Load model configuration.
    # config.yaml should have:
    #   router_config: {...}        - model config for the router LLM
    #   model_configs:
    #     "0.6b": {...}
    #     "1.7b": {...}
    #     "4b":   {...}
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    router_client = ChatCompletionClient.load_component(config["router_config"])

    # Load the task prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    start_time = time.time()

    # Route the task to a model
    chosen = await route_task(router_client, task)
    route_time = time.time()
    print(f"Router chose model: {chosen} (took {route_time - start_time:.2f}s)")

    # Create the chosen model client
    model_client = ChatCompletionClient.load_component(config["model_configs"][chosen])

    # Model context
    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    # Coder
    coder_agent = MagenticOneCoderAgent(
        name="coder",
        model_client=model_client,
    )
    coder_agent._model_context = model_context  # type: ignore

    # Executor
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder"],
    )

    # Termination condition
    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    # Single turn only: coder writes code, executor tests it
    agent_team = RoundRobinGroupChat([coder_agent, executor], max_turns=2, termination_condition=termination)

    # Run the team and stream messages to the console.
    stream = agent_team.run_stream(task=task)
    await Console(stream)

    end_time = time.time()
    print(f"Routed model: {chosen}")
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
