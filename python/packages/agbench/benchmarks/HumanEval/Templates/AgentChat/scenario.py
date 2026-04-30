import asyncio
import os
import time

import yaml
from agbench.scenario_logging import emit_llm_call, emit_run_summary
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext


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
    # Set model context.
    coder_agent._model_context = model_context  # type: ignore

    # Executor
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder"],
    )

    # Termination condition
    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    # Define a team
    agent_team = RoundRobinGroupChat([coder_agent, executor], max_turns=2, termination_condition=termination)

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    # Run the team and stream messages to the console.
    stream = agent_team.run_stream(task=task)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Per-call LLM accounting. AgentChat doesn't wrap the model client, so
    # we don't have per-call wall-clock durations — just token counts derived
    # from each message's models_usage. Emit one [LLM_CALL] per LLM-produced
    # message for the unified summary.
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
            emit_llm_call(
                agent=str(getattr(msg, "source", "coder")),
                duration_s=None,
                prompt_tokens=msg.models_usage.prompt_tokens,
                completion_tokens=msg.models_usage.completion_tokens,
                reasoning_tokens=msg.models_usage.reasoning_tokens,
                turn=turn,
            )

    # Canonical end-of-run summary (parsed by run_cmd into result.json).
    # A "round" in this scenario is one coder->executor cycle: count coder messages.
    rounds = sum(
        1 for m in task_result.messages if getattr(m, "source", None) == "coder"
    )
    emit_run_summary(task_result.messages, rounds=rounds)


asyncio.run(main())
