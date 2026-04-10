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
    agent_team = RoundRobinGroupChat([coder_agent, executor], max_turns=12, termination_condition=termination)

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    # Extract exec_context from code_context.txt so the model knows
    # what variables it will receive and what variable names to use.
    import re

    exec_context_snippet = ""
    with open("code_context.txt", "rt") as fh:
        code_context_text = fh.read()
    m = re.search(r'exec_context\s*=\s*r?"""(.*?)"""', code_context_text, re.DOTALL)
    if m:
        exec_context_snippet = m.group(1).strip()

    # Extract the variable setup lines (everything before [insert])
    var_setup = ""
    if exec_context_snippet:
        lines = exec_context_snippet.split("\n")
        setup_lines = [l for l in lines if l.strip() and l.strip() != "[insert]"]
        var_setup = "\n".join(setup_lines)

    task = f"""Solve the following data science coding problem. The problem provides setup code between <code> tags and asks you to fill in the solution.

IMPORTANT RULES:
1. Your code will be executed in the following environment where input variables are already defined:
```
{var_setup}
```
2. Use EXACTLY the variable names shown above (e.g., if the environment defines `df`, use `df`, not `data` or other names from the problem description).
3. Do NOT repeat imports, variable definitions, or setup code. Only write the solution logic.
4. You MUST store your answer in a variable called `result`.
5. Format your output as a Markdown python code block using triple backticks (```python), NOT <code> tags.

{prompt}
"""

    # Run the team and stream messages to the console.
    stream = agent_team.run_stream(task=task)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Print per-turn token usage summary.
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
