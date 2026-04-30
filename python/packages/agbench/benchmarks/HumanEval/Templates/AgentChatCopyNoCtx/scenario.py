import asyncio
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


def make_model_context(model_client: ChatCompletionClient) -> ChatCompletionContext:
    if model_client.model_info["family"] == ModelFamily.R1:
        return ReasoningModelContext()
    return UnboundedChatCompletionContext()


async def run_turn(
    turn_name: str,
    model_client: ChatCompletionClient,
    task: str,
):
    """Run a single coder+executor turn.

    Returns ``(passed, messages)`` — the boolean is True iff the executor
    emitted TERMINATE; ``messages`` is the list of agent messages produced
    during this turn so the caller can aggregate token usage across turns.
    """
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

    passed = result.stop_reason is not None and "Text 'TERMINATE' mentioned" in result.stop_reason
    return passed, result.messages


async def main() -> None:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load turn configs; pre-create clients before timing starts
    turn_clients = [
        ("turn1", ChatCompletionClient.load_component(config["model_config_turn1"])),
        ("turn2", ChatCompletionClient.load_component(config["model_config_turn2"])),
    ]
    for turn_idx in range(3, 10):
        key = f"model_config_turn{turn_idx}"
        if key in config:
            turn_clients.append((f"turn{turn_idx}", ChatCompletionClient.load_component(config[key])))

    start_time = time.time()

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    # Run turns sequentially; stop early if tests pass.
    # Aggregate messages across all turns so token totals cover the whole run.
    all_messages: list = []
    rounds = 0
    for turn_name, client in turn_clients:
        passed, messages = await run_turn(turn_name, client, task)
        all_messages.extend(messages)
        # Each invocation of ``run_turn`` is one coder->executor cycle.
        rounds += 1
        # Emit one [LLM_CALL] per LLM-produced message in this turn so the
        # per-agent summary distinguishes coder_turn1 / coder_turn2 / ...
        for msg in messages:
            if getattr(msg, "models_usage", None) is None:
                continue
            emit_llm_call(
                agent=str(getattr(msg, "source", f"coder_{turn_name}")),
                duration_s=None,
                prompt_tokens=msg.models_usage.prompt_tokens,
                completion_tokens=msg.models_usage.completion_tokens,
                reasoning_tokens=msg.models_usage.reasoning_tokens,
                turn=turn_name,
            )
        if passed:
            break

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Canonical end-of-run summary parsed by run_cmd into result.json.
    # A "round" here is one configured turn (coder+executor cycle), counting
    # only those that actually ran before early-exit.
    emit_run_summary(all_messages, rounds=rounds)


asyncio.run(main())
