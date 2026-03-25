import asyncio
import json
import os
import time

import yaml
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelFamily,
)
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext


def _decode_token(tp) -> str:
    """Decode a TopLogprob's bytes field to a token string."""
    if tp.bytes is not None:
        try:
            return bytes(tp.bytes).decode("utf-8", errors="replace")
        except Exception:
            return f"<bytes:{tp.bytes}>"
    return f"<logprob:{tp.logprob}>"


class LogprobsWrapper:
    """Wraps a ChatCompletionClient to intercept responses and save logprobs to a file."""

    def __init__(self, client: ChatCompletionClient, label: str, output_dir: str = "logprobs_output"):
        self._client = client
        self._label = label
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._call_count = 0

    def _save_logprobs(self, result: CreateResult) -> None:
        self._call_count += 1
        if result.logprobs is None:
            print(f"WARNING: No logprobs returned for {self._label} call {self._call_count}")
            return
        record = {
            "label": self._label,
            "call": self._call_count,
            "content": str(result.content) if result.content else None,
            "finish_reason": result.finish_reason,
            "logprobs": [
                {
                    "token": lp.token,
                    "logprob": lp.logprob,
                    "top_logprobs": [
                        {
                            "token": _decode_token(tp),
                            "logprob": tp.logprob,
                            "bytes": tp.bytes,
                        }
                        for tp in (lp.top_logprobs or [])
                    ] if lp.top_logprobs else None,
                    "bytes": lp.bytes,
                }
                for lp in result.logprobs
            ],
        }
        output_path = os.path.join(self._output_dir, f"{self._label}_{self._call_count}.json")
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Saved logprobs to {os.path.abspath(output_path)}")

    async def create(self, messages, **kwargs):
        result = await self._client.create(messages, **kwargs)
        self._save_logprobs(result)
        return result

    async def create_stream(self, messages, **kwargs):
        async for chunk in self._client.create_stream(messages, **kwargs):
            if isinstance(chunk, CreateResult):
                self._save_logprobs(chunk)
            yield chunk

    # Delegate all other attributes to the wrapped client
    def __getattr__(self, name):
        return getattr(self._client, name)


def make_model_context(model_client) -> ChatCompletionContext:
    info = model_client.model_info if hasattr(model_client, 'model_info') else model_client._client.model_info
    if info["family"] == ModelFamily.R1:
        return ReasoningModelContext()
    return UnboundedChatCompletionContext()


async def main() -> None:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Inject logprobs via extra_body since the Pydantic config model doesn't have logprobs/top_logprobs fields
    mc = config["model_config"]["config"]
    if "extra_body" not in mc or mc["extra_body"] is None:
        mc["extra_body"] = {}
    mc["extra_body"]["logprobs"] = True
    mc["extra_body"]["top_logprobs"] = 20

    client = ChatCompletionClient.load_component(config["model_config"])

    # Derive task_id and repetition from working directory: .../run_name/HumanEval_32/0/
    cwd = os.getcwd()
    rep = os.path.basename(cwd)
    task_id = os.path.basename(os.path.dirname(cwd))
    run_name = os.path.basename(os.path.dirname(os.path.dirname(cwd)))
    # Central logprobs dir scoped per run: .../Results/run_name/logprobs_output/
    results_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    logprobs_dir = os.path.join(results_dir, run_name, "logprobs_output")
    os.makedirs(logprobs_dir, exist_ok=True)
    label = f"{task_id}_rep{rep}"
    wrapped = LogprobsWrapper(client, label, output_dir=logprobs_dir)

    start_time = time.time()

    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
```
"""

    coder = MagenticOneCoderAgent(
        name="coder",
        model_client=wrapped,
    )
    coder._model_context = make_model_context(wrapped)  # type: ignore

    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder"],
    )

    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    agent_team = RoundRobinGroupChat(
        [coder, executor],
        max_turns=2,
        termination_condition=termination,
    )

    stream = agent_team.run_stream(task=task)
    await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
