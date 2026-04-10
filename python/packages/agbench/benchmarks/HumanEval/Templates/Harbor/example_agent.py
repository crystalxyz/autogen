"""
Minimal external Harbor agent used for the agbench smoke test.

It runs entirely on the host: a single OpenAI-compatible chat completion call,
the response is parsed for a Python code block, and the block is written into
the container at `/logs/agent/solution.py` via `environment.exec`. No
container-side install needed (works around the cluster's lack of fakeroot /
fuse-overlayfs that blocks installed agents like mini-swe-agent).

This is also the reference template for the new `benchmarks/Harbor/` benchmark:
users author similar `agent.py` files and drop them under `Templates/<MyAgent>/`.

Endpoint configuration is read from environment variables (set by the agbench
shim before invoking `harbor run`):
  - OPENAI_BASE_URL  / OPENAI_API_BASE
  - OPENAI_API_KEY
The model name passed via `harbor run -m <model>` should be the bare model id
(no provider prefix) — this agent does NOT route through litellm.
"""

from __future__ import annotations

import os
import re
import shlex

from openai import OpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class ExampleAgent(BaseAgent):
    """One-shot LLM agent that writes a python solution into the container."""

    SUPPORTS_ATIF: bool = False

    @staticmethod
    def name() -> str:
        return "agbench-example"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install — the agent runs on the host.
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "OPENAI_API_BASE"
        )
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        if not base_url:
            raise RuntimeError(
                "OPENAI_BASE_URL / OPENAI_API_BASE not set; agbench shim should "
                "forward the SGLang endpoint into the agent process env."
            )
        if not self.model_name:
            raise RuntimeError("No model_name configured for ExampleAgent")
        # Strip any litellm-style provider prefix (`openai/...`) — we talk to
        # the SGLang server directly via the OpenAI client, which expects the
        # bare model id.
        model = self.model_name.split("/", 1)[1] if self.model_name.startswith("openai/") else self.model_name

        client = OpenAI(base_url=base_url, api_key=api_key)
        self.logger.info(f"ExampleAgent: calling {base_url} model={model}")

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python coding assistant. Read the task and "
                        "respond with a SINGLE python code block that defines "
                        "the requested function. Do not include any prose or "
                        "tests — just the python code block."
                    ),
                },
                {"role": "user", "content": instruction},
            ],
            max_tokens=2048,
            temperature=0.0,
        )

        message = resp.choices[0].message
        text = message.content or ""
        if not text and getattr(message, "reasoning_content", None):
            # SGLang in thinking mode emits the answer as `reasoning_content`
            # when content is None; fall back to that so we still get a code
            # block out of the model.
            text = message.reasoning_content or ""

        self.logger.info(f"ExampleAgent: model returned {len(text)} chars")
        if context is not None:
            context.n_input_tokens = (resp.usage.prompt_tokens if resp.usage else 0)
            context.n_output_tokens = (resp.usage.completion_tokens if resp.usage else 0)

        m = _CODE_BLOCK_RE.search(text)
        code = m.group(1) if m else text  # fall back to raw text if no fence

        # Persist the model output to the agent log dir for debugging, then
        # write the extracted code into the container so the verifier finds it.
        await environment.exec("mkdir -p /logs/agent")
        heredoc = "AGBENCH_SOLUTION_EOF"
        await environment.exec(
            f"cat > /logs/agent/solution.py << '{heredoc}'\n{code}\n{heredoc}"
        )
        await environment.exec(
            f"cat > /logs/agent/raw_response.txt << '{heredoc}'\n{text}\n{heredoc}"
        )
