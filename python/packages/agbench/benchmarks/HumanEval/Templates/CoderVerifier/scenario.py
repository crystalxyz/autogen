import asyncio
import json
import os
from datetime import datetime

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from custom_code_executor import CustomCodeExecutorAgent

# OpenTelemetry configuration for latency tracking
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from reasoning_model_context import ReasoningModelContext

# Create in-memory exporter to collect span data
span_exporter = InMemorySpanExporter()
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))


async def main() -> None:
    # Load model configuration and create separate model clients for Coder and Verifier
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load separate model clients
    coder_model_client = ChatCompletionClient.load_component(config["coder_client"])
    verifier_model_client = ChatCompletionClient.load_component(config["verifier_client"])

    # Model context for Coder
    coder_model_context: ChatCompletionContext
    if coder_model_client.model_info["family"] == ModelFamily.R1:
        coder_model_context = ReasoningModelContext()
    else:
        coder_model_context = UnboundedChatCompletionContext()

    # Model context for Verifier
    verifier_model_context: ChatCompletionContext
    if verifier_model_client.model_info["family"] == ModelFamily.R1:
        verifier_model_context = ReasoningModelContext()
    else:
        verifier_model_context = UnboundedChatCompletionContext()

    # Coder Agent - Generates code solutions using coder_client
    coder_agent = MagenticOneCoderAgent(
        name="Coder",
        model_client=coder_model_client,
        model_client_stream=True,  # Enable streaming for TTFT tracking
    )
    # Set model context for coder
    coder_agent._model_context = coder_model_context  # type: ignore

    # Verifier Agent - Reviews code before execution using verifier_client
    # Using AssistantAgent with custom system message for verification role
    with open("verifier_prompt.txt", "r") as f:
        VERIFIER_SYSTEM_MESSAGE = f.read()

    verifier_agent = AssistantAgent(
        name="Verifier",
        model_client=verifier_model_client,
        model_client_stream=True,  # Enable streaming for TTFT tracking
        system_message=VERIFIER_SYSTEM_MESSAGE,
        description="A code verification expert that reviews code for correctness, edge cases, and quality before execution.",
    )
    # Set model context for verifier
    verifier_agent._model_context = verifier_model_context  # type: ignore

    # Executor - Runs code and provides test results
    # Note: sources=["Coder"] means it will execute code from Coder's messages
    executor = CustomCodeExecutorAgent(
        name="Executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["Coder"],  # Only execute code from Coder, not Verifier
        description="A computer terminal that executes Python code and runs unit tests. Reports test results.",
    )

    # Termination condition: Stop when executor says TERMINATE (tests pass)
    termination = TextMentionTermination(text="TERMINATE", sources=["Executor"])

    # Define the team with Coder → Verifier → Executor flow
    # RoundRobinGroupChat will cycle through agents in order
    # Max turns increased to allow for verification iterations
    agent_team = RoundRobinGroupChat(
        [coder_agent, verifier_agent, executor], max_turns=12, termination_condition=termination
    )

    # Load the HumanEval problem prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    # Construct the task with clear instructions
    task = f"""Complete the following python function.

**Instructions for the team:**
1. **Coder**: Write the complete function implementation as a Python code block
2. **Verifier**: Review the code and either APPROVE or REQUEST_CHANGES
3. If APPROVED, **Executor** will run the tests
4. If tests fail or changes requested, **Coder** should revise based on feedback
5. Continue until tests pass

**Function to implement:**

```python
{prompt}
```

**Coder**: Please provide your implementation now.
"""

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task=task)
    await Console(stream)

    # Export agent latency data from OpenTelemetry spans
    spans = span_exporter.get_finished_spans()
    latency_data = []

    # Track turn numbers per agent for per-turn analysis
    agent_turn_counters = {}

    for span in spans:
        print(f"[SPAN_EXPORT_DEBUG] Span name: {span.name}")
        print(
            f"[SPAN_EXPORT_DEBUG] Span attributes keys: {list(span.attributes.keys()) if span.attributes else 'NONE'}"
        )

        # Filter for agent invocation spans
        if "invoke_agent" in span.name:
            agent_name = span.attributes.get("gen_ai.agent.name")
            print(f"[SPAN_EXPORT_DEBUG] ✓ Found invoke_agent span for: {agent_name}")
            print(f"[SPAN_EXPORT_DEBUG]   Has llm.prompt_tokens: {'llm.prompt_tokens' in span.attributes}")
            print(f"[SPAN_EXPORT_DEBUG]   Has llm.ttft_seconds: {'llm.ttft_seconds' in span.attributes}")

        # Filter for agent invocation spans
        if "invoke_agent" in span.name:
            agent_name = span.attributes.get("gen_ai.agent.name")

            # Track turn number for this agent
            if agent_name not in agent_turn_counters:
                agent_turn_counters[agent_name] = 0
            agent_turn_counters[agent_name] += 1
            turn_number = agent_turn_counters[agent_name]

            # Extract base agent metrics
            agent_data = {
                "agent_name": agent_name,
                "agent_id": span.attributes.get("gen_ai.agent.id"),
                "turn_number": turn_number,  # NEW: Track which turn this is for the agent
                "start_time": datetime.fromtimestamp(span.start_time / 1e9).isoformat(),
                "end_time": datetime.fromtimestamp(span.end_time / 1e9).isoformat(),
                "duration_seconds": (span.end_time - span.start_time) / 1e9,
                "status": span.status.status_code.name,
            }

            # Extract LLM-specific metrics if available
            llm_metrics = {}
            if span.attributes.get("llm.ttft_seconds") is not None:
                llm_metrics["ttft_seconds"] = span.attributes.get("llm.ttft_seconds")
            if span.attributes.get("llm.generation_seconds") is not None:
                llm_metrics["generation_seconds"] = span.attributes.get("llm.generation_seconds")
            if span.attributes.get("llm.prompt_tokens") is not None:
                llm_metrics["prompt_tokens"] = span.attributes.get("llm.prompt_tokens")
            if span.attributes.get("llm.completion_tokens") is not None:
                llm_metrics["completion_tokens"] = span.attributes.get("llm.completion_tokens")
            if span.attributes.get("llm.total_tokens") is not None:
                llm_metrics["total_tokens"] = span.attributes.get("llm.total_tokens")
            if span.attributes.get("llm.tokens_per_second") is not None:
                llm_metrics["tokens_per_second"] = span.attributes.get("llm.tokens_per_second")
            if span.attributes.get("llm.cached") is not None:
                llm_metrics["cached"] = span.attributes.get("llm.cached")
            if span.attributes.get("llm.finish_reason") is not None:
                llm_metrics["finish_reason"] = span.attributes.get("llm.finish_reason")

            # NEW: Calculate TPOT (Time Per Output Token)
            # TPOT = (generation_time - ttft) / completion_tokens
            # This measures pure decoding speed per token
            if (
                "generation_seconds" in llm_metrics
                and "ttft_seconds" in llm_metrics
                and "completion_tokens" in llm_metrics
                and llm_metrics["completion_tokens"] > 0
            ):
                decode_time = llm_metrics["generation_seconds"] - llm_metrics["ttft_seconds"]
                llm_metrics["tpot_seconds"] = decode_time / llm_metrics["completion_tokens"]
                llm_metrics["decode_time_seconds"] = decode_time

            # NEW: Add service time breakdown
            # Service time = TTFT + decode_time (excludes any queueing/overhead)
            if "generation_seconds" in llm_metrics:
                llm_metrics["service_time_seconds"] = llm_metrics["generation_seconds"]

            # NEW: Calculate overhead (total duration - generation time)
            if "generation_seconds" in llm_metrics:
                overhead = agent_data["duration_seconds"] - llm_metrics["generation_seconds"]
                agent_data["overhead_seconds"] = overhead

            if llm_metrics:
                agent_data["llm_metrics"] = llm_metrics

            latency_data.append(agent_data)

    # Sort by start time to see chronological order
    latency_data.sort(key=lambda x: x["start_time"])

    # NEW: Add global turn sequence number after sorting
    for i, data in enumerate(latency_data, 1):
        data["global_turn"] = i

    # Save to JSON file
    with open("latency.json", "w") as f:
        json.dump(latency_data, f, indent=2)

    # Print summary to console
    print("\n" + "=" * 80)
    print("AGENT LATENCY & LLM METRICS SUMMARY (Per-Turn)")
    print("=" * 80)
    if latency_data:
        for data in latency_data:
            agent_name = data["agent_name"]
            global_turn = data["global_turn"]
            agent_turn = data["turn_number"]
            duration = data["duration_seconds"]
            overhead = data.get("overhead_seconds", 0)

            print(f"\nTurn {global_turn:2} | {agent_name:10} (#{agent_turn}) - Duration: {duration:6.2f}s", end="")
            if overhead > 0:
                print(f" (overhead: {overhead:5.2f}s)", end="")
            print(f" [{data['status']}]")

            # Display LLM metrics if available
            if "llm_metrics" in data:
                llm = data["llm_metrics"]

                # Line 1: Timing metrics
                if "ttft_seconds" in llm:
                    print(f"         TTFT: {llm['ttft_seconds']:6.3f}s", end="")
                if "decode_time_seconds" in llm:
                    print(f" | Decode: {llm['decode_time_seconds']:6.2f}s", end="")
                if "tpot_seconds" in llm:
                    print(f" | TPOT: {llm['tpot_seconds'] * 1000:6.2f}ms", end="")
                if "tokens_per_second" in llm:
                    print(f" | {llm['tokens_per_second']:6.1f} tok/s", end="")
                print()  # newline

                # Line 2: Token counts and cache status
                if "prompt_tokens" in llm and "completion_tokens" in llm:
                    print(
                        f"         Tokens: {llm['prompt_tokens']:5} prompt + {llm['completion_tokens']:4} completion = {llm.get('total_tokens', llm['prompt_tokens'] + llm['completion_tokens']):5} total",
                        end="",
                    )
                if "cached" in llm and llm["cached"]:
                    print(" [CACHED]", end="")
                if "finish_reason" in llm:
                    print(f" ({llm['finish_reason']})", end="")
                print()  # newline

        # Calculate total and average per agent
        print("\n" + "=" * 80)
        print("PER-AGENT SUMMARY")
        print("=" * 80)
        agent_totals = {}
        agent_llm_totals = {}

        for data in latency_data:
            agent_name = data["agent_name"]

            # Track duration
            if agent_name not in agent_totals:
                agent_totals[agent_name] = []
            agent_totals[agent_name].append(data["duration_seconds"])

            # Track LLM metrics
            if "llm_metrics" in data:
                if agent_name not in agent_llm_totals:
                    agent_llm_totals[agent_name] = {
                        "ttft": [],
                        "tpot": [],
                        "decode_time": [],
                        "service_time": [],
                        "prompt_tokens": [],
                        "completion_tokens": [],
                        "tokens_per_second": [],
                        "cached_count": 0,
                        "total_count": 0,
                    }
                llm = data["llm_metrics"]
                agent_llm_totals[agent_name]["total_count"] += 1

                if "ttft_seconds" in llm:
                    agent_llm_totals[agent_name]["ttft"].append(llm["ttft_seconds"])
                if "tpot_seconds" in llm:
                    agent_llm_totals[agent_name]["tpot"].append(llm["tpot_seconds"])
                if "decode_time_seconds" in llm:
                    agent_llm_totals[agent_name]["decode_time"].append(llm["decode_time_seconds"])
                if "service_time_seconds" in llm:
                    agent_llm_totals[agent_name]["service_time"].append(llm["service_time_seconds"])
                if "prompt_tokens" in llm:
                    agent_llm_totals[agent_name]["prompt_tokens"].append(llm["prompt_tokens"])
                if "completion_tokens" in llm:
                    agent_llm_totals[agent_name]["completion_tokens"].append(llm["completion_tokens"])
                if "tokens_per_second" in llm:
                    agent_llm_totals[agent_name]["tokens_per_second"].append(llm["tokens_per_second"])
                if llm.get("cached"):
                    agent_llm_totals[agent_name]["cached_count"] += 1

        for agent_name, durations in agent_totals.items():
            avg = sum(durations) / len(durations)
            total = sum(durations)
            print(f"\n{agent_name:10} - {len(durations)} turns, Total: {total:6.2f}s, Avg: {avg:6.2f}s/turn")

            # Print LLM averages if available
            if agent_name in agent_llm_totals:
                llm_data = agent_llm_totals[agent_name]

                # Timing metrics
                if llm_data["ttft"]:
                    avg_ttft = sum(llm_data["ttft"]) / len(llm_data["ttft"])
                    print(f"    Avg TTFT: {avg_ttft:6.3f}s", end="")
                if llm_data["tpot"]:
                    avg_tpot = sum(llm_data["tpot"]) / len(llm_data["tpot"])
                    print(f" | Avg TPOT: {avg_tpot * 1000:6.2f}ms", end="")
                if llm_data["tokens_per_second"]:
                    avg_tps = sum(llm_data["tokens_per_second"]) / len(llm_data["tokens_per_second"])
                    print(f" | Avg Speed: {avg_tps:6.1f} tok/s", end="")
                print()

                # Token counts
                if llm_data["prompt_tokens"] and llm_data["completion_tokens"]:
                    total_prompt = sum(llm_data["prompt_tokens"])
                    total_completion = sum(llm_data["completion_tokens"])
                    avg_prompt = total_prompt / len(llm_data["prompt_tokens"])
                    avg_completion = total_completion / len(llm_data["completion_tokens"])
                    print(
                        f"    Tokens: {total_prompt:6} prompt (avg {avg_prompt:6.0f}) + {total_completion:5} completion (avg {avg_completion:5.0f})"
                    )

                # Cache statistics
                if llm_data["total_count"] > 0:
                    cache_hit_rate = (llm_data["cached_count"] / llm_data["total_count"]) * 100
                    print(
                        f"    Cache: {llm_data['cached_count']}/{llm_data['total_count']} hits ({cache_hit_rate:.1f}% hit rate)"
                    )

        print("=" * 70)
        print(f"Latency data saved to: latency.json")
    else:
        print("No agent latency data collected.")
    print("=" * 70)


asyncio.run(main())
