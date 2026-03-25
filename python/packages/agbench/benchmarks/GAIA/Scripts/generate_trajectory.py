#!/usr/bin/env python3
"""Generate trajectory.jsonl from GAIA ReactAgent benchmark results.

Parses console_log.txt files to extract per-task trajectory metrics
including LLM calls, tool usage, token counts, and scoring.

Usage:
    python Scripts/generate_trajectory.py ../../Results/gaia_validation_level_1__ReactAgent_20260304_140725_662/
"""

import ast
import json
import os
import re
import sys

# Add parent paths so we can import the scorer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Scripts.custom_tabulate import gaia_question_scorer


def parse_chat_completion(text):
    """Parse a ChatCompletion(...) string into a dict with key fields."""
    result = {}

    # Extract created timestamp
    m = re.search(r"created=(\d+)", text)
    if m:
        result["created"] = int(m.group(1))

    # Extract model
    m = re.search(r"model='([^']+)'", text)
    if m:
        result["model"] = m.group(1)

    # Extract finish_reason
    m = re.search(r"finish_reason='([^']+)'", text)
    if m:
        result["finish_reason"] = m.group(1)

    # Extract usage
    m = re.search(r"completion_tokens=(\d+)", text)
    if m:
        result["completion_tokens"] = int(m.group(1))
    m = re.search(r"prompt_tokens=(\d+)", text)
    if m:
        result["prompt_tokens"] = int(m.group(1))
    m = re.search(r"total_tokens=(\d+)", text)
    if m:
        result["total_tokens"] = int(m.group(1))
    m = re.search(r"reasoning_tokens=(\d+)", text)
    if m:
        result["reasoning_tokens"] = int(m.group(1))
    else:
        result["reasoning_tokens"] = 0

    # Extract tool_calls from message
    tool_calls = []
    for tc_match in re.finditer(
        r"ChatCompletionMessageFunctionToolCall\(id='([^']+)',\s*function=Function\(arguments='",
        text,
    ):
        tc_start = tc_match.end()
        # Find the closing: ', name='...')
        tail = re.search(r"',\s*name='([^']+)'\)", text[tc_start:])
        if tail:
            arguments = text[tc_start:tc_start + tail.start()].replace("\\'", "'")
            tool_calls.append({
                "id": tc_match.group(1),
                "name": tail.group(1),
                "arguments": arguments,
            })
    result["tool_calls"] = tool_calls

    # Check for thought content (content starts with '<think>')
    # Avoid regex over huge content; just check if '<think>' appears after content='
    m = re.search(r"message=ChatCompletionMessage\(content='", text)
    if m:
        # Check first 100 chars after content=' for <think>
        snippet = text[m.end():m.end() + 100]
        result["has_thought"] = "<think>" in snippet
    else:
        result["has_thought"] = False

    return result


def parse_tool_call_request(line):
    """Parse a ToolCallRequestEvent line to extract FunctionCall list."""
    calls = []
    for m in re.finditer(
        r"FunctionCall\(id='([^']+)',\s*arguments='((?:[^'\\]|\\.)*)'\s*,\s*name='([^']+)'\)",
        line,
    ):
        calls.append({
            "name": m.group(3),
            "arguments": m.group(2).replace("\\'", "'"),
            "call_id": m.group(1),
        })
    return calls


def parse_tool_call_execution(text):
    """Parse a ToolCallExecutionEvent block to extract FunctionExecutionResult list."""
    results = []
    # Instead of regex over huge content strings, find each FunctionExecutionResult
    # and parse the trailing metadata fields by searching backwards from the closing paren.
    for m in re.finditer(r"FunctionExecutionResult\(content='", text):
        start = m.end()  # position right after content='
        # Find the trailing metadata: ', name='...', call_id='...', is_error=...)
        # Search for the pattern from the end. The tail looks like:
        #   ', name='X', call_id='Y', is_error=Z)
        tail_match = re.search(
            r"',\s*name='([^']+)',\s*call_id='([^']+)',\s*is_error=(True|False)\)",
            text[start:],
        )
        if tail_match:
            content_len = tail_match.start()  # length of content string
            results.append({
                "name": tail_match.group(1),
                "call_id": tail_match.group(2),
                "is_error": tail_match.group(3) == "True",
                "result_chars": content_len,
            })
    return results


def parse_console_log(console_log_text):
    """Parse a console_log.txt into a list of step dicts."""
    steps = []
    lines = console_log_text.split("\n")
    i = 0

    # Separator line
    sep = "=" * 80

    question = None
    final_answer = None

    while i < len(lines):
        line = lines[i]

        # TextMessage (user) - extract question
        if re.match(r"^-+ TextMessage \(user\) -+$", line):
            i += 1
            msg_lines = []
            while i < len(lines) and lines[i] != sep:
                msg_lines.append(lines[i])
            question = "\n".join(msg_lines).strip()
            i += 1
            continue

        # LLM OUTPUT block
        if line.strip() == "LLM OUTPUT:":
            i += 1
            output_lines = []
            while i < len(lines) and lines[i] != sep:
                output_lines.append(lines[i])
                i += 1
            output_text = "\n".join(output_lines)
            if "ChatCompletion(" in output_text:
                cc = parse_chat_completion(output_text)
                step = {
                    "step": len(steps),
                    "type": "llm_call",
                    "created": cc.get("created", 0),
                    "model": cc.get("model", ""),
                    "prompt_tokens": cc.get("prompt_tokens", 0),
                    "completion_tokens": cc.get("completion_tokens", 0),
                    "total_tokens": cc.get("total_tokens", 0),
                    "reasoning_tokens": cc.get("reasoning_tokens", 0),
                    "finish_reason": cc.get("finish_reason", ""),
                    "has_thought": cc.get("has_thought", False),
                    "tool_calls": [
                        {"name": tc["name"], "arguments": tc["arguments"]}
                        for tc in cc.get("tool_calls", [])
                    ],
                }
                steps.append(step)
            i += 1
            continue

        # ThoughtEvent - flag has_thought on the most recent llm_call
        if re.match(r"^-+ ThoughtEvent ", line):
            # The ThoughtEvent comes AFTER the LLM OUTPUT that produced it
            # Mark the last llm_call step
            for s in reversed(steps):
                if s["type"] == "llm_call":
                    s["has_thought"] = True
                    break
            # Skip past the thought content
            i += 1
            continue

        # ToolCallRequestEvent - we already capture tool_calls from ChatCompletion
        # Skip these lines
        if re.match(r"^-+ ToolCallRequestEvent ", line):
            i += 1
            continue

        # ToolCallExecutionEvent
        if re.match(r"^-+ ToolCallExecutionEvent ", line):
            i += 1
            exec_lines = []
            while i < len(lines) and not (
                lines[i] == sep
                or re.match(r"^-+ ", lines[i])
                or lines[i].startswith("LLM INPUT:")
                or lines[i].startswith("LLM OUTPUT:")
            ):
                exec_lines.append(lines[i])
                i += 1
            exec_text = "\n".join(exec_lines)
            tool_results = parse_tool_call_execution(exec_text)
            if tool_results:
                steps.append({
                    "step": len(steps),
                    "type": "tool_execution",
                    "tools": tool_results,
                })
            continue

        i += 1

    # Extract FINAL ANSWER (last match, same as scorer)
    matches = re.findall(r"FINAL ANSWER:(.*?)\n", console_log_text, re.DOTALL)
    if matches:
        final_answer = matches[-1].strip()
        steps.append({
            "step": len(steps),
            "type": "final_answer",
            "answer": final_answer,
        })

    # Extract runtime
    total_time = None
    m = re.search(r"SCENARIO\.PY RUNTIME:\s*([\d.]+)", console_log_text)
    if m:
        total_time = float(m.group(1))

    return {
        "question": question,
        "final_answer": final_answer,
        "total_time_seconds": total_time,
        "steps": steps,
    }


def process_task_dir(task_dir, task_id, repetition):
    """Process a single task/repetition directory and return a trajectory dict."""
    console_log_path = os.path.join(task_dir, "console_log.txt")
    expected_answer_path = os.path.join(task_dir, "expected_answer.txt")

    if not os.path.isfile(console_log_path):
        return None

    with open(console_log_path, "r") as f:
        console_log_text = f.read()

    expected_answer = None
    if os.path.isfile(expected_answer_path):
        with open(expected_answer_path, "r") as f:
            expected_answer = f.read().strip()

    parsed = parse_console_log(console_log_text)

    # Compute aggregates
    llm_steps = [s for s in parsed["steps"] if s["type"] == "llm_call"]
    tool_steps = [s for s in parsed["steps"] if s["type"] == "tool_execution"]

    total_prompt_tokens = sum(s["prompt_tokens"] for s in llm_steps)
    total_completion_tokens = sum(s["completion_tokens"] for s in llm_steps)
    total_tokens = sum(s["total_tokens"] for s in llm_steps)
    num_tool_calls = sum(len(s["tools"]) for s in tool_steps)

    # Get model from first LLM call
    model = llm_steps[0]["model"] if llm_steps else ""

    # Score
    correct = None
    if expected_answer is not None and parsed["final_answer"] is not None:
        correct = gaia_question_scorer(parsed["final_answer"], expected_answer)

    return {
        "task_id": task_id,
        "repetition": repetition,
        "question": parsed["question"],
        "expected_answer": expected_answer,
        "final_answer": parsed["final_answer"],
        "correct": correct,
        "total_time_seconds": parsed["total_time_seconds"],
        "num_steps": len(parsed["steps"]),
        "num_llm_calls": len(llm_steps),
        "num_tool_calls": num_tool_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "model": model,
        "steps": parsed["steps"],
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_directory>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path = os.path.join(results_dir, "trajectory.jsonl")
    count = 0

    with open(output_path, "w") as out_f:
        for task_id in sorted(os.listdir(results_dir)):
            task_path = os.path.join(results_dir, task_id)
            if not os.path.isdir(task_path):
                continue
            # Skip non-UUID directories
            if not re.match(r"^[0-9a-f]{8}-", task_id):
                continue

            for rep_name in sorted(os.listdir(task_path)):
                rep_path = os.path.join(task_path, rep_name)
                if not os.path.isdir(rep_path):
                    continue
                try:
                    repetition = int(rep_name)
                except ValueError:
                    continue

                trajectory = process_task_dir(rep_path, task_id, repetition)
                if trajectory is not None:
                    out_f.write(json.dumps(trajectory) + "\n")
                    count += 1

    print(f"Wrote {count} trajectories to {output_path}")


if __name__ == "__main__":
    main()
