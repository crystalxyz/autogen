import asyncio
import os
import time
import warnings
import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ThoughtEvent, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top 5 results.

    Args:
        query: The search query string.

    Returns:
        A formatted string of search results with titles, URLs, and snippets.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r.get('title', 'No title')}\n   URL: {r.get('href', 'N/A')}\n   {r.get('body', '')}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def visit_url(url: str) -> str:
    """Fetch a webpage and return its text content as markdown.

    Uses markitdown (same as autogen's MultimodalWebSurfer) for high-quality
    HTML-to-markdown conversion, falling back to BeautifulSoup if unavailable.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        The text content of the webpage, truncated to 20000 characters if needed.
    """
    try:
        import io

        import requests

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
            },
        )
        resp.raise_for_status()

        # Use markitdown for HTML-to-markdown conversion (same approach as MultimodalWebSurfer)
        # try:
        from markitdown import MarkItDown                                                     `

        converter = MarkItDown()
        result = converter.convert_stream(io.BytesIO(resp.content), file_extension=".html", url=url)
        text = result.text_content
        # except ImportError:
        #     # Fallback to BeautifulSoup if markitdown is not installed
        #     from bs4 import BeautifulSoup

        #     soup = BeautifulSoup(resp.text, "html.parser")
        #     for tag in soup(["script", "style", "nav", "footer", "header"]):
        #         tag.decompose()
        #     text = soup.get_text(separator="\n", strip=True)

        # if len(text) > 20000:
        #     text = text[:20000] + "\n... [truncated at 20000 characters]"
        return text if text else "Page returned no text content."
    except Exception as e:
        return f"Error fetching URL: {e}"


def read_file(filepath: str) -> str:
    """Read the contents of a text file.

    Args:
        filepath: Path to the file to read.

    Returns:
        The file contents as a string, truncated to 50000 characters if needed.
    """
    try:
        # Check if the file is likely binary
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return f"File '{filepath}' appears to be a binary file. Use Python code execution to process it."
        with open(filepath, "r", errors="replace") as f:
            content = f.read(50000)
        if len(content) == 50000:
            content += "\n... [truncated at 50000 characters]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def list_files(directory: str = ".") -> str:
    """List files and directories in the specified directory.

    Args:
        directory: Path to the directory to list. Defaults to current directory.

    Returns:
        A newline-separated list of file and directory names.
    """
    try:
        entries = os.listdir(directory)
        if not entries:
            return "Directory is empty."
        return "\n".join(sorted(entries))
    except Exception as e:
        return f"Error listing directory: {e}"


FINAL_ANSWER_PROMPT = """
You are a helpful AI assistant that solves tasks using tools.

You have access to the following tools:
- CodeExecutor: Execute Python code to perform computations, data processing, or any programming task.
- web_search: Search the web for information using DuckDuckGo.
- visit_url: Fetch a webpage and return its full text content. Use after web_search to read pages in detail.
- read_file: Read the contents of text files.
- list_files: List files in a directory.

When working on a task:
1. Think step-by-step about what you need to do.
2. Use your tools to gather information and perform actions.
3. When you have the final answer, STOP using tools and respond directly in plain text using the format: FINAL ANSWER: [YOUR FINAL ANSWER]

IMPORTANT: Do NOT use CodeExecutor or any other tool to output your final answer. Simply respond with plain text containing "FINAL ANSWER: [YOUR FINAL ANSWER]". Do NOT call print() or any code to display the answer.

Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()


async def main() -> None:
    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Use react_agent_client if available, otherwise fall back to orchestrator_client
    client_config_key = "qwen3_client"
    model_client = ChatCompletionClient.load_component(config[client_config_key])

    # Read the prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read().strip()
    filename = "__FILE_NAME__".strip()

    # Set up tools
    code_executor = LocalCommandLineCodeExecutor()
    code_execution_tool = PythonCodeExecutionTool(code_executor)

    web_search_tool = FunctionTool(
        web_search, description="Search the web using DuckDuckGo and return the top 5 results."
    )
    visit_url_tool = FunctionTool(
        visit_url,
        description="Fetch a webpage URL and return its text content. Use this after web_search to read full page details.",
    )
    read_file_tool = FunctionTool(read_file, description="Read the contents of a text file.")
    list_files_tool = FunctionTool(list_files, description="List files and directories in the specified directory.")

    # Create the agent
    agent = AssistantAgent(
        name="ReactAgent",
        model_client=model_client,
        tools=[code_execution_tool, web_search_tool, visit_url_tool, read_file_tool, list_files_tool],
        system_message=FINAL_ANSWER_PROMPT,
        reflect_on_tool_use=True,
        max_tool_iterations=15,
    )

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Run the task with latency tracking, writing to console_log.txt
    # Steps match agent -> tool cycles: each ToolCallRequestEvent starts a new step,
    # and the corresponding ToolCallExecutionEvent completes it.
    stream = agent.run_stream(task=task.strip())
    step = 0
    last_time = time.time()
    with open("console_log.txt", "w") as log:

        def log_print(text: str = "") -> None:
            """Print to both stdout and log file."""
            print(text)
            log.write(text + "\n")
            log.flush()

        total_prompt_tokens = 0
        total_completion_tokens = 0

        def log_tokens(msg: object) -> None:
            """Log token usage if available on the message."""
            nonlocal total_prompt_tokens, total_completion_tokens
            usage = getattr(msg, "models_usage", None)
            if usage is not None:
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                log_print(f"[Tokens] prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")

        step_incremented = False  # Track whether step was already incremented for this LLM call

        async for message in stream:
            now = time.time()
            latency = now - last_time
            if isinstance(message, TaskResult):
                log_print(f"\n{'=' * 60}")
                log_print(f"[DONE] Total steps: {step}")
                log_print(f"[Total Tokens] prompt={total_prompt_tokens}, completion={total_completion_tokens}")
                break
            if isinstance(message, ThoughtEvent):
                # ThoughtEvent starts a new LLM call — increment step
                step += 1
                step_incremented = True
                content = str(message.content)
                log_print(f"\n{'=' * 60}")
                log_print(f"[Step {step}] Thought")
                log_print(f"[Latency] {latency:.2f}s")
                log_tokens(message)
                log_print(f"[Content] {content}")
            elif isinstance(message, ToolCallRequestEvent):
                if not step_incremented:
                    # No ThoughtEvent preceded this (non-reasoning model) — increment step
                    step += 1
                step_incremented = False
                log_print(f"\n{'=' * 60}")
                log_print(f"[Step {step}] Agent -> Tool request")
                log_print(f"[Latency] {latency:.2f}s")
                log_tokens(message)
                for tc in message.content:
                    log_print(f"  Tool: {tc.name}({tc.arguments})")
            elif isinstance(message, ToolCallExecutionEvent):
                log_print(f"\n{'=' * 60}")
                log_print(f"[Step {step}] Tool -> Result")
                log_print(f"[Latency] {latency:.2f}s")
                for result in message.content:
                    log_print(f"  [{result.call_id}] {result.content}")
            else:
                msg_type = type(message).__name__
                content = str(getattr(message, "content", ""))
                log_print(f"\n{'=' * 60}")
                log_print(f"[Step {step}] {msg_type}")
                log_print(f"[Latency] {latency:.2f}s")
                log_tokens(message)
                log_print(f"[Content] {content}")
            last_time = now


if __name__ == "__main__":
    asyncio.run(main())
