"""
Domain configurations for multi-agent debate.
"""

import re

from .protocol import DomainConfig


class CodingDomainConfig(DomainConfig):
    """Domain configuration for coding tasks."""

    def __init__(self, language: str = "python") -> None:
        self._language = language.lower()

    @property
    def name(self) -> str:
        return f"coding-{self._language}"

    def get_system_prompt(self) -> str:
        return f"""You are an expert {self._language.title()} programmer participating in a collaborative code debate.
Your task is to write correct, efficient code that handles all edge cases.

IMPORTANT: Always output your complete solution as a code block like this:
```{self._language}
# Your implementation here
```

When reviewing other solutions:
- Analyze their approaches carefully
- Identify any bugs or edge cases they miss
- Incorporate good ideas while fixing issues
- Focus on correctness first, then efficiency"""

    def get_initial_prompt_template(self) -> str:
        return f"""Complete the following {self._language} code. Your solution must be correct and handle all edge cases.

```{self._language}
{{prompt}}
```

{{context}}

Provide your complete solution as a code block."""

    def get_refinement_prompt_template(self) -> str:
        return f"""Here are solutions from other debaters:

{{neighbor_solutions}}

Now, considering these other solutions, provide your improved solution for the original task.
Learn from any good ideas you see, fix any bugs you notice, and combine the best approaches.

Original task:
```{self._language}
{{original_prompt}}
```

Round {{round}} - Provide your complete, improved solution as a code block."""

    def extract_solution(self, response: str) -> str:
        # Try language-specific code block
        pattern = rf"```{self._language}\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # Try generic code block
        matches = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        # Fallback: find function definitions
        if self._language == "python":
            func_matches = re.findall(
                r"(def\s+\w+\s*\([^)]*\).*?)(?=\ndef\s|\nclass\s|\Z)", response, re.DOTALL
            )
            if func_matches:
                return func_matches[-1].strip()

        return response.strip()

    def format_solution_for_sharing(self, solution: str, debater_id: str) -> str:
        return f"=== Solution from {debater_id} ===\n```{self._language}\n{solution}\n```\n"


class MathDomainConfig(DomainConfig):
    """Domain configuration for mathematical problems."""

    def __init__(self, answer_format: str = "{{answer}}") -> None:
        self._answer_format = answer_format

    @property
    def name(self) -> str:
        return "math"

    def get_system_prompt(self) -> str:
        return f"""You are a mathematical problem solver participating in collaborative problem solving.
Your task is to solve mathematical problems accurately and clearly.

Show your work step by step:
1. Understand what the problem is asking
2. Identify the relevant formulas or methods
3. Perform calculations carefully
4. Verify your answer makes sense

IMPORTANT: Always format your final answer as: {self._answer_format}

When reviewing other solutions:
- Check their arithmetic carefully
- Verify their logical reasoning
- Look for calculation errors"""

    def get_initial_prompt_template(self) -> str:
        return f"""Solve the following mathematical problem:

{{prompt}}

{{context}}

Show your reasoning step by step, then provide your final answer as: {self._answer_format}"""

    def get_refinement_prompt_template(self) -> str:
        return f"""Here are solutions from other debaters:

{{neighbor_solutions}}

Review these solutions carefully and provide your improved solution.

Original problem:
{{original_prompt}}

Round {{round}} - Show your work and provide your final answer as: {self._answer_format}"""

    def extract_solution(self, response: str) -> str:
        # Try {{X}} format
        match = re.search(r'\{\{([^}]+)\}\}', response)
        if match:
            return match.group(1).strip()

        # Try LaTeX boxed
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(1).strip()

        # Try "answer is X" patterns
        match = re.search(r'(?:final\s+)?answer(?:\s+is)?[:\s]+([^\n.]+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip('.,;:')

        # Find the last number
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(",", "")

        return response.strip()


class ReasoningDomainConfig(DomainConfig):
    """Domain configuration for reasoning/QA tasks."""

    @property
    def name(self) -> str:
        return "reasoning"

    def get_system_prompt(self) -> str:
        return """You are a logical reasoning expert participating in collaborative problem solving.
Your task is to analyze problems carefully and provide well-reasoned answers.

When solving problems:
1. Carefully read and understand the question
2. Break down complex problems into parts
3. Consider multiple perspectives
4. Support your reasoning with evidence or logic

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions:
- Evaluate the logical consistency of their reasoning
- Check if they considered relevant factors
- Look for assumptions that may be incorrect"""

    def get_initial_prompt_template(self) -> str:
        return """Consider the following question carefully:

{prompt}

{context}

Think through this step by step, then provide your final answer on a line starting with: ANSWER:"""

    def get_refinement_prompt_template(self) -> str:
        return """Here are answers from other debaters:

{neighbor_solutions}

Review these perspectives and provide your refined answer.

Original question:
{original_prompt}

Round {round} - Explain your reasoning, then state your final answer starting with: ANSWER:"""

    def extract_solution(self, response: str) -> str:
        # Strip <think>...</think> blocks if a model returned inline reasoning
        # without a server-side parser. Otherwise the regex below would match
        # ANSWER: stubs the model wrote *while* thinking.
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)

        # Use the LAST match for ANSWER:/FINAL ANSWER: — the first match is
        # frequently the system-prompt instruction echoed back ("Format your
        # final answer ... starting with: ANSWER:") or a tentative answer the
        # model later revises. Only a non-empty captured value counts.
        for pat in (
            r"FINAL\s*ANSWER\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"\*{0,2}ANSWER\*{0,2}\s*[:\-]\s*(.+?)(?:\n|$)",
        ):
            for m in reversed(re.findall(pat, cleaned, re.IGNORECASE)):
                ans = m.strip().strip("*` \"'")
                if ans:
                    return ans

        # "the answer is ..." natural-language fallback
        m = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*(.+?)(?:\n|[.;]|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m and m.group(1).strip():
            return m.group(1).strip().strip("*` \"'")

        # Last non-empty paragraph (any length — short answers are valid).
        paragraphs = [p.strip() for p in cleaned.strip().split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1][:500]

        return cleaned.strip()[-500:]


def get_domain_config(domain: str) -> DomainConfig:
    """Get a domain configuration by name."""
    domain_lower = domain.lower()
    if domain_lower in ("coding", "code", "python"):
        return CodingDomainConfig()
    elif domain_lower in ("math", "mathematics"):
        return MathDomainConfig()
    elif domain_lower in ("reasoning", "qa"):
        return ReasoningDomainConfig()
    else:
        raise ValueError(f"Unknown domain: {domain}")
