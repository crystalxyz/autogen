import re
from typing import List, Sequence

from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent


class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._code_context = ""
        with open("code_context.txt", "rt") as fh:
            self._code_context = fh.read()

    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:
            code_content = block.code

            # If python, wrap the extracted code in the DS-1000 test harness.
            # code_context defines test_execution(solution: str) which takes
            # the solution code as a string, inserts it into an exec context,
            # and runs assertions.
            if block.language and block.language.lower() == "python":
                # Escape the solution code so it can be passed as a string
                escaped_code = code_content.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
                code_content = self._code_context + '\n\n' + f'''
solution_code = """{escaped_code}"""

try:
    test_execution(solution_code)
    # We can search for this string in the output
    print("ALL TESTS PASSED !#!#")
    print("TERMINATE")
except Exception as e:
    print(f"SOME TESTS FAILED - TRY AGAIN !#!# Error: {{e}}")
'''
            new_blocks.append(CodeBlock(code=code_content, language=block.language))

        return new_blocks
