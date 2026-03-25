import re
import subprocess
import sys
from typing import List, Sequence

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_core.code_executor import CodeBlock, CodeExecutor


class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that runs shell scripts (provided in ```sh code blocks) to apply code fixes, then runs the test suite to verify.",
        sources: Sequence[str] | None = None,
        repo_dir: str = "",
        fail_to_pass: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._repo_dir = repo_dir
        self._fail_to_pass = fail_to_pass or []

    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:
            if block.language and block.language.lower() in ("sh", "bash"):
                # Append test execution after the fix script
                test_ids = " ".join(self._fail_to_pass)
                code_content = block.code + f"""

echo "--- Running FAIL_TO_PASS tests ---"
cd {self._repo_dir}
{sys.executable} -m pytest --no-header -rN -x --tb=short -q {test_ids}
PYTEST_EXIT=$?
if [ $PYTEST_EXIT -eq 0 ]; then
    echo "ALL TESTS PASSED !#!#"
    echo "TERMINATE"
else
    echo "SOME TESTS FAILED - TRY AGAIN !#!#"
fi
"""
                new_blocks.append(CodeBlock(code=code_content, language=block.language))
            else:
                new_blocks.append(block)

        return new_blocks
