import re
from typing import List, Sequence
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken


class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        sources: Sequence[str] | None = None,
    ) -> None:
        print("[DEBUG] CustomCodeExecutorAgent __init__ called")
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._test_code = ""
        with open("test.txt", "rt") as fh:
            self._test_code = fh.read()
        print(f"[DEBUG] CustomCodeExecutorAgent initialized with sources={sources}")


    async def extract_code_blocks_from_messages(self, messages: Sequence[BaseChatMessage]) -> List[CodeBlock]:
        """
        Override to check if Verifier approved before extracting code blocks.
        This method is called by on_messages_stream when model_client is None.
        """
        print(f"\n[DEBUG] extract_code_blocks_from_messages called with {len(messages)} messages")

        # Check if the last message (most recent) is from Verifier
        if len(messages) > 0:
            last_msg = messages[-1]
            last_source = getattr(last_msg, 'source', None)
            print(f"[DEBUG] Last message source: '{last_source}'")

            if hasattr(last_msg, 'source') and last_msg.source == "Verifier":
                content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

                # Check for Verifier's directives
                if "REQUEST_CLARIFICATION" in content:
                    print("[DEBUG] ✓ Detected REQUEST_CLARIFICATION - returning empty code blocks (skip execution)")
                    return []
                elif "REQUEST_CHANGES" in content:
                    print("[DEBUG] ✓ Detected REQUEST_CHANGES - returning empty code blocks (skip execution)")
                    return []
                elif "ADVERSARIAL_TEST" in content:
                    print("[DEBUG] ✓ Detected ADVERSARIAL_TEST - returning empty code blocks (skip execution)")
                    return []
                elif "APPROVED" in content:
                    print("[DEBUG] ✓ Detected APPROVED - proceeding with code extraction")
                else:
                    print("[DEBUG] ⚠ Verifier message doesn't contain explicit directive - proceeding anyway")

        # Proceed with normal code extraction
        print("[DEBUG] Calling parent extract_code_blocks_from_messages")
        return await super().extract_code_blocks_from_messages(messages)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken | None = None) -> Response:
        """
        Override on_messages to check if Verifier approved before executing.

        Checks the most recent message from Verifier:
        - If it contains "APPROVED", proceed with execution
        - If it contains "REQUEST_CLARIFICATION", skip execution (Verifier needs clarification from Coder)
        - If it contains "REQUEST_CHANGES", skip execution (Verifier wants code revisions)
        - If it contains "ADVERSARIAL_TEST", skip execution (Verifier wants Coder to trace test cases)
        - Otherwise, proceed with execution (backward compatibility)
        """
        # DEBUG: Print all messages received
        print(f"\n[DEBUG] Executor received {len(messages)} messages:")
        for i, msg in enumerate(messages):
            source = getattr(msg, 'source', 'UNKNOWN')
            content_preview = str(msg.content)[:50] if hasattr(msg, 'content') else 'NO CONTENT'
            print(f"[DEBUG]   Message {i}: source='{source}', content preview: {content_preview}")

        # Check if the previous message is from Verifier
        if len(messages) > 0:
            last_msg = messages[-1]
            last_source = getattr(last_msg, 'source', None)
            print(f"[DEBUG] Last message source: '{last_source}'")

            if hasattr(last_msg, 'source') and last_msg.source == "Verifier":
                # Check if Verifier requested changes
                content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                print(f"[DEBUG] Last message is from Verifier. Content: {content[:200]}")

                if "REQUEST_CLARIFICATION" in content:
                    print("[DEBUG] ✓ Detected REQUEST_CLARIFICATION - skipping execution")
                    # Verifier requested clarification - skip execution
                    return Response(
                        chat_message=TextMessage(
                            content="Skipping execution - Verifier requested clarification. Coder, please answer the Verifier's questions.",
                            source=self.name
                        )
                    )
                elif "REQUEST_CHANGES" in content:
                    print("[DEBUG] ✓ Detected REQUEST_CHANGES - skipping execution")
                    # Verifier requested changes - skip execution
                    return Response(
                        chat_message=TextMessage(
                            content="Skipping execution - Verifier requested changes. Coder, please revise the code based on Verifier's feedback.",
                            source=self.name
                        )
                    )
                elif "ADVERSARIAL_TEST" in content:
                    print("[DEBUG] ✓ Detected ADVERSARIAL_TEST - skipping execution")
                    # Verifier requested adversarial test trace - skip execution
                    return Response(
                        chat_message=TextMessage(
                            content="Skipping execution - Verifier requested adversarial test analysis. Coder, please trace through the provided test cases.",
                            source=self.name
                        )
                    )
                elif "APPROVED" not in content:
                    print("[DEBUG] ⚠ Verifier message doesn't contain APPROVED, REQUEST_CLARIFICATION, REQUEST_CHANGES, or ADVERSARIAL_TEST - continuing anyway")
                    # Verifier didn't explicitly approve or request anything - maybe it's not doing its job
                    # We'll be lenient and still execute, but warn
                    pass  # Continue to execution
                else:
                    print("[DEBUG] ✓ Detected APPROVED - proceeding with execution")
            else:
                print(f"[DEBUG] Last message is NOT from Verifier (source='{last_source}') - proceeding with execution")
        else:
            print("[DEBUG] No messages received - proceeding with execution")

        # Proceed with normal execution
        print("[DEBUG] Calling parent on_messages for execution")
        return await super().on_messages(messages, cancellation_token)


    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        print(f"\n[DEBUG] _extract_markdown_code_blocks called!")
        print(f"[DEBUG] Markdown text preview: {markdown_text[:100]}")

        # NOTE: This method is called when extracting code from messages, BEFORE execution
        # We can't check messages here because we don't have access to the full message history
        # The check needs to happen in on_messages instead

        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        print(f"[DEBUG] Found {len(code_blocks)} code blocks")
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:

            # Handle deepseek
            code_content = block.code
            #m = re.search(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*?)\s*$", code_content, re.DOTALL)
            #if m:
            #    code_content = m.group(2)

            # If python, wrap the extracted code in a unit testing harness
            if block.language and block.language.lower() == "python":
                code_content = self._test_code + """

def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("ALL TESTS PASSED !#!#")
        print("TERMINATE")
    except AssertionError:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")

""" + code_content + """

run_tests(__ENTRY_POINT__)
"""
            new_blocks.append(CodeBlock(code=code_content, language=block.language))

        return new_blocks
