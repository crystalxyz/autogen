import base64
import json
from typing import List, Sequence

from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent


# Wrapper template that gets prepended to the user's solution.
# It runs PUBLIC tests first; if all pass, it runs PRIVATE tests
# (the full test suite) to verify final correctness. Otherwise it
# reports the failures and asks the coder to try again.
WRAPPER_TEMPLATE = r"""
import base64, json, sys, io, traceback
from contextlib import redirect_stdout

USER_CODE = base64.b64decode("__USER_CODE_B64__").decode("utf-8")
PUBLIC_TESTS = json.loads(base64.b64decode("__PUBLIC_TESTS_B64__").decode("utf-8"))
PRIVATE_TESTS = json.loads(base64.b64decode("__PRIVATE_TESTS_B64__").decode("utf-8"))
FN_NAME = base64.b64decode("__FN_NAME_B64__").decode("utf-8") or None


def _coerce(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _run_functional(test):
    inp = test["input"]
    expected = test["output"]
    args = _coerce(inp)
    if not isinstance(args, list):
        args = [args]
    ns = {"__name__": "__solution__"}
    exec(USER_CODE, ns)
    target = None
    if "Solution" in ns:
        try:
            target = getattr(ns["Solution"](), FN_NAME)
        except Exception:
            target = None
    if target is None and FN_NAME and FN_NAME in ns:
        target = ns[FN_NAME]
    if target is None:
        raise RuntimeError(f"Could not locate entry function {FN_NAME!r}")
    actual = target(*args)
    exp = _coerce(expected)
    return actual == exp, actual, exp


def _run_stdin(test):
    inp = test["input"]
    expected = test["output"]
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(inp)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ns = {"__name__": "__main__"}
            exec(USER_CODE, ns)
    finally:
        sys.stdin = old_stdin
    actual = buf.getvalue()
    return actual.strip() == str(expected).strip(), actual, expected


def _run_one(test):
    ttype = (test.get("testtype") or "stdin").lower()
    if ttype in ("functional", "function", "call"):
        return _run_functional(test)
    return _run_stdin(test)


def run_set(tests, label):
    if not tests:
        print(f"{label}: (no tests)")
        return True
    failures = []
    for i, t in enumerate(tests):
        try:
            ok, actual, expected = _run_one(t)
        except Exception as e:
            failures.append((i, f"<exception: {type(e).__name__}: {e}>", t.get("output")))
            continue
        if not ok:
            failures.append((i, actual, expected))
    if failures:
        print(f"{label}: {len(failures)}/{len(tests)} FAILED")
        for i, a, e in failures[:3]:
            a_s = repr(a)[:300]
            e_s = repr(e)[:300]
            print(f"  Test {i}: expected={e_s} got={a_s}")
        return False
    print(f"{label}: ALL {len(tests)} PASSED")
    return True


public_ok = run_set(PUBLIC_TESTS, "PUBLIC TESTS")
if public_ok:
    private_ok = run_set(PRIVATE_TESTS, "PRIVATE TESTS")
    if private_ok:
        print("ALL TESTS PASSED !#!#")
    else:
        print("PRIVATE TESTS FAILED !#!#")
    print("TERMINATE")
else:
    print("PUBLIC TESTS FAILED - TRY AGAIN !#!#")
"""


def _read_file(path: str) -> str:
    with open(path, "rt") as fh:
        return fh.read()


class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = (
            "A python sandbox. Provide a single ```python code block "
            "containing the complete solution. The sandbox will run the "
            "public tests against your code and report failures. Once the "
            "public tests pass, it will additionally run the full hidden "
            "test suite to verify correctness."
        ),
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._public_tests_raw = _read_file("public_tests.json")
        self._private_tests_raw = _read_file("private_tests.json")
        meta = json.loads(_read_file("meta.json"))
        self._fn_name = meta.get("fn_name") or ""

        # Pre-compute the b64 strings that don't change per turn.
        self._public_b64 = base64.b64encode(self._public_tests_raw.encode("utf-8")).decode("ascii")
        self._private_b64 = base64.b64encode(self._private_tests_raw.encode("utf-8")).decode("ascii")
        self._fn_name_b64 = base64.b64encode(self._fn_name.encode("utf-8")).decode("ascii")

    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:
            code_content = block.code
            if block.language and block.language.lower() == "python":
                user_b64 = base64.b64encode(code_content.encode("utf-8")).decode("ascii")
                wrapped = (
                    WRAPPER_TEMPLATE
                    .replace("__USER_CODE_B64__", user_b64)
                    .replace("__PUBLIC_TESTS_B64__", self._public_b64)
                    .replace("__PRIVATE_TESTS_B64__", self._private_b64)
                    .replace("__FN_NAME_B64__", self._fn_name_b64)
                )
                code_content = wrapped
            new_blocks.append(CodeBlock(code=code_content, language=block.language))
        return new_blocks
