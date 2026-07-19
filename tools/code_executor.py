"""
tools/code_executor.py
Sandboxed Python code execution inside Docker.
Hard limits: 10s timeout, no network calls, no file writes outside /tmp.
Output captured and returned as string.
"""

import sys
import io
import ast
import time
import signal
import traceback
import contextlib

MAX_OUTPUT_CHARS = 4000
TIMEOUT_SECONDS  = 10

# Blocked imports — prevent escape from sandbox
BLOCKED_IMPORTS = {
    "subprocess", "os.system", "socket", "requests",
    "urllib", "http", "ftplib", "smtplib", "paramiko",
    "shutil", "pathlib",  # block filesystem writes
}


def _check_ast(code: str) -> str | None:
    """
    Static analysis — block dangerous patterns before execution.
    Returns error string if blocked, None if clean.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if any(name.startswith(b) for b in BLOCKED_IMPORTS):
                    return f"Blocked import: {name}"

        # Block exec/eval
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("exec", "eval", "__import__"):
                return f"Blocked call: {func.id}()"

    return None


def _run_with_timeout(code: str, timeout: int) -> tuple[str, str]:
    """
    Execute code capturing stdout/stderr.
    Returns (output, error).
    """
    stdout_cap = io.StringIO()
    stderr_cap = io.StringIO()

    # Restricted globals — only safe builtins
    safe_builtins = {
        "print": print, "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "sorted": sorted,
        "reversed": reversed, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "str": str, "int": int, "float": float,
        "bool": bool, "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "any": any, "all": all, "type": type, "isinstance": isinstance,
        "repr": repr, "format": format, "chr": chr, "ord": ord,
        "hex": hex, "oct": oct, "bin": bin, "pow": pow, "divmod": divmod,
    }

    # Allow safe imports
    import math, random, datetime, json, re, itertools, functools, collections
    safe_globals = {
        "__builtins__": safe_builtins,
        "math":        math,
        "random":      random,
        "datetime":    datetime,
        "json":        json,
        "re":          re,
        "itertools":   itertools,
        "functools":   functools,
        "collections": collections,
    }

    output = ""
    error  = ""

    def _exec():
        nonlocal output, error
        try:
            with contextlib.redirect_stdout(stdout_cap), \
                 contextlib.redirect_stderr(stderr_cap):
                exec(compile(code, "<spirit_sandbox>", "exec"), safe_globals)
            output = stdout_cap.getvalue()
        except Exception:
            error = traceback.format_exc()

    import threading
    thread = threading.Thread(target=_exec, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return "", f"Execution timed out after {timeout}s."

    return stdout_cap.getvalue(), stderr_cap.getvalue() or error


def code_executor_tool(code: str) -> str:
    """
    Execute Python code in a safe sandbox and return the output.
    Use this for: calculations, data processing, generating results, solving math.
    Allowed: math, random, datetime, json, re, itertools, functools, collections.
    Blocked: network access, file system writes, subprocess, exec/eval.
    Input: valid Python code as a string.
    Output: stdout output, or error message if execution failed.
    """
    # Static analysis first
    block_reason = _check_ast(code)
    if block_reason:
        return f"Execution blocked: {block_reason}"

    output, error = _run_with_timeout(code, TIMEOUT_SECONDS)

    if error:
        return f"Execution error:\n{error[:MAX_OUTPUT_CHARS]}"

    if not output.strip():
        return "Code executed successfully (no output produced)."

    result = output[:MAX_OUTPUT_CHARS]
    if len(output) > MAX_OUTPUT_CHARS:
        result += f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"

    return result
