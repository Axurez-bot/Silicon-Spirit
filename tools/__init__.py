"""
tools/__init__.py — Lazy tool registry. server.py and engine.py use
__import__() with try/except for actual loading. This file is here
so `from tools import X` doesn't fail in user scripts, and so a missing optional dep does not crash the whole server.
"""

_TOOL_MAP = {
    "web_search_tool":     "tools.web_search",
    "file_reader_tool":    "tools.file_reader",
    "file_writer_tool":    "tools.file_writer",
    "code_executor_tool":  "tools.code_executor",
    "system_context_tool": "tools.system_context",
    "notepad_read_tool":   "tools.notepad_tool",
    "notepad_write_tool":  "tools.notepad_tool",
    "notepad_delete_tool": "tools.notepad_tool",
}


def __getattr__(name):
    if name in _TOOL_MAP:
        import importlib
        mod = importlib.import_module(_TOOL_MAP[name])
        return getattr(mod, name)
    if name == "get_system_context":
        from tools.system_context import get_system_context
        return get_system_context
    raise AttributeError(f"module 'tools' has no attribute {name!r}")


__all__ = list(_TOOL_MAP.keys()) + ["get_system_context"]
