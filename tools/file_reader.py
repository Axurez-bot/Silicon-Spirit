"""
tools/file_reader.py
Reads files inside Docker — whitelisted to /app/ only (maps to ai_league/).
Handles text, JSON, YAML, CSV. Binary files get metadata only.
Write/create/delete operations require explicit Creator approval via approval system.
"""

import os
import json
import csv
from pathlib import Path

# ============================================================
# CONTAINMENT BOUNDARY
# /app/ inside Docker = ai_league/ on Windows host
# Spirit cannot read, write, or reference anything outside this
# ============================================================

ALLOWED_ROOT = Path("/app")
MAX_CHARS    = 8000

# Paths Spirit can READ freely (no approval needed)
READ_WHITELIST = [
    "/app/spirit_memory",
    "/app/static",
    "/app/voices",
    "/app/tools",
    "/app/config",
    "/tmp",
]

# Paths Spirit can NEVER access even with approval
HARD_BLACKLIST = [
    "/app/.env",           # API keys — never exposed
    "/app/models",         # model weights — read-only by system only
]


def _is_within_boundary(path: Path) -> tuple[bool, str]:
    """Check if path is within the ai_league boundary."""
    try:
        resolved = path.resolve()
        # Must be inside /app
        resolved.relative_to(ALLOWED_ROOT)
        return True, ""
    except ValueError:
        return False, f"ACCESS DENIED: {path} is outside the ai_league boundary."


def _is_blacklisted(path: Path) -> tuple[bool, str]:
    resolved = str(path.resolve())
    for bl in HARD_BLACKLIST:
        if resolved.startswith(bl):
            return True, f"ACCESS DENIED: {bl} is a protected system path."
    return False, ""


def _is_read_whitelisted(path: Path) -> bool:
    resolved = str(path.resolve())
    return any(resolved.startswith(w) for w in READ_WHITELIST)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            with open(path) as f:
                data = json.load(f)
            return json.dumps(data, indent=2)[:MAX_CHARS]
        except Exception as e:
            return f"JSON parse error: {e}"

    if suffix == ".csv":
        try:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            lines = [", ".join(row) for row in rows[:50]]
            result = "\n".join(lines)
            if len(rows) > 50:
                result += f"\n... ({len(rows) - 50} more rows)"
            return result
        except Exception as e:
            return f"CSV parse error: {e}"

    text_suffixes = {".txt", ".md", ".yaml", ".yml", ".log", ".py", ".json"}
    if suffix in text_suffixes or suffix == "":
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_CHARS)
            if len(content) == MAX_CHARS:
                content += f"\n... [truncated at {MAX_CHARS} chars]"
            return content
        except Exception as e:
            return f"Read error: {e}"

    stat = path.stat()
    return (
        f"Binary file: {path.name}\n"
        f"Size: {stat.st_size / 1024:.1f} KB"
    )


def file_reader_tool(file_path: str) -> str:
    """
    Read a file from within the ai_league project folder only.
    Allowed: /app/spirit_memory/, /app/static/, /app/voices/, /app/tools/, /app/config/, /tmp/
    Blocked: anything outside /app/, .env, model weights.
    Input: absolute file path string.
    Output: file contents or metadata for binary files.
    """
    path = Path(file_path.strip())

    # Boundary check
    ok, err = _is_within_boundary(path)
    if not ok:
        return err

    # Blacklist check
    bl, err = _is_blacklisted(path)
    if bl:
        return err

    if not path.exists():
        return f"File not found: {file_path}"

    if path.is_dir():
        items = list(path.iterdir())
        lines = [f"Directory: {path}\n"]
        for item in sorted(items)[:50]:
            size = f"{item.stat().st_size/1024:.1f}KB" if item.is_file() else "DIR"
            lines.append(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name} ({size})")
        if len(items) > 50:
            lines.append(f"  ... ({len(items) - 50} more)")
        return "\n".join(lines)

    # Free read if whitelisted
    if _is_read_whitelisted(path):
        return _read_file(path)

    # For other paths inside /app — allowed but log it
    print(f"[FileReader] Reading non-whitelisted path: {path}")
    return _read_file(path)
