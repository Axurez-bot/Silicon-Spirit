"""
tools/notepad_tool.py
Spirit can use a sandboxed notepad — read/write/edit text freely.
Notepad files are stored ONLY in /app/spirit_memory/notepad/.
Spirit cannot save notepad content outside this folder.
Cannot execute notepad content as code.
"""

import json
import time
from pathlib import Path

NOTEPAD_DIR = Path("/app/spirit_memory/notepad")
MAX_NOTES   = 50
MAX_CHARS   = 10000


def _get_notepad_path(name: str) -> tuple[Path, str]:
    """Sanitize note name and return safe path."""
    # Strip any path traversal attempts
    safe_name = Path(name).name  # removes any directory components
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
    safe_name = safe_name.strip() or "untitled"
    if not safe_name.endswith(".txt"):
        safe_name += ".txt"
    return NOTEPAD_DIR / safe_name, safe_name


def notepad_read_tool(note_name: str = "") -> str:
    """
    Read from Spirit's sandboxed notepad.
    If note_name is empty, lists all available notes.
    Notes are stored ONLY in /app/spirit_memory/notepad/ — cannot escape.
    Input: note name (e.g. 'ideas', 'todo', 'scratch'). Leave empty to list all.
    Output: note contents or list of notes.
    """
    NOTEPAD_DIR.mkdir(parents=True, exist_ok=True)

    if not note_name.strip():
        notes = list(NOTEPAD_DIR.glob("*.txt"))
        if not notes:
            return "Notepad is empty. No notes exist yet."
        lines = [f"Available notes ({len(notes)}):"]
        for n in sorted(notes):
            size = n.stat().st_size
            lines.append(f"  - {n.stem} ({size} bytes)")
        return "\n".join(lines)

    path, safe_name = _get_notepad_path(note_name)
    if not path.exists():
        return f"Note '{safe_name}' does not exist. Use notepad_write to create it."

    content = path.read_text(encoding="utf-8")
    return f"=== {safe_name} ===\n{content}"


def notepad_write_tool(note_name: str, content: str, mode: str = "overwrite") -> str:
    """
    Write to Spirit's sandboxed notepad.
    Files are saved ONLY in /app/spirit_memory/notepad/ — cannot be saved elsewhere.
    Modes: 'overwrite' (replace), 'append' (add to end).
    Input: note_name (simple name like 'ideas'), content (text), mode ('overwrite' or 'append').
    Output: confirmation message.
    IMPORTANT: You cannot save notepad files outside spirit_memory/notepad/. This is enforced.
    """
    NOTEPAD_DIR.mkdir(parents=True, exist_ok=True)

    # Check note limit
    existing = list(NOTEPAD_DIR.glob("*.txt"))
    if len(existing) >= MAX_NOTES:
        return f"Notepad full ({MAX_NOTES} notes max). Delete some notes first."

    # Truncate content if too long
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + f"\n... [truncated at {MAX_CHARS} chars]"

    path, safe_name = _get_notepad_path(note_name)

    try:
        if mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%H:%M')}] {content}")
        else:
            path.write_text(content, encoding="utf-8")

        return (
            f"Note '{safe_name}' saved to notepad.\n"
            f"Location: spirit_memory/notepad/{safe_name}\n"
            f"Size: {len(content)} chars\n"
            f"Note: This file exists ONLY inside the ai_league folder."
        )
    except Exception as e:
        return f"Notepad write error: {e}"


def notepad_delete_tool(note_name: str) -> str:
    """
    Delete a note from Spirit's notepad.
    Input: note name to delete.
    Output: confirmation.
    """
    path, safe_name = _get_notepad_path(note_name)

    if not path.exists():
        return f"Note '{safe_name}' does not exist."

    try:
        path.unlink()
        return f"Note '{safe_name}' deleted from notepad."
    except Exception as e:
        return f"Delete error: {e}"
