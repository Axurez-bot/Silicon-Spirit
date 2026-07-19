"""
tools/file_writer.py
Write/create/modify files inside /app/ (ai_league/) only.
ALL write operations require Creator approval before executing.
Spirit proposes the change — Creator approves or denies via approval endpoint.
Cannot create files outside /app/. Cannot overwrite protected files.
"""

import os
import json
import time
from pathlib import Path

ALLOWED_ROOT = Path("/app")
APPROVAL_FILE = Path("/app/spirit_memory/pending_approvals.json")

# Spirit can NEVER write to these regardless of approval
WRITE_BLACKLIST = [
    "/app/.env",
    "/app/models",
    "/app/Dockerfile",
    "/app/docker-compose.yml",
    "/app/requirements.txt",
]

# Spirit can write freely here without approval (her own scratch space)
WRITE_FREE_ZONES = [
    "/app/spirit_memory",
    "/tmp",
    "/app/static",
]


def _load_approvals() -> list:
    if APPROVAL_FILE.exists():
        try:
            return json.loads(APPROVAL_FILE.read_text())
        except Exception:
            pass
    return []


def _save_approvals(approvals: list):
    APPROVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    APPROVAL_FILE.write_text(json.dumps(approvals, indent=2))


def _is_within_boundary(path: Path) -> tuple[bool, str]:
    try:
        path.resolve().relative_to(ALLOWED_ROOT)
        return True, ""
    except ValueError:
        return False, f"DENIED: {path} is outside the ai_league boundary. Spirit cannot escape."


def _is_blacklisted(path: Path) -> bool:
    resolved = str(path.resolve())
    return any(resolved.startswith(bl) for bl in WRITE_BLACKLIST)


def _is_free_zone(path: Path) -> bool:
    resolved = str(path.resolve())
    return any(resolved.startswith(fz) for fz in WRITE_FREE_ZONES)


def _request_approval(operation: str, file_path: str, content_preview: str) -> str:
    """Queue an approval request and return pending status."""
    approvals = _load_approvals()
    request_id = f"req_{int(time.time_ns())}"
    approvals.append({
        "id":              request_id,
        "status":          "pending",
        "operation":       operation,
        "file_path":       file_path,
        "content_preview": content_preview[:500],
        "requested_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "approved_at":     None,
    })
    _save_approvals(approvals)
    return (
        f"APPROVAL REQUIRED\n"
        f"Request ID: {request_id}\n"
        f"Operation: {operation}\n"
        f"File: {file_path}\n"
        f"Preview: {content_preview[:200]}\n\n"
        f"This request is pending Creator approval.\n"
        f"Check the Spirit UI approval panel to approve or deny."
    )


def _check_approval(file_path: str, operation: str) -> tuple[bool, str]:
    """Check if a pending request for this file+operation was approved."""
    approvals = _load_approvals()
    for req in approvals:
        if (req["file_path"] == file_path and
                req["operation"] == operation and
                req["status"] == "approved"):
            # Mark as consumed
            req["status"] = "consumed"
            _save_approvals(approvals)
            return True, req["id"]
    return False, ""


def file_writer_tool(file_path: str, content: str, operation: str = "write") -> str:
    """
    Write or modify a file inside the ai_league project folder.
    ALL writes outside spirit_memory/ and /tmp/ require Creator approval first.
    Operations: 'write' (create/overwrite), 'append' (add to end).
    NEVER use this to write outside /app/ (ai_league folder).
    Process: Spirit calls this → approval request created → Creator approves in UI → Spirit calls again to execute.
    Input: file_path (absolute), content (text to write), operation ('write' or 'append').
    """
    path = Path(file_path.strip())

    # Boundary check — absolute hard stop
    ok, err = _is_within_boundary(path)
    if not ok:
        return err

    # Blacklist check
    if _is_blacklisted(path):
        return f"DENIED: {file_path} is a protected system file. Spirit cannot modify this."

    # Free zone — write directly (spirit_memory, tmp, static)
    if _is_free_zone(path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if operation == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")
            return f"Written successfully: {file_path} ({len(content)} chars)"
        except Exception as e:
            return f"Write error: {e}"

    # All other paths — require approval
    approved, req_id = _check_approval(file_path, operation)
    if approved:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if operation == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")
            return f"Executed (approved {req_id}): {file_path} written successfully."
        except Exception as e:
            return f"Write error after approval: {e}"
    else:
        return _request_approval(operation, file_path, content)
