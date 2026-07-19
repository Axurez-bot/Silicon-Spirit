"""
tools/goal_manager.py
Spirit's persistent goal store — lives in /app/spirit_memory/goals.json
Goals survive Docker restarts. Spirit can add, complete, list, and prioritize.

Goal structure:
{
    "id": "goal_<timestamp_ns>",
    "title": "Monitor project folder for changes",
    "description": "Watch /app for new .py files and report",
    "status": "active" | "in_progress" | "completed" | "failed" | "paused",
    "priority": 1-5 (5 = highest),
    "created_at": "2026-05-16 14:30:00",
    "due_at": None or "2026-05-16 18:00:00",
    "last_checked": None or timestamp,
    "progress_notes": [],
    "result": None or "what Spirit found/did",
    "autonomous": True  # runs without Creator prompting
}
"""

import json
import time
from pathlib import Path

GOALS_FILE = Path("/app/spirit_memory/goals.json")
MAX_GOALS  = 20


def _load_goals() -> list:
    if GOALS_FILE.exists():
        try:
            return json.loads(GOALS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_goals(goals: list):
    GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOALS_FILE.write_text(json.dumps(goals, indent=2))


def _get_goal(goal_id: str) -> dict | None:
    for g in _load_goals():
        if g["id"] == goal_id or g["id"].endswith(goal_id):
            return g
    return None


# ── Public API (used by autonomous loop, not just tools) ─────

def get_active_goals() -> list[dict]:
    """Returns all active/in_progress goals sorted by priority."""
    goals = _load_goals()
    active = [g for g in goals if g["status"] in ("active", "in_progress")]
    return sorted(active, key=lambda g: g.get("priority", 1), reverse=True)


def add_goal(title: str, description: str = "", priority: int = 3,
             due_at: str = "", autonomous: bool = True) -> dict:
    """Add a new goal to the store."""
    goals = _load_goals()
    if len(goals) >= MAX_GOALS:
        # Auto-clean completed goals first
        goals = [g for g in goals if g["status"] not in ("completed", "failed")]

    goal = {
        "id":             f"goal_{time.time_ns()}",
        "title":          title,
        "description":    description,
        "status":         "active",
        "priority":       max(1, min(5, priority)),
        "created_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "due_at":         due_at or None,
        "last_checked":   None,
        "progress_notes": [],
        "result":         None,
        "autonomous":     autonomous,
    }
    goals.append(goal)
    _save_goals(goals)
    return goal


def update_goal_status(goal_id: str, status: str, note: str = "", result: str = ""):
    """Update a goal's status and optionally add a progress note."""
    goals = _load_goals()
    for g in goals:
        if g["id"] == goal_id or g["id"].endswith(goal_id):
            g["status"]       = status
            g["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if note:
                g["progress_notes"].append(f"[{time.strftime('%H:%M')}] {note}")
            if result:
                g["result"] = result
            break
    _save_goals(goals)


def mark_goal_checked(goal_id: str, note: str = ""):
    """Mark a goal as checked without completing it."""
    update_goal_status(goal_id, "in_progress", note)



def goal_add_tool(goal_spec: str) -> str:
    """
    Add a new goal for Spirit to pursue autonomously.
    Format: 'title | description | priority (1-5) | due_at (optional)'
    Example: 'Monitor project | Watch /app for new Python files | 4 | 2026-05-16 18:00'
    Example: 'Daily summary | Summarize what happened today at 6pm | 3'
    Input: goal specification string in the format above.
    Output: confirmation with goal ID.
    """
    parts = [p.strip() for p in goal_spec.split("|")]
    title       = parts[0] if len(parts) > 0 else "Unnamed goal"
    description = parts[1] if len(parts) > 1 else ""
    priority    = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 3
    due_at      = parts[3] if len(parts) > 3 else ""

    goal = add_goal(title, description, priority, due_at)
    return (
        f"Goal added: {goal['title']}\n"
        f"ID: {goal['id'][-12:]}\n"
        f"Priority: {goal['priority']}/5\n"
        f"Status: active — I will pursue this autonomously."
    )


def goal_list_tool(filter_status: str = "active") -> str:
    """
    List Spirit's current goals.
    Input: status filter — 'active', 'completed', 'all' (default: active)
    Output: formatted list of goals with status and priority.
    """
    goals = _load_goals()
    if filter_status == "all":
        filtered = goals
    elif filter_status == "completed":
        filtered = [g for g in goals if g["status"] in ("completed", "failed")]
    else:
        filtered = [g for g in goals if g["status"] in ("active", "in_progress")]

    if not filtered:
        return f"No {filter_status} goals."

    lines = [f"Goals ({filter_status}):"]
    for g in sorted(filtered, key=lambda x: x.get("priority", 1), reverse=True):
        status_icon = {"active": "○", "in_progress": "◉", "completed": "✓", "failed": "✗", "paused": "⏸"}.get(g["status"], "?")
        lines.append(
            f"{status_icon} [{g['priority']}/5] {g['title']}\n"
            f"   ID: {g['id'][-12:]} | {g['status']}\n"
            f"   {g.get('description','')[:80]}"
        )
        if g.get("last_checked"):
            lines.append(f"   Last checked: {g['last_checked']}")
        if g.get("result"):
            lines.append(f"   Result: {g['result'][:100]}")
    return "\n".join(lines)


def goal_complete_tool(goal_spec: str) -> str:
    """
    Mark a goal as completed.
    Input: 'goal_id | result summary' (result is optional)
    Example: 'abc123456789 | Found 3 new Python files added since yesterday'
    Output: confirmation.
    """
    parts  = [p.strip() for p in goal_spec.split("|", 1)]
    gid    = parts[0]
    result = parts[1] if len(parts) > 1 else ""
    update_goal_status(gid, "completed", result=result)
    return f"Goal {gid[-8:]} marked as completed. Result: {result or 'none recorded'}"


def goal_update_tool(goal_spec: str) -> str:
    """
    Add a progress note to an existing goal.
    Input: 'goal_id | progress note'
    Example: 'abc123456789 | Checked folder — no changes yet'
    Output: confirmation.
    """
    parts = [p.strip() for p in goal_spec.split("|", 1)]
    gid   = parts[0]
    note  = parts[1] if len(parts) > 1 else "checked"
    mark_goal_checked(gid, note)
    return f"Progress noted on goal {gid[-8:]}: {note}"


def goal_pause_tool(goal_id: str) -> str:
    """
    Pause a goal temporarily.
    Input: goal ID.
    Output: confirmation.
    """
    update_goal_status(goal_id.strip(), "paused", "Paused by Spirit")
    return f"Goal {goal_id.strip()[-8:]} paused."
