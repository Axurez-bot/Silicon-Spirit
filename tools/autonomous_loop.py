"""
tools/autonomous_loop.py
The autonomy engine — ticks Spirit's goal queue on a configurable interval.
For each active goal, Spirit reasons about what to do next and acts.
Results are queued for Creator to see, but Spirit doesn't wait for approval
to check, research, or observe — only to write files outside her sandbox.

Architecture:
  _autonomy_loop() runs in a background thread (started by server.py)
  → loads active goals
  → for each goal, calls _process_goal()
  → _process_goal() invokes Spirit's autonomous_node in engine.py
  → results queued in /app/spirit_memory/autonomous_log.json
  → server.py /autonomous/results endpoint delivers them to client
"""

import os
import json
import time
import threading
from pathlib import Path

AUTONOMY_LOG    = Path("/app/spirit_memory/autonomous_log.json")
AUTONOMY_RESULT_QUEUE: list = []

TICK_INTERVAL   = int(os.getenv("AUTONOMY_TICK_SECS", "300"))   # default: 5 min
_loop_running   = False
_loop_thread    = None


def _load_log() -> list:
    if AUTONOMY_LOG.exists():
        try:
            return json.loads(AUTONOMY_LOG.read_text())
        except Exception:
            pass
    return []


def _append_log(entry: dict):
    log = _load_log()
    log.append(entry)
    AUTONOMY_LOG.parent.mkdir(parents=True, exist_ok=True)
    AUTONOMY_LOG.write_text(json.dumps(log[-200:], indent=2))  # keep last 200


def _process_goal(goal: dict) -> str | None:
    """
    Ask Spirit to work on a goal autonomously.
    Returns a reply string if Spirit has something to report, None if she's still working.
    """
    try:
        from engine import autonomous_node, SpiritState
        from tools.file_watcher import get_pending_events

        # Check for file events relevant to this goal
        file_events = get_pending_events()
        file_context = ""
        if file_events:
            file_context = "\n\nFile change events detected:\n" + "\n".join(
                f"  [{e['type']}] {e['path']} at {e['timestamp']}"
                for e in file_events[:10]
            )

        state = {
            "messages": [("user", f"[AUTONOMOUS_TICK] Working on goal: {goal['title']}\n"
                                   f"Description: {goal['description']}\n"
                                   f"Priority: {goal['priority']}/5\n"
                                   f"Goal ID: {goal['id']}{file_context}")],
            "personality": "Evil Neuro",
            "route":        "autonomous",
            "goal_id":      goal["id"],
            "goal_title":   goal["title"],
        }

        result = autonomous_node(state)
        reply  = result.get("autonomous_reply")
        action = result.get("autonomous_action", "checked")

        # Log the action
        _append_log({
            "goal_id":    goal["id"],
            "goal_title": goal["title"],
            "action":     action,
            "reply":      reply,
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Update goal status
        from tools.goal_manager import mark_goal_checked, update_goal_status
        if result.get("goal_completed"):
            update_goal_status(goal["id"], "completed",
                               note=action, result=reply or "")
        else:
            mark_goal_checked(goal["id"], action)

        return reply

    except Exception as e:
        print(f"[Autonomy] Goal processing failed: {e}")
        return None


def _autonomy_loop():
    """Main autonomy tick loop — runs as daemon thread."""
    global _loop_running
    print(f"[Autonomy] Loop started. Tick interval: {TICK_INTERVAL}s")

    while _loop_running:
        try:
            from tools.goal_manager import get_active_goals
            goals = get_active_goals()

            if goals:
                print(f"[Autonomy] Tick — {len(goals)} active goal(s)")
                for goal in goals:
                    if not _loop_running:
                        break
                    reply = _process_goal(goal)
                    if reply:
                        AUTONOMY_RESULT_QUEUE.append({
                            "goal_id":    goal["id"],
                            "goal_title": goal["title"],
                            "reply":      reply,
                            "timestamp":  time.strftime("%H:%M"),
                            "priority":   goal.get("priority", 1),
                        })
                        print(f"[Autonomy] Goal '{goal['title']}' has a report: {reply[:60]}")
        except Exception as e:
            print(f"[Autonomy] Tick error: {e}")

        # Sleep in small chunks so we can respond to stop signal
        for _ in range(TICK_INTERVAL):
            if not _loop_running:
                break
            time.sleep(1)


def start_autonomy_loop():
    """Start the autonomous goal loop. Called by server.py on startup."""
    global _loop_running, _loop_thread
    if _loop_thread and _loop_thread.is_alive():
        return
    _loop_running = True
    _loop_thread  = threading.Thread(target=_autonomy_loop, daemon=True)
    _loop_thread.start()
    print("[Autonomy] Background loop started.")


def stop_autonomy_loop():
    global _loop_running
    _loop_running = False


def get_pending_results() -> list[dict]:
    """Drain and return queued autonomous results for the client."""
    results = list(AUTONOMY_RESULT_QUEUE)
    AUTONOMY_RESULT_QUEUE.clear()
    return results


def get_autonomy_log(limit: int = 50) -> list[dict]:
    """Return recent autonomy log entries."""
    return _load_log()[-limit:]
