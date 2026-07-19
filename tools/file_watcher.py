"""
tools/file_watcher.py
Advanced multi-tier kernel file event monitoring matrix.
Bridges host directory mutations directly to the LangGraph ReAct proactive audit engine.
"""

import os
import time
import json
import threading
import requests
from pathlib import Path

WATCH_STATE_FILE = Path("/app/spirit_memory/file_watches.json")
EVENT_QUEUE_FILE = Path("/app/spirit_memory/file_events.json")

ALLOWED_WATCH_ROOTS = [
    "/app/spirit_memory",
    "/app/static",
    "/app"  # Expanded to catch workspace script saves
]

_watch_thread = None
_watch_running = False
_observer_instance = None
state_lock = threading.Lock()

# --- Helper State Persistence Functions ---

def _load_watches() -> list:
    if WATCH_STATE_FILE.exists():
        try:
            return json.loads(WATCH_STATE_FILE.read_text())
        except Exception:
            pass
    return []

def _save_watches(watches: list):
    with state_lock:
        WATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WATCH_STATE_FILE.write_text(json.dumps(watches, indent=2))

def _load_events() -> list:
    if EVENT_QUEUE_FILE.exists():
        try:
            return json.loads(EVENT_QUEUE_FILE.read_text())
        except Exception:
            pass
    return []

def _save_events(events: list):
    with state_lock:
        EVENT_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVENT_QUEUE_FILE.write_text(json.dumps(events[-100:], indent=2))

def _is_allowed(path: str) -> bool:
    resolved = str(Path(path).resolve())
    return any(resolved.startswith(str(Path(root).resolve())) for root in ALLOWED_WATCH_ROOTS)

# --- Automated Engine Bridge Callback ---

def _trigger_proactive_pipeline(event_type: str, file_path: Path):
    """
    Asynchronously pipes file modification events to the FastAPI security server.
    Ignores structural system locks and database state files to avoid tracking loops.
    """
    if file_path.suffix != '.py' or 'spirit_memory' in file_path.parts:
        return

    print(f"[Kernel Event] Detect {event_type.upper()}: {file_path.name}")
    
    # Save event globally to queue file
    events = _load_events()
    events.append({
        "type": event_type,
        "path": str(file_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notified": True
    })
    _save_events(events)

    # Dispatches non-blocking update to server gateway
    def dispatch():
        try:
            # Targets container network loopback context
            requests.post(
                "http://127.0.0.1:8501/security/proactive_audit",
                json={"filename": file_path.name},
                timeout=2
            )
        except Exception as e:
            print(f"[Proactive Bridge Warning] Request dispatch failed: {e}")

    threading.Thread(target=dispatch, daemon=True).start()

# --- Kernel Watchdog Implementation Fallback Layers ---

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class SpiritCoreEngineHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                _trigger_proactive_pipeline("modified", Path(event.src_path))
        def on_created(self, event):
            if not event.is_directory:
                _trigger_proactive_pipeline("created", Path(event.src_path))
        def on_deleted(self, event):
            if not event.is_directory:
                _trigger_proactive_pipeline("deleted", Path(event.src_path))

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

def _fallback_poll_loop():
    """Optimized low-overhead system ticker running sub-second differentials."""
    global _watch_running
    snapshots = {}

    while _watch_running:
        watches = _load_watches()
        for watch in watches:
            path = Path(watch.get("path", ""))
            if not path.exists():
                continue

            prev_snap = snapshots.get(str(path), {})
            curr_snap = {}
            
            try:
                for entry in path.rglob("*.py"):
                    if entry.is_file() and 'spirit_memory' not in entry.parts:
                        curr_snap[str(entry)] = entry.stat().st_mtime
            except Exception:
                continue

            if prev_snap:
                for fp, mtime in curr_snap.items():
                    if fp not in prev_snap:
                        _trigger_proactive_pipeline("created", Path(fp))
                    elif mtime != prev_snap.get(fp):
                        _trigger_proactive_pipeline("modified", Path(fp))
                for fp in prev_snap:
                    if fp not in curr_snap:
                        _trigger_proactive_pipeline("deleted", Path(fp))

            snapshots[str(path)] = curr_snap
        time.sleep(1.5)

# --- Primary Operational Interface Methods ---

def start_watcher():
    global _watch_thread, _watch_running, _observer_instance
    if _watch_running:
        return

    _watch_running = True
    watches = _load_watches()
    
    # Pre-seed root tracking paths if file context is blank
    if not watches:
        watches.append({"id": "watch_default_app", "path": "/app", "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        _save_watches(watches)

    if HAS_WATCHDOG:
        try:
            _observer_instance = Observer()
            handler = SpiritCoreEngineHandler()
            for watch in watches:
                w_path = watch.get("path")
                if Path(w_path).exists():
                    _observer_instance.schedule(handler, w_path, recursive=True)
            _observer_instance.start()
            print("[FileWatcher] High-Performance Native Kernel Watchdog Active.")
            return
        except Exception as e:
            print(f"[FileWatcher Initialization Warning] Dynamic library fail: {e}. Transitioning to fallback polling...")

    # Instantiates fallback high-frequency state polling
    _watch_thread = threading.Thread(target=_fallback_poll_loop, daemon=True)
    _watch_thread.start()
    print("[FileWatcher] High-Frequency Dynamic Ticker Array Active.")

def get_pending_events() -> list[dict]:
    events = _load_events()
    pending = [e for e in events if not e.get("notified")]
    for e in events:
        e["notified"] = True
    _save_events(events)
    return pending

# --- Registered System Agent Tool Wrappers ---

def watch_folder_tool(path: str) -> str:
    """
    Register a folder for Spirit to monitor for file changes.
    Allowed paths: /app/spirit_memory/, /app/static/, /app/
    Input: absolute folder path.
    Output: confirmation with watch ID.
    """
    path = path.strip()
    if not _is_allowed(path):
        return f"Access Denied: Path target boundaries violation for '{path}'."
    if not Path(path).exists():
        return f"Target validation failed: Path resource '{path}' does not resolve."

    watches = _load_watches()
    if any(w["path"] == path for w in watches):
        return f"Resource active: Watch rule already registered for path '{path}'."

    watch_id = f"watch_{time.time_ns()}"
    watches.append({
        "id": watch_id,
        "path": path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_watches(watches)
    
    # Recycle system watch bindings dynamically
    global _watch_running, _observer_instance
    if _observer_instance:
        try:
            _observer_instance.stop()
            _observer_instance.join()
        except Exception:
            pass
    _watch_running = False
    start_watcher()
    
    return f"Watch registration successful.\nTarget: {path}\nIdentifier: {watch_id[-12:]}\nReal-Time telemetry link established."

def check_file_events_tool(query: str = "") -> str:
    """
    Check for recent file change events Spirit has detected.
    Input: ignored.
    Output: list of file changes since last check.
    """
    events = get_pending_events()
    if not events:
        return "No unread filesystem adjustments present in the tracking array."

    lines = [f"Filesystem Mutation Log Summary ({len(events)} transactions):"]
    for e in events[-20:]:
        lines.append(f"  [{e['type'].upper()}] -> File resource location: {e['path']} verified at {e['timestamp']}")
    return "\n".join(lines)

def list_watches_tool(query: str = "") -> str:
    """
    List all active folder watches.
    Input: ignored.
    Output: list of watched folders.
    """
    watches = _load_watches()
    if not watches:
        return "Active tracking array contains zero records."
    lines = [f"Active Engine Listeners Summary ({len(watches)} nodes):"]
    for w in watches:
        lines.append(f"  Node Array [{w['id'][-8:]}] -> Target Mountpoint: '{w['path']}' | Initialized: {w['created_at']}")
    return "\n".join(lines)