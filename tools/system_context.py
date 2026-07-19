"""
tools/system_context.py
Builds a system context string injected into every Spirit prompt.
Runs server-side inside Docker — gives Spirit awareness of:
  - Current date/time (Malaysia timezone)
  - Day of week
  - Server uptime
  - Available memory stats

Note: active window + clipboard are client-side only (spirit_app.pyw injects those).
"""

import time
import datetime
import os
import platform

_START_TIME = time.time()


def get_system_context() -> str:
    """
    Returns a context block injected into Spirit's system prompt on every request.
    Called from engine.py's build_system_prompt().
    """
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))  # MYT

    uptime_secs  = int(time.time() - _START_TIME)
    uptime_hours = uptime_secs // 3600
    uptime_mins  = (uptime_secs % 3600) // 60

    # Memory info (Linux inside Docker)
    mem_info = ""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {k.strip(): v.strip() for k, v in
               (line.split(":") for line in lines if ":" in line)}
        total = int(mem.get("MemTotal", "0 kB").split()[0]) // 1024
        avail = int(mem.get("MemAvailable", "0 kB").split()[0]) // 1024
        mem_info = f"RAM: {avail}MB free of {total}MB"
    except Exception:
        mem_info = "RAM: unknown"

    return (
        f"[SYSTEM CONTEXT]\n"
        f"Time: {now.strftime('%H:%M')} MYT | "
        f"Date: {now.strftime('%A, %d %B %Y')} | "
        f"Uptime: {uptime_hours}h {uptime_mins}m | "
        f"{mem_info}"
    )


def system_context_tool(query: str = "") -> str:
    """
    Get current system information: time, date, uptime, memory.
    Use this when Creator asks what time it is, how long Spirit has been running,
    or any question about the current environment.
    Input: ignored — always returns current system state.
    Output: current time (MYT), date, server uptime, RAM usage.
    """
    return get_system_context()
