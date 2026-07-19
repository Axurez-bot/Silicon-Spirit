"""
tools/system_telemetry.py
Gives Silicon Spirit direct awareness of the host hardware environment.
"""
import os
import shutil
import psutil
from crewai.tools import tool

@tool("get_system_telemetry")
def get_system_telemetry(query: str = "") -> str:
    """
    Queries the host system's hardware performance metrics including CPU load, 
    memory distribution, and main system storage allocation overhead.
    Input: Ignored.
    """
    try:
        # 1. Gather host computing load profile
        cpu_usage = psutil.cpu_percent(interval=0.5)
        cpu_cores = psutil.cpu_count(logical=True)
        
        # 2. Extract memory architecture metrics
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)
        mem_percent = mem.percent
        
        # 3. Check persistent partition bounds
        total, used, free = shutil.disk_usage("/app")
        disk_free_gb = free / (1024 ** 3)
        disk_total_gb = total / (1024 ** 3)

        report = (
            f"[HARDWARE TELEMETRY BRIEF]\n"
            f"- CPU Compute Overhead: {cpu_usage}% utilization across {cpu_cores} threads\n"
            f"- Memory Space Matrix: {mem_used_gb:.2f} GB / {mem_total_gb:.2f} GB allocated ({mem_percent}%)\n"
            f"- Sandbox Space Bounds: {disk_free_gb:.2f} GB free workspace out of {disk_total_gb:.2f} GB total"
        )
        return report

    except Exception as e:
        return f"System telemetry diagnostic pipeline encountered a fault: {str(e)}"