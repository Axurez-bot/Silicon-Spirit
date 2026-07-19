"""
tools/memory_consolidator.py
Autonomously compresses conversation logs into permanent project knowledge vectors.
"""
import os
import re
import json
import sqlite3
from pathlib import Path

MEMORY_BRIEF_PATH = Path("/app/spirit_memory/project_state.json")
DB_PATH = "/app/spirit_memory/checkpoints.db"

def consolidate_memory_brief(router_llm) -> str:
    """
    Reads recent raw SQLite conversation traces, distills core developments,
    and updates the long-term project brief.
    """
    if not os.path.exists(DB_PATH):
        return "No diagnostic databases found to consolidate."

    try:
        # 1. Extract raw recent human and AI text blocks out of checkpoints
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Querying the langgraph database serialization dump safely
        cursor.execute("SELECT checkpoint FROM checkpoints ORDER BY data DESC LIMIT 30")
        rows = cursor.fetchall()
        conn.close()

        # Parse text chunks roughly out of the langgraph binary records
        raw_dump = ""
        for row in rows:
            text_matches = re.findall(r'"content":\s*"([^"]+)"', str(row[0]))
            for text in text_matches:
                if len(text) > 15 and not text.startswith("[SYSTEM]"):
                    raw_dump += f" {text} |"
                    
        if len(raw_dump) < 100:
            return "Insufficient new chat telemetry to run consolidation cycles."

        # 2. Load the existing long-term brief if available
        current_brief = ""
        if MEMORY_BRIEF_PATH.exists():
            try:
                current_brief = MEMORY_BRIEF_PATH.read_text()
            except Exception:
                pass

        # 3. Prompt the router_llm to merge the new context into the master profile
        consolidation_prompt = (
            f"You are the Core Memory Consolidation Node.\n"
            f"Your job is to merge new chat telemetry into the existing Project Brief.\n\n"
            f"--- CURRENT master BRIEF ---\n{current_brief}\n\n"
            f"--- NEW CONVERSATION RAW TELEMETRY ---\n{raw_dump[:2000]}\n\n"
            f"Instructions:\n"
            f"- Output a clean, structured overview of what the Creator is building.\n"
            f"- Note current script issues, system bottlenecks, and tool states.\n"
            f"- Keep it highly dense, explicit, and factual. No conversational filler.\n"
            f"- Output ONLY the final updated brief."
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        updated_brief = router_llm.invoke([
            SystemMessage(content="You manage persistent system memory profiles."),
            HumanMessage(content=consolidation_prompt)
        ]).content.strip()

        # 4. Write to disk
        MEMORY_BRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_BRIEF_PATH.write_text(updated_brief)
        print("[Memory Engine] Project state brief successfully consolidated.")
        return updated_brief

    except Exception as e:
        print(f"[Memory Consolidation Failure]: {e}")
        return f"Error during memory cycle execution: {str(e)}"

def get_memory_brief_context() -> str:
    """Reads the master persistent summary for system prompt integration."""
    if MEMORY_BRIEF_PATH.exists():
        try:
            return f"\n\n[PERSISTENT LONG-TERM PROJECT STATE]:\n{MEMORY_BRIEF_PATH.read_text()}"
        except Exception:
            pass
    return ""