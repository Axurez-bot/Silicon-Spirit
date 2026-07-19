"""
tools/moondream_vision.py
FEATURE 3: Screen vision via Moondream2 running in Ollama.
Spirit can see a screenshot and describe/analyze what's on screen.
Client-side: spirit_app.pyw captures screenshot and sends base64 to server.
Server-side: this tool sends to Ollama moondream2 for vision analysis.
"""

import base64
import requests
import json
from pathlib import Path
from crewai.tools import tool

# at the top of the file
import os
OLLAMA_BASE  = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
VISION_MODEL = "moondream"
MAX_IMG_BYTES = 5 * 1024 * 1024  # 5MB limit


def analyze_image_base64(image_b64: str, prompt: str = "") -> str:
    """
    Send a base64 image to Moondream via Ollama /api/generate.
    Returns description string.
    """
    if not prompt:
        prompt = "Describe what you see on this screen concisely. Focus on what the user is working on."

    try:
        payload = {
            "model":  VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "Could not analyze image.")
    except requests.exceptions.ConnectionError:
        return "Moondream offline — Ollama not reachable."
    except Exception as e:
        return f"Vision error: {e}"


def analyze_image_file(image_path: str, prompt: str = "") -> str:
    """Analyze an image from a file path."""
    path = Path(image_path)
    if not path.exists():
        return f"Image not found: {image_path}"
    if path.stat().st_size > MAX_IMG_BYTES:
        return f"Image too large (max 5MB): {path.stat().st_size / 1024 / 1024:.1f}MB"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return analyze_image_base64(b64, prompt)


@tool("screen_vision")
def screen_vision_tool(image_path_or_prompt: str) -> str:
    """
    Analyze what's on the Creator's screen using Moondream vision model.
    Requires: 'ollama pull moondream' to be run first.
    Input: either a file path to a screenshot, OR a description prompt.
    Output: natural language description of screen contents.
    Use when Creator asks 'what do you see', 'look at my screen', or similar.
    """
    # Check if it's a file path
    if image_path_or_prompt.startswith("/") or image_path_or_prompt.endswith((".png", ".jpg", ".jpeg")):
        return analyze_image_file(image_path_or_prompt)

    # Otherwise treat as prompt for last uploaded screenshot
    screenshot_path = Path("/app/static/latest_screenshot.png")
    if screenshot_path.exists():
        return analyze_image_file(str(screenshot_path), image_path_or_prompt)

    return "No screenshot available. Creator needs to share their screen first."