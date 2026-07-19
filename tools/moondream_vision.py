"""
tools/moondream_vision.py — Stub. Real implementation comes later.
For now /vision returns 200 with a placeholder.
"""
import os

def analyze_image_base64(image_b64: str, prompt: str = "") -> str:
    return "Vision offline (moondream stub). Will be implemented in a later phase."

def analyze_image_file(image_path: str, prompt: str = "") -> str:
    return f"Vision offline (moondream stub) for {image_path}."
