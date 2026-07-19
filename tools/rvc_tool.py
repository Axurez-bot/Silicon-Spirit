"""
tools/rvc_tool.py — Spirit tool wrapper for RVC voice conversion.

The agent can call this to convert audio OR to speak text in a custom voice
(by first calling Kokoro, then RVC).
"""
import os
from pathlib import Path


def rvc_convert_tool(input_or_text: str) -> str:
    """
    Convert audio to the trained voice (Chisa), or speak text in that voice.

    Input can be:
      - A path to a WAV file (will be converted)
      - Plain text (will be synthesized with Kokoro, then converted)

    Output: path to the converted WAV file.
    """
    from rvc_inference import convert_voice

    MODEL = "/app/models/spirit_voice/Chisa_Voice.pth"
    INDEX = "/app/models/spirit_voice/Chisa_Voice.index"

    # If it's a file path, convert it directly
    if input_or_text.strip().endswith((".wav", ".mp3", ".flac")) and Path(input_or_text).exists():
        out = convert_voice(input_or_text, model_path=MODEL, index_path=INDEX)
        return f"Converted audio saved to: {out}"

    # Otherwise treat as text: synthesize with Kokoro, then convert
    try:
        from server import tts_kokoro_wav, VoiceConfig
        cfg = VoiceConfig()
        wav = tts_kokoro_wav(input_or_text, voice=cfg.voice, speed=cfg.speed)
        if wav is None:
            return "RVC tool: Kokoro synthesis failed, cannot convert."
        out = convert_voice(str(wav), model_path=MODEL, index_path=INDEX)
        # Clean up intermediate Kokoro file
        try:
            os.unlink(wav)
        except OSError:
            pass
        return f"Spoken in Chisa's voice, saved to: {out}"
    except Exception as e:
        return f"RVC tool error: {e}"


def rvc_speak_tool(text: str) -> str:
    """
    Speak text in Chisa's voice. Shortcut for rvc_convert_tool on text.
    """
    return rvc_convert_tool(text)
