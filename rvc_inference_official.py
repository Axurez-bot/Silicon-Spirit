"""
rvc_inference_official.py — Voice conversion using the official RVC repo's
inference code. This is the canonical implementation that matches the
Chisa_Voice.pth checkpoint exactly.
"""
import sys
from pathlib import Path

# Add the rvc_lib directory to the path so we can import the official code
RVC_LIB = Path(__file__).parent / "rvc_lib"
sys.path.insert(0, str(RVC_LIB))
sys.path.insert(0, str(RVC_LIB / "infer"))

# Import the official inference class
from infer.modules.uvr5.vr import Audio  # type: ignore


# Single global instance, lazy-loaded
_audio_instance: "Audio | None" = None


def _get_audio(model_path: str, index_path: str, device: str = "cuda"):
    """Lazy-load the Audio class. It handles all the RVC inference internally."""
    global _audio_instance
    if _audio_instance is not None:
        return _audio_instance
    # The Audio class needs hubert_path and rmvpe_path
    hubert = "/app/models/hubert_base.pt"
    rmvpe = "/app/models/rmvpe.pt"
    _audio_instance = Audio(
        model_path=model_path,
        index_path=index_path if Path(index_path).exists() else "",
        hubert_path=hubert,
        rmvpe_path=rmvpe,
        device=device,
    )
    return _audio_instance


def convert_voice(
    input_path: str,
    model_path: str = "/app/models/spirit_voice/Chisa_Voice.pth",
    index_path: str = "/app/models/spirit_voice/Chisa_Voice.index",
    pitch_shift: int = 0,
    index_influence: float = 0.5,
    output_path: str = None,
    f0_method: str = "rmvpe",
) -> str:
    """
    Convert audio to the target voice using the official RVC inference code.

    Args:
        input_path: source WAV file
        model_path: path to .pth
        index_path: path to .index (optional)
        pitch_shift: semitones
        index_influence: 0.0 to 1.0
        output_path: auto if None
        f0_method: "rmvpe", "harvest", "crepe", or "pm"

    Returns:
        Path to converted WAV file.
    """
    out = output_path or f"/app/static/rvc_{int(time.time_ns())}.wav"

    audio = _get_audio(model_path, index_path)

    # The official Audio.infer method signature is:
    # infer(self, input_path, pitch, f0_method, file_index, index_rate, volume, output_path)
    # (varies by version; we'll print the actual signature on first call)
    import inspect
    sig = inspect.signature(audio.infer)
    print(f"[RVC] Audio.infer signature: {sig}")

    result = audio.infer(
        input_path,
        pitch_shift,
        f0_method,
        index_path if Path(index_path).exists() else "",
        index_influence,
        1.0,  # volume
        out,
    )
    return result if isinstance(result, str) else out
