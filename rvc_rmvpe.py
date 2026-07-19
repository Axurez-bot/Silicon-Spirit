"""
rvc_rmvpe.py — F0 (pitch) extraction using pyworld.
pyworld is reliable, well-tested, and doesn't have API surprises.
RVC models can be trained on pyworld F0; if your model was, this works.
"""
import numpy as np
import torch
from pathlib import Path


def extract_f0(
    audio: np.ndarray,
    sr: int = 40000,
    device: str = "cuda",
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 (Hz) and unvoiced mask from mono audio using pyworld.

    Returns:
        f0: (T,) float32, F0 in Hz. 0 = unvoiced.
        uv: (T,) float32, 1 = unvoiced, 0 = voiced.
    """
    import pyworld as pw
    import librosa

    # pyworld expects float64
    audio_f64 = audio.astype(np.float64)

    # pyworld can fail on very short or silent audio
    try:
        f0, t = pw.harvest(audio_f64, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=5.0)
    except Exception as e:
        print(f"[F0] pyworld.harvest failed: {e}; returning zero F0")
        return np.zeros(audio.shape[0], dtype=np.float32), np.ones(audio.shape[0], dtype=np.float32)

    # f0 is at frame rate (~200 Hz for 5ms period); upsample to audio rate
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
    f0_t = torch.nn.functional.interpolate(f0_t, size=audio.shape[0], mode="linear", align_corners=False)
    f0_out = f0_t.squeeze().numpy().astype(np.float32)

    # Build unvoiced mask
    uv = (f0_out < f0_min).astype(np.float32)
    f0_out = np.where((f0_out < f0_min) | (f0_out > f0_max), 0.0, f0_out).astype(np.float32)
    return f0_out, uv


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """Shift F0 by N semitones. Preserves 0 (unvoiced)."""
    if semitones == 0:
        return f0
    factor = 2.0 ** (semitones / 12.0)
    out = f0.copy()
    voiced = f0 > 0
    out[voiced] = f0[voiced] * factor
    return out
