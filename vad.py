"""
vad.py — Silero VAD v4 wrapper. ONNX if available, RMS energy fallback.
"""
import os
import urllib.request
from pathlib import Path
import numpy as np

MODEL_DIR  = Path(os.getenv("SPIRIT_MODEL_DIR", "/app/models"))
MODEL_PATH = MODEL_DIR / "silero_vad.onnx"
MODEL_URL  = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


def _download_model() -> bool:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:
        return True
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[VAD] Downloading {MODEL_URL} -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000
    except Exception as e:
        print(f"[VAD] Download failed: {e}")
        return False


class _EnergyVAD:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def is_speech(self, audio_chunk: np.ndarray, sr: int = 16000) -> bool:
        if audio_chunk is None or audio_chunk.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
        return rms > self.threshold

    def reset(self) -> None:
        pass


class SileroVAD:
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold   = threshold
        self.sample_rate = sample_rate
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._session = None

        if _ORT_AVAILABLE and _download_model():
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(MODEL_PATH),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                print(f"[VAD] Silero ONNX loaded (threshold={threshold})")
            except Exception as e:
                print(f"[VAD] Session failed, using energy fallback: {e}")
                self._session = None
        else:
            print("[VAD] ONNX Runtime missing, using energy fallback")

        self._fallback = None if self._session is not None else _EnergyVAD()

    def is_speech(self, audio_chunk: np.ndarray, sr: int = None) -> bool:
        if audio_chunk is None or audio_chunk.size < 512:
            return False
        audio = audio_chunk.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if audio.max() > 1.5 or audio.min() < -1.5:
            audio = audio / 32768.0

        if self._fallback is not None:
            return self._fallback.is_speech(audio, sr or self.sample_rate)

        try:
            outs = self._session.run(
                None,
                {
                    "input": audio.reshape(1, -1).astype(np.float32),
                    "h":     self._h,
                    "c":     self._c,
                },
            )
            out     = outs[0]
            self._h = outs[1]
            self._c = outs[2]
            prob    = float(out[0][0]) if out.ndim == 2 else float(out[0])
            return prob > self.threshold
        except Exception as e:
            print(f"[VAD] Inference failed: {e}")
            return self._fallback.is_speech(audio, sr) if self._fallback else False

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        if self._fallback:
            self._fallback.reset()
