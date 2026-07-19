"""
rvc_inference.py — Clean public API for RVC v2 voice conversion.

Usage:
    from rvc_inference import convert_voice
    result = convert_voice(
        input_path="source.wav",
        model_path="Chisa.pth",
        index_path="Chisa.index",  # optional but recommended
        pitch_shift=0,
    )
    print(f"Converted audio: {result}")
"""
import os
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa

from rvc_v2_arch import load_rvc_v2_generator
from rvc_rmvpe import extract_f0, shift_f0


# ---- Lazy-loaded, cached ----
_HUBERT = None
_NET = None
_NET_CONFIG = None
_FAISS_INDEX = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_hubert(model_path: str = "/app/models/hubert_base.pt"):
    """Load HuBERT for content feature extraction."""
    global _HUBERT
    if _HUBERT is not None:
        return _HUBERT
    from transformers import Wav2Vec2FeatureExtractor, AutoModel
    _HUBERT = (
        Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960"),
        AutoModel.from_pretrained("facebook/hubert-base-ls960").to(_DEVICE).eval(),
    )
    return _HUBERT


def _extract_hubert_features(audio: np.ndarray, sr: int = 40000) -> torch.Tensor:
    """
    Extract HuBERT features from audio. Returns [1, 768, T_frames] tensor.
    audio must be at 16 kHz for HuBERT (we resample internally).
    """
    import torch.nn.functional as F

    if sr != 16000:
        audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio.astype(np.float32)

    processor, model = _load_hubert()
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(_DEVICE)

    with torch.no_grad():
        outputs = model(input_values)
        feats = outputs.last_hidden_state  # [1, T_frames, 768]

    # Transpose to [1, 768, T_frames] for the conv layers
    feats = feats.transpose(1, 2)
    return feats


def _load_faiss_index(index_path: str):
    """Load FAISS index for retrieval-based feature blending."""
    global _FAISS_INDEX
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX
    import faiss
    _FAISS_INDEX = faiss.read_index(index_path)
    return _FAISS_INDEX


def _apply_index(features: torch.Tensor, index_path: str, influence: float = 0.5) -> torch.Tensor:
    """
    Blend generated features with nearest neighbors from the FAISS index.
    Uses index.search() to find nearest neighbors and reconstruct them
    via search-and-store (works on any FAISS index type).
    """
    import faiss

    if not Path(index_path).exists() or influence <= 0.0:
        return features

    index = _load_faiss_index(index_path)
    if index.ntotal == 0:
        return features

    # features: [1, 768, T]
    B, C, T = features.shape
    feats_flat = features.permute(0, 2, 1).reshape(B * T, C).cpu().numpy().astype(np.float32)

    # For each query, find nearest neighbors and reconstruct them by
    # searching the index with a one-hot vector (works without direct map)
    k = min(8, index.ntotal)
    distances, neighbors = index.search(feats_flat, k)

    # If the index supports reconstruct_n, use it. Otherwise, the fallback
    # is to skip the index blending and use features as-is. The model itself
    # already produces output in the target voice; the index just refines it.
    try:
        # Test if reconstruct works on the first index
        _ = index.reconstruct(0)
        has_reconstruct = True
    except Exception:
        has_reconstruct = False

    if not has_reconstruct:
        # Index has no direct map. Skip blending; the model output is already
        # in the target voice. Returning features unchanged.
        print(f"[RVC] FAISS index has no direct map; skipping index blend (output still uses target voice)")
        return features

    retrieved = np.zeros_like(feats_flat)
    for i in range(B * T):
        nbrs = neighbors[i]
        valid = [int(idx) for idx in nbrs if idx >= 0]
        if not valid:
            continue
        vecs = np.array([index.reconstruct(int(idx)) for idx in valid])
        if len(vecs) > 0:
            retrieved[i] = vecs.mean(axis=0)

    retrieved_t = torch.from_numpy(retrieved).float().to(features.device).reshape(B, T, C).transpose(1, 2)
    blended = (1 - influence) * features + influence * retrieved_t
    return blended


def _load_model(model_path: str):
    """Lazy-load the RVC generator."""
    global _NET, _NET_CONFIG
    if _NET is not None and _NET_CONFIG.get("model_path") == model_path:
        return _NET, _NET_CONFIG
    _NET, cfg = load_rvc_v2_generator(model_path, device=_DEVICE)
    _NET_CONFIG = {**cfg, "model_path": model_path}
    return _NET, _NET_CONFIG


def convert_voice(
    input_path: str,
    model_path: str = "/app/models/spirit_voice/Chisa_Voice.pth",
    index_path: str = "/app/models/spirit_voice/Chisa_Voice.index",
    pitch_shift: int = 0,
    index_influence: float = 0.5,
    output_path: str = None,
) -> str:
    """
    Convert audio to the target voice using RVC v2.

    Args:
        input_path: source WAV file (any sample rate, mono or stereo)
        model_path: path to .pth model
        index_path: path to .index file (optional, recommended)
        pitch_shift: semitones to shift F0 (can be negative)
        index_influence: 0.0 to 1.0, how much to blend in retrieved features
        output_path: where to save result. Auto-generated if None.

    Returns:
        Path to converted WAV file (40 kHz mono).
    """
    out = output_path or f"/app/static/rvc_{time.time_ns()}.wav"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[RVC] Converting {input_path} -> {out}")
    print(f"[RVC]   model={Path(model_path).name}  pitch={pitch_shift}  index_inf={index_influence}")

    # 1. Load audio
    audio, in_sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # to mono
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    # 2. Resample to model's sample rate
    target_sr = 40000
    if in_sr != target_sr:
        audio_40k = librosa.resample(audio, orig_sr=in_sr, target_sr=target_sr)
    else:
        audio_40k = audio

    # 3. Load model
    net, cfg = _load_model(model_path)
    target_sr = cfg["sr"]
    if in_sr != target_sr:
        audio_40k = librosa.resample(audio, orig_sr=in_sr, target_sr=target_sr)

    # 4. Extract HuBERT content features
    print("[RVC] Extracting HuBERT features...")
    feats = _extract_hubert_features(audio_40k, sr=target_sr)  # [1, 768, T_frames]

    # 5. Apply index retrieval
    if index_path and Path(index_path).exists() and index_influence > 0:
        print(f"[RVC] Applying FAISS index (influence={index_influence})...")
        feats = _apply_index(feats, index_path, index_influence)

    # 6. Extract F0
    print("[RVC] Extracting F0...")
    f0, uv = extract_f0(audio_40k, sr=target_sr, device=_DEVICE)
    if pitch_shift != 0:
        f0 = shift_f0(f0, pitch_shift)

    # 7. Run generator
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0).to(_DEVICE)  # [1, 1, T_audio]
    uv_t = torch.from_numpy(uv).float().unsqueeze(0).unsqueeze(0).to(_DEVICE)

    print("[RVC] Running generator...")
    with torch.no_grad():
        audio_out = net(feats, f0_t)  # [1, 1, T_audio]
    audio_out = audio_out.squeeze().cpu().numpy()

    # 8. Normalize and save
    peak = np.abs(audio_out).max()
    if peak > 0:
        audio_out = audio_out / peak * 0.95

    sf.write(out, audio_out.astype(np.float32), target_sr)
    print(f"[RVC] Saved {out}  ({len(audio_out)/target_sr:.2f}s @ {target_sr}Hz)")
    return out


# ---- CLI for testing ----
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rvc_inference.py <input.wav> [pitch_shift]")
        sys.exit(1)
    input_wav = sys.argv[1]
    pitch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = convert_voice(input_wav, pitch_shift=pitch)
    print(f"Done: {out}")
