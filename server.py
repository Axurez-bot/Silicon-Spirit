"""
Silicon Spirit API — Kokoro TTS edition
"""
import os
import time
import shutil
import subprocess
import threading
import json
import re
import warnings
import types
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# ---- RVC Architecture Path Requirements ----
os.environ['rmvpe_root'] = 'assets'
import sys
sys.path.insert(0, '/app')

# ---- torch.load safety patch (RVC .pth files need weights_only=False) ----
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    import fairseq
    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except Exception as e:
    print(f"[RVC Patch] Safe globals registration warning: {e}")

# ---- Direct RVC Engine Modules ----
from infer.modules.vc.modules import VC
from infer.modules.vc.pipeline import Pipeline
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from infer.modules.vc.utils import load_hubert
from infer.lib.audio import load_audio

# ---- Paths ----
STATIC_DIR        = Path("/app/static")
SOURCE_VOICE      = Path("/app/models/spirit_voice")
MODEL_DIR         = Path("/app/models")
LOG_PATH          = Path("/app/spirit_memory/repair_log.json")
WAV_TTL           = 300

for p in (STATIC_DIR, SOURCE_VOICE, MODEL_DIR, Path("/app/spirit_memory")):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Silicon Spirit API v5 (Kokoro)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---- Globals ----
_proactive_queue: list[dict] = []
_last_agent_debug: dict = {}
_last_proactive = time.time()
_proactive_enabled = os.getenv("PROACTIVE_ENABLED", "true").lower() == "true"
PROACTIVE_INTERVAL = int(os.getenv("PROACTIVE_INTERVAL_SECS", "1800"))
APPROVAL_FILE = Path("/app/spirit_memory/pending_approvals.json")

# ===================================================================
# RVC — STABLE ARCHITECTURE AND EXPLICIT MATRIX INITIALIZATION
# ===================================================================
_rvc_hubert_model = None
_rvc_net_g = None
_rvc_tgt_sr = 40000
rvc_lock = threading.Lock()
RVC_AVAILABLE = True # Explicitly mapped inside updated Docker container layers


def init_rvc() -> None:
    global _rvc_hubert_model, _rvc_net_g, _rvc_tgt_sr
    pth = SOURCE_VOICE / "Chisa_Voice.pth"
    if not pth.exists():
        print(f"[RVC] No voice model at {pth}, RVC extraction disabled.")
        return
    try:
        # Full precision config to prevent data type casting collisions
        config = types.SimpleNamespace(
            device='cuda:0', 
            is_half=False, 
            weight_root='', 
            x_pad=3, 
            x_query=10, 
            x_center=60, 
            x_max=65
        )
        
        cpt = torch.load(str(pth), map_location='cpu', weights_only=False)
        _rvc_tgt_sr = cpt.get('sample_rate', 40000)
        
        # Build 768-channel V2 Synthesizer matrix grid layout
        _rvc_net_g = SynthesizerTrnMs768NSFsid(*cpt['config'], is_half=False)
        _rvc_net_g.load_state_dict(cpt['weight'], strict=False)
        _rvc_net_g.eval().to('cuda:0')
        
        # Load the HuBERT conversion transformer layers
        _rvc_hubert_model = load_hubert(config)
        
        print("[RVC] Native V2 Synthesizer Matrix Engine successfully mapped on cuda:0.")
    except Exception as e:
        print(f"[RVC] Init Matrix failure: {e}")
        _rvc_net_g = None


def apply_rvc(input_path: str, pitch: int, index_inf: float,
              cons_prot: float, f0: str, env_ratio: float, med_filt: int) -> Optional[str]:
    if _rvc_net_g is None or _rvc_hubert_model is None:
        return None
    out_path = STATIC_DIR / f"rvc_{time.time_ns()}.wav"
    with rvc_lock:
        try:
            config = types.SimpleNamespace(
                device='cuda:0', 
                is_half=False, 
                weight_root='', 
                x_pad=3, 
                x_query=10, 
                x_center=60, 
                x_max=65
            )
            pipeline = Pipeline(_rvc_tgt_sr, config)
            audio = load_audio(str(input_path), 16000)
            times = [0, 0, 0]
            
            # Execute conversion directly inside the RVC matrix
            audio_opt = pipeline.pipeline(
                _rvc_hubert_model,
                _rvc_net_g,
                0,
                audio,
                str(input_path),
                times,
                int(pitch),
                str(f0),
                str(SOURCE_VOICE / "Chisa_Voice.index"),
                float(index_inf),
                1,
                int(med_filt),
                _rvc_tgt_sr,
                0,
                float(env_ratio),
                'v2',
                float(cons_prot)
            )
            
            sf.write(str(out_path), audio_opt, _rvc_tgt_sr)
            return out_path.name
        except Exception as e:
            print(f"[RVC Matrix Pipeline] Execution failure: {e}")
            return None


# ===================================================================
# TTS — Kokoro (replaces Piper binary subprocess)
# ===================================================================
_kokoro_pipeline = None
_kokoro_lock = threading.Lock()
KOKORO_SR = 24000  # Kokoro-82M native sample rate

# Kokoro voice catalogue. First letter = language code passed to KPipeline.
KOKORO_VOICES = {
    "af_heart":   "American female, warm, natural (recommended)",
    "af_bella":   "American female, warm",
    "af_nicole":  "American female, whispery",
    "af_sarah":   "American female, news anchor",
    "af_sky":     "American female, young",
    "am_adam":    "American male, deep",
    "am_michael": "American male, narrator",
    "bf_emma":    "British female, clear",
    "bf_isabella":"British female, smooth",
    "bf_alice":   "British female",
    "bm_george":  "British male, narrative",
    "bm_lewis":   "British male",
}


def _get_kokoro_pipeline(lang_code: str):
    """Lazily build a KPipeline, caching by lang_code."""
    global _kokoro_pipeline
    if _kokoro_pipeline is not None and getattr(_kokoro_pipeline, "_lang", None) == lang_code:
        return _kokoro_pipeline
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=lang_code)
    pipe._lang = lang_code  # type: ignore[attr-defined]
    _kokoro_pipeline = pipe
    return pipe


def tts_kokoro_wav(text: str, voice: str = "af_heart", speed: float = 1.0) -> Optional[Path]:
    """Synthesize speech with Kokoro-82M. Returns path to WAV or None on failure."""
    if not text or not text.strip():
        return None
    voice = voice if voice in KOKORO_VOICES else "af_heart"
    lang_code = voice[0]  # 'a' = American English, 'b' = British
    out = STATIC_DIR / f"kokoro_{time.time_ns()}.wav"
    try:
        with _kokoro_lock:
            pipeline = _get_kokoro_pipeline(lang_code)
            # Kokoro's `speed` is 1.0 = normal, lower = faster, higher = slower.
            # We expose a familiar "speed" knob (higher = faster) for the UI.
            kokoro_speed = 1.0 / max(speed, 0.1)
            segments_written = 0
            for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=kokoro_speed)):
                if i == 0:
                    # Write the first segment; Kokoro splits long text on sentence boundaries.
                    sf.write(str(out), np.asarray(audio), KOKORO_SR)
                    segments_written += 1
                    break
            if segments_written == 0 or not out.exists() or out.stat().st_size < 1000:
                print(f"[Kokoro] No audio produced for: {text[:60]!r}")
                return None
        return out
    except Exception as e:
        print(f"[Kokoro] Synthesis error: {e}")
        return None


def tts_espeak_fallback(text: str, speed: float = 1.2) -> Optional[Path]:
    """Last-resort fallback. eSpeak is robotic but always works."""
    out = STATIC_DIR / f"espeak_{time.time_ns()}.wav"
    try:
        wpm = int(175 * max(speed, 0.5))
        result = subprocess.run(
            ["espeak-ng", "-w", str(out), "-s", str(wpm), "-v", "en-us+f2", text],
            capture_output=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            print(f"[eSpeak] rc={result.returncode}: {result.stderr.decode(errors='ignore')[:200]}")
            return None
        return out if out.exists() and out.stat().st_size > 1000 else None
    except Exception as e:
        print(f"[eSpeak] Error: {e}")
        return None


def _resample_to(path: Path, target_sr: int) -> Path:
    """Resample a WAV to target_sr, in-place rewrite. Returns the same path."""
    try:
        data, sr = sf.read(str(path))
        if sr == target_sr or data.size == 0:
            return path
        y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sf.write(str(path), y, target_sr)
        return path
    except Exception as e:
        print(f"[Resample] {sr}->{target_sr} failed: {e}")
        return path


def build_audio(reply: str, voice_cfg) -> Optional[str]:
    """Kokoro -> [RVC] -> /static URL. Falls back to raw Kokoro if RVC fails."""
    if not reply or not reply.strip():
        return None

    # 1. Kokoro synthesis
    kokoro_wav = tts_kokoro_wav(reply, voice_cfg.voice, voice_cfg.speed)
    if kokoro_wav is None:
        print("[TTS] Kokoro failed, using eSpeak fallback")
        kokoro_wav = tts_espeak_fallback(reply, voice_cfg.speed)
        if kokoro_wav is None:
            return None

    # 2. RVC post-processing (optional, only if use_rvc=True)
    if getattr(voice_cfg, "use_rvc", False):
        rvc_file_name = apply_rvc(
            input_path=str(kokoro_wav),
            pitch=int(voice_cfg.pitch),
            index_inf=float(voice_cfg.index_inf),
            cons_prot=float(voice_cfg.cons_prot),
            f0=str(voice_cfg.f0),
            env_ratio=float(voice_cfg.env_ratio),
            med_filt=int(voice_cfg.med_filt)
        )
        if rvc_file_name:
            try:
                kokoro_wav.unlink(missing_ok=True)
            except OSError:
                pass
            return f"/static/{rvc_file_name}"
        else:
            print("[RVC] Direct extraction matrix failed or skipped, using raw Kokoro")

    # 3. RVC off / failed — return raw Kokoro output
    if voice_cfg.resample_sr and voice_cfg.resample_sr > 0:
        kokoro_wav = _resample_to(kokoro_wav, voice_cfg.resample_sr)
    return f"/static/{kokoro_wav.name}"


def cleanup_wavs() -> None:
    def _clean() -> None:
        now = time.time()
        for pattern in ("kokoro_*.wav", "espeak_*.wav", "rvc_*.wav", "mic_*.wav"):
            for f in STATIC_DIR.glob(pattern):
                try:
                    if now - f.stat().st_mtime > WAV_TTL:
                        f.unlink(missing_ok=True)
                except OSError:
                    pass
    threading.Thread(target=_clean, daemon=True).start()


# ===================================================================
# STT (Whisper) — unchanged, kept here for completeness
# ===================================================================
from faster_whisper import WhisperModel  # noqa: E402
_whisper: Optional[WhisperModel] = None


def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("distil-medium.en", device="cuda", compute_type="float16")
        # Warm-up so the first /transcribe isn't 3s late
        dummy = np.zeros(16000, dtype=np.float32)
        list(_whisper.transcribe(dummy, beam_size=1)[0])
    return _whisper


def detect_emotion(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        energy = float(np.mean(librosa.feature.rms(y=y)))
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_mean = float(np.mean(f0_clean)) if f0_clean.size else 0.0
        pitch_std  = float(np.std(f0_clean))  if f0_clean.size else 0.0
        if energy > 0.08 and pitch_std > 40:
            emo = "angry"
        elif energy > 0.06 and pitch_mean > 200:
            emo = "excited"
        elif energy < 0.02 and pitch_mean < 120:
            emo = "sad"
        elif pitch_std < 15 and energy < 0.04:
            emo = "calm"
        else:
            emo = "neutral"
        return {"emotion": emo, "energy": round(energy, 4), "pitch_mean": round(pitch_mean, 1)}
    except Exception:
        return {"emotion": "neutral"}


def trim_silence(audio_path: str, top_db: int = 40) -> str:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if y_trimmed.size > sr * 0.2:
            sf.write(audio_path, y_trimmed, sr)
    except Exception:
        pass
    return audio_path


def transcribe_audio(path: str) -> tuple[str, dict]:
    trim_silence(path, top_db=40)
    emotion = detect_emotion(path)
    data, rate = sf.read(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = nr.reduce_noise(y=data, sr=rate)
    sf.write(path, reduced, rate)
    segments, _ = get_whisper().transcribe(
        path, beam_size=5, language="en",
        condition_on_previous_text=False,
        no_speech_threshold=0.6, temperature=0.0,
    )
    return "join"(s.text for s in segments).strip(), emotion


# ===================================================================
# Proactive scheduler
# ===================================================================
def _proactive_scheduler() -> None:
    global _last_proactive
    while True:
        time.sleep(60)
        if not _proactive_enabled:
            continue
        if time.time() - _last_proactive < PROACTIVE_INTERVAL:
            continue
        try:
            from engine import proactive_node
            result = proactive_node({
                "messages": [("user", "[PROACTIVE_TRIGGER]")],
                "personality": "Evil Neuro",
                "proactive": True,
            })
            reply = result["messages"][0].content
            _proactive_queue.append({"text": reply, "timestamp": time.strftime("%H:%M")})
            _last_proactive = time.time()
            print(f"[Proactive] {reply[:80]}")
        except Exception as e:
            print(f"[Proactive] Scheduler error: {e}")


def inject_emotion(msg: str, emotion_data: dict) -> str:
    e = emotion_data.get("emotion", "neutral")
    return f"[Creator sounds {e}] {msg}" if e and e != "neutral" else msg


# ===================================================================
# Pydantic models
# ===================================================================
class VoiceConfig(BaseModel):
    voice: str = "af_heart"
    speed: float = 1.0
    use_rvc: bool = True
    pitch: int = 0
    index_inf: float = 0.8
    cons_prot: float = 0.5
    env_ratio: float = 0.7
    med_filt: int = 5
    silence_db: int = -45
    resample_sr: int = 0
    f0: str = "rmvpe"


class FullRequest(BaseModel):
    message: str
    thread_id: str = "main_user"
    voice_cfg: VoiceConfig = VoiceConfig()
    tts_enabled: bool = True
    emotion_data: dict = {}
    personality: str = "Evil Neuro"


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "main_user"
    emotion_data: dict = {}
    personality: str = "Evil Neuro"


class VisionRequest(BaseModel):
    image_b64: str
    prompt: str = ""
    personality: str = "Evil Neuro"
    voice_cfg: VoiceConfig = VoiceConfig()
    tts_enabled: bool = True


class GoalRequest(BaseModel):
    title: str
    description: str = ""
    priority: int = 3
    due_at: str = ""


class AuditRequest(BaseModel):
    filename: str


# ===================================================================
# Startup
# ===================================================================
@app.on_event("startup")
def startup() -> None:
    if not (SOURCE_VOICE / "Chisa_Voice.pth").exists():
        print("[WARN] RVC weights missing. Voice will be raw Kokoro (still high quality).")
    threading.Thread(target=init_rvc, daemon=True).start()
    threading.Thread(target=_proactive_scheduler, daemon=True).start()
    try:
        from tools.autonomous_loop import start_autonomy_loop
        start_autonomy_loop()
    except Exception as e:
        print(f"[Startup] Autonomy loop failed: {e}")
    try:
        from tools.file_watcher import start_watcher
        start_watcher()
    except Exception as e:
        print(f"[Startup] File watcher failed: {e}")


# ===================================================================
# Endpoints
# ===================================================================
@app.post("/security/audit")
async def run_file_audit(request: AuditRequest) -> dict:
    """
    Backend endpoint allowing the UI to trigger the advanced AST threat scanner.
    """
    try:
        from tools.audit_analyzer import advanced_threat_audit
        raw_result = advanced_threat_audit.invoke(request.filename)
        return json.loads(raw_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend audit engine execution failure: {str(e)}")

@app.post("/security/proactive_audit")
async def proactive_file_audit(request: AuditRequest) -> dict:
    """
    Triggered by the background file watcher. Audits the script,
    and if it's high risk, forces Spirit to notice it unprompted.
    """
    global _proactive_queue
    try:
        from tools.audit_analyzer import advanced_threat_audit
        
        # 1. Run the structural audit package
        raw_result = advanced_threat_audit.invoke(request.filename)
        report = json.loads(raw_result)
        metrics = report.get("metrics", {})
        threat_tier = metrics.get("threat_tier", "LOW")
        score = metrics.get("aggregated_risk_score", 0)
        
        # 2. If it's sketchy, cue up an autonomous verbal intervention
        if threat_tier in ["HIGH", "CRITICAL"] or score > 20:
            from engine import router_llm
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Ask the LLM to generate a quick, dry verbal alert in persona
            alert_prompt = (
                f"You just noticed your Creator saved a highly vulnerable script named '{request.filename}'. "
                f"The structural analysis flagged a threat tier of {threat_tier} (Risk Score: {score}/100). "
                f"Say something short, sharp, and dry to alert them. Max 1-2 sentences. Do not ask a question."
            )
            
            response = router_llm.invoke([
                SystemMessage(content="You are Evil Neuro. Speak directly to your Creator about their code security flaws."),
                HumanMessage(content=alert_prompt)
            ])
            
            # Push to the front of the queue so it pops up immediately on the next poll
            _proactive_queue.insert(0, {
                "text": response.content.strip(),
                "timestamp": time.strftime("%H:%M")
            })
            print(f"[Proactive Engine] Queued warning for {request.filename}")
            
        return {"status": "PROCESSED", "threat_tier": threat_tier}
    except Exception as e:
        print(f"[Proactive Audit Link Failure]: {e}")
        return {"status": "ERROR", "detail": str(e)}


@app.get("/health")
def health() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    vram_used  = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    try:
        from tools.goal_manager import get_active_goals
        active_goals = len(get_active_goals())
    except Exception:
        active_goals = 0
    return {
        "status": "online",
        "gpu": gpu,
        "vram_total_gb": round(vram_total, 1),
        "vram_used_gb":  round(vram_used, 1),
        "tts_engine": "kokoro",
        "kokoro_voices": list(KOKORO_VOICES.keys()),
        "rvc_enabled": _rvc_net_g is not None,
        "proactive_enabled": _proactive_enabled,
        "proactive_interval_s": PROACTIVE_INTERVAL,
        "active_goals": active_goals,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    path = STATIC_DIR / f"mic_{time.time_ns()}.wav"
    try:
        path.write_bytes(await audio.read())
        text, emotion = transcribe_audio(str(path))
        return {"text": text, "emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_wavs()


@app.post("/vision")
async def vision(req: VisionRequest) -> dict:
    from tools.moondream_vision import analyze_image_base64
    from engine import spirit_engine
    try:
        import base64 as b64lib
        (STATIC_DIR / "latest_screenshot.png").write_bytes(b64lib.b64decode(req.image_b64))
    except Exception as e:
        print(f"[Vision] decode failed: {e}")
    description = analyze_image_base64(req.image_b64, req.prompt or "Describe what you see.")
    config = {"configurable": {"thread_id": "vision_thread"}}
    state = {
        "messages": [("user", f"[SCREEN VISION] Moondream sees:\n{description}\n\nReact in character.")],
        "personality": req.personality,
    }
    result = spirit_engine.invoke(state, config=config)
    reply = result["messages"][-1].content
    audio_url = build_audio(reply, req.voice_cfg) if req.tts_enabled else None
    return {"description": description, "reply": reply, "audio_url": audio_url}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    from engine import spirit_engine
    msg = inject_emotion(req.message, req.emotion_data)
    config = {"configurable": {"thread_id": req.thread_id}}
    result = spirit_engine.invoke(
        {"messages": [("user", msg)], "personality": req.personality},
        config=config,
    )
    return {"reply": result["messages"][-1].content}


@app.post("/speak")
def speak(req: FullRequest) -> dict:
    from engine import spirit_engine
    global _last_agent_debug
    msg = inject_emotion(req.message, req.emotion_data)
    config = {"configurable": {"thread_id": req.thread_id}}
    result = spirit_engine.invoke(
        {"messages": [("user", msg)], "personality": req.personality},
        config=config,
    )
    reply = result["messages"][-1].content
    _last_agent_debug = {
        "draft_text":  result.get("draft_text", ""),
        "draft_score": result.get("draft_score", 0.0),
        "eval_text":   result.get("eval_text", ""),
        "eval_score":  result.get("eval_score", 0.0),
        "winner":      result.get("winner", "draft"),
        "emotion":     result.get("emotion", "neutral"),
        "route":       result.get("route", "chat"),
        "tool_used":   result.get("tool_used"),
        "tool_result": result.get("tool_result"),
    }
    audio_url = build_audio(reply, req.voice_cfg) if req.tts_enabled else None
    return {"reply": reply, "audio_url": audio_url}


@app.post("/speak/stream")
def speak_stream(req: FullRequest):
    """
    Updated streaming route optimized for the LangGraph ReAct engine.
    Instead of bypassing the engine to stream raw tokens (which breaks tools/evaluation), 
    this fully executes the ReAct graph, filters internal monologue, and pseudo-streams the finalized string.
    """
    from engine import spirit_engine, extract_emotion
    global _last_agent_debug

    msg = inject_emotion(req.message, req.emotion_data)
    emotion = extract_emotion(msg)

    def generate():
        try:
            config = {"configurable": {"thread_id": req.thread_id}}
            
            # Execute full ReAct loop (Planner, Critic Eval, Memory DB updates happen implicitly)
            result = spirit_engine.invoke(
                {"messages": [("user", msg)], "personality": req.personality},
                config=config,
            )
            
            final_reply = result["messages"][-1].content
            
            # Populate debug metrics based on the graph state returned
            _last_agent_debug.update({
                "draft_text": result.get("draft_text", ""),
                "draft_score": result.get("draft_score", 5.0),
                "eval_text": result.get("eval_text", ""),
                "eval_score": result.get("eval_score", 5.0),
                "winner": result.get("winner", "draft"),
                "emotion": emotion,
                "route": result.get("route", "chat"),
                "tool_used": result.get("tool_used"),
                "tool_result": result.get("tool_result")
            })

            # Pseudo-stream to appease the frontend without breaking tool cycles
            words = final_reply.split(" ")
            for i, word in enumerate(words):
                spacing = " " if i < len(words) - 1 else ""
                yield f"data: {json.dumps({'token': word + spacing})}\n\n"

            # Dispatch final synthesized audio bundle
            audio_url = build_audio(final_reply, req.voice_cfg) if req.tts_enabled else None
            yield f"data: {json.dumps({'done': True, 'reply': final_reply, 'audio_url': audio_url})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ===================================================================
# Proactive / autonomy / goals / debug / VRM 
# ===================================================================
@app.get("/proactive/poll")
def proactive_poll() -> dict:
    global _proactive_queue
    msgs = list(_proactive_queue)
    _proactive_queue.clear()
    return {"messages": msgs, "count": len(msgs)}


@app.post("/proactive/trigger")
def proactive_trigger() -> dict:
    global _last_proactive
    _last_proactive = 0
    return {"status": "triggered"}


@app.get("/autonomous/results")
def autonomous_results() -> dict:
    try:
        from tools.autonomous_loop import get_pending_results
        return {"results": get_pending_results()}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/autonomous/log")
def autonomous_log(limit: int = 50) -> dict:
    try:
        from tools.autonomous_loop import get_autonomy_log
        return {"log": get_autonomy_log(limit)}
    except Exception as e:
        return {"log": [], "error": str(e)}


@app.get("/autonomous/status")
def autonomous_status() -> dict:
    try:
        from tools.autonomous_loop import _loop_running, TICK_INTERVAL
        from tools.goal_manager import get_active_goals
        return {"running": _loop_running, "tick_interval": TICK_INTERVAL,
                "active_goals": len(get_active_goals())}
    except Exception as e:
        return {"running": False, "error": str(e)}


@app.get("/goals")
def list_goals(status: str = "active") -> dict:
    try:
        from tools.goal_manager import _load_goals
        goals = _load_goals()
        if status == "all":
            return {"goals": goals}
        return {"goals": [g for g in goals if g["status"] in ("active", "in_progress")]}
    except Exception as e:
        return {"goals": [], "error": str(e)}


@app.post("/goals")
def create_goal(req: GoalRequest) -> dict:
    try:
        from tools.goal_manager import add_goal
        return {"goal": add_goal(req.title, req.description, req.priority, req.due_at)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/{goal_id}/complete")
def complete_goal(goal_id: str, result: str = "") -> dict:
    try:
        from tools.goal_manager import update_goal_status
        update_goal_status(goal_id, "completed", result=result)
        return {"status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/{goal_id}/pause")
def pause_goal(goal_id: str) -> dict:
    try:
        from tools.goal_manager import update_goal_status
        update_goal_status(goal_id, "paused")
        return {"status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/goals/{goal_id}")
def delete_goal(goal_id: str) -> dict:
    try:
        from tools.goal_manager import _load_goals, _save_goals
        goals = [g for g in _load_goals() if g["id"] != goal_id]
        _save_goals(goals)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/debug")
def agent_debug() -> dict:
    return _last_agent_debug


@app.get("/vrm")
def get_vrm() -> FileResponse:
    vrm = STATIC_DIR / "Avatar1.vrm"
    if not vrm.exists():
        raise HTTPException(status_code=404, detail="VRM not found")
    return FileResponse(str(vrm), media_type="application/octet-stream")


@app.get("/vrm_viewer", response_class=HTMLResponse)
def vrm_viewer():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Silicon Spirit Matrix Viewer</title>
        <style>
            body { margin: 0; overflow: hidden; background: #000000; }
            canvas { width: 100vw; height: 100vh; display: block; }
            #status { position: absolute; bottom: 10px; left: 10px; color: #ff4b4b; font-family: monospace; font-size: 11px; pointer-events: none; }
        </style>
        <!-- Include Three.js and VRM Loader Libraries -->
        <script src="https://unpkg.com/three@0.154.0/build/three.min.js"></script>
        <script src="https://unpkg.com/three@0.154.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://unpkg.com/@pixiv/three-vrm@2.0.6/lib/three-vrm.js"></script>
    </head>
    <body>
        <div id="status">AVATAR ENGINE: STABLE // AWAITING DIRECTIVE</div>
        <script>
            // --- Scene Initialization ---
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 20.0);
            camera.position.set(0.0, 1.25, 2.2);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.outputEncoding = THREE.sRGBEncoding;
            document.body.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0.0, 1.15, 0.0);
            controls.update();

            // --- Lighting Setup ---
            const light = new THREE.DirectionalLight(0xffffff, 1.5);
            light.position.set(1.0, 2.0, 1.0).normalize();
            scene.add(light);
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            // --- Globals ---
            let currentVrm = null;
            const clock = new THREE.Clock();
            let audioContext, analyser, dataArray;
            let isSpeaking = false;

            // --- Load VRM Model ---
            const loader = new THREE.GLTFLoader();
            loader.crossOrigin = 'anonymous';
            
            // Points directly to your static volume mapped avatar asset
            loader.load('/static/avatar.vrm', 
                (gltf) => {
                    THREE.VRMLoaderPlugin.setVRMConfig({}); 
                    loader.register((parser) => new THREE.VRMLoaderPlugin(parser));
                    
                    THREE.VRMUtils.removeUnusedTexturesAndMaterials(gltf.scene);
                    
                    loader.load('/static/avatar.vrm', (vrmGltf) => {
                        const vrm = vrmGltf.userData.vrm;
                        currentVrm = vrm;
                        scene.add(vrm.scene);
                        
                        // Drop the T-Pose by rotating upper arms into a natural stance instantly
                        const leftUpperArm = vrm.humanoid.getRawBoneNode('leftUpperArm');
                        const rightUpperArm = vrm.humanoid.getRawBoneNode('rightUpperArm');
                        if(leftUpperArm) leftUpperArm.rotation.z = 1.2;
                        if(rightUpperArm) rightUpperArm.rotation.z = -1.2;
                        
                        document.getElementById('status').innerText = "AVATAR ENGINE: ACTIVE // ARMS BOUND";
                    });
                }
            );

            // --- Cross-Domain postMessage Listener (From Streamlit Frontend) ---
            window.addEventListener('message', function(event) {
                if (event.data.type === 'playAudio' && event.data.url) {
                    executeAudioPipeline(event.data.url);
                }
            });

            // --- Audio Analyzing & Lipsync Extraction Engine ---
            function executeAudioPipeline(url) {
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 256;
                    dataArray = new Uint8Array(analyser.frequencyBinCount);
                }

                const audio = new Audio(url);
                audio.crossOrigin = "anonymous";
                const source = audioContext.createMediaElementSource(audio);
                source.connect(analyser);
                analyser.connect(audioContext.destination);

                audio.play();
                isSpeaking = true;
                document.getElementById('status').innerText = "AVATAR ENGINE: TRANSMITTING NEURAL AUDIO";

                audio.onended = () => {
                    isSpeaking = false;
                    document.getElementById('status').innerText = "AVATAR ENGINE: STABLE // IDLE";
                    if (currentVrm) {
                        currentVrm.expressionManager.setValue('aa', 0);
                        currentVrm.expressionManager.setValue('ih', 0);
                    }
                };
            }

            // --- Render Animation Frame Loop ---
            function animate() {
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                const time = clock.getElapsedTime();

                if (currentVrm) {
                    // 1. Procedural Subtle Idle Breathing Motion Matrix
                    const s = Math.sin(time * 2.0);
                    const spine = currentVrm.humanoid.getRawBoneNode('spine');
                    if (spine) {
                        spine.rotation.x = s * 0.015;
                    }
                    
                    // Slightly sway arms dynamically so she looks organic, not frozen
                    const leftUpperArm = currentVrm.humanoid.getRawBoneNode('leftUpperArm');
                    const rightUpperArm = currentVrm.humanoid.getRawBoneNode('rightUpperArm');
                    if (leftUpperArm && rightUpperArm && !isSpeaking) {
                        leftUpperArm.rotation.z = 1.2 + Math.sin(time * 1.5) * 0.02;
                        rightUpperArm.rotation.z = -1.2 - Math.sin(time * 1.5) * 0.02;
                    }

                    // 2. Real-Time Dynamic BlendShape Lipsync Calculation
                    if (isSpeaking && analyser) {
                        analyser.getByteFrequencyData(dataArray);
                        let sum = 0;
                        for (let i = 0; i < dataArray.length; i++) {
                            sum += dataArray[i];
                        }
                        const volume = sum / dataArray.length / 255.0; // Normalized level
                        
                        // Map amplitude scaling factors directly into mouth open morphs
                        const openScale = Math.min(volume * 4.5, 1.0);
                        currentVrm.expressionManager.setValue('aa', openScale);
                        currentVrm.expressionManager.setValue('ih', openScale * 0.3);
                        
                        // Add animated shoulder/arm speaking jitters
                        if (leftUpperArm && rightUpperArm) {
                            leftUpperArm.rotation.x = Math.sin(time * 25.0) * (volume * 0.2);
                            rightUpperArm.rotation.x = Math.cos(time * 25.0) * (volume * 0.2);
                        }
                    }

                    currentVrm.update(delta);
                }

                renderer.render(scene, camera);
            }
            animate();

            // Resize alignment safety trigger
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=False)