"""
Silicon Spirit – Streamlit Web Frontend (Hardened V2 Cyberpunk Design)
Communicates with the existing FastAPI backend (Docker).
"""

import streamlit as st
import requests
import json
import time
import base64
import os
from pathlib import Path

# Injects the dynamic environment target variable required for internal Docker network bridges
API_BASE = os.getenv("SPIRIT_API", "http://localhost:8000")
ANALYSIS_SANDBOX = Path("/app/spirit_memory/analysis_sandbox")

# Force custom cyberpunk terminal theme variables 
st.set_page_config(
    page_title="SILICON SPIRIT // CORE", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Threat Lab Styling Injection
st.markdown("""
<style>
    /* Global layout overrides */
    .stApp {
        background-color: #0d0e12;
        color: #e2e8f0;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    
    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #07080a !important;
        border-right: 1px solid #ff4b4b22;
        min-width: 340px !important;
    }
    
    /* Visual component metrics blocks */
    div[data-testid="stMetricValue"] {
        font-family: 'Consolas', monospace;
        color: #ff4b4b !important;
    }
    
    /* Custom message bubble frames */
    .chat-bubble-you {
        padding: 12px 16px;
        background-color: #161920;
        border-left: 3px solid #94a3b8;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #cbd5e1;
    }
    .chat-bubble-spirit {
        padding: 12px 16px;
        background-color: #1c1517;
        border-left: 3px solid #ff4b4b;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #fca5a5;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.05);
    }
    
    /* Laboratory Container Cards */
    .lab-card {
        background-color: #0f1115;
        border: 1px solid #ff4b4b11;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Ensure secure isolation staging directory exists locally
try:
    ANALYSIS_SANDBOX.mkdir(parents=True, exist_ok=True)
except Exception as e:
    pass

# ---------- Sidebar (System Configuration Node) ----------
with st.sidebar:
    st.markdown("<h2 style='color:#ff4b4b; margin-bottom:0;'>⚙ SILICON SPIRIT</h2>", unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b;'>CRITICAL INFRASTRUCTURE CORE v5.1</small><br><br>", unsafe_allow_html=True)
    
    # Health check metric board
    try:
        health = requests.get(f"{API_BASE}/health", timeout=2).json()
        st.markdown(f"**Host GPU:** `{health.get('gpu', '?')}`")
        
        col_vram, col_goals = st.columns(2)
        with col_vram:
            st.metric("VRAM Usage", f"{health.get('vram_used',0):.1f} GB")
        with col_goals:
            active_goals = health.get('active_goals', 0)
            st.metric("Active Tasks", active_goals)
    except:
        st.error("🚨 CORRUPT SYSTEM: Backend node unreachable.")
        st.stop()

    # VRM Avatar view framing - Using localhost routing to match host browser access
    st.markdown("---")
    vrm_url = "http://localhost:8501/vrm_viewer"
    st.iframe(vrm_url, width=300, height=360)

    # Runtime Parameter Controls Matrix
    st.markdown("---")
    personalities = ["Evil Spirit", "Cold Spirit", "Assistant Mode", "Yandere"]
    personality = st.selectbox("Core Personality Matrix", personalities, index=0)

    voice_speed = st.slider("Speech Rate Scalar", 0.8, 1.5, 1.2, 0.05)
    voice_pitch = st.slider("Frequency Modulation (Pitch)", -12, 12, 4, 1)
    
    if st.button("👁 CAPTURE IMAGE BUFFER (SHARE SCREEN)", use_container_width=True):
        st.session_state.screen_shared = True
        st.toast("Screen sync payload compiled.")

# ---------- Main Workspace Grid Structure ----------
st.markdown("<h1 style='letter-spacing: -1px; margin-bottom: 20px;'>🧠 NEURAL SYSTEM TERMINAL</h1>", unsafe_allow_html=True)

# Main chat window architecture definition
if "messages" not in st.session_state:
    st.session_state.messages = []

# Section layout split: 65% Chat Framework & Log Matrix, 35% Hardened Testing Sandbox
left_pane, right_pane = st.columns([13, 7], gap="medium")

with left_pane:
    # Notification Banner Matrix if a file target is actively initialized
    if st.session_state.get("audit_target"):
        st.markdown(
            f"<div style='background-color:#221510; border:1px solid #ea580c; padding:12px; border-radius:4px; margin-bottom:15px; color:#fdba74;'>"
            f"⚠️ <b>SANDBOX INTERACTION ACTIVE:</b> Spirit execution context is locked onto target script "
            f"<code>{st.session_state.audit_target}</code>.</div>", 
            unsafe_allow_html=True
        )

    # Live Chat Display Container
    st.markdown("### 📡 TRANSMISSION HISTORY")
    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.messages:
            st.markdown("<p style='color:#475569; text-align:center; margin-top:180px;'>[ NO ACTIVE TRANSMISSIONS FOUND ]</p>", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "You":
                    st.markdown(f"<div class='chat-bubble-you'><b>▸ Operator:</b> {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble-spirit'><b>◈ Spirit:</b> {msg['content']}</div>", unsafe_allow_html=True)

    # Input Control Bar Layout
    default_placeholder = "Input operator message directives..." if not st.session_state.get("audit_target") else f"Direct analysis of script target {st.session_state.audit_target}..."
    
    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input("Direct Directive Input", key="input", label_visibility="collapsed", placeholder=default_placeholder)
    with col_send:
        send_btn = st.button("EXECUTE", use_container_width=True, type="primary")

    if send_btn and user_input:
        final_prompt = user_input
        if st.session_state.get("audit_target"):
            final_prompt = (
                f"[SYSTEM INTERACTION - CODE FILE SCAN LOADED]\n"
                f"Target Staged File Path: /app/spirit_memory/analysis_sandbox/{st.session_state.audit_target}\n"
                f"User Prompt: {user_input}"
            )
            
        st.session_state.messages.append({"role": "You", "content": user_input})
        full_reply = ""
        
        with st.spinner("Processing neural calculation parameters..."):
            payload = {
                "message": final_prompt,
                "thread_id": "streamlit",
                "tts_enabled": True,
                "emotion_data": {},
                "personality": personality,
                "voice_cfg": {
                    "voice": "en_GB-cori-high.onnx",
                    "speed": voice_speed,
                    "pitch": voice_pitch,
                    "index_inf": 0.6,
                    "cons_prot": 0.5,
                    "env_ratio": 0.5,
                    "med_filt": 3,
                    "silence_db": -45,
                    "resample_sr": 0,
                    "f0": "crepe"
                }
            }
            
            try:
                response = requests.post(f"{API_BASE}/speak/stream", json=payload, stream=True)
                placeholder = st.empty()
                audio_url = None
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            full_reply += data["token"]
                        elif "audio_url" in data:
                            audio_url = data["audio_url"]
                        elif data.get("done"):
                            break
                            
                st.session_state.messages.append({"role": "Spirit", "content": full_reply})
                
                if audio_url:
                    full_audio_url = f"{API_BASE}{audio_url}"
                    st.markdown(f"""
                    <script>
                        var iframe = document.querySelector('iframe');
                        if (iframe && iframe.contentWindow) {{
                            iframe.contentWindow.postMessage({{type: 'playAudio', url: '{full_audio_url}'}}, '*');
                        }}
                    </script>
                    """, unsafe_allow_html=True)
                    
                if st.session_state.get("audit_target"):
                    st.session_state.audit_target = None
                    st.session_state.last_audit_report = None
                    
                st.rerun()
            except Exception as e:
                st.error(f"Network routing transactional failure: {e}")

    # Dynamic system automation logging block
    st.markdown("<br>", unsafe_allow_html=True)
    tab_autonomy, tab_goals = st.tabs(["📜 LIVE ENGINE AUTONOMY LOG", "🎯 COMPILING OBJECTIVE MATRICES"])
    
    with tab_autonomy:
        try:
            log_resp = requests.get(f"{API_BASE}/autonomous/log?limit=5", timeout=3).json()
            for entry in log_resp.get("log", []):
                st.markdown(f"`{entry.get('timestamp')}` **{entry.get('goal_title')}** → `{entry.get('action')}`")
        except:
            st.text("Awaiting operational metric initialization profiles...")

    with tab_goals:
        try:
            goals_resp = requests.get(f"{API_BASE}/goals", timeout=3).json()
            for g in goals_resp.get("goals", []):
                st.markdown(f"**[{g['status'].upper()}]** Priority ({g['priority']}/5) — {g['title']}")
        except:
            st.text("Objective allocation registry empty.")

with right_pane:
    st.markdown("### 🛠 HARDENED ADVANCED TESTING SANDBOX")
    
    # Lab container framing panel
    st.markdown("<div class='lab-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Stage target verification payload scripts", 
        type=["py", "txt", "json", "md", "env", "sqlite3"],
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        try:
            secure_filename = Path(uploaded_file.name).name
            save_path = ANALYSIS_SANDBOX / secure_filename
            save_path.write_bytes(uploaded_file.getvalue())
            
            st.success(f"📦 Staged in sandbox boundary: `{secure_filename}`")
            
            with st.expander("🔍 Inspect Binary Structure Buffer"):
                if secure_filename.endswith((".py", ".txt", ".json", ".md", ".env")):
                    file_text = uploaded_file.getvalue().decode("utf-8")
                    st.code(file_text[:800] + ("\n..." if len(file_text) > 800 else ""), language="python")
                else:
                    st.info(f"Opaque target array initialized ({len(uploaded_file.getvalue())} bytes).")
            
            if st.button("⚡ EXECUTE ABSTRACT SYNTAX THREAT EVALUATION", use_container_width=True, type="primary"):
                with st.spinner("Compiling tree nodes for vulnerability matrix mapping..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/security/audit", 
                            json={"filename": secure_filename},
                            timeout=15
                        ).json()
                        st.session_state.last_audit_report = response
                        st.session_state.audit_target = secure_filename
                        st.toast("Security assessment registry metric compiled.")
                    except Exception as err:
                        st.error(f"Audit connection refused: {err}")
        except Exception as e:
            st.error(f"Sandbox staging constraint collision: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Dynamic metrics profile engine presentation layout
    if "last_audit_report" in st.session_state and st.session_state.last_audit_report:
        report = st.session_state.last_audit_report
        if report.get("status") == "COMPLETED":
            metrics = report.get("metrics", {})
            tier = metrics.get('threat_tier')
            color_map = {"LOW": "#00ff88", "MEDIUM": "#ffaa00", "HIGH": "#ff4b4b", "CRITICAL": "#7f1d1d"}
            
            st.markdown(f"""
            <div style='background-color:#0f1115; border:1px solid #ff4b4b33; padding:16px; border-radius:6px;'>
                <h4 style='margin-top:0; color:#ff4b4b;'>🚨 SECURITY THREAT PROFILE DEFINED</h4>
                <p style='margin:4px 0;'>Aggregated Threat Index: <b style='color:{color_map.get(tier, "#e2e8f0")}'>{tier}</b></p>
                <p style='margin:4px 0;'>Risk Matrix Weighting: <b>{metrics.get('aggregated_risk_score')}/100</b></p>
                <p style='margin:4px 0;'>Raw Line Count: <b>{metrics.get('total_lines')}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📂 VIEW PARSED OBJECT node VISITS"):
                st.json(report.get("findings", []))
        elif report.get("status") in ("FAILED", "ERROR"):
            st.error(f"Structural analysis initialization failure: {report.get('reason')}")

# ---------- Javascript Vision Subsystem Pipeline Layer ----------
if st.session_state.get("screen_shared", False):
    st.markdown("""
    <script>
    (function captureScreen() {
        navigator.mediaDevices.getDisplayMedia({ video: true })
            .then(stream => {
                let video = document.createElement('video');
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    let canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    let ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    let imageData = canvas.toDataURL('image/png');
                    fetch('/vision', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData.split(',')[1] })
                    });
                    stream.getTracks().forEach(track => track.stop());
                };
            }).catch(err => console.error(err));
    })();
    </script>
    """, unsafe_allow_html=True)
    st.session_state.screen_shared = False