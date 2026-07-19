"""
vrm_widget.py – VRM avatar with non-blocking audio playback.

The previous version used `while pygame.mixer.get_busy(): pygame.time.wait(100)`
which blocked the PyQt event loop, causing the window to be marked as
"Not Responding" by Windows during RVC inference. This version uses a QTimer
to poll audio status without blocking the event loop.
"""

import tempfile
import requests
import os
import pygame
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSignal, QUrl, QTimer  # CHANGED: added QTimer

# ---------- VRM HTML/JS – FULL ORIGINAL SCRIPT ----------
_VRM_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#050505;overflow:hidden}
#stage{width:100vw;height:100vh;display:block}
#status{
    position:fixed;bottom:8px;left:10px;
    color:#ff4b4b55;font-family:monospace;font-size:10px;pointer-events:none
}
#ebadge{
    position:fixed;top:8px;right:10px;
    color:#ff4b4b33;font-family:monospace;font-size:10px;pointer-events:none
}
</style>
</head><body>
<canvas id="stage"></canvas>
<div id="status">Initializing...</div>
<div id="ebadge">NEUTRAL</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.145.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.145.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1.0.0/lib/three-vrm.js"></script>
<script>
// ─────────────────────────────────────────────────────────────
// RENDERER SETUP
// ─────────────────────────────────────────────────────────────
const canvas   = document.getElementById('stage');
const statusEl = document.getElementById('status');
const ebadge   = document.getElementById('ebadge');

const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputEncoding = THREE.sRGBEncoding;

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(26, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(0, 1.38, 2.6);
camera.lookAt(0, 1.28, 0);

// Lighting
const keyLight = new THREE.DirectionalLight(0xfff5e0, 1.8);
keyLight.position.set(1.2, 2.5, 2.0);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xe0f0ff, 0.5);
fillLight.position.set(-1.5, 1.0, -1.0);
scene.add(fillLight);

scene.add(new THREE.AmbientLight(0xffffff, 0.85));

// ─────────────────────────────────────────────────────────────
// STATE CONFIGURATION
// ─────────────────────────────────────────────────────────────
let vrm   = null;
const clk = new THREE.Clock();

let isTalking   = false;
let audioLevel  = 0.0;
let vowelIdx    = 0;
let vowelTimer  = 0.0;
const vtarget   = [0, 0, 0, 0, 0]; 
const vcurrent  = [0, 0, 0, 0, 0]; 

const VOWEL_BIAS = [0.35, 0.25, 0.15, 0.15, 0.10];

let curEmotion = 'neutral';
const emoT = {joy:0, angry:0, sorrow:0, fun:0};
const emoC = {joy:0, angry:0, sorrow:0, fun:0};

let blinkTimer    = 0.0;
let blinkInterval = 3.0 + Math.random() * 2.5;
let blinking      = false;
let blinkPhase    = 0.0;

let idleT = 0.0;
let lx=0, ly=0, ltx=0, lty=0, lookTimer=0;

// Blendshape lookup
let BS = (typeof THREE_VRM !== 'undefined' && THREE_VRM.VRMSchema) ? THREE_VRM.VRMSchema.BlendShapePresetName : null;
if(!BS && THREE.VRMSchema) { BS = THREE.VRMSchema.BlendShapePresetName; }

function sbs(name, v) {
    if (!vrm || !vrm.blendShapeProxy || !name) return;
    try { vrm.blendShapeProxy.setValue(name, Math.max(0, Math.min(1, v))); } catch(e){}
}

// ─────────────────────────────────────────────────────────────
// EXPOSED PYTHON WINDOW INTERFACES
// ─────────────────────────────────────────────────────────────
window.setTalking = function(v) {
    isTalking = !!v;
    if (!v) {
        audioLevel = 0;
        for (let i = 0; i < 5; i++) vtarget[i] = 0;
    }
};

window.setAudioLevel = function(v) {
    audioLevel = Math.max(0, Math.min(1, parseFloat(v) || 0));
    if (audioLevel > 0.02) isTalking = true;
};

window.setEmotion = function(e) {
    curEmotion = e;
    ebadge.textContent = e.toUpperCase();
    Object.keys(emoT).forEach(k => emoT[k] = 0);
    switch(e) {
        case 'excited': emoT.joy=0.6; emoT.fun=0.35; break;
        case 'happy':   emoT.joy=0.75; break;
        case 'angry':   emoT.angry=0.65; break;
        case 'sad':     emoT.sorrow=0.6; break;
        case 'calm':    emoT.fun=0.18; break;
    }
};

// ─────────────────────────────────────────────────────────────
// REALTIME ANIMATION UPDATES
// ─────────────────────────────────────────────────────────────
function updateLipSync(dt) {
    if (!vrm || !vrm.blendShapeProxy || !BS) return;
    const VOWELS = [BS.A, BS.I, BS.U, BS.E, BS.O];

    if (isTalking && audioLevel > 0.02) {
        vowelTimer -= dt;
        if (vowelTimer <= 0) {
            let r = Math.random(), cum = 0;
            vowelIdx = 0;
            for (let i = 0; i < VOWEL_BIAS.length; i++) {
                cum += VOWEL_BIAS[i];
                if (r < cum) { vowelIdx = i; break; }
            }
            for (let i = 0; i < 5; i++) vtarget[i] = 0;
            vtarget[vowelIdx] = Math.min(0.95, audioLevel * 0.9);
            vowelTimer = 0.065 + (1 - audioLevel) * 0.10;
        }
    } else {
        for (let i = 0; i < 5; i++) vtarget[i] = 0;
        if (audioLevel < 0.01) isTalking = false;
    }

    const spd = isTalking ? 22 : 10;
    for (let i = 0; i < 5; i++) {
        vcurrent[i] += (vtarget[i] - vcurrent[i]) * Math.min(1, spd * dt);
        sbs(VOWELS[i], vcurrent[i]);
    }
}

function updateBlink(dt) {
    if (!vrm || !BS) return;
    blinkTimer += dt;
    if (!blinking && blinkTimer >= blinkInterval) {
        blinking = true; blinkPhase = 0; blinkTimer = 0;
        blinkInterval = 2.2 + Math.random() * 4.8;
        if (Math.random() < 0.15) blinkInterval = 0.11;
    }
    if (blinking) {
        blinkPhase += dt * 13;
        sbs(BS.Blink, Math.max(0, Math.sin(blinkPhase * Math.PI)));
        if (blinkPhase >= 1) { blinking = false; sbs(BS.Blink, 0); }
    }
}

function updateEmotions(dt) {
    if (!vrm || !BS) return;
    const spd = 2.5;
    for (const [k, t] of Object.entries(emoT)) {
        emoC[k] += (t - emoC[k]) * Math.min(1, spd * dt);
    }
    sbs(BS.Joy,    emoC.joy);
    sbs(BS.Angry,  emoC.angry);
    sbs(BS.Sorrow, emoC.sorrow);
    sbs(BS.Fun,    emoC.fun);
}

function updateIdle(dt) {
    if (!vrm || !vrm.humanoid) return;
    idleT += dt;

    try {
        const spine = vrm.humanoid.getNormalizedBoneNode('spine');
        if (spine && spine.rotation) {
            spine.rotation.x = Math.sin(idleT * 0.72) * 0.010;
            spine.rotation.z = Math.sin(idleT * 0.36) * 0.006;
        }
    } catch(e) { console.error("Spine animation error:", e); }

    try {
        const head = vrm.humanoid.getNormalizedBoneNode('head');
        if (head && head.rotation) {
            const tMult = isTalking ? (1.0 + audioLevel * 0.7) : 1.0;
            const eMult = curEmotion==='excited' ? 1.9 : curEmotion==='angry' ? 1.4 : 1.0;
            head.rotation.x = Math.sin(idleT * 1.08) * 0.022 * tMult + (isTalking ? audioLevel * 0.032 : 0);
            head.rotation.y = Math.sin(idleT * 0.55) * 0.038 * tMult * eMult + lx * 0.25;
            head.rotation.z = Math.sin(idleT * 0.44) * 0.011;
        }
    } catch(e) { console.error("Head animation error:", e); }

    try {
        const neck = vrm.humanoid.getNormalizedBoneNode('neck');
        if (neck && neck.rotation) {
            neck.rotation.x = Math.sin(idleT * 1.08) * 0.007;
            neck.rotation.y = lx * 0.10;
        }
    } catch(e) { console.error("Neck animation error:", e); }

    try {
        const sl = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
        const sr = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
        
        if (sl && sl.rotation) sl.rotation.z =  1.0 + Math.sin(idleT * 0.72) * 0.016;
        if (sr && sr.rotation) sr.rotation.z = -1.0 - Math.sin(idleT * 0.72) * 0.016;
    } catch(e) { console.error("Arm animation error:", e); }

    try {
        if (vrm.scene) {
            vrm.scene.position.y = Math.sin(idleT * 0.62) * 0.006;
            vrm.scene.rotation.z = Math.sin(idleT * 0.36) * 0.004;
        }
    } catch(e) {}
}

function updateEyes(dt) {
    if (!vrm || !vrm.humanoid) return;
    lookTimer -= dt;
    if (lookTimer <= 0) {
        ltx = Math.random() < 0.35 ? (Math.random() - 0.5) * 0.55 : 0;
        lty = Math.random() < 0.20 ? (Math.random() - 0.5) * 0.28 : 0;
        lookTimer = 1.8 + Math.random() * 3.8;
    }
    lx += (ltx - lx) * Math.min(1, 3 * dt);
    ly += (lty - ly) * Math.min(1, 3 * dt);

    try {
        const el = vrm.humanoid.getNormalizedBoneNode('leftEye');
        const er = vrm.humanoid.getNormalizedBoneNode('rightEye');
        if (el && el.rotation) { el.rotation.y = lx * 0.32; el.rotation.x = ly * 0.22; }
        if (er && er.rotation) { er.rotation.y = lx * 0.32; er.rotation.x = ly * 0.22; }
    } catch(e) {
        console.error("Eye animation error:", e);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clk.getDelta(), 0.05);
    if (vrm) {
        updateLipSync(dt);
        updateBlink(dt);
        updateEmotions(dt);
        updateIdle(dt);
        updateEyes(dt);
        vrm.update(dt);
    }
    renderer.render(scene, camera);
}

function loadVRM(src) {
    statusEl.textContent = 'Loading avatar...';
    const lib = (typeof THREE_VRM !== 'undefined') ? THREE_VRM : THREE;
    if (!lib || !lib.VRMLoaderPlugin) {
        statusEl.textContent = 'Error: VRMLoaderPlugin missing.';
        return;
    }
    const loader = new THREE.GLTFLoader();
    loader.register((parser) => new lib.VRMLoaderPlugin(parser));
    fetch(src).then(r => r.arrayBuffer()).then(buf => {
        loader.parse(buf, '', (gltf) => {
            const vrmInstance = gltf.userData.vrm;
            if (vrm) scene.remove(vrm.scene);
            vrm = vrmInstance;
            scene.add(vrm.scene);
            vrm.scene.rotation.y = Math.PI;
            const box = new THREE.Box3().setFromObject(vrm.scene);
            const c   = box.getCenter(new THREE.Vector3());
            vrm.scene.position.x -= c.x;
            vrm.scene.position.z -= c.z;
            statusEl.textContent = '[SPIRIT ONLINE]';
            animate();
        });
    }).catch(e => console.error(e));
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

loadVRM('__VRM_SRC__');
</script>
</body></html>"""

def _make_placeholder():
    return """<!DOCTYPE html><html><head>
<style>body{background:#050505;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0;font-family:monospace}
.m{color:#ff4b4b22;font-size:12px;text-align:center;line-height:2.4}
.s{color:#ff4b4b44;font-size:10px}</style>
</head><body><div class="m">AVATAR OFFLINE<br>
Place Avatar1.vrm in /app/static/<br>
<span class="s">Ensure Docker volume is mounted</span>
</div></body></html>"""


class VRMWidget(QWebEngineView):
    talking_changed = pyqtSignal(bool)

    def __init__(self, api_base: str = "http://localhost:8501"):
        super().__init__()
        self.api_base = api_base
        self.setMinimumHeight(380)
        self.page().javaScriptConsoleMessage = self._on_js_console
        self._load_avatar()
        # NEW: state for non-blocking audio polling
        self._audio_timer = None
        self._audio_done_callback = None
        self._audio_tmp = None
        self._current_sound = None

    def _on_js_console(self, level, msg, line, source):
        if level >= 1:
            print(f"[VRM-JS] {msg}")

    def _load_avatar(self):
        try:
            r = requests.get(f"{self.api_base}/vrm", stream=True, timeout=5)
            if r.status_code == 200:
                html = _VRM_HTML_TEMPLATE.replace("__VRM_SRC__", f"{self.api_base}/vrm")
                self.setHtml(html, QUrl(self.api_base + "/"))
                return
        except Exception as e:
            print(f"[VRM] Load error: {e}")
        self.setHtml(_make_placeholder())

    def reload_avatar(self):
        self._load_avatar()

    def set_talking(self, v: bool):
        js = f"if(window.setTalking)window.setTalking({'true' if v else 'false'});"
        self.page().runJavaScript(js)
        self.talking_changed.emit(v)

    def set_audio_level(self, level: float):
        self.page().runJavaScript(f"if(window.setAudioLevel)window.setAudioLevel({level:.4f});")

    def set_emotion(self, emotion: str):
        safe = emotion.strip().replace("'", "").replace('"', "")
        self.page().runJavaScript(f"if(window.setEmotion)window.setEmotion('{safe}');")

    def play_audio_with_lipsync(self, audio_url: str, device: int = 3, on_done=None, **kwargs):
        # CHANGED: download happens here (still synchronous, but fast)
        full_url = f"{self.api_base}{audio_url}"
        try:
            r = requests.get(full_url, timeout=15)
            if r.status_code != 200:
                print(f"[VRM] Download failed: HTTP {r.status_code}")
                if on_done:
                    on_done()
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(r.content)
                tmp = f.name
        except Exception as e:
            print(f"[VRM] Download error: {e}")
            if on_done:
                on_done()
            return

        # Initialize pygame mixer once
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000, size=-16, channels=1)

        self.set_talking(True)
        # CHANGED: keep a reference so it doesn't get garbage collected mid-playback
        self._current_sound = pygame.mixer.Sound(tmp)
        self._current_sound.play()
        self._audio_tmp = tmp
        self._audio_done_callback = on_done

        # CHANGED: use a QTimer to poll audio status instead of blocking
        # This keeps the PyQt event loop responsive during RVC inference
        if self._audio_timer is not None:
            self._audio_timer.stop()
        self._audio_timer = QTimer(self)
        self._audio_timer.setInterval(50)  # poll every 50ms
        self._audio_timer.timeout.connect(self._check_audio_done)
        self._audio_timer.start()

    def _check_audio_done(self):
        """NEW: called by QTimer to check if audio finished playing.
        Replaces the blocking while-loop and keeps the event loop responsive."""
        if not pygame.mixer.get_busy():
            # Audio finished — clean up
            if self._audio_timer is not None:
                self._audio_timer.stop()
                self._audio_timer = None
            self.set_talking(False)
            try:
                if self._audio_tmp and os.path.exists(self._audio_tmp):
                    os.unlink(self._audio_tmp)
            except OSError:
                pass
            self._audio_tmp = None
            self._current_sound = None
            if self._audio_done_callback:
                cb = self._audio_done_callback
                self._audio_done_callback = None
                cb()

    def stop_audio(self):
        """Stop any playing audio immediately."""
        if self._audio_timer is not None:
            self._audio_timer.stop()
            self._audio_timer = None
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        self.set_talking(False)
        if self._audio_tmp and os.path.exists(self._audio_tmp):
            try:
                os.unlink(self._audio_tmp)
            except OSError:
                pass
        self._audio_tmp = None
        self._current_sound = None
        if self._audio_done_callback:
            cb = self._audio_done_callback
            self._audio_done_callback = None
            cb()

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)
