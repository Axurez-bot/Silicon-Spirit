# ============================================================
# SILICON SPIRIT — DOCKERFILE (Kokoro + RVC, stable PyTorch 2.7 cu128)
# ============================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    HF_HUB_VERBOSITY=error \
    HF_HUB_DISABLE_SYMLINKS=1 \
    LANGGRAPH_ALLOWED_OBJECTS=messages \
    KOKORO_MODEL_DIR=/app/models/kokoro \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ---- System deps (espeak-ng is REQUIRED by Kokoro's G2P / misaki) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential curl git wget \
    ffmpeg libportaudio2 libsndfile1 espeak-ng espeak-ng-data \
    libasound2-dev portaudio19-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    && ldconfig \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# misaki looks here on some distros
RUN mkdir -p /usr/share && \
    ln -sfn /usr/lib/x86_64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data || true

# ---- Python tooling ----
RUN pip3 install --no-cache-dir --upgrade "pip<25" "setuptools<75" wheel uv

# ---- PyTorch STABLE 2.7 with CUDA 12.8 support ----
# This is a real release. No more nightly dependency hell.
# Change from 2.7.0 to 2.6.0
RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---- HuggingFace CLI ----
RUN pip3 install --no-cache-dir "huggingface_hub==0.24.7"

# ---- Pre-download Kokoro + RVC support models ----
RUN mkdir -p /app/models/kokoro && \
    python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('hexgrad/Kokoro-82M', local_dir='/app/models/kokoro')" \
    || echo "[WARN] Kokoro pre-download failed; will retry at runtime"

RUN mkdir -p /app/models && \
    (wget -q -O /app/models/hubert_base.pt \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt || true) && \
    (wget -q -O /app/models/rmvpe.pt \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt || true)

# ---- Application code ----
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- RVC Legacy Dependency Architecture Injection ----
# Downgrading pip temporarily relaxes strict modern metadata parsing rules.
# This successfully bypasses the broken 'PyYAML>=5.1.*' wildcard string block in fairseq.
RUN pip3 install --no-cache-dir pip==24.0 && \
    pip3 install --no-cache-dir "omegaconf<2.1" hydra-core torchcrepe && \
    pip3 install --no-cache-dir fairseq==0.12.2 && \
    pip3 install --no-cache-dir --upgrade "pip<25"

COPY . .

# ---- Ensure correct Asset linking & Directory setup post-copy ----
# We wipe any accidental assets directory copied from the host and build exactly what the V2 engine needs
RUN rm -rf /app/assets && \
    mkdir -p /app/assets/hubert /app/static /app/models/spirit_voice /app/spirit_memory && \
    ln -sfn /app/models/hubert_base.pt /app/assets/hubert/hubert_base.pt && \
    ln -sfn /app/models/rmvpe.pt /app/assets/rmvpe.pt && \
    cp -n ./models/spirit_voice/Chisa_Voice.* /app/models/spirit_voice/ 2>/dev/null || true

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8501", "--workers", "1"]