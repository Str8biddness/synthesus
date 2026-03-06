# Synthesus 2.0 Installation Guide

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| g++ | 13+ | C++ kernel compiler |
| Python | 3.11+ | API server & modules |
| SQLite3 | any | Context memory database |
| make | any | Build orchestration |

## Quick Start (CPU-only, 3 steps)

```bash
# 1. Install Python dependencies
pip install fastapi "uvicorn[standard]" sse-starlette python-multipart httpx pydantic svgwrite

# 2. Download required model files
bash download_models.sh

# 3. Build and run
bash build.sh --rebuild
```

Then open: **http://localhost:5000/dashboard**

---

## Full Installation

### Step 1: System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential g++ libsqlite3-dev python3 python3-pip
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y gcc-c++ sqlite-devel python3 python3-pip
```

### Step 2: Python Requirements (Required)

```bash
pip install fastapi "uvicorn[standard]" sse-starlette python-multipart httpx pydantic svgwrite
```

### Step 3: Python Requirements (Optional - Full Capability)

```bash
# Right hemisphere SLM (TinyLlama)
pip install llama-cpp-python

# Text-to-speech
pip install piper-tts

# Speech-to-text
pip install pywhispercpp
```

### Step 4: Download Model Files

```bash
# Automated download (recommended)
bash download_models.sh

# Manual download locations:
# TinyLlama GGUF (638 MB) -> models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Piper voice files (~116 MB) -> models/en_US-ryan-high.onnx + .json
#   https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/high
```

### Step 5: Build

```bash
bash build.sh --rebuild
```

---

## GPU Acceleration (Optional)

For NVIDIA GPU support, install CUDA toolkit (11.8+) before building.
The build script auto-detects GPU and enables CUDA compilation.

```bash
# Install llama-cpp with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=ON" pip install llama-cpp-python
bash build.sh --rebuild
```

---

## Verify Installation

```bash
# Health check
curl http://localhost:5000/health

# Run test query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello Synthesus", "mode": "dual"}'
```

---

## Architecture Overview

```
[FastAPI Server :5000]
        |
   [HemiReconciler]  L=0.50 / R=0.30 / V=0.20
      /         \
[Left Hemi]  [Right Hemi]
 (C++ Kernel)  (TinyLlama SLM)
  9 Reasoners    Pattern-Based
  PPBRS + SINN   Character Engine
  KN Database    UNPC Engine
```

## Directory Structure

```
synthesus/
  api/              # FastAPI server
  automation/       # Watchdog & telemetry
  characters/       # NPC character files (bio.json + patterns.json)
  core/             # HemiReconciler, PPBRSRouter, ContextMemory
  kernel/           # ThreadPool, MessageBus, MemoryAllocator
  memory/           # KNDatabase, Working/Episodic/LongTerm/SelfPerception
  models/           # Model files (gitignored)
  modules/          # Python fallback, web scraper, vehicle AI
  onnx_bridge/      # GPU/CPU hardware router
  reasoning/        # 9 reasoning modules (SINN, PPBRS, Symbolic...)
  static/           # Web dashboard
  unpc_engine/      # Universal NPC Character Engine
  vcu/              # 11 Virtual Control Units
  vendor/           # SQLite3 amalgamation
```

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | Run: `pip install fastapi uvicorn` |
| `synthesus_kernel not found` | Run: `bash build.sh --rebuild` |
| TinyLlama slow (~800ms) | Expected on CPU; GPU reduces to ~80ms |
| Port 5000 in use | Change port in `api/fastapi_server.py` |
| SQLite WAL errors | Delete `context.db` and restart |

## License

MIT License - AIVM LLC
