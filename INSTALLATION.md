# Synthesus 2.0 Installation Guide

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.11+ | API server & modules |
| g++ | 13+ | C++ kernel compiler (optional) |
| SQLite3 | any | Context memory database |

## Quick Start (CPU-only, 3 steps)

```bash
# 1. Clone and create venv
git clone https://github.com/Str8biddness/synthesus.git
cd synthesus
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install Python dependencies (no Torch, no sentence-transformers!)
pip install fastapi "uvicorn[standard]" sse-starlette python-multipart httpx pydantic svgwrite numpy scipy scikit-learn faiss-cpu

# 3. Run the API server
uvicorn api.gateway:app --host 0.0.0.0 --port 5000
```

Then check: **http://localhost:5000/health**

---

## Full Installation

### Step 1: System Dependencies (Linux only)

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential g++ libsqlite3-dev python3 python3-pip
```

**Fedora/RHEL:**
```bash
sudo dnf install -y gcc-c++ sqlite-devel python3 python3-pip
```

### Step 2: Python Requirements (Required)

```bash
pip install fastapi "uvicorn[standard]" sse-starlette python-multipart httpx pydantic svgwrite numpy scipy scikit-learn faiss-cpu
```

All embedding is handled by the built-in **SwarmEmbedder** (`ml/swarm_embedder.py`),
which uses scikit-learn's TF-IDF + TruncatedSVD pipeline. No PyTorch, no HuggingFace
Transformers, no sentence-transformers needed.

### Step 3: Python Requirements (Optional — Full Capability)

```bash
# Right hemisphere SLM (TinyLlama / Qwen GGUF — optional)
pip install llama-cpp-python

# Text-to-speech
pip install piper-tts

# Speech-to-text
pip install pywhispercpp

# Model persistence (save/load fitted SwarmEmbedder)
pip install joblib
```

### Step 4: Download Model Files (Optional)

Only needed if you want the right-hemisphere SLM (TinyLlama / Qwen):

```bash
bash download_models.sh
```

### Step 5: Build C++ Kernel (Optional)

```bash
bash build.sh --rebuild
```

---

## Verify Installation

```bash
# Health check
curl http://localhost:5000/health
# → {"status": "operational", "kernel": "down", "slm": "down", "rag_vectors": 0, "version": "2.0.0"}

# Run test query (requires API key header)
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: test-key-1234567890" \
  -d '{"query": "Hello Synthesus"}'
```

---

## Architecture Overview

```
[FastAPI Server :5000]
        |
   [HemiReconciler]
      /         \
[Left Hemi]  [Right Hemi]
 (C++ Kernel)  (ML Swarm micro-models)
  Pattern Engine   SwarmEmbedder (TF-IDF+SVD)
  PPBRS Router     Optional SLM (TinyLlama/Qwen)
  KN Database      RAG Pipeline + FAISS
```

## ML Swarm Embedding

The heavyweight `sentence-transformers` + PyTorch stack has been replaced with
`SwarmEmbedder`, a lightweight TF-IDF + SVD pipeline:

| Property | Old (sentence-transformers) | New (SwarmEmbedder) |
|---|---|---|
| Dependencies | PyTorch (~2 GB), sentence-transformers | scikit-learn (~30 MB) |
| Model size | ~80 MB (MiniLM) | ~50 KB (fitted TF-IDF) |
| Embedding dim | 384 | 128 |
| Latency | 5–15 ms/query | <1 ms/query |
| Accuracy | Neural semantic | Char n-gram statistical |
| GPU needed? | Optional CUDA | Never |

The SwarmEmbedder fits lazily on the first corpus it encounters and can be
swapped for an ONNX micro-model later without changing any API contracts.

## Directory Structure

```
synthesus/
  api/              # FastAPI server (gateway.py)
  automation/       # Watchdog & telemetry
  characters/       # NPC character files (bio.json + patterns.json)
  cognitive/        # SemanticMatcher, CognitiveEngine
  core/             # HemisphereBridge, RAGPipeline, PatternEngine
  kernel/           # ThreadPool, MessageBus, MemoryAllocator
  memory/           # KNDatabase, Working/Episodic/LongTerm
  ml/               # SwarmEmbedder, IntentClassifier, SentimentAnalyzer
  models/           # Model files (gitignored)
  modules/          # Python fallback, web scraper, vehicle AI
  onnx_bridge/      # GPU/CPU hardware router
  reasoning/        # 9 reasoning modules (SINN, PPBRS, Symbolic...)
  scripts/          # Embedding migration pipeline
  static/           # Web dashboard
  unpc_engine/      # Universal NPC Character Engine
  vcu/              # 11 Virtual Control Units
  vendor/           # SQLite3 amalgamation
```

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | Run: `pip install fastapi uvicorn` |
| `ModuleNotFoundError: sklearn` | Run: `pip install scikit-learn` |
| `ModuleNotFoundError: faiss` | Run: `pip install faiss-cpu` |
| Port 5000 in use | Change port: `uvicorn api.gateway:app --port 8000` |
| SLM disabled warning | Install `llama-cpp-python` + download GGUF model |

## License

MIT License - AIVM LLC