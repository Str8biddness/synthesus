# Synthesus 2.0 Installation Guide

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | API server & modules |
| g++ | 13+ | C++ kernel compiler (optional) |
| SQLite3 | any | Context memory database |

## Quick Start (CPU-only, 3 steps)

```bash
# 1. Clone and create venv
git clone https://github.com/Str8biddness/synthesus.git
cd synthesus
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install fastapi "uvicorn[standard]" httpx pydantic numpy scipy scikit-learn faiss-cpu python-dotenv rich tenacity

# 3. Run the production server
uvicorn api.production_server:app --host 0.0.0.0 --port 5000
```

Then open: **http://localhost:5000** (dashboard) or check **http://localhost:5000/api/v1/health**

---

## How It Works

Synthesus runs on **two hemispheres** — no LLMs, no SLMs, no cloud APIs.

### Left Hemisphere — Pattern Matching
Pure pattern matching: tokenized triggers, confidence scoring, fallback cascades.
Resolves most queries **under 1ms**.

### Right Hemisphere — 9 Cognitive Modules
1. **Conversation Tracking** — turn-by-turn context memory
2. **Emotion State Machine** — 7-state emotional model with decay and transitions
3. **Relationship System** — per-player trust, rapport, and interaction history
4. **World-State Awareness** — react to game events, time of day, weather
5. **Knowledge Graph** — structured domain knowledge per character
6. **Personality Bank** — trait-driven response modulation
7. **Context Recall** — episodic and semantic memory retrieval
8. **Response Composition** — assemble responses from multiple sources
9. **Escalation Gating** — route complex queries to deeper processing

Together, they create NPCs that **remember, feel, and react**.

### ML Swarm — 7 Specialized Micro-Models
Replaces what used to require a 0.6B parameter language model.
Instead of one big model, we use 7 specialized micro-models:

| Micro-Model | Purpose | Location |
|---|---|---|
| Intent Classifier | Parse player intent from query text | `ml/intent_classifier.py` |
| Sentiment Analyzer | Detect emotional tone of player input | `ml/sentiment_analyzer.py` |
| Context Embedder (SwarmEmbedder) | TF-IDF + SVD text embeddings for FAISS | `ml/swarm_embedder.py` |
| Behavior Predictor | Anticipate player next actions | `ml/` (planned) |
| Loot Balancer | Fair reward distribution | `ml/` (planned) |
| Dialogue Ranker | Rank candidate responses | `ml/` (planned) |
| Emotion Detector | Classify player emotional state | `ml/` (planned) |

**Total footprint: ~458 KB. Total inference: under 1ms.**
That's shippable on a PS5 or mid-tier gaming PC.

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
pip install fastapi "uvicorn[standard]" httpx pydantic numpy scipy scikit-learn faiss-cpu python-dotenv rich tenacity
```

All embedding is handled by the built-in **SwarmEmbedder** (`ml/swarm_embedder.py`),
which uses scikit-learn's TF-IDF + TruncatedSVD pipeline. No PyTorch, no HuggingFace
Transformers, no sentence-transformers, no cloud LLM APIs needed.

### Step 3: Python Requirements (Optional)

```bash
# Model persistence (save/load fitted SwarmEmbedder across restarts)
pip install joblib
```

### Step 4: Build C++ Kernel (Optional)

```bash
bash build.sh --rebuild
```

### Step 5: Run Embedding Migration (Optional — one-time)

Populates the FAISS index with 78K+ patterns from HuggingFace datasets and character files:

```bash
pip install datasets  # HuggingFace datasets library
python scripts/embedding_pipeline.py
```

This builds `data/faiss.index` and `data/faiss_metadata.json`. The pipeline supports
checkpoint resume — safe to interrupt and restart.

---

## Running the Servers

### Production Server (port 5000) — recommended

The full API server with RAG, cognitive engine, character routing, and dashboard UI:

```bash
uvicorn api.production_server:app --host 0.0.0.0 --port 5000
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI (Test Console) |
| `/api/v1/query` | POST | Main query — cognitive → RAG → fallback |
| `/api/v1/chat` | POST | Multi-turn conversation |
| `/api/v1/characters` | GET | List all characters |
| `/api/v1/characters/{id}` | GET | Character details |
| `/api/v1/health` | GET | System health & stats |
| `/docs` | GET | Interactive Swagger UI |

### Character Studio (port 8500) — optional

Web-based NPC creator with live chat preview:

```bash
python studio/character_studio.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Character Studio UI |
| `/api/session/create` | POST | Create character editing session |
| `/api/session/{id}/chat` | POST | Chat with character in session |
| `/api/session/{id}/genome` | PUT | Update character personality/traits |
| `/api/session/{id}/export` | GET | Export character package |
| `/api/session/{id}/save` | POST | Save character to disk |

---

## Verify Installation

```bash
# Health check (production server)
curl http://localhost:5000/api/v1/health
# → {"status": "healthy", "version": "2.0.0", "rag": {"enabled": true, "vectors": 78017}, ...}

# Run test query (no auth required for demo rate limit)
curl -X POST http://localhost:5000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello Synthesus", "character": "synth"}'

# With API key for higher rate limits
curl -X POST http://localhost:5000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-synth-dev-key" \
  -d '{"query": "What do you sell?", "character": "synth"}'
```

---

## Architecture

```
                     PLAYER INPUT
                         │
              ┌──────────┴──────────┐
              │                     │
       LEFT HEMISPHERE       RIGHT HEMISPHERE
       Pattern Matching      9 Cognitive Modules
              │                     │
       • Tokenized triggers  • Conversation Tracker
       • Confidence scoring  • Emotion State Machine
       • Fallback cascades   • Relationship System
       • <1ms resolution     • World-State Reactor
              │              • Knowledge Graph
              │              • Personality Bank
              │              • Context Recall
              │              • Response Compositor
              │              • Escalation Gate
              │                     │
              └──────────┬──────────┘
                    RECONCILER
                         │
                 ML SWARM (7 models)
                    ~458 KB total
                     <1ms inference
                         │
                     RESPONSE
```

## ML Swarm Embedding

| Property | Old (sentence-transformers) | New (SwarmEmbedder) |
|---|---|---|
| Dependencies | PyTorch (~2 GB), sentence-transformers | scikit-learn (~30 MB) |
| Model size | ~80 MB (MiniLM) | ~50 KB (fitted TF-IDF) |
| Embedding dim | 384 | 128 |
| Latency | 5–15 ms/query | <1 ms/query |
| Accuracy | Neural semantic | Char n-gram statistical |
| GPU needed? | Never | Never |

---

## Directory Structure

```
synthesus/
  api/              # FastAPI servers (production_server.py, gateway.py)
  characters/       # NPC character files (bio.json + patterns.json)
  cognitive/        # 9 Right Hemisphere modules (CognitiveEngine + subsystems)
  core/             # HemisphereBridge, RAGPipeline, PatternEngine
  data/             # FAISS index, metadata, migration checkpoints
  kernel/           # C++ ThreadPool, MessageBus, MemoryAllocator
  memory/           # KNDatabase, Working/Episodic/LongTerm memory
  ml/               # ML Swarm micro-models (SwarmEmbedder, IntentClassifier, ...)
  modules/          # Python fallback, web scraper, vehicle AI
  onnx_bridge/      # GPU/CPU hardware router
  reasoning/        # 9 reasoning modules (SINN, PPBRS, Symbolic...)
  scripts/          # Embedding migration pipeline, benchmarks
  sdk/              # Python, Unity, Unreal SDKs
  static/           # Dashboard UI (dashboard.html)
  studio/           # Character Studio (studio_ui.html + character_studio.py)
  unpc_engine/      # Universal NPC Character Engine + archetypes
  vcu/              # 11 Virtual Control Units
  vendor/           # SQLite3 amalgamation
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | Run: `pip install fastapi uvicorn` |
| `ModuleNotFoundError: sklearn` | Run: `pip install scikit-learn` |
| `ModuleNotFoundError: faiss` | Run: `pip install faiss-cpu` |
| `ModuleNotFoundError: cognitive` | Run from project root: `cd synthesus` |
| Port 5000 in use | Change port: `uvicorn api.production_server:app --port 8000` |
| No FAISS index found | Run: `python scripts/embedding_pipeline.py` |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYNTHESUS_API_KEY` | `sk-synth-dev-key` | Admin API key for authenticated rate limits |
| `PORT` | `5000` | Server port |

## License

MIT License - AIVM LLC