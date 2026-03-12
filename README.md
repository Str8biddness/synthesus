# Synthesus 2.0

> **AIVM Synthesus** — Dual-Hemisphere Synthetic Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]()

Synthesus 2.0 is a production-grade synthetic intelligence system developed by **AIVM LLC**. It powers NPCs that remember, feel, and react — without a single large language model. The engine runs on a dual-hemisphere architecture with an ML Swarm of specialized micro-models, bringing believable AI characters to games and interactive applications at under 1ms latency on consumer hardware.

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

### Left Hemisphere — Pattern Matching
Pure pattern matching: tokenized triggers, confidence scoring, fallback cascades. The left hemisphere resolves most queries under 1ms.

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

Together, they create NPCs that remember, feel, and react.

### ML Swarm — 7 Specialized Micro-Models
Replaces what used to require a 0.6B parameter language model. Instead of one big model, we use 7 specialized micro-models — intent classifiers, sentiment analyzers, behavior predictors, loot balancers, dialogue rankers, emotion detectors, and context embedders. Total footprint: **~458 KB**. Total inference: **under 1ms**. That's shippable on a PS5 or mid-tier gaming PC.

---

### Key Subsystems

| Subsystem | Language | Purpose |
|---|---|---|
| `ml/` | Python | ML Swarm: 7 micro-models — intent, sentiment, embeddings, etc. |
| `cognitive/` | Python | Right Hemisphere: 9 cognitive modules (emotion, memory, relationships...) |
| `core/` | Python | Left Hemisphere: PatternEngine, RAGPipeline, HemisphereBridge |
| `api/` | Python | FastAPI production server + gateway |
| `kernel/` | C++ | Thread pool, memory allocator, message bus |
| `reasoning/` | C++ | PPBRS, causal, Bayesian, symbolic, SINN, planner |
| `memory/` | C++ | Episodic, working, long-term, self-perception, KN DB |
| `vcu/` | C++ | 11 Virtual Control Units (emotion, executive, language...) |
| `unpc_engine/` | Python | Universal NPC Character Engine with archetype genome system |
| `studio/` | Python/HTML | Character Studio — build, test, and export NPC characters |

---

## Features

- **Dual-Hemisphere Processing** — Left (pattern) and Right (cognitive) hemispheres run in parallel and reconcile
- **9 Cognitive Modules** — Conversation, emotion, relationships, world-state, knowledge, personality, recall, composition, escalation
- **ML Swarm** — 7 specialized micro-models (~458 KB total) replacing heavyweight LLMs
- **Synthetic RAG Pipeline** — FAISS-backed semantic retrieval with 78K+ embedded patterns
- **Pattern-Based Reasoning (PPBRS)** — Left-hemisphere fast-path pattern matching engine
- **Universal NPC Character Engine (UNPC)** — Genome-based character generation with 12+ archetypes
- **Character Studio** — Web-based NPC creator with live chat preview and personality sliders
- **Knowledge Node Database (KN-DB)** — Persistent semantic memory with episodic recall
- **FastAPI REST API** — Full async API with rate limiting, auth, and Swagger docs
- **Zero GPU Required** — Runs entirely on CPU at under 1ms per query
- **Docker Support** — Multi-stage container build

---

## Quick Start

### Prerequisites

- Python 3.10+
- CMake 3.18+ (optional — for C++ kernel)

### Installation

```bash
# Clone and create venv
git clone https://github.com/Str8biddness/synthesus.git
cd synthesus
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi "uvicorn[standard]" httpx pydantic numpy scipy scikit-learn faiss-cpu python-dotenv rich tenacity

# Start the production server
uvicorn api.production_server:app --host 0.0.0.0 --port 5000
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions, all endpoints, and the Character Studio.

### Docker

```bash
docker build -t synthesus:2.0 .
docker run -p 5000:5000 synthesus:2.0
```

---

## API Usage

### Query a Character

```bash
curl -X POST http://localhost:5000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What do you sell?", "character": "synth"}'
```

### Multi-Turn Conversation

```bash
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about the ancient war", "character": "sage", "session_id": "my-session"}'
```

### List Characters

```bash
curl http://localhost:5000/api/v1/characters
```

### Health Check

```bash
curl http://localhost:5000/api/v1/health
```

---

## Available Archetypes

| Archetype | Response Style | Dominant Traits |
|---|---|---|
| `warrior` | aggressive_direct | motor, executive |
| `sage` | philosophical_deliberate | language, memo |
| `doctor` | clinical_empathetic | language, social |
| `detective` | analytical_skeptical | executive, perception |
| `merchant` | persuasive_pragmatic | social, language |
| `teacher` | instructional_patient | language, social |
| `scientist` | precise_curious | executive, memo |
| `soldier` | direct_tactical | motor, executive |
| `noble` | formal_authoritative | executive, social |
| `trickster` | playful_deflective | language, social |
| `software_engineer` | technical_methodical | executive, memo |

Custom archetypes can be added by creating JSON files in `unpc_engine/archetypes/`.

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run benchmarks
python scripts/benchmark.py

# Build C++ kernel (optional)
cmake -B build && cmake --build build
```

---

## Project Structure

```
synthesus/
├── api/              # FastAPI servers (production_server.py, gateway.py)
├── characters/       # NPC character files (bio.json + patterns.json)
├── cognitive/        # Right Hemisphere: 9 cognitive modules
├── core/             # Left Hemisphere: PatternEngine, RAG, Bridge
├── data/             # FAISS index, metadata, checkpoints
├── kernel/           # C++ kernel (threads, memory, bus)
├── memory/           # C++ memory systems
├── ml/               # ML Swarm micro-models (~458 KB)
├── modules/          # Extension modules
├── onnx_bridge/      # ONNX Runtime bridge
├── reasoning/        # C++ reasoning engines
├── scripts/          # Migration, benchmarks, testing
├── sdk/              # Python, Unity, Unreal SDKs
├── static/           # Dashboard UI
├── studio/           # Character Studio (web NPC creator)
├── unpc_engine/      # NPC character engine + archetypes
├── vcu/              # 11 Virtual Control Units
└── vendor/           # SQLite3 amalgamation
```

---

## License

MIT License - Copyright (c) 2024-2026 AIVM LLC / Str8biddness

---

## Author

Built by **dakin ellegood** / AIVM LLC
