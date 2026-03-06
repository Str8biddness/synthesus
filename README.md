# Synthesus 2.0

> **AIVM Synthesus** — Dual-Hemisphere Synthetic Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17%2B-green)]()

Synthesus 2.0 is a production-grade synthetic intelligence system developed by **AIVM LLC**. It implements a biologically-inspired dual-hemisphere architecture that separates fast pattern-based intuition (Right Hemisphere) from deliberate logical reasoning (Left Hemisphere), fusing both pathways to generate coherent, contextually-aware AI behavior.

---

## Architecture

```
                    INPUT
                      |
          +-----------+-----------+
          |                       |
   LEFT HEMISPHERE          RIGHT HEMISPHERE
   (Logical / RAG)        (Pattern / Intuitive)
          |                       |
   - PPBRS Reasoner         - 11 VCUs
   - Causal Engine          - Pattern Engine
   - Symbolic Logic         - Emotion VCU
   - Bayesian Nets          - Social VCU
   - RAG Pipeline           - Language VCU
   - SINN Core              - Motor/Sensory VCU
          |                       |
          +-------[BRIDGE]--------+
                      |
               RECONCILER
                      |
                  RESPONSE
```

### Key Subsystems

| Subsystem | Language | Purpose |
|-----------|----------|--------|
| `kernel/` | C++ | Thread pool, memory allocator, message bus |
| `reasoning/` | C++ | PPBRS, causal, Bayesian, symbolic, SINN, planner |
| `memory/` | C++ | Episodic, working, long-term, self-perception, KN DB |
| `vcu/` | C++ | 11 Virtual Control Units (emotion, executive, language...) |
| `core/` | Python | Runtime bridge, RAG pipeline, pattern engine, character factory |
| `onnx_bridge/` | C++/Python | Hardware-accelerated model inference via ONNX Runtime |
| `unpc_engine/` | Python | Universal NPC Character Engine with archetype genome system |
| `api/` | Python | FastAPI REST gateway with Pydantic schemas |
| `automation/` | C++ | Watchdog, telemetry, health monitoring |

---

## Features

- **Dual-Hemisphere Processing** — Left (logic) and Right (intuition) hemispheres run in parallel and reconcile outputs
- **11 Virtual Control Units (VCUs)** — Emotion, Executive, Language, Memory, Motor, Perception, Sensory, Social, CT (x2), and Language processing
- **Synthetic RAG Pipeline** — Retrieval-Augmented Generation with no dependency on external LLM APIs
- **Pattern-Based Reasoning System (PPBRS)** — Right-hemisphere fast-path pattern matching engine
- **Universal NPC Character Engine (UNPC)** — Genome-based character generation supporting 12+ archetypes
- **ONNX Runtime Integration** — CPU/CUDA/ROCm/TensorRT hardware acceleration
- **Knowledge Node Database (KN-DB)** — Persistent semantic memory with episodic recall
- **FastAPI REST Gateway** — Full async API with streaming support
- **Docker Support** — Multi-stage container build

---

## Quick Start

### Prerequisites

- Python 3.10+
- CMake 3.18+
- GCC/Clang with C++17 support
- ONNX Runtime 1.17+

### Installation

```bash
# Clone the repo
git clone https://github.com/Str8biddness/synthesus.git
cd synthesus

# Run the automated installer
bash build.sh

# Download ONNX models
bash download_models.sh

# Start the API server
python -m uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

### Docker

```bash
docker build -t synthesus:2.0 .
docker run -p 8000:8000 synthesus:2.0
```

---

## API Usage

### Process Input

```bash
curl -X POST http://localhost:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, who are you?", "character_id": "synth"}'
```

### Spawn NPC Character

```bash
curl -X POST http://localhost:8000/character/spawn \
  -H 'Content-Type: application/json' \
  -d '{"archetype": "warrior", "name": "Ragnok"}'
```

### Python SDK

```python
from core.synthesus import Synthesus
import asyncio

engine = Synthesus()
engine.initialize()

# Process a query
result = asyncio.run(engine.process("Tell me about the ancient war", character_id="sage"))
print(result["text"])

# Spawn a character
npc = engine.spawn_character("warrior", name="Ragnok", traits={"aggression": 0.9})
print(npc)
```

---

## Available Archetypes

| Archetype | Response Style | Dominant VCUs |
|-----------|---------------|---------------|
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
python scripts/test_synthesus.py

# Run benchmarks
python scripts/benchmark.py

# Build C++ components only
cmake -B build && cmake --build build
```

---

## Project Structure

```
synthesus/
|-- main.cpp                  # C++ IPC entry point
|-- core/                     # Python runtime layer
|-- kernel/                   # C++ kernel (threads, memory, bus)
|-- reasoning/                # C++ reasoning engines
|-- memory/                   # C++ memory systems
|-- vcu/                      # C++ Virtual Control Units
|-- onnx_bridge/              # ONNX Runtime bridge
|-- unpc_engine/              # NPC character engine
|-- api/                      # FastAPI REST gateway
|-- automation/               # Watchdog & telemetry
|-- characters/               # Pre-built character genomes
|-- models/                   # ONNX model weights (downloaded)
|-- scripts/                  # Test & benchmark scripts
|-- static/                   # Dashboard HTML
```

---

## License

MIT License - Copyright (c) 2024-2026 AIVM LLC / Str8biddness

---

## Author

Built by **dakin ellegood** / AIVM LLC
