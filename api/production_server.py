#!/usr/bin/env python3
"""
Synthesus 2.0 — Production API Server
AIVM LLC

Production-grade FastAPI server integrating:
- FAISS RAG pipeline for semantic retrieval (78K+ patterns)
- Character routing with personality, boundaries, ethics
- Cognitive engine for NPC dialogue (emotion, memory, relationships)
- API key authentication with rate limiting
- Health monitoring and telemetry

Revenue endpoints:
  POST /api/v1/query          — Main query endpoint (character-routed + RAG)
  POST /api/v1/chat           — Multi-turn conversation
  GET  /api/v1/characters     — List available characters
  GET  /api/v1/characters/{id} — Character details
  GET  /api/v1/health         — System health
  GET  /                      — Dashboard UI
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import sys

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("synthesus.api")

# ─── Config ──────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJ_ROOT))

# Direct imports — bypass core/__init__.py to avoid cascading import errors
import importlib.util

def _import_module_direct(module_name: str, file_path: str):
    """Import a module directly from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_rag_mod = _import_module_direct("core.rag_pipeline", str(PROJ_ROOT / "core" / "rag_pipeline.py"))
RAGPipeline = _rag_mod.RAGPipeline

try:
    from cognitive.cognitive_engine import CognitiveEngine
    HAS_COGNITIVE = True
except (ImportError, Exception) as e:
    logger.warning(f"Cognitive engine not available: {e}")
    HAS_COGNITIVE = False
    CognitiveEngine = None
CHARACTERS_DIR = PROJ_ROOT / "characters"
DATA_DIR = PROJ_ROOT / "data"
STATIC_DIR = PROJ_ROOT / "static"

API_KEY_HEADER = "X-API-Key"
DEMO_RATE_LIMIT = 10  # requests per minute for unauthenticated
AUTH_RATE_LIMIT = 60   # requests per minute for authenticated

# Admin key (set via env or default for dev)
ADMIN_KEY = os.environ.get("SYNTHESUS_API_KEY", "sk-synth-dev-key")

# ─── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Synthesus 2.0 API",
    description="AIVM Synthesus — Dual-Hemisphere Synthetic Intelligence Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────────────
_rag: Optional[RAGPipeline] = None
_character_cache: Dict[str, Dict[str, Any]] = {}
_cognitive_engines: Dict[str, CognitiveEngine] = {}
_conversations: Dict[str, List[Dict]] = defaultdict(list)  # session_id -> messages
_rate_limits: Dict[str, List[float]] = defaultdict(list)  # ip/key -> timestamps
_request_count = 0
_start_time = time.time()

# ─── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _rag
    logger.info("Starting Synthesus 2.0 Production Server...")
    
    # Load RAG pipeline
    index_path = DATA_DIR / "faiss.index"
    meta_path = DATA_DIR / "faiss_metadata.json"
    if index_path.exists():
        logger.info("Loading RAG pipeline...")
        _rag = RAGPipeline(
            index_path=str(index_path),
            metadata_path=str(meta_path),
            top_k=5,
            score_threshold=0.5,
        )
        logger.info(f"RAG loaded: {_rag.total_vectors} vectors")
    else:
        logger.warning("No FAISS index found — RAG disabled")
    
    # Pre-load characters
    for char_dir in CHARACTERS_DIR.iterdir():
        if char_dir.is_dir() and (char_dir / "bio.json").exists():
            _load_character(char_dir.name)
    logger.info(f"Loaded {len(_character_cache)} characters")
    logger.info("Server ready.")


def _load_character(char_id: str) -> Optional[Dict[str, Any]]:
    if char_id in _character_cache:
        return _character_cache[char_id]
    char_dir = CHARACTERS_DIR / char_id
    bio_path = char_dir / "bio.json"
    if not bio_path.exists():
        return None
    
    with open(bio_path) as f:
        bio = json.load(f)
    
    patterns = {}
    pat_path = char_dir / "patterns.json"
    if pat_path.exists():
        with open(pat_path) as f:
            patterns = json.load(f)
    
    personality = {}
    pers_path = char_dir / "personality.json"
    if pers_path.exists():
        with open(pers_path) as f:
            personality = json.load(f)
    
    knowledge = {}
    know_path = char_dir / "knowledge.json"
    if know_path.exists():
        with open(know_path) as f:
            knowledge = json.load(f)
    
    _character_cache[char_id] = {
        "bio": bio,
        "patterns": patterns,
        "personality": personality,
        "knowledge": knowledge,
    }
    return _character_cache[char_id]


def _get_cognitive_engine(char_id: str) -> Optional[CognitiveEngine]:
    if char_id in _cognitive_engines:
        return _cognitive_engines[char_id]
    char_data = _load_character(char_id)
    if not char_data:
        return None
    try:
        engine = CognitiveEngine(
            character_id=char_id,
            bio=char_data["bio"],
            patterns=char_data["patterns"],
            char_dir=str(CHARACTERS_DIR / char_id),
        )
        _cognitive_engines[char_id] = engine
        return engine
    except Exception as e:
        logger.error(f"Failed to create engine for {char_id}: {e}")
        return None


# ─── Rate Limiting ───────────────────────────────────────────────────
def _check_rate_limit(key: str, limit: int) -> bool:
    now = time.time()
    window = [t for t in _rate_limits[key] if now - t < 60]
    _rate_limits[key] = window
    if len(window) >= limit:
        return False
    _rate_limits[key].append(now)
    return True


async def get_auth(request: Request, x_api_key: Optional[str] = Header(None)):
    """Auth dependency — returns (is_authenticated, rate_limit_key)."""
    if x_api_key and x_api_key == ADMIN_KEY:
        return True, f"auth:{x_api_key[:8]}"
    client_ip = request.client.host if request.client else "unknown"
    return False, f"ip:{client_ip}"


# ─── Request / Response Models ───────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The query text")
    character: str = Field(default="synth", description="Character ID to route to")
    mode: str = Field(default="auto", description="Processing mode: auto|cognitive|rag|pattern")
    session_id: Optional[str] = Field(default=None, description="Session ID for multi-turn")
    player_id: str = Field(default="default", description="Player/user ID for relationship tracking")
    include_sources: bool = Field(default=False, description="Include RAG source citations")
    include_debug: bool = Field(default=False, description="Include debug telemetry")

class QueryResponse(BaseModel):
    response: str
    confidence: float
    character: str
    source: str  # "rag", "pattern", "cognitive", "fallback"
    session_id: str
    latency_ms: float
    sources: Optional[List[Dict]] = None
    emotion: Optional[str] = None
    relationship: Optional[Dict] = None
    debug: Optional[Dict] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class CharacterInfo(BaseModel):
    id: str
    name: str
    role: str
    description: str
    domains: List[str]
    personality_traits: List[str]
    ethics_disclosure: str


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard UI."""
    dashboard = STATIC_DIR / "dashboard.html"
    if dashboard.exists():
        return dashboard.read_text()
    return f"""
    <html><head><title>Synthesus 2.0</title></head>
    <body style="background:#0a0a0a;color:#e0e0e0;font-family:system-ui;padding:40px">
    <h1 style="color:#00ff88">Synthesus 2.0</h1>
    <p>AIVM Dual-Hemisphere Synthetic Intelligence Engine</p>
    <p>FAISS Index: {_rag.total_vectors if _rag else 0} vectors</p>
    <p>Characters: {len(_character_cache)}</p>
    <p>API Docs: <a href="/docs" style="color:#00ff88">/docs</a></p>
    <p>Status: <a href="/api/v1/health" style="color:#00ff88">/api/v1/health</a></p>
    </body></html>
    """


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, auth=Depends(get_auth)):
    """Main query endpoint — RAG + character routing."""
    global _request_count
    _request_count += 1
    t0 = time.time()
    
    is_auth, rate_key = auth
    limit = AUTH_RATE_LIMIT if is_auth else DEMO_RATE_LIMIT
    if not _check_rate_limit(rate_key, limit):
        raise HTTPException(429, "Rate limit exceeded. Add X-API-Key header for higher limits.")
    
    session_id = req.session_id or str(uuid.uuid4())
    char_id = req.character
    query_text = req.query.strip()
    
    # ── Step 1: Try cognitive engine (full NPC brain) ──
    if HAS_COGNITIVE and req.mode in ("cognitive", "auto"):
        engine = _get_cognitive_engine(char_id)
        if engine:
            result = engine.process_query(
                player_id=req.player_id,
                query=query_text,
                thinking_layer_available=False,
            )
            if result.get("confidence", 0) > 0.7:  # High threshold — only use cognitive for strong matches
                latency = (time.time() - t0) * 1000
                # Store in conversation
                _conversations[session_id].append({"role": "user", "content": query_text})
                _conversations[session_id].append({"role": "assistant", "content": result["response"]})
                return QueryResponse(
                    response=result["response"],
                    confidence=result["confidence"],
                    character=char_id,
                    source="cognitive",
                    session_id=session_id,
                    latency_ms=round(latency, 2),
                    emotion=result.get("emotion"),
                    relationship=result.get("relationship") if req.include_debug else None,
                    debug=result.get("debug") if req.include_debug else None,
                )
    
    # ── Step 2: RAG semantic retrieval ──
    if _rag and req.mode in ("rag", "auto"):
        rag_result = await _rag.retrieve(query_text, character_id=char_id)
        if rag_result.get("context"):
            # Use the top match's response directly
            sources = rag_result.get("sources", [])
            top = sources[0] if sources else {}
            top_score = top.get("score", 0)
            
            if top_score >= 0.65:
                # High confidence RAG match — use directly
                # Find the full response from metadata
                response_text = ""
                for src in sources:
                    pattern = src.get("pattern", "")
                    # Find this pattern's response in metadata
                    if pattern:
                        response_text = _find_response_for_pattern(pattern)
                        if response_text:
                            break
                
                if not response_text:
                    response_text = rag_result["context"].split("\nA: ")[-1].split("\n")[0] if "\nA: " in rag_result["context"] else rag_result["context"][:500]
                
                latency = (time.time() - t0) * 1000
                _conversations[session_id].append({"role": "user", "content": query_text})
                _conversations[session_id].append({"role": "assistant", "content": response_text})
                return QueryResponse(
                    response=response_text,
                    confidence=round(top_score, 4),
                    character=char_id,
                    source="rag",
                    session_id=session_id,
                    latency_ms=round(latency, 2),
                    sources=sources if req.include_sources else None,
                )
    
    # ── Step 3: Character pattern fallback ──
    char_data = _load_character(char_id)
    if char_data:
        patterns = char_data.get("patterns", {})
        bio = char_data.get("bio", {})
        fallback_name = bio.get("name", char_id)
        fallback_text = f"I'm {fallback_name}. I don't have a specific answer for that yet, but I'm always learning. Could you try rephrasing?"
        
        latency = (time.time() - t0) * 1000
        return QueryResponse(
            response=fallback_text,
            confidence=0.3,
            character=char_id,
            source="fallback",
            session_id=session_id,
            latency_ms=round(latency, 2),
        )
    
    # ── Step 4: Global fallback ──
    latency = (time.time() - t0) * 1000
    return QueryResponse(
        response="I couldn't process that request. Please try again.",
        confidence=0.1,
        character=char_id,
        source="fallback",
        session_id=session_id,
        latency_ms=round(latency, 2),
    )


def _find_response_for_pattern(pattern: str) -> str:
    """Find the response text for a given pattern from RAG metadata."""
    if not _rag or not _rag._metadata:
        return ""
    # Binary search would be better but linear is fine for now
    for m in _rag._metadata:
        if m.get("pattern", "") == pattern:
            return m.get("response", "")
    return ""


@app.post("/api/v1/chat")
async def chat_endpoint(req: QueryRequest, auth=Depends(get_auth)):
    """Multi-turn chat — maintains conversation context."""
    session_id = req.session_id or str(uuid.uuid4())
    
    # Get conversation history
    history = _conversations.get(session_id, [])
    
    # Build context from history
    context = ""
    if history:
        recent = history[-6:]  # Last 3 turns
        context = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    
    # Route through main query
    result = await query_endpoint(QueryRequest(
        query=req.query,
        character=req.character,
        mode=req.mode,
        session_id=session_id,
        player_id=req.player_id,
        include_sources=req.include_sources,
        include_debug=req.include_debug,
    ), auth)
    
    return result


@app.get("/api/v1/characters")
async def list_characters():
    """List all available characters."""
    chars = []
    for char_id, data in _character_cache.items():
        bio = data.get("bio", {})
        chars.append({
            "id": char_id,
            "name": bio.get("name", char_id),
            "role": bio.get("role", ""),
            "description": bio.get("description", bio.get("backstory", ""))[:200],
        })
    return {"characters": chars, "count": len(chars)}


@app.get("/api/v1/characters/{char_id}")
async def get_character(char_id: str):
    """Get detailed character info."""
    data = _load_character(char_id)
    if not data:
        raise HTTPException(404, f"Character '{char_id}' not found")
    
    bio = data.get("bio", {})
    personality = data.get("personality", {})
    knowledge = data.get("knowledge", {})
    
    return {
        "id": char_id,
        "bio": bio,
        "personality_summary": {
            "traits": personality.get("traits", []),
            "voice": personality.get("voice", {}),
        },
        "knowledge_domains": list(knowledge.get("domains", {}).keys()) if isinstance(knowledge.get("domains"), dict) else [],
        "ethics": "Rule 1: This character is a synthetic intelligence created by AIVM. It will always disclose its nature when asked.",
    }


@app.get("/api/v1/health")
async def health():
    """System health and stats."""
    uptime = time.time() - _start_time
    return {
        "status": "healthy",
        "version": "2.0.0",
        "uptime_seconds": round(uptime, 1),
        "rag": {
            "enabled": _rag is not None,
            "vectors": _rag.total_vectors if _rag else 0,
        },
        "characters_loaded": len(_character_cache),
        "cognitive_engines_active": len(_cognitive_engines),
        "active_sessions": len(_conversations),
        "total_requests": _request_count,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/stats")
async def stats():
    """Detailed system statistics."""
    return {
        "rag_stats": _rag.get_stats() if _rag else {},
        "characters": list(_character_cache.keys()),
        "engines": list(_cognitive_engines.keys()),
        "sessions": len(_conversations),
        "requests": _request_count,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
