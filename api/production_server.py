#!/usr/bin/env python3
"""
Synthesus 2.0 — Production API Server
AIVM LLC

Production-grade FastAPI server integrating:
- ML Swarm: 7 specialized micro-models (~458 KB, <1ms inference)
- FAISS RAG pipeline for semantic retrieval (78K+ patterns)
- Character routing with personality, boundaries, ethics
- Cognitive engine for NPC dialogue (emotion, memory, relationships)
- API key authentication with rate limiting
- Health monitoring and telemetry

Endpoints:
  POST /api/v1/query          — Main query endpoint (ML → cognitive → RAG → fallback)
  POST /api/v1/chat           — Multi-turn conversation
  GET  /api/v1/characters     — List available characters
  GET  /api/v1/characters/{id} — Character details
  GET  /api/v1/health         — System health + ML Swarm status
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
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("synthesus.api")

# ─── Config ──────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJ_ROOT))

# ─── Dynamic Module Imports ──────────────────────────────────────────
def _import_module_direct(name: str, path: str):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE execution to avoid dataclass/decorator issues
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Dynamically import project modules
_rag_mod = _import_module_direct("core.rag_pipeline", str(PROJ_ROOT / "core" / "rag_pipeline.py"))
RAGPipeline = _rag_mod.RAGPipeline

_factory_mod = _import_module_direct("character_factory_v2", str(PROJ_ROOT / "character_factory_v2.py"))
CharacterFactory = _factory_mod.CharacterFactory
CharacterSpec = _factory_mod.CharacterSpec

try:
    from cognitive.cognitive_engine import CognitiveEngine
    HAS_COGNITIVE = True
except (ImportError, Exception) as e:
    logger.warning(f"Cognitive engine not available: {e}")
    HAS_COGNITIVE = False
    CognitiveEngine = None

# ─── ML Swarm Models ─────────────────────────────────────────────────
try:
    from ml.intent_classifier import IntentClassifier
    from ml.sentiment_analyzer import SentimentAnalyzer
    from ml.emotion_detector import EmotionDetector
    from ml.behavior_predictor import BehaviorPredictor
    from ml.loot_balancer import LootBalancer
    from ml.dialogue_ranker import DialogueRanker
    HAS_ML_SWARM = True
except (ImportError, Exception) as e:
    logger.warning(f"ML Swarm not available: {e}")
    HAS_ML_SWARM = False
CHARACTERS_DIR = PROJ_ROOT / "characters"
DATA_DIR = PROJ_ROOT / "data"
STATIC_DIR = PROJ_ROOT / "static"
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "faiss_metadata.json"

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

# Mount static directory for CSS/JS assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global State ────────────────────────────────────────────────────
_rag: Optional[RAGPipeline] = None
_character_cache: Dict[str, Dict[str, Any]] = {}
_cognitive_engines: Dict[str, CognitiveEngine] = {}
_conversations: Dict[str, List[Dict]] = defaultdict(list)  # session_id -> messages
_rate_limits: Dict[str, List[float]] = defaultdict(list)  # ip/key -> timestamps
_request_count = 0
_start_time = time.time()

# ML Swarm instances (initialized at startup)
_intent_classifier = None
_sentiment_analyzer = None
_emotion_detector = None
_behavior_predictor = None
_loot_balancer = None
_dialogue_ranker = None

# ─── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _rag, _intent_classifier, _sentiment_analyzer, _emotion_detector
    global _behavior_predictor, _loot_balancer, _dialogue_ranker
    logger.info("Starting Synthesus 2.0 Production Server...")

    # ── Initialize ML Swarm ──
    if HAS_ML_SWARM:
        logger.info("Training ML Swarm micro-models...")
        t0 = time.time()
        _intent_classifier = IntentClassifier()
        _intent_classifier.train()
        _sentiment_analyzer = SentimentAnalyzer()
        _sentiment_analyzer.train()
        _emotion_detector = EmotionDetector()
        _behavior_predictor = BehaviorPredictor()
        _loot_balancer = LootBalancer()
        _dialogue_ranker = DialogueRanker()
        ml_ms = (time.time() - t0) * 1000
        logger.info(f"ML Swarm ready: 6 models trained in {ml_ms:.0f}ms")
    else:
        logger.warning("ML Swarm not available — running without ML classification")

    # ── Load RAG pipeline ──
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

    # ── Pre-load characters ──
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


class PatternIngest(BaseModel):
    # Support both a list of patterns and a single pattern object
    patterns: Optional[List[Dict[str, Any]]] = None
    # For single pattern direct POST
    pattern: Optional[str] = None
    response: Optional[str] = None
    domain: Optional[str] = None
    # Character association
    character_id: Optional[str] = Field(None, description="Target character for these patterns")
    create_character: bool = Field(False, description="Whether to create the character if it doesn't exist")
    source: Optional[str] = None
    module: Optional[str] = None

@app.post("/api/patterns/ingest")
async def ingest_patterns(req: Request):
    """Ingest new patterns into the FAISS index and metadata."""
    global _rag
    
    # Auto-initialize RAG if it's missing
    if not _rag:
        logger.info("Initializing fresh RAG pipeline for ingestion...")
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _rag = RAGPipeline(
                index_path=str(INDEX_PATH),
                metadata_path=str(METADATA_PATH),
                top_k=5,
                score_threshold=0.5,
            )
            # If still no index (RAGPipeline didn't create one), force it
            if not hasattr(_rag, "_index") or _rag._index is None:
                _rag._index = faiss.IndexFlatIP(128)
                _rag._metadata = []
        except Exception as e:
            logger.error(f"Failed to auto-init RAG: {e}")
            raise HTTPException(500, f"RAG initialization failed: {e}")
    
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    # Normalize to a list of patterns
    new_patterns = []
    if isinstance(body, list):
        new_patterns = body
    elif isinstance(body, dict):
        if "patterns" in body and isinstance(body["patterns"], list):
            new_patterns = body["patterns"]
        else:
            # Single object
            new_patterns = [body]
    
    if not new_patterns:
        return {"status": "success", "added": 0}
    
    # ── Step 1: Extract and normalize patterns ──
    char_id = body.get("character_id")
    create_char = body.get("create_character", False)
    
    normalized = []
    for p in new_patterns:
        text = p.get("pattern") or p.get("phrase") or ""
        resp = p.get("response") or p.get("response_template") or ""
        src = p.get("source") or "zo_computer_agent"
        dom = p.get("domain") or p.get("module") or "general"
        # Per-pattern character override
        p_char_id = p.get("character_id") or char_id
        
        if text:
            item = {
                "pattern": text,
                "response": resp,
                "source": src,
                "domain": dom
            }
            if p_char_id:
                item["character_id"] = p_char_id
            normalized.append(item)
    
    if not normalized:
        return {"status": "success", "added": 0}
    
    # ── Step 2: Handle character creation if needed ──
    if char_id and create_char:
        char_dir = CHARACTERS_DIR / char_id
        if not char_dir.exists():
            logger.info(f"Bootstrapping character: {char_id}")
            try:
                factory = CharacterFactory(characters_dir=str(CHARACTERS_DIR))
                spec = CharacterSpec(
                    name=char_id.capitalize(),
                    id=char_id,
                    archetype="scholar",  # Default archetype
                    backstory=f"A synthetic intelligence specialized in {normalized[0].get('domain', 'general knowledge')}."
                )
                factory.generate(spec)
                _load_character(char_id)
            except Exception as e:
                logger.error(f"Failed to bootstrap character {char_id}: {e}")
                # We continue even if bootstrap fails, patterns will just be tagged with the ID
    
    # ── Step 3: Add to RAG pipeline ──
    try:
        # Use the RAGPipeline's built-in method which handles embedding and indexing
        added_count = _rag.add_patterns(
            patterns=normalized,
            # We don't pass global character_id here because it's already in the normalized items
        )
        
        # ── Step 4: Final save ──
        _rag.save_index()
            
        logger.info(f"Ingested {added_count} new patterns (Target: {char_id or 'global'})")
        return {
            "status": "success", 
            "added": added_count, 
            "total": _rag.total_vectors,
            "character_id": char_id
        }
    except Exception as e:
        logger.error(f"Failed to ingest patterns: {e}")
        raise HTTPException(500, f"Ingestion failed: {e}")

# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard UI."""
    dashboard = STATIC_DIR / "index.html"
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
    """Main query endpoint — ML Swarm → Cognitive → RAG → Fallback."""
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

    # ── Step 0: ML Swarm Preprocessing ──
    # Classify intent, sentiment, and player emotion BEFORE cognitive engine
    ml_context = {}
    if HAS_ML_SWARM and _intent_classifier:
        intent, intent_conf = _intent_classifier.predict(query_text)
        sentiment, sent_conf = _sentiment_analyzer.predict(query_text)
        player_emotion = _emotion_detector.detect(query_text)

        # Build conversation features for behavior prediction
        conv_history = _conversations.get(session_id, [])
        turn_count = len([m for m in conv_history if m["role"] == "user"])
        avg_msg_len = (
            sum(len(m["content"].split()) for m in conv_history if m["role"] == "user") / max(turn_count, 1)
        )
        question_ratio = (
            sum(1 for m in conv_history if m["role"] == "user" and m["content"].strip().endswith("?"))
            / max(turn_count, 1)
        )
        behavior = _behavior_predictor.predict({
            "turn_count": turn_count,
            "avg_msg_length": avg_msg_len,
            "sentiment_trend": 0.0 if sentiment == "neutral" else (0.3 if sentiment == "positive" else -0.3),
            "topic_switches": 0,
            "time_between_msgs": 5.0,
            "question_ratio": question_ratio,
        })

        ml_context = {
            "intent": intent,
            "intent_confidence": intent_conf,
            "sentiment": sentiment,
            "sentiment_confidence": sent_conf,
            "player_emotion": player_emotion["primary"],
            "emotion_intensity": player_emotion["intensity"],
            "emotion_scores": player_emotion["scores"],
            "predicted_action": behavior["predicted_action"],
            "engagement_score": behavior["engagement_score"],
            "escalation_risk": behavior["escalation_risk"],
        }

    # ── Step 1: Try cognitive engine (full NPC brain) ──
    if HAS_COGNITIVE and req.mode in ("cognitive", "auto"):
        engine = _get_cognitive_engine(char_id)
        if engine:
            result = engine.process_query(
                player_id=req.player_id,
                query=query_text,
                thinking_layer_available=False,
                ml_context=ml_context,
            )
            if result.get("confidence", 0) > 0.7:  # High threshold — only use cognitive for strong matches
                latency = (time.time() - t0) * 1000
                # Store in conversation
                _conversations[session_id].append({"role": "user", "content": query_text})
                _conversations[session_id].append({"role": "assistant", "content": result["response"]})

                debug_data = result.get("debug", {}) if req.include_debug else None
                if debug_data and ml_context:
                    debug_data["ml_swarm"] = ml_context

                return QueryResponse(
                    response=result["response"],
                    confidence=result["confidence"],
                    character=char_id,
                    source="cognitive",
                    session_id=session_id,
                    latency_ms=round(latency, 2),
                    emotion=result.get("emotion"),
                    relationship=result.get("relationship") if req.include_debug else None,
                    debug=debug_data,
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
        "ml_swarm": {
            "enabled": HAS_ML_SWARM,
            "models": 7 if HAS_ML_SWARM else 0,
            "footprint_kb": 458 if HAS_ML_SWARM else 0,
        },
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

      # ─── Pattern Ingest Endpoint ──────────────────────────────────────────────────
      _pe_mod = _import_module_direct('core.pattern_engine', str(PROJ_ROOT / 'core' / 'pattern_engine.py'))
      PatternEngine = _pe_mod.PatternEngine
  _pattern_engine: Optional[Any] = None

class PatternIngestRequest(BaseModel):
      phrase: str
      source: str = 'zo_enrichment'
      module: str = 'general'
      confidence: float = 0.75
      character_id: str = 'global'
      response_template: str = ''

  @app.post('/api/patterns/ingest')
async def ingest_pattern(req: PatternIngestRequest):
      """Ingest an enrichment pattern from Zo Computer or external automation."""
      global _pattern_engine
      if _pattern_engine is None:
                _pattern_engine = PatternEngine(db_path=str(DATA_DIR / 'patterns.db'))
            response = req.response_template or f'[Enriched: {req.phrase}] Domain: {req.module}. Source: {req.source}.'
    pattern = _pattern_engine.add_pattern(
              character_id=req.character_id,
              pattern_type=req.module,
              trigger=req.phrase,
              response_template=response,
              weight=req.confidence,
              metadata={'source': req.source, 'module': req.module},
          )
    logger.info(f'Pattern ingested: {pattern.id} | trigger={pattern.trigger} | char={pattern.character_id}')
    return {
              'status': 'ok',
              'pattern_id': pattern.id,
              'trigger': pattern.trigger,
              'character_id': pattern.character_id,
              'pattern_type': pattern.pattern_type,
              'weight': pattern.weight,
          }
        "requests": _request_count,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
