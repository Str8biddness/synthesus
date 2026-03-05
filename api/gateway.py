#!/usr/bin/env python3
"""
Synthesus 2.0 API Gateway
AIVM LLC - Dual-Hemisphere Synthetic Intelligence

Routes queries to the appropriate hemisphere:
 - Left Hemisphere: C++ PPBRS kernel (pattern matching, logic, planning)
 - Right Hemisphere: SLM (Qwen3-0.6B -> Phi-4-mini, generative tasks)
 - Metacognition: Agreement tracking, confidence calibration
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from synthesus.core.hemisphere_bridge import HemisphereBridge
from synthesus.core.rag_pipeline import RAGPipeline
from synthesus.characters.character_loader import CharacterLoader
from synthesus.metacognition.confidence import ConfidenceCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Synthesus 2.0 API",
    description="AIVM Dual-Hemisphere Synthetic Intelligence - Real Life NPCs",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
bridge = HemisphereBridge()
rag = RAGPipeline()
character_loader = CharacterLoader()
calibrator = ConfidenceCalibrator()


class QueryRequest(BaseModel):
    query: str
    character_id: Optional[str] = None
    session_id: Optional[str] = None
    hemisphere: Optional[str] = "auto"  # "left", "right", "both", "auto"
    use_rag: bool = True
    max_tokens: int = 512


class QueryResponse(BaseModel):
    response: str
    hemisphere_used: str
    confidence: float
    agreement_score: Optional[float] = None
    character_id: Optional[str] = None
    latency_ms: float
    rag_sources: Optional[list] = None


async def verify_api_key(x_api_key: str = Header(...)):
    """Validate API key from request header."""
    # TODO: Replace with real key validation from DB
    if not x_api_key or len(x_api_key) < 16:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.get("/health")
async def health_check():
    """API health check endpoint."""
    kernel_status = bridge.ping_kernel()
    slm_status = bridge.ping_slm()
    return {
        "status": "operational",
        "kernel": "up" if kernel_status else "down",
        "slm": "up" if slm_status else "down",
        "version": "2.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query_synthesus(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main query endpoint. Routes to appropriate hemisphere based on query type.
    Uses RAG pipeline for context retrieval when enabled.
    """
    start_time = time.time()

    # Load character context if specified
    character_context = None
    if request.character_id:
        character_context = character_loader.load(request.character_id)
        if not character_context:
            raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    # RAG context retrieval
    rag_sources = None
    rag_context = ""
    if request.use_rag:
        rag_result = await rag.retrieve(request.query, character_id=request.character_id)
        rag_context = rag_result.get("context", "")
        rag_sources = rag_result.get("sources", [])

    # Route to hemisphere
    result = await bridge.route_query(
        query=request.query,
        hemisphere=request.hemisphere,
        character_context=character_context,
        rag_context=rag_context,
        max_tokens=request.max_tokens
    )

    # Calibrate confidence
    confidence = calibrator.calibrate(
        result["raw_confidence"],
        result["hemisphere_used"],
        agreement_score=result.get("agreement_score")
    )

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        response=result["response"],
        hemisphere_used=result["hemisphere_used"],
        confidence=confidence,
        agreement_score=result.get("agreement_score"),
        character_id=request.character_id,
        latency_ms=latency_ms,
        rag_sources=rag_sources
    )


@app.get("/characters")
async def list_characters(api_key: str = Depends(verify_api_key)):
    """List all available character genomes."""
    return character_loader.list_all()


@app.get("/characters/{character_id}")
async def get_character(
    character_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get character genome manifest."""
    character = character_loader.load(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    return character.get_manifest()


@app.get("/kernel/status")
async def kernel_status(api_key: str = Depends(verify_api_key)):
    """Get detailed kernel statistics."""
    return bridge.get_kernel_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
