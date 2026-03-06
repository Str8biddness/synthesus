"""Synthesus 2.0 API Schemas - Pydantic models for request/response validation"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import time


class ProcessRequest(BaseModel):
    """Request model for the /process endpoint."""
    text: str = Field(..., description="Input text to process", min_length=1)
    character_id: str = Field("default", description="Character profile ID")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    stream: bool = Field(False, description="Whether to stream response tokens")


class ProcessResponse(BaseModel):
    """Response model for the /process endpoint."""
    text: str = Field(..., description="Generated response text")
    character_id: str = Field(..., description="Character that responded")
    session_id: str = Field(..., description="Session ID")
    hemisphere_data: Optional[Dict[str, Any]] = Field(None, description="Hemisphere debug info")
    reasoning_trace: Optional[List[str]] = Field(None, description="Reasoning steps")
    timestamp: float = Field(default_factory=time.time)
    processing_ms: Optional[float] = Field(None, description="Processing time in ms")


class SpawnCharacterRequest(BaseModel):
    """Request model for the /character/spawn endpoint."""
    archetype: str = Field(..., description="Archetype name (e.g., 'warrior', 'doctor')")
    name: str = Field(..., description="Character display name")
    traits: Optional[Dict[str, float]] = Field(None, description="Trait overrides 0.0-1.0")
    backstory: Optional[str] = Field(None, description="Custom backstory")
    world_id: Optional[str] = Field(None, description="World/game ID")


class CharacterResponse(BaseModel):
    """Response model for character operations."""
    character_id: str
    name: str
    archetype: str
    traits: Dict[str, float]
    created_at: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str = "ok"
    version: str = "2.0.0"
    uptime_seconds: float
    subsystems: Dict[str, str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
