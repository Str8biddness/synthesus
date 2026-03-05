# Synthesus 2.0 - FastAPI Server
# Full REST + SSE streaming server for the ZO kernel
from __future__ import annotations
import asyncio
import subprocess
import json
import os
from typing import AsyncIterator, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Synthesus 2.0", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])

KERNEL_BIN = os.path.join(os.path.dirname(__file__), "..", "zo_kernel")
_kernel_proc: Optional[subprocess.Popen] = None

def get_kernel():
    global _kernel_proc
    if _kernel_proc is None or _kernel_proc.poll() is not None:
        if os.path.exists(KERNEL_BIN):
            _kernel_proc = subprocess.Popen(
                [KERNEL_BIN], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
    return _kernel_proc

class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    context: str = ""

class QueryResponse(BaseModel):
    response: str
    confidence: float
    module: str
    source: str = "kernel"

@app.get("/", response_class=HTMLResponse)
async def root():
    static_path = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.html")
    if os.path.exists(static_path):
        with open(static_path) as f:
            return f.read()
    return "<h1>Synthesus 2.0</h1><p>Dashboard not found.</p>"

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    kernel = get_kernel()
    if kernel is None:
        # Python fallback
        return QueryResponse(
            response=f"[FALLBACK] Processed: {req.query}",
            confidence=0.5, module="python_fallback", source="fallback"
        )
    try:
        kernel.stdin.write(req.query + "\n")
        kernel.stdin.flush()
        line = kernel.stdout.readline().strip()
        data = json.loads(line)
        return QueryResponse(
            response=data.get("r", ""),
            confidence=data.get("c", 0.0),
            module=data.get("m", "unknown"),
            source="kernel"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    kernel = get_kernel()
    return {"status": "ok", "kernel": kernel is not None and kernel.poll() is None}

@app.get("/stream")
async def stream(q: str):
    async def generator() -> AsyncIterator[dict]:
        kernel = get_kernel()
        if kernel:
            kernel.stdin.write(q + "\n")
            kernel.stdin.flush()
            line = kernel.stdout.readline().strip()
            yield {"data": line}
        else:
            yield {"data": json.dumps({"r": f"[FALLBACK] {q}", "c": 0.5, "m": "fallback"})}
    return EventSourceResponse(generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)