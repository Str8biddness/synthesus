#!/usr/bin/env python3
"""
HemisphereBridge - Dual-Hemisphere Orchestration Layer
AIVM Synthesus 2.0

Manages communication between:
  Left Hemisphere:  C++ PPBRS kernel (zo_kernel) via stdin/stdout pipe
  Right Hemisphere: SLM (Qwen3-0.6B / Phi-4-mini) via llama-cpp-python
  Metacognition:    Agreement tracking, confidence reconciliation
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)


class HemisphereMode(Enum):
    LEFT = "left"       # Pattern-based kernel only
    RIGHT = "right"     # SLM only
    BOTH = "both"       # Parallel then reconcile
    AUTO = "auto"       # Bridge decides based on query


@dataclass
class HemisphereResult:
    response: str
    hemisphere_used: str
    raw_confidence: float
    agreement_score: Optional[float] = None
    left_response: Optional[str] = None
    right_response: Optional[str] = None
    latency_ms: float = 0.0


class HemisphereBridge:
    """
    Orchestrates the dual-hemisphere Synthesus architecture.
    Left: C++ PPBRS kernel (1000+ QPS pattern matching)
    Right: Qwen3-0.6B SLM (generative, context-aware)
    """

    def __init__(
        self,
        kernel_bin: str = "./build/zo_kernel",
        slm_model_path: str = "./models/qwen3-0.6b-q4_k_m.gguf",
        kernel_timeout: float = 2.0,
        slm_n_ctx: int = 32768,
        slm_n_threads: int = 8,
        agreement_threshold: float = 0.65,
    ):
        self.kernel_bin = kernel_bin
        self.slm_model_path = slm_model_path
        self.kernel_timeout = kernel_timeout
        self.agreement_threshold = agreement_threshold

        # Stats
        self._queries_total = 0
        self._left_wins = 0
        self._right_wins = 0
        self._agreement_count = 0
        self._agreement_sum = 0.0

        # Initialize SLM (optional — llama-cpp-python may not be installed)
        self.slm = None
        if Llama is not None:
            try:
                logger.info(f"Loading SLM from {slm_model_path}...")
                self.slm = Llama(
                    model_path=slm_model_path,
                    n_ctx=slm_n_ctx,
                    n_threads=slm_n_threads,
                    verbose=False
                )
                logger.info("SLM loaded.")
            except Exception as e:
                logger.warning(f"SLM failed to load (non-fatal): {e}")
                self.slm = None
        else:
            logger.warning("llama-cpp-python not installed — right hemisphere SLM disabled.")

        # Kernel process (persistent pipe)
        self._kernel_proc: Optional[subprocess.Popen] = None
        self._start_kernel()

    def _start_kernel(self):
        """Launch the C++ PPBRS kernel as a persistent subprocess."""
        try:
            self._kernel_proc = subprocess.Popen(
                [self.kernel_bin],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            logger.info(f"Kernel started (PID {self._kernel_proc.pid})")
        except FileNotFoundError:
            logger.warning(f"Kernel binary not found at {self.kernel_bin}. Left hemisphere disabled.")
            self._kernel_proc = None

    def ping_kernel(self) -> bool:
        """Check if kernel process is alive."""
        if self._kernel_proc is None:
            return False
        return self._kernel_proc.poll() is None

    def ping_slm(self) -> bool:
        """Check if SLM is loaded."""
        return self.slm is not None

    def _query_kernel(self, query: str, character_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send query to C++ kernel via stdin pipe.
        Returns dict with response, confidence, hemisphere_id.
        """
        if not self.ping_kernel():
            return {"response": "", "confidence": 0.0, "found": False}

        payload = json.dumps({"query": query, "character_id": character_id or ""}) + "\n"
        try:
            self._kernel_proc.stdin.write(payload)
            self._kernel_proc.stdin.flush()
            line = self._kernel_proc.stdout.readline()
            result = json.loads(line.strip())
            return result
        except Exception as e:
            logger.error(f"Kernel query error: {e}")
            return {"response": "", "confidence": 0.0, "found": False}

    def _query_slm(
        self,
        query: str,
        character_context: Optional[Dict] = None,
        rag_context: str = "",
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Query the right hemisphere SLM."""
        system_prompt = "You are Synthesus, an advanced synthetic intelligence."
        if character_context:
            persona = character_context.get("personality", {}).get("voice", {})
            system_prompt = (
                f"You are {character_context['name']}. "
                f"Tone: {persona.get('base_tone', 'professional')}. "
                f"Speak as this character naturally."
            )

        messages = [{"role": "system", "content": system_prompt}]
        if rag_context:
            messages.append({"role": "system", "content": f"Context:\n{rag_context}"})
        messages.append({"role": "user", "content": query})

        try:
            output = self.slm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            response = output["choices"][0]["message"]["content"].strip()
            return {"response": response, "confidence": 0.75}
        except Exception as e:
            logger.error(f"SLM query error: {e}")
            return {"response": "", "confidence": 0.0}

    def _calculate_agreement(self, left_resp: str, right_resp: str) -> float:
        """
        Simple agreement score between hemispheres.
        Uses token overlap as proxy - full semantic comparison via FAISS in production.
        """
        if not left_resp or not right_resp:
            return 0.0
        left_tokens = set(left_resp.lower().split())
        right_tokens = set(right_resp.lower().split())
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(intersection) / len(union)  # Jaccard similarity

    async def route_query(
        self,
        query: str,
        hemisphere: str = "auto",
        character_context: Optional[Dict] = None,
        rag_context: str = "",
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Route query to appropriate hemisphere(s) and reconcile.
        AUTO mode: tries left first (fast), falls back to right if low confidence.
        """
        self._queries_total += 1
        start = time.time()

        character_id = character_context.get("character_id") if character_context else None
        mode = HemisphereMode(hemisphere)

        left_result = None
        right_result = None

        if mode in (HemisphereMode.LEFT, HemisphereMode.BOTH, HemisphereMode.AUTO):
            left_result = self._query_kernel(query, character_id)

        # AUTO: only call right if left didn't find a good match
        if mode == HemisphereMode.AUTO:
            if left_result and left_result.get("confidence", 0) >= self.agreement_threshold:
                # Left hemisphere is confident - use it
                self._left_wins += 1
                return {
                    "response": left_result["response"],
                    "hemisphere_used": "left",
                    "raw_confidence": left_result["confidence"],
                    "agreement_score": None,
                    "latency_ms": (time.time() - start) * 1000
                }
            else:
                # Escalate to right hemisphere
                right_result = self._query_slm(query, character_context, rag_context, max_tokens)
                self._right_wins += 1
                return {
                    "response": right_result["response"],
                    "hemisphere_used": "right",
                    "raw_confidence": right_result["confidence"],
                    "agreement_score": None,
                    "latency_ms": (time.time() - start) * 1000
                }

        elif mode == HemisphereMode.BOTH:
            # Run both in parallel
            loop = asyncio.get_event_loop()
            right_result = await loop.run_in_executor(
                None,
                lambda: self._query_slm(query, character_context, rag_context, max_tokens)
            )
            agreement = self._calculate_agreement(
                left_result.get("response", ""),
                right_result.get("response", "")
            )
            self._agreement_count += 1
            self._agreement_sum += agreement

            # Use left if agreement is high (kernel is ground truth), else right
            if agreement >= self.agreement_threshold:
                best_response = left_result["response"]
                hemisphere_used = "left+right(agreed)"
                self._left_wins += 1
            else:
                best_response = right_result["response"]
                hemisphere_used = "right(escalated)"
                self._right_wins += 1

            return {
                "response": best_response,
                "hemisphere_used": hemisphere_used,
                "raw_confidence": max(left_result.get("confidence", 0), right_result.get("confidence", 0)),
                "agreement_score": agreement,
                "left_response": left_result.get("response"),
                "right_response": right_result.get("response"),
                "latency_ms": (time.time() - start) * 1000
            }

        elif mode == HemisphereMode.LEFT:
            self._left_wins += 1
            return {
                "response": left_result.get("response", ""),
                "hemisphere_used": "left",
                "raw_confidence": left_result.get("confidence", 0),
                "agreement_score": None,
                "latency_ms": (time.time() - start) * 1000
            }

        elif mode == HemisphereMode.RIGHT:
            right_result = self._query_slm(query, character_context, rag_context, max_tokens)
            self._right_wins += 1
            return {
                "response": right_result.get("response", ""),
                "hemisphere_used": "right",
                "raw_confidence": right_result.get("confidence", 0),
                "agreement_score": None,
                "latency_ms": (time.time() - start) * 1000
            }

    def get_kernel_stats(self) -> Dict[str, Any]:
        """Return kernel and bridge statistics."""
        avg_agreement = (
            self._agreement_sum / self._agreement_count
            if self._agreement_count > 0 else 0.0
        )
        return {
            "kernel_alive": self.ping_kernel(),
            "slm_loaded": self.ping_slm(),
            "queries_total": self._queries_total,
            "left_wins": self._left_wins,
            "right_wins": self._right_wins,
            "avg_agreement_score": round(avg_agreement, 4),
            "agreement_threshold": self.agreement_threshold
        }

    def __del__(self):
        if self._kernel_proc:
            self._kernel_proc.terminate()
