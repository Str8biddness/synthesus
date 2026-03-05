# core/reasoning_core.py
# Synthesus 2.0 - Reasoning Core
# Orchestrates multi-step reasoning chains using hemisphere bridge + pattern engine

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hemisphere_bridge import HemisphereBridge
from .pattern_engine import PatternEngine, PatternMatch
from .els_bridge import ELSBridge


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    step_id: str
    step_type: str           # "left", "right", "synthesis", "pattern_recall"
    input_text: str
    output_text: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    session_id: str
    character_id: str
    query: str
    final_response: str
    steps: List[ReasoningStep] = field(default_factory=list)
    total_latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ReasoningCore
# ---------------------------------------------------------------------------

class ReasoningCore:
    """
    Orchestrates reasoning for a Synthesus character.

    Pipeline:
      1. Recall relevant patterns
      2. Left-hemisphere analytical pass
      3. Right-hemisphere creative/intuitive pass
      4. Synthesis via HemisphereBridge
      5. Log interaction via ELSBridge
      6. Discover new patterns from successful interactions
    """

    def __init__(
        self,
        character_id: str,
        hemisphere_bridge: Optional[HemisphereBridge] = None,
        pattern_engine: Optional[PatternEngine] = None,
        els_bridge: Optional[ELSBridge] = None,
        left_model: str = "left",
        right_model: str = "right",
    ):
        self.character_id = character_id
        self.hb = hemisphere_bridge or HemisphereBridge()
        self.pe = pattern_engine or PatternEngine()
        self.els = els_bridge or ELSBridge()
        self.left_model = left_model
        self.right_model = right_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        query: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k_patterns: int = 3,
    ) -> ReasoningResult:
        session_id = session_id or str(uuid.uuid4())
        steps: List[ReasoningStep] = []
        t_start = time.perf_counter()

        # 1. Pattern recall
        pattern_context = self._recall_patterns(query, top_k_patterns, steps)

        # 2. Build enriched prompt
        enriched = self._build_prompt(query, context, pattern_context)

        # 3. Left hemisphere (analytical)
        left_out = self._run_hemisphere(
            "left", enriched, steps, model=self.left_model
        )

        # 4. Right hemisphere (creative)
        right_out = self._run_hemisphere(
            "right", enriched, steps, model=self.right_model
        )

        # 5. Synthesis
        final = self._synthesize(left_out, right_out, steps)

        total_ms = (time.perf_counter() - t_start) * 1000

        result = ReasoningResult(
            session_id=session_id,
            character_id=self.character_id,
            query=query,
            final_response=final,
            steps=steps,
            total_latency_ms=round(total_ms, 2),
            success=True,
        )

        # 6. Log + learn
        self._log_and_learn(query, final, result.success)

        return result

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _recall_patterns(
        self,
        query: str,
        top_k: int,
        steps: List[ReasoningStep],
    ) -> str:
        t0 = time.perf_counter()
        matches: List[PatternMatch] = self.pe.match(
            self.character_id, query, top_k=top_k
        )
        latency = (time.perf_counter() - t0) * 1000

        if not matches:
            return ""

        recalled = "\n".join(
            f"[Pattern {i+1} | score={m.score:.3f}]: {m.pattern.response_template[:120]}"
            for i, m in enumerate(matches)
        )

        steps.append(ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="pattern_recall",
            input_text=query,
            output_text=recalled,
            confidence=matches[0].score if matches else 0.0,
            latency_ms=round(latency, 2),
            metadata={"num_patterns": len(matches)},
        ))
        return recalled

    def _run_hemisphere(
        self,
        side: str,
        prompt: str,
        steps: List[ReasoningStep],
        model: str,
    ) -> str:
        t0 = time.perf_counter()
        if side == "left":
            out = self.hb.left(prompt)
        else:
            out = self.hb.right(prompt)
        latency = (time.perf_counter() - t0) * 1000

        steps.append(ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type=side,
            input_text=prompt[:200],
            output_text=out[:500],
            latency_ms=round(latency, 2),
        ))
        return out

    def _synthesize(
        self,
        left_out: str,
        right_out: str,
        steps: List[ReasoningStep],
    ) -> str:
        t0 = time.perf_counter()
        combined = self.hb.synthesize(left_out, right_out)
        latency = (time.perf_counter() - t0) * 1000

        steps.append(ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="synthesis",
            input_text=f"left={left_out[:100]} | right={right_out[:100]}",
            output_text=combined[:500],
            latency_ms=round(latency, 2),
        ))
        return combined

    def _build_prompt(
        self,
        query: str,
        context: Optional[str],
        pattern_context: str,
    ) -> str:
        parts = []
        if pattern_context:
            parts.append(f"--- Recalled Patterns ---\n{pattern_context}\n")
        if context:
            parts.append(f"--- Context ---\n{context}\n")
        parts.append(f"--- Query ---\n{query}")
        return "\n".join(parts)

    def _log_and_learn(
        self, query: str, response: str, success: bool
    ) -> None:
        try:
            self.els.capture(
                character_id=self.character_id,
                user_input=query,
                character_response=response,
                outcome_success=success,
            )
            self.pe.discover(
                character_id=self.character_id,
                interaction_text=f"{query} {response}",
                outcome_success=success,
            )
        except Exception:
            pass  # Non-fatal: logging/learning failures should not break reasoning

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "pattern_stats": self.pe.stats(self.character_id),
            "els_stats": self.els.stats(self.character_id),
        }
