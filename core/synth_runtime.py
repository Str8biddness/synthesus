# core/synth_runtime.py
# Synthesus 2.0 - Synth Runtime
# Top-level runtime that wires all subsystems and exposes a clean public API

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .character_factory import CharacterFactory, Character
from .hemisphere_bridge import HemisphereBridge
from .pattern_engine import PatternEngine
from .els_bridge import ELSBridge
from .memory_store import MemoryStore
from .reasoning_core import ReasoningCore, ReasoningResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SynthRuntime
# ---------------------------------------------------------------------------

class SynthRuntime:
    """
    Synthesus 2.0 top-level runtime.

    Usage:
        runtime = SynthRuntime()
        runtime.create_character("synth", "Synth", "default")
        result = runtime.respond("synth", "Hello, tell me about reasoning.")
        print(result.final_response)
    """

    def __init__(
        self,
        characters_dir: str = "characters",
        data_dir: str = "data",
        left_model: str = "left",
        right_model: str = "right",
    ):
        self.characters_dir = characters_dir
        self.data_dir = data_dir
        self.left_model = left_model
        self.right_model = right_model

        # Shared subsystems
        self._factory = CharacterFactory(characters_dir=characters_dir)
        self._pattern_engine = PatternEngine(db_path=f"{data_dir}/patterns.db")
        self._els_bridge = ELSBridge(
            db_path=f"{data_dir}/interactions.db",
            patterns_path=f"{data_dir}/candidate_patterns.json",
        )
        self._memory_store = MemoryStore(db_path=f"{data_dir}/memory.db")
        self._hemisphere_bridge = HemisphereBridge()

        # Per-character reasoning cores (lazy)
        self._cores: Dict[str, ReasoningCore] = {}

        logger.info("SynthRuntime initialized")

    # ------------------------------------------------------------------
    # Character management
    # ------------------------------------------------------------------

    def create_character(
        self,
        character_id: str,
        name: str,
        archetype: str = "default",
        traits: Optional[List[str]] = None,
        backstory: str = "",
        **kwargs,
    ) -> Character:
        char = self._factory.create(
            character_id=character_id,
            name=name,
            archetype=archetype,
            traits=traits or [],
            backstory=backstory,
            **kwargs,
        )
        logger.info(f"Created character: {character_id}")
        return char

    def load_character(self, character_id: str) -> Optional[Character]:
        return self._factory.load(character_id)

    def list_characters(self) -> List[str]:
        return self._factory.list_ids()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def respond(
        self,
        character_id: str,
        user_input: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ReasoningResult:
        """Main inference endpoint. Returns a full ReasoningResult."""
        core = self._get_core(character_id)
        result = core.reason(
            query=user_input,
            context=context,
            session_id=session_id,
        )

        # Store episodic memory
        self._memory_store.store(
            character_id=character_id,
            content=f"User: {user_input}\nSynth: {result.final_response}",
            memory_type="episodic",
            importance=0.5,
        )

        return result

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        character_id: str,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> None:
        self._memory_store.store(
            character_id=character_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
        )

    def recall(
        self,
        character_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        memories = self._memory_store.recall(
            character_id=character_id,
            query=query,
            top_k=top_k,
        )
        return [m.content for m in memories]

    # ------------------------------------------------------------------
    # Pattern management
    # ------------------------------------------------------------------

    def add_pattern(
        self,
        character_id: str,
        trigger: str,
        response_template: str,
        pattern_type: str = "reasoning",
        weight: float = 1.0,
    ) -> None:
        self._pattern_engine.add_pattern(
            character_id=character_id,
            pattern_type=pattern_type,
            trigger=trigger,
            response_template=response_template,
            weight=weight,
        )

    def review_candidates(
        self,
        character_id: str,
        approve_all: bool = False,
    ) -> int:
        """Approve pending ELS candidate patterns into the pattern engine."""
        candidates = self._els_bridge.get_candidates(
            character_id=character_id, status="pending"
        )
        approved = []
        for c in candidates:
            if approve_all or c.get("score", 0) > 0.6:
                approved.append(c)
        if approved:
            return self._els_bridge.integrate_patterns(
                character_id=character_id, approved=approved
            )
        return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, character_id: str) -> Dict[str, Any]:
        core = self._get_core(character_id)
        return {
            **core.stats(),
            "memory": self._memory_store.stats(character_id),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_core(self, character_id: str) -> ReasoningCore:
        if character_id not in self._cores:
            self._cores[character_id] = ReasoningCore(
                character_id=character_id,
                hemisphere_bridge=self._hemisphere_bridge,
                pattern_engine=self._pattern_engine,
                els_bridge=self._els_bridge,
                left_model=self.left_model,
                right_model=self.right_model,
            )
        return self._cores[character_id]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_runtime: Optional[SynthRuntime] = None


def get_runtime(**kwargs) -> SynthRuntime:
    """Return the module-level singleton runtime, creating it if needed."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = SynthRuntime(**kwargs)
    return _default_runtime
