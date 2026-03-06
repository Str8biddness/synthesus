"""
Synthesus 2.0 — Pytest Fixtures & Character Auto-Discovery

Automatically discovers all characters under characters/ and provides
shared fixtures for the test harness.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHARACTERS_DIR = ROOT / "characters"


# ──────────────────────────────────────────────────
# Character genome loader
# ──────────────────────────────────────────────────

class CharacterGenome:
    """Fully loaded character genome — all 4 files."""

    def __init__(self, char_id: str, char_dir: Path):
        self.id = char_id
        self.dir = char_dir
        self.bio: Dict[str, Any] = self._load("bio.json")
        self.patterns: Dict[str, Any] = self._load("patterns.json", default={})
        self.knowledge: Dict[str, Any] = self._load("knowledge.json", default={})
        self.personality: Dict[str, Any] = self._load("personality.json", default={})

        # Derived properties
        self.name: str = self.bio.get("name", self.bio.get("display_name", char_id))
        self.archetype: str = self.bio.get("archetype", self.bio.get("role", "unknown"))
        self.domains: List[str] = self.bio.get("pattern_domains", [])
        self.setting: str = self.bio.get("setting", "unknown")
        self.is_full: bool = self._has_file("knowledge.json") and self._has_file("personality.json")

        # Pre-extract test triggers from genome
        self.synthetic_patterns = self.patterns.get("synthetic_patterns", [])
        self.generic_patterns = self.patterns.get("generic_patterns", [])
        self.knowledge_entities = self.knowledge.get("entities", {})
        # Personality uses "responses" dict keyed by intent name
        self.personality_responses = self.personality.get("responses", {})
        self.fallback_text = self.patterns.get("fallback", "")

    def _load(self, filename: str, default=None) -> Dict:
        path = self.dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        if default is not None:
            return default
        raise FileNotFoundError(f"{path} not found")

    def _has_file(self, filename: str) -> bool:
        return (self.dir / filename).exists()

    def get_test_triggers(self) -> List[Dict[str, Any]]:
        """Extract testable trigger→response pairs from all patterns."""
        triggers = []
        for pat in self.synthetic_patterns:
            t_list = pat.get("trigger", [])
            if isinstance(t_list, str):
                t_list = [t_list]
            for t in t_list:
                triggers.append({
                    "input": t,
                    "expected_response": pat.get("response_template", ""),
                    "pattern_id": pat.get("id", "unknown"),
                    "confidence": pat.get("confidence", 0.5),
                    "source": "synthetic",
                })
        for pat in self.generic_patterns:
            t_list = pat.get("trigger", [])
            if isinstance(t_list, str):
                t_list = [t_list]
            for t in t_list:
                triggers.append({
                    "input": t,
                    "expected_response": pat.get("response_template", ""),
                    "pattern_id": pat.get("id", "unknown"),
                    "confidence": pat.get("confidence", 0.5),
                    "source": "generic",
                })
        return triggers

    def get_knowledge_queries(self) -> List[Dict[str, Any]]:
        """Extract testable queries from knowledge graph entities.
        
        Knowledge entities are stored as a dict keyed by entity name:
        { "tomas": { "entity_type": "person", "display_name": "Tomás", ... } }
        """
        queries = []
        entities = self.knowledge.get("entities", {})
        # Handle both dict (keyed) and list formats
        if isinstance(entities, dict):
            items = [(k, v) for k, v in entities.items()]
        elif isinstance(entities, list):
            items = [(e.get("name", f"entity_{i}"), e) for i, e in enumerate(entities)]
        else:
            return queries

        for name, entity_data in items:
            if isinstance(entity_data, str):
                # Simple string value — skip
                continue
            display = entity_data.get("display_name", name)
            etype = entity_data.get("entity_type", entity_data.get("type", ""))
            aliases = entity_data.get("aliases", [])
            description = entity_data.get("description", "")
            queries.append({
                "input": f"Tell me about {display}",
                "entity_name": display,
                "entity_type": etype,
                "description": description,
            })
            if aliases:
                for alias in aliases[:2]:
                    queries.append({
                        "input": f"What is {alias}?",
                        "entity_name": display,
                        "entity_type": etype,
                        "description": description,
                    })
        return queries

    def get_personality_queries(self) -> List[Dict[str, Any]]:
        """Extract testable personality intent triggers.
        
        Personality uses a 'responses' dict keyed by intent name:
        { "song": [{"text": "..."}], "joke": [{"text": "..."}], ... }
        """
        queries = []
        # Map intent names to natural language triggers
        _INTENT_TRIGGERS = {
            "song": ["Sing me a song", "Do you know any songs?"],
            "joke": ["Tell me a joke", "Do you know any jokes?"],
            "favorite": ["What's your favorite thing?", "What do you like?"],
            "opinion": ["What's your opinion?", "What do you think?"],
            "personal": ["Tell me about yourself", "Are you happy?"],
            "philosophical": ["What is the meaning of life?", "What happens when we die?"],
            "compliment_response": ["You're amazing!", "You're the best!"],
            "insult_response": ["You're an idiot", "You're terrible"],
            "creative_request": ["Tell me a story", "Tell me a riddle"],
            "rumor": ["Heard any rumors?", "What's the gossip?"],
            "advice": ["Give me some advice", "Any tips?"],
        }
        for intent_name, responses in self.personality_responses.items():
            triggers = _INTENT_TRIGGERS.get(intent_name, [f"Tell me about {intent_name}"])
            for t in triggers[:2]:
                queries.append({
                    "input": t,
                    "intent": intent_name,
                    "expected_responses": [r.get("text", "") for r in responses if isinstance(r, dict)],
                })
        return queries

    def __repr__(self):
        return f"<CharacterGenome {self.id} ({self.archetype}) full={self.is_full}>"


# ──────────────────────────────────────────────────
# Auto-discover all characters
# ──────────────────────────────────────────────────

def discover_characters() -> List[CharacterGenome]:
    """Scan characters/ directory and load all valid character genomes."""
    chars = []
    if not CHARACTERS_DIR.exists():
        return chars
    for entry in sorted(CHARACTERS_DIR.iterdir()):
        if entry.is_dir() and (entry / "bio.json").exists():
            # Skip schema directory
            if entry.name == "schema":
                continue
            try:
                genome = CharacterGenome(entry.name, entry)
                chars.append(genome)
            except Exception as e:
                print(f"WARNING: Failed to load character {entry.name}: {e}")
    return chars


ALL_CHARACTERS = discover_characters()
FULL_CHARACTERS = [c for c in ALL_CHARACTERS if c.is_full]
STUB_CHARACTERS = [c for c in ALL_CHARACTERS if not c.is_full]


# ──────────────────────────────────────────────────
# Pytest fixtures
# ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def all_characters():
    """All discovered characters (full + stubs)."""
    return ALL_CHARACTERS


@pytest.fixture(scope="session")
def full_characters():
    """Only characters with complete genomes (bio+patterns+knowledge+personality)."""
    return FULL_CHARACTERS


@pytest.fixture(scope="session")
def stub_characters():
    """Characters with partial genomes (bio+patterns only)."""
    return STUB_CHARACTERS


@pytest.fixture(scope="session")
def server_url():
    """The base URL for the running Synthesus server."""
    return os.getenv("SYNTHESUS_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def api_client(server_url):
    """HTTP client for the Synthesus API."""
    import httpx
    client = httpx.Client(base_url=server_url, timeout=15.0)
    yield client
    client.close()
