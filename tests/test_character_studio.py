"""
Tests for Phase 16: Character Studio Web UI

Tests cover:
- Session lifecycle (create, update, delete)
- Genome editing (personality traits, name, role)
- Pattern generation from bio
- Chat with character in session
- Export and save operations
- API health endpoint
- Edge cases (missing sessions, invalid data)
"""

import json
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Patch CHARACTERS_DIR before importing the app
_test_char_dir = tempfile.mkdtemp(prefix="studio_test_chars_")

import studio.character_studio as cs
cs.CHARACTERS_DIR = Path(_test_char_dir)

from studio.character_studio import app, _sessions, _generate_patterns, _create_engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Clear sessions and test files between tests."""
    _sessions.clear()
    # Clean test characters dir
    for p in Path(_test_char_dir).iterdir():
        if p.is_dir():
            shutil.rmtree(p)
    yield
    _sessions.clear()


# ══════════════════════════════════════
# Pattern Generation Tests
# ══════════════════════════════════════

class TestPatternGeneration:
    def test_generates_basic_patterns(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Welcome!",
               "personality": {"chattiness": 0.5, "friendliness": 0.5, "humor": 0.3}}
        patterns = _generate_patterns(bio)
        assert "synthetic_patterns" in patterns
        assert len(patterns["synthetic_patterns"]) >= 3  # greeting, name, role, farewell

    def test_greeting_uses_bio(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Ahoy there!",
               "personality": {}}
        patterns = _generate_patterns(bio)
        greet = next(p for p in patterns["synthetic_patterns"] if p["topic"] == "greeting")
        assert greet["response_template"] == "Ahoy there!"

    def test_backstory_adds_pattern(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Hi",
               "backstory": "Born in the mountains", "personality": {}}
        patterns = _generate_patterns(bio)
        topics = [p["topic"] for p in patterns["synthetic_patterns"]]
        assert "backstory" in topics

    def test_humor_trait_adds_joke(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Hi",
               "personality": {"humor": 0.8}}
        patterns = _generate_patterns(bio)
        topics = [p["topic"] for p in patterns["synthetic_patterns"]]
        assert "humor" in topics

    def test_friendly_trait_adds_help(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Hi",
               "personality": {"friendliness": 0.8}}
        patterns = _generate_patterns(bio)
        topics = [p["topic"] for p in patterns["synthetic_patterns"]]
        assert "help" in topics

    def test_aggressive_trait_adds_combat(self):
        bio = {"name": "Orc", "role": "warrior", "greeting": "What?!",
               "personality": {"aggression": 0.8, "friendliness": 0.2}}
        patterns = _generate_patterns(bio)
        topics = [p["topic"] for p in patterns["synthetic_patterns"]]
        assert "combat" in topics

    def test_knowledge_domain_adds_pattern(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Hi",
               "knowledge_domain": "herbalism", "personality": {}}
        patterns = _generate_patterns(bio)
        topics = [p["topic"] for p in patterns["synthetic_patterns"]]
        assert "knowledge" in topics

    def test_has_fallback(self):
        bio = {"name": "Tom", "role": "merchant", "greeting": "Hi", "personality": {}}
        patterns = _generate_patterns(bio)
        assert "fallback" in patterns
        assert len(patterns["fallback"]) > 0


# ══════════════════════════════════════
# Session Lifecycle Tests
# ══════════════════════════════════════

class TestSessionLifecycle:
    def test_create_session(self):
        resp = client.post("/api/session/create", json={
            "name": "Test NPC", "role": "guard"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["bio"]["name"] == "Test NPC"
        assert data["pattern_count"] >= 3

    def test_create_session_with_personality(self):
        resp = client.post("/api/session/create", json={
            "name": "Happy Bot",
            "role": "assistant",
            "personality": {"chattiness": 0.9, "friendliness": 0.9, "humor": 0.8}
        })
        data = resp.json()
        assert data["bio"]["personality"]["chattiness"] == 0.9

    def test_delete_session(self):
        resp = client.post("/api/session/create", json={"name": "Temp"})
        sid = resp.json()["session_id"]
        resp = client.delete(f"/api/session/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_session(self):
        resp = client.delete("/api/session/nonexistent")
        assert resp.status_code == 404


# ══════════════════════════════════════
# Genome Editing Tests
# ══════════════════════════════════════

class TestGenomeEditing:
    def _create_session(self):
        resp = client.post("/api/session/create", json={
            "name": "Editable", "role": "merchant"
        })
        return resp.json()["session_id"]

    def test_update_name(self):
        sid = self._create_session()
        resp = client.put(f"/api/session/{sid}/genome", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["bio"]["name"] == "Renamed"

    def test_update_personality(self):
        sid = self._create_session()
        resp = client.put(f"/api/session/{sid}/genome", json={
            "personality": {"chattiness": 1.0, "friendliness": 0.0}
        })
        data = resp.json()
        assert data["bio"]["personality"]["chattiness"] == 1.0

    def test_update_regenerates_patterns(self):
        sid = self._create_session()
        resp = client.put(f"/api/session/{sid}/genome", json={
            "backstory": "Once upon a time...",
            "knowledge_domain": "magic",
        })
        assert resp.json()["status"] == "regenerated"
        assert resp.json()["pattern_count"] >= 4  # Added backstory + knowledge

    def test_update_nonexistent_session(self):
        resp = client.put("/api/session/fake/genome", json={"name": "X"})
        assert resp.status_code == 404

    def test_add_custom_patterns(self):
        sid = self._create_session()
        resp = client.put(f"/api/session/{sid}/genome", json={
            "custom_patterns": [
                {"id": "custom_001", "triggers": ["secret"], "response_template": "Shh!", "topic": "secret"}
            ]
        })
        # Pattern count should include the custom pattern
        assert resp.json()["pattern_count"] >= 4


# ══════════════════════════════════════
# Chat Tests
# ══════════════════════════════════════

class TestChat:
    def _create_session(self, name="Chatty", greeting="Hello!"):
        resp = client.post("/api/session/create", json={
            "name": name, "role": "merchant", "greeting": greeting
        })
        return resp.json()["session_id"]

    def test_chat_basic(self):
        sid = self._create_session()
        resp = client.post(f"/api/session/{sid}/chat", json={
            "message": "hello"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_returns_debug(self):
        sid = self._create_session()
        resp = client.post(f"/api/session/{sid}/chat", json={
            "message": "hello"
        })
        data = resp.json()
        assert "confidence" in data
        assert "emotion" in data
        assert "source" in data

    def test_chat_with_greeting(self):
        sid = self._create_session(greeting="Ahoy matey!")
        resp = client.post(f"/api/session/{sid}/chat", json={
            "message": "hi there"
        })
        # Should get some response (may be greeting or pattern match)
        assert len(resp.json()["response"]) > 0

    def test_chat_nonexistent_session(self):
        resp = client.post("/api/session/fake/chat", json={"message": "hello"})
        assert resp.status_code == 404

    def test_multi_turn_conversation(self):
        sid = self._create_session()
        for msg in ["hello", "what's your name?", "bye"]:
            resp = client.post(f"/api/session/{sid}/chat", json={"message": msg})
            assert resp.status_code == 200
            assert len(resp.json()["response"]) > 0


# ══════════════════════════════════════
# Export & Save Tests
# ══════════════════════════════════════

class TestExportSave:
    def _create_session(self):
        resp = client.post("/api/session/create", json={
            "name": "Exportable", "role": "merchant"
        })
        return resp.json()["session_id"]

    def test_export(self):
        sid = self._create_session()
        resp = client.get(f"/api/session/{sid}/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "bio" in data
        assert "patterns" in data
        assert data["export_format"] == "synthesus_v2"

    def test_export_nonexistent(self):
        resp = client.get("/api/session/fake/export")
        assert resp.status_code == 404

    def test_save_to_disk(self):
        sid = self._create_session()
        resp = client.post(f"/api/session/{sid}/save")
        assert resp.status_code == 200
        data = resp.json()
        assert "saved_to" in data
        # Verify files exist
        saved_dir = Path(data["saved_to"])
        assert (saved_dir / "bio.json").exists()
        assert (saved_dir / "patterns.json").exists()

    def test_save_nonexistent(self):
        resp = client.post("/api/session/fake/save")
        assert resp.status_code == 404


# ══════════════════════════════════════
# Listing Endpoints
# ══════════════════════════════════════

class TestListings:
    def test_list_characters_empty(self):
        resp = client.get("/api/characters")
        assert resp.status_code == 200
        assert resp.json()["characters"] == []

    def test_list_characters_after_save(self):
        # Create and save
        resp = client.post("/api/session/create", json={
            "name": "Listed NPC", "role": "guard"
        })
        sid = resp.json()["session_id"]
        client.post(f"/api/session/{sid}/save")

        resp = client.get("/api/characters")
        chars = resp.json()["characters"]
        assert len(chars) == 1
        assert chars[0]["name"] == "Listed NPC"

    def test_list_archetypes(self):
        resp = client.get("/api/archetypes")
        assert resp.status_code == 200
        assert "archetypes" in resp.json()

    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_studio_ui_serves(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Character Studio" in resp.text


# ══════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════

class TestEdgeCases:
    def test_empty_name(self):
        resp = client.post("/api/session/create", json={
            "name": "", "role": ""
        })
        assert resp.status_code == 200  # Should still work

    def test_unicode_name(self):
        resp = client.post("/api/session/create", json={
            "name": "日本語NPC", "role": "samurai"
        })
        data = resp.json()
        assert data["bio"]["name"] == "日本語NPC"

    def test_very_long_backstory(self):
        resp = client.post("/api/session/create", json={
            "name": "Wordy",
            "backstory": "x" * 10000,
        })
        assert resp.status_code == 200

    def test_extreme_personality(self):
        resp = client.post("/api/session/create", json={
            "name": "Extreme",
            "personality": {"chattiness": 1.0, "aggression": 1.0, "humor": 1.0,
                           "friendliness": 0.0, "formality": 0.0, "curiosity": 0.0}
        })
        assert resp.status_code == 200
