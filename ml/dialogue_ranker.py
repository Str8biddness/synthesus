#!/usr/bin/env python3
"""
DialogueRanker — ML Swarm Micro-Model #6
AIVM Synthesus 2.0

Ranks candidate NPC responses by contextual relevance, personality fit,
and conversation flow quality. Used by the ResponseCompositor to select
the best response when multiple candidates are available.

Scoring dimensions:
- Relevance: how well does the response match the query?
- Personality fit: does the tone match the NPC's character?
- Flow: does it follow naturally from the conversation?
- Variety: penalize repetition of recent responses

Footprint: ~12 KB, <1ms inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DialogueRanker:
    """
    Ranks candidate responses for NPC dialogue selection.

    Usage:
        ranker = DialogueRanker()
        ranked = ranker.rank(
            candidates=["Welcome!", "What do you want?", "Hello there."],
            query="hi",
            personality={"friendliness": 0.8},
            recent_responses=["Hello there."],
        )
        # → [("Welcome!", 0.82), ("Hello there.", 0.45), ("What do you want?", 0.31)]
    """

    # Personality-tone mapping for scoring
    TONE_KEYWORDS = {
        "friendly": ["welcome", "friend", "glad", "happy", "please", "love", "great", "wonderful"],
        "hostile": ["leave", "get out", "fool", "dare", "challenge", "fight", "enemy"],
        "formal": ["greetings", "indeed", "certainly", "shall", "permit", "acknowledge"],
        "casual": ["hey", "yo", "sup", "cool", "yeah", "nah", "gonna", "wanna"],
        "mysterious": ["perhaps", "secrets", "whisper", "shadow", "hidden", "ancient", "unknown"],
        "humorous": ["ha", "joke", "funny", "laugh", "ridiculous", "hilarious"],
    }

    def __init__(
        self,
        relevance_weight: float = 0.35,
        personality_weight: float = 0.25,
        flow_weight: float = 0.20,
        variety_weight: float = 0.20,
    ):
        self.relevance_weight = relevance_weight
        self.personality_weight = personality_weight
        self.flow_weight = flow_weight
        self.variety_weight = variety_weight

    def rank(
        self,
        candidates: List[str],
        query: str = "",
        personality: Optional[Dict[str, float]] = None,
        recent_responses: Optional[List[str]] = None,
        context_keywords: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate responses by composite score.

        Returns:
            List of (response, score) tuples sorted by score descending.
        """
        if not candidates:
            return []

        personality = personality or {}
        recent_responses = recent_responses or []
        context_keywords = context_keywords or []

        scored = []
        for resp in candidates:
            relevance = self._score_relevance(resp, query, context_keywords)
            personality_fit = self._score_personality(resp, personality)
            flow = self._score_flow(resp, query)
            variety = self._score_variety(resp, recent_responses)

            composite = (
                relevance * self.relevance_weight +
                personality_fit * self.personality_weight +
                flow * self.flow_weight +
                variety * self.variety_weight
            )
            scored.append((resp, round(composite, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_relevance(self, response: str, query: str, context_keywords: List[str]) -> float:
        """Score based on keyword overlap with query and context."""
        if not query:
            return 0.5

        resp_words = set(response.lower().split())
        query_words = set(query.lower().split())

        # Direct overlap
        overlap = len(resp_words & query_words)
        base_score = min(overlap / max(len(query_words), 1), 1.0)

        # Context keyword bonus
        if context_keywords:
            ctx_set = set(w.lower() for w in context_keywords)
            ctx_overlap = len(resp_words & ctx_set)
            base_score += min(ctx_overlap / max(len(ctx_set), 1), 0.5) * 0.3

        return min(base_score, 1.0)

    def _score_personality(self, response: str, personality: Dict[str, float]) -> float:
        """Score personality/tone match."""
        if not personality:
            return 0.5

        resp_lower = response.lower()
        score = 0.5  # neutral baseline

        # Check tone alignment
        friendliness = personality.get("friendliness", 0.5)
        formality = personality.get("formality", 0.5)
        humor = personality.get("humor", 0.3)
        aggression = personality.get("aggression", 0.1)

        # Friendly keywords
        friendly_hits = sum(1 for kw in self.TONE_KEYWORDS["friendly"] if kw in resp_lower)
        hostile_hits = sum(1 for kw in self.TONE_KEYWORDS["hostile"] if kw in resp_lower)

        if friendliness > 0.6:
            score += friendly_hits * 0.1
            score -= hostile_hits * 0.15
        elif aggression > 0.5:
            score += hostile_hits * 0.1
            score -= friendly_hits * 0.05

        # Formality
        formal_hits = sum(1 for kw in self.TONE_KEYWORDS["formal"] if kw in resp_lower)
        casual_hits = sum(1 for kw in self.TONE_KEYWORDS["casual"] if kw in resp_lower)

        if formality > 0.6:
            score += formal_hits * 0.08
            score -= casual_hits * 0.1
        elif formality < 0.4:
            score += casual_hits * 0.08
            score -= formal_hits * 0.1

        return max(0.0, min(score, 1.0))

    def _score_flow(self, response: str, query: str) -> float:
        """Score conversational flow (length appropriateness, question matching)."""
        if not query:
            return 0.5

        # Questions should get substantive answers
        is_question = query.strip().endswith("?")
        resp_len = len(response.split())

        if is_question:
            # Penalize very short answers to questions
            if resp_len < 3:
                return 0.2
            elif resp_len < 8:
                return 0.6
            else:
                return 0.8
        else:
            # Statements / greetings — shorter responses are fine
            if resp_len < 15:
                return 0.7
            else:
                return 0.5

    def _score_variety(self, response: str, recent_responses: List[str]) -> float:
        """Penalize responses that duplicate recent ones."""
        if not recent_responses:
            return 1.0

        resp_lower = response.lower().strip()
        for recent in recent_responses:
            if resp_lower == recent.lower().strip():
                return 0.0  # exact duplicate
            # Partial overlap penalty
            resp_words = set(resp_lower.split())
            recent_words = set(recent.lower().split())
            if resp_words and recent_words:
                overlap = len(resp_words & recent_words) / len(resp_words)
                if overlap > 0.8:
                    return 0.2

        return 1.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "DialogueRanker",
            "scoring_dimensions": ["relevance", "personality", "flow", "variety"],
            "tone_categories": list(self.TONE_KEYWORDS.keys()),
            "footprint_kb": 12,
        }
