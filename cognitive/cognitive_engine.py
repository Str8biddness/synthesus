"""
Cognitive Engine — The NPC Right Hemisphere
Orchestrates all 9 cognitive modules into a single NPC brain.

The CognitiveEngine is the main entry point. It:
1. Receives ML Swarm signals (intent, sentiment, player emotion)
2. Runs the query through all 9 cognitive modules
3. Returns a fully assembled, context-aware response
4. Reports whether it handled locally or needs escalation

Left Hemisphere: Hybrid token + semantic matching.
- Token matcher: fast keyword/substring overlap
- Semantic matcher: SwarmEmbedder (TF-IDF + SVD) + FAISS cosine similarity
- Hybrid score = max(token_score, semantic_score * confidence)

ML Swarm integration: IntentClassifier, SentimentAnalyzer, EmotionDetector,
BehaviorPredictor feed signals into emotion state machine, escalation gate,
and response composition. Total cost: <1ms for ML + ~2-5ms for cognitive.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .conversation_tracker import ConversationTracker, Topic
from .emotion_state_machine import EmotionStateMachine, EmotionState
from .response_compositor import ResponseCompositor
from .relationship_tracker import RelationshipTracker
from .world_state_reactor import WorldStateReactor, WorldReaction
from .escalation_gate import EscalationGate
from .personality_bank import PersonalityBank
from .knowledge_graph import KnowledgeGraph, load_knowledge_from_file, load_knowledge_from_dict
from .context_recall import ContextRecall
from .semantic_matcher import SemanticMatcher
from .goal_stack import GoalStack
from .proactive_engine import ProactiveEngine


# Stop words (duplicated for self-contained pattern matching)
_STOP = {
    "the", "a", "an", "is", "it", "of", "in", "to", "and", "or", "i", "me", "my",
    "you", "your", "do", "does", "did", "can", "could", "would", "should",
    "what", "how", "why", "when", "where", "who", "that", "this", "if",
    "about", "for", "with", "on", "at", "by", "from", "up", "out", "so",
    "be", "been", "am", "are", "was", "were", "have", "has", "had", "not",
    "but", "just", "more", "very", "ever", "one", "thing", "like", "any",
    "some", "no", "all", "them", "they", "we", "he", "she", "its",
    "don't", "there", "i'll", "got", "get", "know", "think", "tell",
    "really", "much", "way", "too", "also", "here", "now", "then",
    "going", "want", "need", "make", "let", "go", "come", "see",
    "take", "give", "say", "said", "look", "well", "back", "even",
    "still", "us", "our", "his", "her", "their", "him", "it's", "i'm",
}


def _tokenize(text: str) -> set:
    return set(re.findall(r'[a-z]+', text.lower())) - _STOP


class CognitiveEngine:
    """
    The NPC Right Hemisphere.
    
    Combines:
    - ConversationTracker (multi-turn context)
    - EmotionStateMachine (emotional reactions)
    - ResponseCompositor (varied response assembly)
    - RelationshipTracker (persistent relationships)
    - WorldStateReactor (world event awareness)
    - EscalationGate (smart routing to Thinking Layer)
    - PersonalityBank (pre-authored creative responses)
    - KnowledgeGraph (entity knowledge base)
    - ContextRecall (NPC references its own prior statements)
    
    Plus the left hemisphere pattern matcher.
    """

    def __init__(
        self,
        character_id: str,
        bio: Dict[str, Any],
        patterns: Dict[str, Any],
        persist_dir: Optional[str] = None,
        char_dir: Optional[str] = None,
    ):
        self.character_id = character_id
        self.bio = bio
        self.patterns = patterns
        self._persist_dir = Path(persist_dir) if persist_dir else None

        # Resolve the character directory for loading config files
        self._char_dir = Path(char_dir) if char_dir else None

        # Module 8: Knowledge Graph — load from knowledge.json if it exists
        knowledge = self._load_knowledge(bio, character_id)
        self.knowledge = KnowledgeGraph(knowledge=knowledge)

        # Extract known entities from bio, patterns, AND the knowledge graph
        known_entities = self._extract_known_entities(bio, patterns, self.knowledge)

        # Initialize all 9 modules
        self.tracker = ConversationTracker(known_entities=known_entities)
        self.emotion = EmotionStateMachine()
        self.compositor = ResponseCompositor()
        self.relationships = RelationshipTracker(
            npc_id=character_id,
            persist_path=str(self._persist_dir / "relationships.json") if self._persist_dir else None,
        )
        self.world = WorldStateReactor(
            reactions=self._build_world_reactions(bio)
        )
        self.gate = EscalationGate()

        # Module 7: Personality Bank — load from personality.json if it exists
        archetype = bio.get("archetype", bio.get("role", "merchant")).lower()
        personality_file = str(self._char_dir / "personality.json") if self._char_dir else None
        self.personality = PersonalityBank(
            archetype=archetype,
            personality_file=personality_file,
        )

        # Module 9: Context Recall — NPC references its own prior statements
        self.recall = ContextRecall()

        # Module 14 & 15: Agentic Behavior
        self.goal_stack = GoalStack()
        self.proactive_engine = ProactiveEngine()

        # Pre-process patterns for fast matching
        self._synthetic = patterns.get("synthetic_patterns", [])
        self._generic = patterns.get("generic_patterns", [])
        self._fallback_text = patterns.get(
            "fallback",
            f"I am {bio.get('name', character_id)}. Could you rephrase?"
        )

        # Module 10: Semantic Matcher — Left Hemisphere v2
        # Pre-embeds all triggers into FAISS for cosine similarity search.
        # Gracefully degrades to token-only if model loading fails.
        self.semantic = SemanticMatcher(similarity_floor=0.35)
        try:
            self.semantic.build_index(self._synthetic, self._generic)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"SemanticMatcher init failed, falling back to token-only: {e}"
            )
            self.semantic._enabled = False

        # Stats
        self._total_queries = 0
        self._local_handled = 0
        self._escalated = 0
        self._knowledge_handled = 0
        self._personality_handled = 0
        self._recall_handled = 0
        self._semantic_wins = 0  # Times semantic beat token matching

    @staticmethod
    def _extract_known_entities(
        bio: Dict, patterns: Dict, knowledge_graph: Optional[KnowledgeGraph] = None,
    ) -> Dict[str, str]:
        """Extract named entities from bio, patterns, and knowledge graph.
        
        Entity sources (in priority order):
        1. Knowledge graph entities (most reliable — typed and aliased)
        2. NPC's own name from bio
        3. Capitalized proper nouns found in pattern response_templates
        """
        entities = {}

        # Source 1: Knowledge graph provides typed, aliased entities
        if knowledge_graph:
            kg_entities = knowledge_graph.get_known_entities()
            entities.update(kg_entities)

        # Source 2: NPC's own name from bio
        name = bio.get("name", bio.get("display_name", ""))
        if name:
            for part in name.split():
                if len(part) > 2:
                    entities[part] = "SELF"

        # Source 3: Scan pattern response_templates for capitalized proper nouns
        for pat_list in [patterns.get("synthetic_patterns", []),
                         patterns.get("generic_patterns", [])]:
            for pat in pat_list:
                text = pat.get("response_template", "")
                words = text.split()
                for j, w in enumerate(words):
                    if j == 0:
                        continue
                    prev = words[j - 1] if j > 0 else ""
                    if prev.endswith(('.', '!', '?', '"')):
                        continue
                    clean = re.sub(r'[^a-zA-Z]', '', w)
                    if clean and clean[0].isupper() and len(clean) > 2:
                        low = clean.lower()
                        if low not in _STOP and clean not in entities:
                            # Default to NPC if not already typed by KG
                            entities[clean] = "NPC"

        return entities

    def _load_knowledge(self, bio: Dict, character_id: str) -> Dict:
        """Load knowledge graph from character's knowledge.json file.
        
        Falls back to empty dict if no file exists.
        """
        if self._char_dir:
            kg_path = self._char_dir / "knowledge.json"
            if kg_path.exists():
                return load_knowledge_from_file(str(kg_path))
        
        # Check if bio has inline knowledge data
        if "knowledge" in bio and isinstance(bio["knowledge"], dict):
            return load_knowledge_from_dict(bio["knowledge"])
        
        return {}

    @staticmethod
    def _build_world_reactions(bio: Dict) -> List[WorldReaction]:
        """Build default world reactions based on character role."""
        reactions = [
            # Town under attack: NPC is afraid, shop patterns disabled
            WorldReaction(
                flag_name="TOWN_UNDER_ATTACK",
                flag_value=True,
                emotion_override=EmotionState.AFRAID,
                disabled_patterns=set(),  # Individual chars override this
                greeting_override="Thank the gods you're here! The town is under attack!",
            ),
            # Night time: shop closed
            WorldReaction(
                flag_name="TIME_OF_DAY",
                flag_value="night",
                greeting_override="Shop's closed for the night. Come back in the morning.",
            ),
            # Player is a known criminal
            WorldReaction(
                flag_name="PLAYER_REPUTATION",
                flag_value="criminal",
                emotion_override=EmotionState.SUSPICIOUS,
            ),
        ]
        return reactions

    def _match_pattern_token(self, query: str) -> Tuple[Optional[Dict], float]:
        """
        Token-based pattern matching (Left Hemisphere v1).
        Fast keyword/substring overlap with geometric mean scoring.
        Returns (pattern_dict, score).
        """
        q = query.strip().lower()
        q_tokens = _tokenize(q)

        best_match = None
        best_score = 0.0

        for pat_list, is_generic in [(self._synthetic, False), (self._generic, True)]:
            for pat in pat_list:
                triggers = pat.get("trigger", [])
                if isinstance(triggers, str):
                    triggers = [triggers]
                conf = pat.get("confidence", 0.5)

                for t in triggers:
                    t_lower = t.lower().strip()
                    t_tokens = _tokenize(t_lower)

                    # Exact match
                    if t_lower == q:
                        return pat, 1.0

                    score = 0.0
                    overlap = q_tokens & t_tokens
                    n_overlap = len(overlap)

                    # Full-trigger substring
                    if t_lower in q and len(t_lower) >= 4:
                        specificity = len(t_lower) / max(len(q), 1)
                        score = conf * (0.7 + 0.3 * specificity)

                    # Token overlap
                    elif t_tokens and q_tokens:
                        min_required = 1 if (len(t_tokens) <= 2 or len(q_tokens) <= 2) else 2
                        if n_overlap >= min_required:
                            trigger_cov = n_overlap / len(t_tokens)
                            query_cov = n_overlap / len(q_tokens)
                            geo_mean = (trigger_cov * query_cov) ** 0.5
                            score = geo_mean * conf

                    if is_generic:
                        score *= 0.7

                    if score > best_score:
                        best_score = score
                        best_match = pat

        return best_match, best_score

    def _match_pattern(self, query: str) -> Tuple[Optional[Dict], float]:
        """
        Hybrid pattern matching (Left Hemisphere v2).
        
        Runs BOTH token matching and semantic matching in parallel,
        then takes the better result. This ensures:
        - Exact/substring matches still get 1.0 scores (token wins)
        - Paraphrases, slang, indirect refs get caught (semantic wins)
        - No regression on existing behavior
        
        Semantic score is scaled by the pattern's confidence value
        to maintain consistency with the token scorer.
        
        Returns (pattern_dict, hybrid_score).
        """
        # Run token matcher (always available, ~0.1ms)
        token_match, token_score = self._match_pattern_token(query)

        # Short-circuit: perfect token match needs no semantic check
        if token_score >= 1.0:
            return token_match, token_score

        # Run semantic matcher (if available, ~12ms)
        if not self.semantic._enabled:
            return token_match, token_score

        sem_pat, sem_trigger, sem_cosine, sem_generic = self.semantic.get_best_match(query)

        if sem_pat is None:
            return token_match, token_score

        # Scale semantic cosine score by pattern confidence
        # (mirrors how token scorer uses conf multiplier)
        sem_conf = sem_pat.get("confidence", 0.5)
        sem_score = sem_cosine * sem_conf
        if sem_generic:
            sem_score *= 0.7

        # Take the better result
        if sem_score > token_score:
            self._semantic_wins += 1
            return sem_pat, sem_score

        return token_match, token_score

    def process_query(
        self,
        player_id: str,
        query: str,
        thinking_layer_available: bool = False,
        ml_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The main entry point. Process a player query through the full
        cognitive engine pipeline.

        Args:
            player_id: Unique player identifier
            query: The player's query text
            thinking_layer_available: Whether escalation to thinking layer is possible
            ml_context: Pre-computed ML Swarm signals from the production server:
                - intent: classified player intent (e.g. "greeting", "shop_buy", "lore")
                - sentiment: emotional valence ("positive", "negative", "neutral", etc.)
                - player_emotion: detected emotion ("joy", "anger", "fear", etc.)
                - emotion_intensity: 0-1 intensity of detected emotion
                - predicted_action: next likely player action
                - engagement_score: 0-1 engagement probability
                - escalation_risk: 0-1 escalation risk score

        Returns:
            Dict with response, confidence, emotion, relationship, debug info.
        """
        start_time = time.time()
        self._total_queries += 1
        ml_context = ml_context or {}

        # ── Step 1: Conversation Tracker ──
        conv_context = self.tracker.process(player_id, query)
        keywords = conv_context["keywords"]

        # ── Step 2: Emotion State Machine ──
        # Feed ML-detected player emotion into the NPC's emotional response
        emotion_result = self.emotion.process(player_id, keywords)
        current_emotion = emotion_result["emotion"]

        # If ML Swarm detected strong player emotion, influence NPC reaction
        if ml_context.get("emotion_intensity", 0) > 0.5:
            player_emo = ml_context.get("player_emotion", "neutral")
            # Threatening players make NPC suspicious/afraid
            if player_emo in ("anger", "disgust"):
                from .emotion_state_machine import EmotionState
                if hasattr(EmotionState, 'SUSPICIOUS'):
                    self.emotion.force_state(player_id, EmotionState.SUSPICIOUS)
                    current_emotion = EmotionState.SUSPICIOUS
            elif player_emo == "fear":
                from .emotion_state_machine import EmotionState
                if hasattr(EmotionState, 'CONCERNED'):
                    self.emotion.force_state(player_id, EmotionState.CONCERNED)
                    current_emotion = EmotionState.CONCERNED

        # ── Step 3: Relationship Tracker ──
        rel_result = self.relationships.process(player_id, keywords)

        # ── Step 4: World State Reactor & Proactive Engine ──
        world_result = self.world.process()

        # Build global context for proactive/goal evaluation
        context = {
            "player_id": player_id,
            "query": query,
            "topic": conv_context["active_topic"],
            "world_flags": world_result.get("active_flags", {}),
            "relationship": {
                "trust": rel_result["trust"],
                "fondness": rel_result["fondness"],
                "respect": rel_result["respect"],
                "debt": rel_result["debt"],
                "tier": next((k for k, v in rel_result["tier"].items() if v), "stranger"),
                "interactions": rel_result.get("interactions", 0),
            },
            "last_interaction_time": self.tracker.get_state(player_id).last_interaction,
        }

        # Check for proactive greeting override (goals/events/time)
        prefix_greeting = None
        if not self.goal_stack.get_active_goals(): # Only if no urgent goals blocking
            prefix_greeting = self.proactive_engine.check(player_id, context)

        # Apply world emotion override if set
        if world_result["emotion_override"] is not None:
            current_emotion = world_result["emotion_override"]
            self.emotion.force_state(player_id, current_emotion)

        # Check for greeting override (world events / proactive)
        greeting = prefix_greeting or world_result.get("greeting_override")
        if conv_context["turn_count"] == 1 and greeting:
            response = greeting
            self.tracker.record_npc_response(player_id, response)
            self.recall.record_response(player_id, response)
            self._local_handled += 1
            return self._build_result(
                response=response,
                source="cognitive_engine",
                confidence=0.95,
                emotion=current_emotion,
                rel_result=rel_result,
                conv_context=conv_context,
                world_result=world_result,
                match_score=0.95,
                pattern_id="world_greeting_override",
                start_time=start_time,
            )

        # ── Step 4b: Context Recall Priority Check ──
        # If the player explicitly references something the NPC said earlier
        # ("you mentioned...", "you said...", "remember when..."),
        # context recall gets FIRST shot before pattern matching.
        # This prevents pattern triggers (e.g., "tomás") from hijacking
        # legitimate recall queries like "you mentioned Tomás earlier".
        emotion_str_pre = current_emotion.value if hasattr(current_emotion, 'value') else str(current_emotion)
        if self.recall._is_recall_query(query):
            cr_early = self.recall.process(
                player_id=player_id,
                query=query,
                emotion=emotion_str_pre,
            )
            if cr_early and cr_early.get("recall_type") != "not_found":
                response = cr_early["response"]
                self.tracker.record_npc_response(player_id, response)
                self._local_handled += 1
                self._recall_handled += 1
                return self._build_result(
                    response=response,
                    source="context_recall",
                    confidence=cr_early["confidence"],
                    emotion=current_emotion,
                    rel_result=rel_result,
                    conv_context=conv_context,
                    world_result=world_result,
                    match_score=cr_early["confidence"],
                    pattern_id=f"cr_{cr_early['recall_type']}",
                    start_time=start_time,
                )

        # ── Step 5: Pattern Matching (Left Hemisphere) ──
        matched_pattern, match_score = self._match_pattern(query)

        # Check if pattern is disabled by world state
        if matched_pattern:
            pat_id = matched_pattern.get("id", "")
            if pat_id in world_result.get("disabled_patterns", set()):
                matched_pattern = None
                match_score = 0.0

            # Check for world state response override
            if pat_id in world_result.get("pattern_overrides", {}):
                matched_pattern = dict(matched_pattern)  # Copy
                matched_pattern["response_template"] = world_result["pattern_overrides"][pat_id]

        # ── Step 6: Escalation Gate ──
        escalation = self.gate.evaluate(
            match_confidence=match_score,
            keywords=keywords,
            conversation_depth=conv_context.get("conversation_depth", 0),
            emotion_intensity=emotion_result["intensity"],
            query_text=query,
        )

        # Boost escalation if ML Swarm detected high escalation risk
        ml_escalation_risk = ml_context.get("escalation_risk", 0)
        if ml_escalation_risk > 0.5:
            escalation.total_score = min(escalation.total_score + ml_escalation_risk * 0.3, 1.0)

        # ── Decision: Local or Escalate? ──
        if matched_pattern and match_score >= 0.55:
            # Local handling via cognitive engine
            # Merge all context for the compositor
            full_context = {
                **conv_context,
                "emotion": current_emotion,
                **rel_result,
                "world_state": world_result.get("world_state", {}),
            }

            response = self.compositor.compose(
                pattern=matched_pattern,
                context=full_context,
                emotion=current_emotion,
                player_id=player_id,
            )

            self.tracker.record_npc_response(player_id, response)
            self.recall.record_response(player_id, response)  # Feed Module 9
            self._local_handled += 1

            return self._build_result(
                response=response,
                source="cognitive_engine",
                confidence=match_score,
                emotion=current_emotion,
                rel_result=rel_result,
                escalation=None,
                conv_context=conv_context,
                world_result=world_result,
                match_score=match_score,
                pattern_id=matched_pattern.get("id", "unknown"),
                start_time=start_time,
                ml_context=ml_context,
            )

        elif escalation.should_escalate and thinking_layer_available:
            # Escalate to thinking layer
            self._escalated += 1
            return self._build_result(
                response=None,  # Caller must fill via deeper processing
                source="escalated",
                confidence=match_score,
                emotion=current_emotion,
                rel_result=rel_result,
                escalation={
                    "should_escalate": True,
                    "score": escalation.total_score,
                    "signals": [
                        {"name": s.name, "weight": s.weight, "score": s.score, "reason": s.reason}
                        for s in escalation.signals
                    ],
                },
                conv_context=conv_context,
                world_result=world_result,
                match_score=match_score,
                pattern_id=None,
                start_time=start_time,
            )

        else:
            # ── NEW: 3-Module Fallback Cascade ──
            # Before stalling, try the 3 new brain modules in order:
            #   1. Knowledge Graph  → entity-specific knowledge
            #   2. Personality Bank → creative/personal responses
            #   3. Context Recall   → reference to own prior statements
            # Only if ALL THREE miss → use the old fallback/stall.

            emotion_str = current_emotion.value if hasattr(current_emotion, 'value') else str(current_emotion)
            player_trust = rel_result.get("trust", 50.0)

            # ── Module 8: Knowledge Graph ──
            kg_result = self.knowledge.lookup(
                query=query,
                player_trust=player_trust,
                emotion=emotion_str,
            )
            if kg_result:
                response = kg_result["response"]
                self.tracker.record_npc_response(player_id, response)
                self.recall.record_response(player_id, response)
                self._local_handled += 1
                self._knowledge_handled += 1
                return self._build_result(
                    response=response,
                    source="knowledge_graph",
                    confidence=kg_result["confidence"],
                    emotion=current_emotion,
                    rel_result=rel_result,
                    conv_context=conv_context,
                    world_result=world_result,
                    match_score=kg_result["confidence"],
                    pattern_id=f"kg_{kg_result['entity_id']}",
                    start_time=start_time,
                )

            # ── Module 7: Personality Bank ──
            pb_result = self.personality.get_response(
                query=query,
                keywords=set(keywords) if isinstance(keywords, list) else keywords,
                emotion=emotion_str,
            )
            if pb_result:
                response = pb_result["response"]
                self.tracker.record_npc_response(player_id, response)
                self.recall.record_response(player_id, response)
                self._local_handled += 1
                self._personality_handled += 1
                return self._build_result(
                    response=response,
                    source="personality_bank",
                    confidence=pb_result["confidence"],
                    emotion=current_emotion,
                    rel_result=rel_result,
                    conv_context=conv_context,
                    world_result=world_result,
                    match_score=pb_result["confidence"],
                    pattern_id=f"pb_{pb_result['intent']}",
                    start_time=start_time,
                )

            # ── Module 9: Context Recall ──
            cr_result = self.recall.process(
                player_id=player_id,
                query=query,
                emotion=emotion_str,
            )
            if cr_result:
                response = cr_result["response"]
                self.tracker.record_npc_response(player_id, response)
                # Don't record recall responses back into recall (avoid loops)
                self._local_handled += 1
                self._recall_handled += 1
                return self._build_result(
                    response=response,
                    source="context_recall",
                    confidence=cr_result["confidence"],
                    emotion=current_emotion,
                    rel_result=rel_result,
                    conv_context=conv_context,
                    world_result=world_result,
                    match_score=cr_result["confidence"],
                    pattern_id=f"cr_{cr_result['recall_type']}",
                    start_time=start_time,
                )

            # ── All 3 modules missed → Original fallback ──
            if escalation.should_escalate:
                # Use stall response
                response = escalation.fallback_response or self._fallback_text
            else:
                response = self._fallback_text

            self.tracker.record_npc_response(player_id, response)
            self.recall.record_response(player_id, response)
            self._local_handled += 1

            return self._build_result(
                response=response,
                source="fallback",
                confidence=0.3,
                emotion=current_emotion,
                rel_result=rel_result,
                escalation={
                    "should_escalate": escalation.should_escalate,
                    "score": escalation.total_score,
                    "thinking_layer_available": thinking_layer_available,
                },
                conv_context=conv_context,
                world_result=world_result,
                match_score=match_score,
                pattern_id=None,
                start_time=start_time,
                context=context,  # Pass context for goal evaluation
                ml_context=ml_context,
            )

    def _build_result(
        self,
        response, source, confidence, emotion, rel_result,
        conv_context, world_result, match_score, pattern_id,
        start_time, escalation=None, context=None, ml_context=None,
    ) -> Dict[str, Any]:
        
        # ── Autonomous Goal Injection (Agentic Layer) ──
        # If we have a response and a context, check goals
        if response and context:
            goal = self.goal_stack.evaluate(
                current_turn=conv_context["turn_count"], 
                world_flags=context.get("world_flags", {})
            )
            if goal:
                if goal.goal_type.value == "warn":
                    response = goal.response_injection
                else:
                    # Mention or steer — prepend to response
                    response = f"{goal.response_injection} {response}"
                self.goal_stack.mark_mentioned(goal.goal_id, conv_context["turn_count"])

        latency = (time.time() - start_time) * 1000
        return {
            "response": response,
            "source": source,
            "confidence": round(confidence, 4),
            "emotion": emotion.value if hasattr(emotion, 'value') else str(emotion),
            "character": self.character_id,
            "relationship": {
                "trust": rel_result["trust"],
                "fondness": rel_result["fondness"],
                "respect": rel_result["respect"],
                "debt": rel_result["debt"],
                "interactions": rel_result.get("interactions", 0),
                "is_first_meeting": rel_result.get("is_first_meeting", True),
                "tier": {k: v for k, v in rel_result["tier"].items() if v},  # Only true flags
            },
            "escalation": escalation,
            "debug": {
                "modules_active": [
                    "conversation_tracker",
                    "emotion_state_machine",
                    "response_compositor",
                    "relationship_tracker",
                    "world_state_reactor",
                    "escalation_gate",
                    "personality_bank",
                    "knowledge_graph",
                    "context_recall",
                    "semantic_matcher",
                    "goal_stack",
                    "proactive_engine",
                ],
                "pattern_matched": pattern_id,
                "match_score": round(match_score, 4),
                "topic": (conv_context["active_topic"].value
                          if hasattr(conv_context["active_topic"], 'value')
                          else str(conv_context["active_topic"])),
                "turn_count": conv_context["turn_count"],
                "entities_mentioned": [
                    e.name for e in conv_context.get("entities_mentioned", [])
                ],
                "pronoun_resolution": conv_context.get("pronoun_resolution"),
                "world_flags": world_result.get("active_flags", {}),
                "latency_ms": round(latency, 2),
                "ml_context": ml_context if ml_context else None,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        stats = {
            "character_id": self.character_id,
            "total_queries": self._total_queries,
            "local_handled": self._local_handled,
            "escalated": self._escalated,
            "knowledge_handled": self._knowledge_handled,
            "personality_handled": self._personality_handled,
            "recall_handled": self._recall_handled,
            "semantic_wins": self._semantic_wins,
            "local_pct": (
                round(self._local_handled / self._total_queries * 100, 1)
                if self._total_queries > 0 else 0
            ),
        }
        # Include semantic matcher stats
        if hasattr(self, 'semantic'):
            stats["semantic_matcher"] = self.semantic.get_stats()
            
        stats["goal_stack"] = self.goal_stack.get_stats()
        stats["proactive_engine"] = self.proactive_engine.get_stats()
        
        return stats

    @classmethod
    def from_character_dir(cls, char_dir: str, persist_dir: Optional[str] = None) -> "CognitiveEngine":
        """Load a CognitiveEngine from a character directory.
        
        Looks for:
          - bio.json (required)
          - patterns.json (optional)
          - knowledge.json (optional, loaded by _load_knowledge)
          - personality.json (optional, loaded by PersonalityBank)
        """
        char_path = Path(char_dir)
        bio_path = char_path / "bio.json"
        pat_path = char_path / "patterns.json"

        with open(bio_path) as f:
            bio = json.load(f)
        patterns = {}
        if pat_path.exists():
            with open(pat_path) as f:
                patterns = json.load(f)

        char_id = bio.get("id", bio.get("character_id", char_path.name))
        
        engine = cls(
            character_id=char_id,
            bio=bio,
            patterns=patterns,
            persist_dir=persist_dir,
            char_dir=str(char_path),
        )
        
        # Load agentic profiles
        if "goals" in bio:
            engine.goal_stack.load_from_config(bio["goals"])
        if "proactive_triggers" in bio:
            engine.proactive_engine.load_from_config(bio["proactive_triggers"])
        else:
            engine.proactive_engine.add_default_triggers()
            
        return engine
