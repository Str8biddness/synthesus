# cognitive/ — The NPC Right Hemisphere
# Ten lightweight modules that give NPCs behavioral intelligence
# without any SLM inference. Module 10 (SemanticMatcher) adds ~80MB
# shared model for semantic understanding. Zero GPU.

from .conversation_tracker import ConversationTracker
from .emotion_state_machine import EmotionStateMachine
from .response_compositor import ResponseCompositor
from .relationship_tracker import RelationshipTracker
from .world_state_reactor import WorldStateReactor
from .escalation_gate import EscalationGate
from .personality_bank import PersonalityBank, load_personality_from_file
from .knowledge_graph import KnowledgeGraph, load_knowledge_from_file, load_knowledge_from_dict
from .context_recall import ContextRecall
from .semantic_matcher import SemanticMatcher
from .cognitive_engine import CognitiveEngine

__all__ = [
    "ConversationTracker",
    "EmotionStateMachine",
    "ResponseCompositor",
    "RelationshipTracker",
    "WorldStateReactor",
    "EscalationGate",
    "PersonalityBank",
    "KnowledgeGraph",
    "load_knowledge_from_file",
    "load_knowledge_from_dict",
    "load_personality_from_file",
    "ContextRecall",
    "SemanticMatcher",
    "CognitiveEngine",
]
