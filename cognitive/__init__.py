# cognitive/ — The NPC Right Hemisphere
# Nine lightweight modules that give NPCs behavioral intelligence
# without any SLM inference. <1ms each, zero GPU.

from .conversation_tracker import ConversationTracker
from .emotion_state_machine import EmotionStateMachine
from .response_compositor import ResponseCompositor
from .relationship_tracker import RelationshipTracker
from .world_state_reactor import WorldStateReactor
from .escalation_gate import EscalationGate
from .personality_bank import PersonalityBank
from .knowledge_graph import KnowledgeGraph
from .context_recall import ContextRecall
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
    "ContextRecall",
    "CognitiveEngine",
]
