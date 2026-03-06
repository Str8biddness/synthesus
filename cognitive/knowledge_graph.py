"""
Module 8: Knowledge Graph
"Who is Tomás? What do you know about Blackhollow?"

A structured entity knowledge base for each NPC. Stores what the NPC
KNOWS about people, places, items, factions, and events — with
relationship-aware response selection.

The NPC doesn't look things up in a database — it "remembers" things
it personally knows, colored by its own perspective and emotion.

Cost: ~0.1ms per query, ~20 KB RAM per NPC, zero GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class EntityType(Enum):
    PERSON = "person"
    PLACE = "place"
    ITEM = "item"
    FACTION = "faction"
    EVENT = "event"
    CONCEPT = "concept"


class KnowledgeDepth(Enum):
    """How well the NPC knows this entity."""
    INTIMATE = "intimate"      # Family, close associates
    FAMILIAR = "familiar"      # Regular interactions
    ACQUAINTED = "acquainted"  # Knows of them
    RUMOR = "rumor"            # Heard about, uncertain
    UNKNOWN = "unknown"        # Doesn't know


@dataclass
class EntityKnowledge:
    """What one NPC knows about one entity."""
    entity_id: str
    entity_type: EntityType
    display_name: str
    depth: KnowledgeDepth = KnowledgeDepth.ACQUAINTED

    # Core knowledge (what the NPC says when asked directly)
    description: str = ""           # Main response when asked about this entity
    relationship_to_npc: str = ""   # "my driver", "a regular customer", "the duke"

    # Emotion-variant descriptions
    emotion_variants: Dict[str, str] = field(default_factory=dict)

    # Related entities (for multi-hop queries)
    related_entities: List[str] = field(default_factory=list)

    # Alternate names / synonyms for matching
    aliases: List[str] = field(default_factory=list)

    # Topics this entity is relevant to (for context-aware surfacing)
    topics: List[str] = field(default_factory=list)

    # Trust-gated knowledge (only shared at certain relationship levels)
    secret_description: str = ""    # Shared only with trusted players
    trust_threshold: float = 70.0   # Trust needed for secret info


def build_garen_knowledge() -> Dict[str, EntityKnowledge]:
    """Build Garen Ironfoot's personal knowledge graph."""
    return {
        # ── PEOPLE ──
        "tomas": EntityKnowledge(
            entity_id="tomas",
            entity_type=EntityType.PERSON,
            display_name="Tomás",
            depth=KnowledgeDepth.INTIMATE,
            description="Tomás? He's my best driver — been with me fifteen years. Honest as they come, knows every road between here and Silvermoor. You can trust him because I trust him, and I don't trust easily.",
            relationship_to_npc="my most trusted driver",
            emotion_variants={
                "afraid": "Tomás... he's out there on the Northern Road right now. If anything's happened to him... *voice breaks* He's been with me fifteen years. Like a son to me.",
                "sad": "*quiet voice* Tomás. Fifteen years he's driven for me. Best man I've ever employed. I pray to the gods he's safe.",
                "friendly": "Ah, Tomás! Best driver in the business. Been with me fifteen years — reliable, honest, knows every shortcut between here and Silvermoor. If you meet him on the road, tell him Garen says hello.",
            },
            related_entities=["silvermoor", "caravan", "northern_road"],
            aliases=["tomas", "your driver", "the driver"],
            topics=["quest", "caravan"],
            secret_description="Between you and me... Tomás has a gambling problem. Owes money to some unsavory people in Silvermoor. I've been helping him pay it off quietly. Don't mention it to him.",
            trust_threshold=75.0,
        ),
        "elara": EntityKnowledge(
            entity_id="elara",
            entity_type=EntityType.PERSON,
            display_name="Elara",
            depth=KnowledgeDepth.INTIMATE,
            description="*pauses* Elara was my wife. Finest woman in all of Ironhaven — could spot a forged coin from across the room and keep the books better than any scribe. I lost her to the winter fever five years ago. I keep her ring in the safe. Safest place I know.",
            relationship_to_npc="my late wife",
            emotion_variants={
                "friendly": "*warm but sad smile* Elara... she was the heart of this shop. Everything good about Ironfoot's Emporium started with her. Five years gone, and I still set two cups at tea time. Old habits.",
                "sad": "*long silence* ...I don't talk about Elara much. It still hurts. She was everything.",
            },
            related_entities=["ironhaven"],
            aliases=["elara", "your wife", "wife"],
            topics=["backstory", "personal"],
        ),
        "aldren": EntityKnowledge(
            entity_id="aldren",
            entity_type=EntityType.PERSON,
            display_name="Aldren",
            depth=KnowledgeDepth.FAMILIAR,
            description="Aldren? He's the weaponsmith two streets over. Good work, fair prices — though he charges more than he should for enchantment. We've been friendly rivals for twenty years. He makes the weapons, I sell everything else. Works out well enough.",
            relationship_to_npc="friendly rival, fellow merchant",
            related_entities=["ironhaven"],
            aliases=["aldren", "the weaponsmith", "the blacksmith", "weaponsmith", "blacksmith"],
            topics=["shopping", "world_info"],
        ),
        "brennan": EntityKnowledge(
            entity_id="brennan",
            entity_type=EntityType.PERSON,
            display_name="Brennan",
            depth=KnowledgeDepth.FAMILIAR,
            description="Brennan was my partner on the Frostpeak Run back in '14. Tough as nails, terrible sense of direction. We nearly froze to death on that mountain pass. He retired to a farm outside Redstone. I owe him my life — he pulled me out of an avalanche.",
            relationship_to_npc="old partner, saved my life",
            related_entities=["frostpeak", "redstone"],
            aliases=["brennan"],
            topics=["backstory"],
        ),
        "thessaly": EntityKnowledge(
            entity_id="thessaly",
            entity_type=EntityType.PERSON,
            display_name="Thessaly",
            depth=KnowledgeDepth.ACQUAINTED,
            description="Old Thessaly? She runs the herbalist shop near the market square. Bit eccentric — talks to her plants, claims she can read fortunes in tea leaves. But her healing potions are the real deal. I send customers her way when they need remedies.",
            relationship_to_npc="fellow shopkeeper, acquaintance",
            related_entities=["ironhaven"],
            aliases=["thessaly", "the herbalist", "herb woman"],
            topics=["world_info", "shopping"],
        ),
        "the_duke": EntityKnowledge(
            entity_id="the_duke",
            entity_type=EntityType.PERSON,
            display_name="Duke Aldric",
            depth=KnowledgeDepth.ACQUAINTED,
            description="Duke Aldric rules Ironhaven — or tries to. He's got the Merchant's Alliance in his pocket through tariff agreements. Fair enough ruler, I suppose, though the taxes have been climbing. The Alliance has agreements with him — tariff protections, caravan escorts. Those matter more to me than politics.",
            relationship_to_npc="the local ruler, trade agreements",
            emotion_variants={
                "suspicious": "*lowers voice* The duke's been... different lately. Quiet. His guards have been asking questions around the market. I don't like it.",
            },
            related_entities=["ironhaven", "merchants_alliance"],
            aliases=["duke", "duke aldric", "the duke", "aldric", "ruler"],
            topics=["world_info", "politics"],
            secret_description="*looks around* I'll tell you something — the duke's coffers are thinner than he lets on. I've seen the trade manifests. Tax revenue is down, and he's been taking loans from... foreign interests. That's not public knowledge, and I'd appreciate if it stayed that way.",
            trust_threshold=70.0,
        ),
        "mirella": EntityKnowledge(
            entity_id="mirella",
            entity_type=EntityType.PERSON,
            display_name="Mirella",
            depth=KnowledgeDepth.FAMILIAR,
            description="Mirella handles my books — sharpest mind with numbers I've ever seen. She's been with me since Elara passed. Quiet woman, keeps to herself, but she catches discrepancies in the ledger that I'd miss entirely.",
            relationship_to_npc="my bookkeeper",
            related_entities=["elara"],
            aliases=["mirella", "the bookkeeper"],
            topics=["backstory"],
        ),

        # ── PLACES ──
        "ironhaven": EntityKnowledge(
            entity_id="ironhaven",
            entity_type=EntityType.PLACE,
            display_name="Ironhaven",
            depth=KnowledgeDepth.INTIMATE,
            description="Ironhaven — my home for forty years. It's a trade city, built where the Northern Road meets the river. Not the biggest city, but the busiest market this side of the mountains. The Merchant's Alliance runs the commerce, the duke runs the politics, and somehow we all make it work.",
            relationship_to_npc="my home city",
            related_entities=["the_duke", "merchants_alliance", "northern_road"],
            aliases=["ironhaven"],
            topics=["world_info"],
        ),
        "silvermoor": EntityKnowledge(
            entity_id="silvermoor",
            entity_type=EntityType.PLACE,
            display_name="Silvermoor",
            depth=KnowledgeDepth.FAMILIAR,
            description="Silvermoor is five days' ride south along the trade road. Beautiful city — white stone walls, famous for silk, dyes, and enchanted fabrics. Half my best inventory comes from there. The Silvermoor Weavers' Guild is the finest in the realm.",
            relationship_to_npc="major trade partner city",
            related_entities=["tomas", "caravan"],
            aliases=["silvermoor"],
            topics=["world_info", "quest"],
        ),
        "blackhollow": EntityKnowledge(
            entity_id="blackhollow",
            entity_type=EntityType.PLACE,
            display_name="Blackhollow",
            depth=KnowledgeDepth.ACQUAINTED,
            description="*voice drops* Blackhollow is a stretch of forest on the Northern Road, about two days' ride from here. Dark place — the trees grow thick and the road narrows. Three caravans have gone missing there in the past two months. Merchants talk about shadows in the treeline. I don't know what's out there, but I don't send drivers through after dark anymore.",
            relationship_to_npc="dangerous area on my trade route",
            emotion_variants={
                "afraid": "*grips counter* Blackhollow... that's where my caravans keep disappearing. Something is very wrong in those woods. Don't go there. Please.",
            },
            related_entities=["northern_road", "tomas", "caravan"],
            aliases=["blackhollow", "the hollow", "that forest"],
            topics=["quest", "world_info"],
        ),
        "northern_road": EntityKnowledge(
            entity_id="northern_road",
            entity_type=EntityType.PLACE,
            display_name="The Northern Road",
            depth=KnowledgeDepth.FAMILIAR,
            description="The Northern Road runs from Ironhaven through Blackhollow and up to the mountain passes. Main trade route for half the merchants in the Alliance. Used to be safe — now I'm not so sure. Three caravans missed their schedule this month.",
            relationship_to_npc="my primary trade route",
            related_entities=["blackhollow", "ironhaven"],
            aliases=["northern road", "the road", "trade road", "the road north"],
            topics=["quest", "world_info"],
        ),
        "frostpeak": EntityKnowledge(
            entity_id="frostpeak",
            entity_type=EntityType.PLACE,
            display_name="Frostpeak Pass",
            depth=KnowledgeDepth.FAMILIAR,
            description="Frostpeak Pass — the mountain crossing north of the trade routes. Brutal in winter, barely passable even in summer. I nearly died there in '14 with my partner Brennan. Beautiful though — you can see three kingdoms from the summit on a clear day.",
            relationship_to_npc="nearly died there, memorable trade route",
            related_entities=["brennan"],
            aliases=["frostpeak", "the pass", "mountain pass", "frostpeak pass"],
            topics=["backstory", "world_info"],
        ),
        "redstone": EntityKnowledge(
            entity_id="redstone",
            entity_type=EntityType.PLACE,
            display_name="Redstone",
            depth=KnowledgeDepth.ACQUAINTED,
            description="Redstone is a mining town east of here. Iron, copper, some gemstones. Rough place, rough people, but they pay well for supplies. My old partner Brennan retired to a farm just outside it.",
            relationship_to_npc="trade town, Brennan's retirement home",
            related_entities=["brennan"],
            aliases=["redstone"],
            topics=["world_info"],
        ),

        # ── FACTIONS ──
        "merchants_alliance": EntityKnowledge(
            entity_id="merchants_alliance",
            entity_type=EntityType.FACTION,
            display_name="The Merchant's Alliance",
            depth=KnowledgeDepth.INTIMATE,
            description="The Merchant's Alliance is the trade guild that runs commerce in this region. I'm the Guild Master — have been for twelve years. We negotiate tariffs with the duke, organize caravan escorts, settle trade disputes. It's not glamorous work, but without us, this city's economy would collapse in a month.",
            relationship_to_npc="I'm the Guild Master",
            related_entities=["the_duke", "ironhaven"],
            aliases=["merchant's alliance", "the alliance", "trade guild", "the guild", "merchants alliance"],
            topics=["world_info", "backstory"],
        ),

        # ── EVENTS ──
        "caravan_disappearances": EntityKnowledge(
            entity_id="caravan_disappearances",
            entity_type=EntityType.EVENT,
            display_name="The Missing Caravans",
            depth=KnowledgeDepth.INTIMATE,
            description="Three caravans have vanished on the Northern Road in the past two months. Two came back empty — drivers said they were robbed by something fast, something they couldn't see clearly. The third hasn't come back at all. Now my own caravan with Tomás is overdue. The Alliance is getting nervous. We'll be hiring sellswords by the dozen if this doesn't stop.",
            relationship_to_npc="my biggest concern right now",
            emotion_variants={
                "afraid": "The caravans... four gone now, counting mine. People are dying on that road, and nobody knows why. The duke's guards say they'll investigate, but I haven't seen them lift a finger.",
            },
            related_entities=["northern_road", "blackhollow", "tomas", "merchants_alliance"],
            aliases=["missing caravans", "caravan disappearances", "the disappearances", "vanishing caravans", "caravan go missing", "caravan missing", "missing caravan"],
            topics=["quest"],
        ),

        # ── ITEMS / CONCEPTS ──
        "starfire_essence": EntityKnowledge(
            entity_id="starfire_essence",
            entity_type=EntityType.ITEM,
            display_name="Starfire Essence",
            depth=KnowledgeDepth.FAMILIAR,
            description="Starfire Essence? Rare alchemical compound — glows with an inner light, used in high-end enchantments. I've only had three bottles pass through my shop in forty years. Worth more than its weight in gold. If you find any, bring it to me first — I'll give you a fair price.",
            relationship_to_npc="rare and valuable trade good",
            aliases=["starfire essence", "starfire", "the essence"],
            topics=["shopping", "world_info"],
        ),
    }


class KnowledgeGraph:
    """
    Module 8 of the Cognitive Engine.
    Structured entity knowledge for NPC-aware responses.
    """

    def __init__(self, knowledge: Optional[Dict[str, EntityKnowledge]] = None):
        self.entities: Dict[str, EntityKnowledge] = knowledge or {}
        # Build alias index for fast lookup
        self._alias_index: Dict[str, str] = {}
        for eid, ek in self.entities.items():
            self._alias_index[ek.display_name.lower()] = eid
            for alias in ek.aliases:
                self._alias_index[alias.lower()] = eid

    def lookup(
        self,
        query: str,
        player_trust: float = 50.0,
        emotion: str = "neutral",
    ) -> Optional[Dict[str, Any]]:
        """
        Look up an entity in the knowledge graph.

        Searches for entity names/aliases in the query text.
        Returns the NPC's knowledge about the best-matching entity.

        Args:
            query: Player's message
            player_trust: Current trust score (for gated knowledge)
            emotion: Current NPC emotion (for variant selection)

        Returns:
            {
                "response": str,
                "entity_id": str,
                "entity_name": str,
                "entity_type": str,
                "depth": str,
                "source": "knowledge_graph",
                "confidence": float,
                "has_secret": bool,
                "related": [str],
            }
            or None if no entity found.
        """
        q_lower = query.lower()

        # Find the longest matching alias (prefer "Merchant's Alliance" over "the")
        best_eid = None
        best_len = 0

        for alias, eid in self._alias_index.items():
            if alias in q_lower and len(alias) > best_len:
                # Ensure it's a word boundary match for short aliases
                if len(alias) <= 3:
                    # Skip very short aliases unless exact word
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if not re.search(pattern, q_lower):
                        continue
                best_eid = eid
                best_len = len(alias)

        if not best_eid or best_eid not in self.entities:
            return None

        entity = self.entities[best_eid]

        # Select response text
        response = entity.description
        if emotion in entity.emotion_variants:
            response = entity.emotion_variants[emotion]

        # Check for trust-gated secret knowledge
        has_secret = bool(entity.secret_description)
        if has_secret and player_trust >= entity.trust_threshold:
            response += f" {entity.secret_description}"

        # Confidence based on knowledge depth
        depth_confidence = {
            KnowledgeDepth.INTIMATE: 0.95,
            KnowledgeDepth.FAMILIAR: 0.85,
            KnowledgeDepth.ACQUAINTED: 0.75,
            KnowledgeDepth.RUMOR: 0.60,
            KnowledgeDepth.UNKNOWN: 0.30,
        }

        return {
            "response": response,
            "entity_id": entity.entity_id,
            "entity_name": entity.display_name,
            "entity_type": entity.entity_type.value,
            "depth": entity.depth.value,
            "source": "knowledge_graph",
            "confidence": depth_confidence.get(entity.depth, 0.70),
            "has_secret": has_secret and player_trust >= entity.trust_threshold,
            "related": entity.related_entities,
        }

    def get_entity(self, entity_id: str) -> Optional[EntityKnowledge]:
        """Direct entity access by ID."""
        return self.entities.get(entity_id)

    def get_related(self, entity_id: str) -> List[EntityKnowledge]:
        """Get entities related to a given entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        return [self.entities[rid] for rid in entity.related_entities
                if rid in self.entities]

    def list_entities(self, entity_type: Optional[EntityType] = None) -> List[EntityKnowledge]:
        """List all known entities, optionally filtered by type."""
        entities = list(self.entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities
