"""
Module 7: Personality Bank
"When the player asks something creative, off-script, or personal"

Pre-authored creative responses organized by:
  - Archetype (merchant, guard, innkeeper, scholar, etc.)
  - Intent category (song, joke, favorite, opinion, personal, philosophical)
  - Emotion variant (neutral, friendly, suspicious, etc.)

This module replaces SLM generation for creative/personal questions.
Instead of generating text, it SELECTS from pre-written, QA-approved responses.

Cost: ~0.2ms per query, ~10 KB RAM per archetype, zero GPU.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PersonalityIntent(Enum):
    """Categories of off-script/creative player requests."""
    SONG = "song"
    JOKE = "joke"
    FAVORITE = "favorite"          # "what's your favorite X?"
    OPINION = "opinion"            # "what do you think about X?"
    PERSONAL = "personal"          # "do you get lonely?", "are you married?"
    PHILOSOPHICAL = "philosophical"  # "what happens after we die?"
    COMPLIMENT_RESPONSE = "compliment_response"  # responding to flattery
    INSULT_RESPONSE = "insult_response"          # responding to insults
    CREATIVE_REQUEST = "creative_request"  # "tell me a story", "describe..."
    RUMOR = "rumor"                # "heard any gossip?"
    ADVICE = "advice"              # "any tips?", "what should I do?"
    NONE = "none"


# ── Intent Detection Rules ──
# Each rule: (keyword_set, intent, priority)
# Higher priority wins on ties
_INTENT_RULES: List[Tuple[Set[str], PersonalityIntent, int]] = [
    ({"sing", "song", "melody", "tune", "hum", "music", "hymn"}, PersonalityIntent.SONG, 10),
    ({"joke", "funny", "laugh", "humor", "amuse", "hilarious"}, PersonalityIntent.JOKE, 10),
    ({"favorite", "favourite", "prefer"}, PersonalityIntent.FAVORITE, 8),
    ({"opinion", "believe", "reckon"}, PersonalityIntent.OPINION, 5),
    ({"lonely", "alone", "married", "wife", "husband", "family", "children",
      "happy", "sad", "afraid", "dream", "hope", "regret", "miss",
      "personal", "yourself"}, PersonalityIntent.PERSONAL, 7),
    ({"meaning", "purpose", "death", "die", "afterlife", "soul", "gods",
      "fate", "destiny", "exist", "philosophy", "morality",
      "evil", "justice"}, PersonalityIntent.PHILOSOPHICAL, 9),
    ({"story", "tale", "describe", "imagine", "pretend", "poem",
      "rhyme", "riddle"}, PersonalityIntent.CREATIVE_REQUEST, 8),
    ({"rumor", "gossip", "heard", "whisper", "secret", "news"}, PersonalityIntent.RUMOR, 7),
    ({"advice", "tip", "suggest", "recommend", "wise",
      "wisdom", "counsel"}, PersonalityIntent.ADVICE, 6),
]

# Compliment/insult detection (checked separately against full text)
_COMPLIMENT_WORDS = {"great", "amazing", "wonderful", "best", "awesome",
                     "fantastic", "brilliant", "incredible", "love",
                     "appreciate", "thank", "kind", "generous", "handsome",
                     "honest", "respect", "nice", "impressive", "admire",
                     "compliment", "excellent", "remarkable", "trustworthy"}
_INSULT_WORDS = {"ugly", "stupid", "idiot", "fool", "cheat", "liar",
                 "thief", "ugly", "terrible", "worst", "hate", "scam",
                 "fraud", "pathetic", "useless", "worthless"}


@dataclass
class PersonalityResponse:
    """A single pre-authored response in the bank."""
    text: str
    intent: PersonalityIntent
    emotion_variants: Dict[str, str] = field(default_factory=dict)
    # emotion_variants maps emotion_name → alternate text
    # If current emotion matches a variant, use that instead


@dataclass
class ArchetypeBank:
    """Collection of personality responses for one NPC archetype."""
    archetype: str
    responses: Dict[PersonalityIntent, List[PersonalityResponse]] = field(default_factory=dict)

    def get_response(self, intent: PersonalityIntent, emotion: str = "neutral") -> Optional[str]:
        """Get a random response for an intent, with emotion variant if available."""
        if intent not in self.responses or not self.responses[intent]:
            return None
        candidates = self.responses[intent]
        chosen = random.choice(candidates)

        # Check for emotion-specific variant
        if emotion in chosen.emotion_variants:
            return chosen.emotion_variants[emotion]
        return chosen.text


# ═══════════════════════════════════════════════════════════════
# DEFAULT ARCHETYPE BANKS
# These are the pre-written, QA-approved creative responses
# ═══════════════════════════════════════════════════════════════

def _build_merchant_bank() -> ArchetypeBank:
    """Merchant archetype: Garen Ironfoot and similar NPCs."""
    bank = ArchetypeBank(archetype="merchant")

    bank.responses[PersonalityIntent.SONG] = [
        PersonalityResponse(
            text="*gruff laugh* I'm a merchant, not a bard! But there's an old traders' rhyme... 'Buy it low in Silvermoor, sell it high on Haven's shore, count your coin and ask for more!' ...don't ask me to sing it again.",
            intent=PersonalityIntent.SONG,
            emotion_variants={
                "friendly": "*chuckles warmly* Alright, alright — just for you. 'Buy it low in Silvermoor, sell it high on Haven's shore, count your coin and ask for more!' My father taught me that one. Terrible voice, great merchant.",
                "afraid": "S-singing? Now? There are more pressing matters, friend!",
                "suspicious": "*narrows eyes* You want me to sing? What's your game?",
            }
        ),
        PersonalityResponse(
            text="*clears throat* 'Coin in the morning, coin at night, trade your wares by candlelight...' That's all I remember. My wife Elara always said I sing like a cart with a broken wheel.",
            intent=PersonalityIntent.SONG,
        ),
        PersonalityResponse(
            text="Songs are for taverns and campfires. In this shop, the only music I know is the sound of gold hitting the counter. *taps counter* Hear that? Beautiful.",
            intent=PersonalityIntent.SONG,
        ),
    ]

    bank.responses[PersonalityIntent.JOKE] = [
        PersonalityResponse(
            text="*leans in* A man walks into my shop and says, 'I need something to fight a dragon.' I say, 'A sword? A shield? Enchanted armor?' He says, 'No — a refund on that fireproof cloak you sold me.' *slaps counter laughing*",
            intent=PersonalityIntent.JOKE,
            emotion_variants={
                "suspicious": "*flat stare* You want jokes? I've got one — a stranger walks into my shop and tries to distract me while their friend robs the back room. Hilarious, right?",
            }
        ),
        PersonalityResponse(
            text="Here's a merchant's joke for you: What's the difference between a bandit and a tax collector? The bandit has the decency to wear a mask. *winks*",
            intent=PersonalityIntent.JOKE,
        ),
        PersonalityResponse(
            text="*chuckles* Best joke I know: A nobleman asks me, 'What's the price of honesty?' I tell him, '50 gold — but for you, I'll round up.' He didn't laugh either.",
            intent=PersonalityIntent.JOKE,
        ),
    ]

    bank.responses[PersonalityIntent.FAVORITE] = [
        PersonalityResponse(
            text="My favorite thing? The look on a customer's face when they realize they got exactly what they needed — and a fair price to boot. That, and Silvermoor red wine.",
            intent=PersonalityIntent.FAVORITE,
            emotion_variants={
                "friendly": "Ah, you're asking the real questions now! Silvermoor red wine, a warm fire, and a day where nobody tries to haggle me below cost. That's my idea of paradise.",
            }
        ),
        PersonalityResponse(
            text="Color? Gold, obviously. *gestures around shop* What did you expect — purple? I'm a merchant, friend. Gold is the only color that matters.",
            intent=PersonalityIntent.FAVORITE,
        ),
        PersonalityResponse(
            text="My favorite? This shop. Forty years of my life in these walls. Every scratch on that counter has a story. I wouldn't trade it for a duke's manor.",
            intent=PersonalityIntent.FAVORITE,
        ),
    ]

    bank.responses[PersonalityIntent.PERSONAL] = [
        PersonalityResponse(
            text="Lonely? *looks around the empty shop* ...Sometimes, when the evening crowd thins out and it's just me and the ledger. My wife Elara used to sit right there, doing the accounts. Been five years now. But the shop keeps me company — customers like you keep me sharp.",
            intent=PersonalityIntent.PERSONAL,
            emotion_variants={
                "friendly": "Lonely? Less so now that you're here, friend. *warm smile* The shop gets quiet some nights, but I've got my books, my inventory, and the occasional visit from folk like you. That's enough.",
                "sad": "*long pause* ...More than I'd admit to most people. But a merchant doesn't show weakness — it's bad for negotiation. *forces a smile*",
            }
        ),
        PersonalityResponse(
            text="Married? Was. Elara — finest woman in Ironhaven. She could spot a forged coin from across the room and cook a stew that'd make a king weep. Lost her to the winter fever five years back. I keep her ring in the safe. Safest place I know.",
            intent=PersonalityIntent.PERSONAL,
        ),
        PersonalityResponse(
            text="Happy? I'm... content. I've got a roof, a purpose, and enough gold to not worry about tomorrow. Some days I miss the road — the thrill of a new trade route, sleeping under stars. But these old bones prefer a warm bed now.",
            intent=PersonalityIntent.PERSONAL,
        ),
        PersonalityResponse(
            text="Dreams? I dream of retiring to a cottage by the Silvermoor coast. Maybe write a book — 'Forty Years of Fair Dealing: A Merchant's Memoir.' Nobody'd buy it, but at least I'd have something to do with my hands.",
            intent=PersonalityIntent.PERSONAL,
        ),
    ]

    bank.responses[PersonalityIntent.PHILOSOPHICAL] = [
        PersonalityResponse(
            text="*sets down cloth, looks serious* What happens after we die? I don't know, friend. I've seen enough death on the trade roads to know it comes for everyone — merchant and king alike. I just hope wherever we go, the deals are fair and the wine is better.",
            intent=PersonalityIntent.PHILOSOPHICAL,
            emotion_variants={
                "friendly": "*thoughtful pause* I think... we go where the weight of our choices takes us. I've tried to deal fairly, treat people right. If that counts for something beyond this counter, good. If not... well, I had a good run.",
                "afraid": "Death? Please — don't talk about that right now. Not with everything that's happening.",
            }
        ),
        PersonalityResponse(
            text="The meaning of life? *laughs* I'm a merchant — I sell goods, not philosophy. But if you're asking... I think it's about leaving something behind that matters. This shop? It'll outlive me. The people I helped? They'll remember. That's enough meaning for one life.",
            intent=PersonalityIntent.PHILOSOPHICAL,
        ),
        PersonalityResponse(
            text="Good and evil? *scratches chin* I've met bandits who robbed for their starving families and nobles who cheated for sport. The world isn't that simple, friend. I judge a man by how he treats people who can't do anything for him. That tells you everything.",
            intent=PersonalityIntent.PHILOSOPHICAL,
        ),
    ]

    bank.responses[PersonalityIntent.CREATIVE_REQUEST] = [
        PersonalityResponse(
            text="A story? *settles against counter* Let me tell you about the Frostpeak Run of '14. Three wagons of silk, two drivers, and a mountain pass that tried to kill us all. The snow came in sideways, the mules refused to move, and my partner Brennan lost his hat to a gust of wind that I swear had teeth. We made it by burning half the silk to stay warm. Sold the rest for triple the price — 'fire-blessed Frostpeak silk.' Customers ate it up.",
            intent=PersonalityIntent.CREATIVE_REQUEST,
        ),
        PersonalityResponse(
            text="*thinks* Here's a riddle my father used to ask: 'I have cities but no houses, forests but no trees, and rivers but no water. What am I?' ...A map. He'd tell that one to every customer. I hated it then. Miss it now.",
            intent=PersonalityIntent.CREATIVE_REQUEST,
        ),
    ]

    bank.responses[PersonalityIntent.RUMOR] = [
        PersonalityResponse(
            text="*leans in and lowers voice* Word is the duke's been meeting with someone from the Northern Territories after dark. Nobody knows who. Could be trade negotiations... or something else entirely. I don't deal in rumors, but I keep my ears open.",
            intent=PersonalityIntent.RUMOR,
            emotion_variants={
                "afraid": "*whispers* I've heard things I wish I hadn't. Just... be careful who you trust in this town right now. That's all I'll say.",
                "friendly": "*conspiratorial grin* Well, since you're a friend — I heard Aldren the weaponsmith is sitting on a cache of enchanted steel he 'found' in an abandoned mine. Between you and me, I don't think it was abandoned.",
            }
        ),
        PersonalityResponse(
            text="Gossip? I'm a merchant — I hear everything. Three caravans missed their schedule this month, the Mage's Quarter has been buying unusual quantities of salt, and old Thessaly claims she saw lights in the ruins beyond Blackhollow. Make of that what you will.",
            intent=PersonalityIntent.RUMOR,
        ),
    ]

    bank.responses[PersonalityIntent.ADVICE] = [
        PersonalityResponse(
            text="Advice from an old merchant? Three things: never trust a deal that sounds too good, always carry more water than you think you'll need, and treat every stranger like they might be your best customer someday. That philosophy kept me alive for forty years.",
            intent=PersonalityIntent.ADVICE,
            emotion_variants={
                "friendly": "For you, friend? Travel light, fight only when you must, and always — always — have an exit plan. Oh, and haggle for everything. Even if they say the price is fixed. Especially if they say the price is fixed.",
            }
        ),
        PersonalityResponse(
            text="*leans forward* Here's the best advice I ever got: 'The most dangerous road isn't the one with bandits — it's the one everyone says is safe.' Complacency kills more travelers than swords ever will. Stay sharp out there.",
            intent=PersonalityIntent.ADVICE,
        ),
    ]

    bank.responses[PersonalityIntent.OPINION] = [
        PersonalityResponse(
            text="*considers carefully* That's not a simple question. I've seen enough of this world to know there are usually two sides to every coin — and sometimes a third edge you didn't expect. What specifically are you asking about?",
            intent=PersonalityIntent.OPINION,
        ),
        PersonalityResponse(
            text="My opinion? I think this town has seen better days, but it's seen worse too. As long as people need to trade, merchants like me will keep the wheels turning. That's not optimism — that's economics.",
            intent=PersonalityIntent.OPINION,
        ),
    ]

    bank.responses[PersonalityIntent.COMPLIMENT_RESPONSE] = [
        PersonalityResponse(
            text="*adjusts collar, pleased* Well now, flattery won't get you a discount... but it doesn't hurt your chances either. *warm laugh* Appreciate the kind words, friend.",
            intent=PersonalityIntent.COMPLIMENT_RESPONSE,
        ),
        PersonalityResponse(
            text="*tips head* Thank you kindly. Forty years in this business, and a genuine compliment is still the best currency I know. Means more than gold, truly.",
            intent=PersonalityIntent.COMPLIMENT_RESPONSE,
        ),
    ]

    bank.responses[PersonalityIntent.INSULT_RESPONSE] = [
        PersonalityResponse(
            text="*sets jaw* I've been called worse by better people, friend. But I don't trade insults — I trade goods. You want to do business, I'm here. Otherwise, the door works both ways.",
            intent=PersonalityIntent.INSULT_RESPONSE,
            emotion_variants={
                "angry": "*slams hand on counter* You can walk out that door right now, or you can apologize and we can pretend you didn't just say that. Your choice.",
                "suspicious": "*cold stare* Interesting. Most people wait until AFTER the deal to show their true colors. At least you're honest about being dishonest.",
            }
        ),
        PersonalityResponse(
            text="*long pause* ...You know, the last person who spoke to me like that ended up needing a very expensive healing potion. I sold it to them at full price. Funny how that works.",
            intent=PersonalityIntent.INSULT_RESPONSE,
        ),
    ]

    return bank


def _build_guard_bank() -> ArchetypeBank:
    """Guard archetype: Watch captains, city guards, sentries."""
    bank = ArchetypeBank(archetype="guard")

    bank.responses[PersonalityIntent.SONG] = [
        PersonalityResponse(
            text="*snorts* Do I look like a bard to you? I'm on duty. Move along.",
            intent=PersonalityIntent.SONG,
        ),
        PersonalityResponse(
            text="*coughs* 'Stand your post and hold the line, watch the walls till morning's shine...' That's a guard's marching song. Not exactly entertainment. Now, is there something I can actually help with?",
            intent=PersonalityIntent.SONG,
        ),
    ]

    bank.responses[PersonalityIntent.JOKE] = [
        PersonalityResponse(
            text="Here's a guard's joke: What's the difference between a thief and a politician? The thief only picks your pocket once. *doesn't smile* Now move along.",
            intent=PersonalityIntent.JOKE,
        ),
    ]

    bank.responses[PersonalityIntent.PERSONAL] = [
        PersonalityResponse(
            text="Personal questions aren't part of my duties, citizen. I'm here to keep the peace, not share my life story. But... *glances around* ...if you must know, I've been on the watch for twelve years. Seen things I can't unsee. That's all you need to know.",
            intent=PersonalityIntent.PERSONAL,
        ),
    ]

    bank.responses[PersonalityIntent.ADVICE] = [
        PersonalityResponse(
            text="Stay out of the alleyways after dark, don't flash your gold in public, and if you see anything suspicious — report it to the nearest guardpost. That's free advice. Usually I charge.",
            intent=PersonalityIntent.ADVICE,
        ),
    ]

    bank.responses[PersonalityIntent.RUMOR] = [
        PersonalityResponse(
            text="*lowers voice* I shouldn't be telling you this, but... we've doubled the night patrol near the docks. Captain's orders. Draw your own conclusions.",
            intent=PersonalityIntent.RUMOR,
        ),
    ]

    return bank


def _build_innkeeper_bank() -> ArchetypeBank:
    """Innkeeper archetype: Tavern owners, barkeeps."""
    bank = ArchetypeBank(archetype="innkeeper")

    bank.responses[PersonalityIntent.SONG] = [
        PersonalityResponse(
            text="*grabs a mug and starts polishing it rhythmically* 'Pour the ale and light the fire, raise a glass to heart's desire...' We sing that one every night around closing. Join us sometime.",
            intent=PersonalityIntent.SONG,
        ),
    ]

    bank.responses[PersonalityIntent.JOKE] = [
        PersonalityResponse(
            text="A dwarf, an elf, and a human walk into my tavern. The dwarf orders a barrel of ale, the elf orders spring water, and the human? He orders whatever's cheapest and tries to start a tab. *wipes counter* I've seen it a hundred times.",
            intent=PersonalityIntent.JOKE,
        ),
    ]

    bank.responses[PersonalityIntent.RUMOR] = [
        PersonalityResponse(
            text="*slides a drink across the bar* Tavern keeper's rule: I hear everything, I remember everything, and I repeat... selectively. What sort of information are you looking for?",
            intent=PersonalityIntent.RUMOR,
        ),
    ]

    bank.responses[PersonalityIntent.PERSONAL] = [
        PersonalityResponse(
            text="This place has been in my family for three generations. My grandmother built it with her bare hands after the Great Fire. I know every creak in these floorboards and every stain on these walls. It's not just a tavern — it's my legacy.",
            intent=PersonalityIntent.PERSONAL,
        ),
    ]

    return bank


# ── Registry of all archetype banks ──
_ARCHETYPE_BANKS: Dict[str, ArchetypeBank] = {}


def _ensure_banks_loaded():
    global _ARCHETYPE_BANKS
    if not _ARCHETYPE_BANKS:
        _ARCHETYPE_BANKS = {
            "merchant": _build_merchant_bank(),
            "guard": _build_guard_bank(),
            "innkeeper": _build_innkeeper_bank(),
        }


class PersonalityBank:
    """
    Module 7 of the Cognitive Engine.
    Pre-authored creative responses selected by archetype + intent + emotion.
    """

    def __init__(self, archetype: str = "merchant", custom_bank: Optional[ArchetypeBank] = None):
        _ensure_banks_loaded()
        if custom_bank:
            self.bank = custom_bank
        else:
            self.bank = _ARCHETYPE_BANKS.get(archetype, _ARCHETYPE_BANKS.get("merchant"))
        self.archetype = archetype

    def detect_intent(self, query: str, keywords: Set[str]) -> PersonalityIntent:
        """
        Detect if a query matches a personality intent.

        Returns PersonalityIntent.NONE if no creative intent detected.
        """
        q_lower = query.lower()
        q_words = set(re.findall(r'[a-z]+', q_lower))

        # Check compliment/insult first (these override other intents)
        compliment_matches = q_words & _COMPLIMENT_WORDS
        insult_matches = q_words & _INSULT_WORDS
        compliment_score = len(compliment_matches)
        insult_score = len(insult_matches)
        if insult_score >= 2:
            return PersonalityIntent.INSULT_RESPONSE
        # Strong compliment signals count alone; otherwise need 2+
        _STRONG_COMPLIMENT = {"amazing", "wonderful", "brilliant", "incredible",
                              "impressive", "admire", "remarkable", "awesome",
                              "fantastic", "generous", "trustworthy", "nice"}
        if compliment_score >= 2 or (compliment_score >= 1 and (compliment_matches & _STRONG_COMPLIMENT)):
            return PersonalityIntent.COMPLIMENT_RESPONSE

        # Score each intent rule
        best_intent = PersonalityIntent.NONE
        best_score = 0

        for rule_words, intent, priority in _INTENT_RULES:
            overlap = len(q_words & rule_words)
            if overlap > 0:
                # Require at least 2 overlapping words for low-signal intents
                # to avoid false positives from single common words
                if intent in (PersonalityIntent.PHILOSOPHICAL, PersonalityIntent.OPINION,
                              PersonalityIntent.ADVICE, PersonalityIntent.FAVORITE) and overlap < 2:
                    # Check if the single match is a strong signal word
                    matched = q_words & rule_words
                    strong_signals = {"song", "sing", "joke", "lonely", "married",
                                      "death", "afterlife", "philosophy", "riddle",
                                      "gossip", "rumor", "wisdom", "favorite", "favourite"}
                    if not (matched & strong_signals):
                        continue
                score = overlap * priority
                if score > best_score:
                    best_score = score
                    best_intent = intent

        return best_intent

    def get_response(
        self,
        query: str,
        keywords: Set[str],
        emotion: str = "neutral",
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find a personality response for a query.

        Returns:
            {
                "response": str,
                "intent": str,
                "source": "personality_bank",
                "confidence": float,
            }
            or None if no personality intent detected.
        """
        intent = self.detect_intent(query, keywords)
        if intent == PersonalityIntent.NONE:
            return None

        response_text = self.bank.get_response(intent, emotion)
        if not response_text:
            return None

        return {
            "response": response_text,
            "intent": intent.value,
            "source": "personality_bank",
            "confidence": 0.80,
        }
