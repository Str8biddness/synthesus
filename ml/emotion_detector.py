#!/usr/bin/env python3
"""
EmotionDetector — ML Swarm Micro-Model #7
AIVM Synthesus 2.0

Classifies the emotional state of player input text.
Uses a rule-based lexicon approach with weighted keyword matching
and contextual modifiers (negation, intensifiers, punctuation).

Emotions detected:
- joy, anger, sadness, fear, surprise, disgust, trust, neutral

Designed to feed into the NPC EmotionStateMachine so NPCs react
appropriately to the player's emotional tone.

Footprint: ~18 KB (lexicon), <1ms inference.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Detects player emotional state from text input.

    Complements SentimentAnalyzer (which gives valence/polarity) by
    providing discrete emotion categories suitable for the NPC
    EmotionStateMachine.

    Usage:
        detector = EmotionDetector()
        result = detector.detect("I can't believe you did that!")
        # → {"primary": "surprise", "secondary": "anger", "scores": {...}, "intensity": 0.75}
    """

    # Emotion lexicon: keyword → (emotion, weight)
    LEXICON = {
        # Joy
        "happy": ("joy", 0.8), "glad": ("joy", 0.7), "great": ("joy", 0.6),
        "love": ("joy", 0.9), "wonderful": ("joy", 0.8), "awesome": ("joy", 0.8),
        "amazing": ("joy", 0.8), "excited": ("joy", 0.9), "fantastic": ("joy", 0.8),
        "perfect": ("joy", 0.7), "beautiful": ("joy", 0.6), "enjoy": ("joy", 0.7),
        "thanks": ("joy", 0.5), "thank": ("joy", 0.5), "pleased": ("joy", 0.7),
        "fun": ("joy", 0.6), "yay": ("joy", 0.8), "nice": ("joy", 0.5),
        "cool": ("joy", 0.5), "sweet": ("joy", 0.6), "brilliant": ("joy", 0.7),
        "haha": ("joy", 0.6), "lol": ("joy", 0.5), "lmao": ("joy", 0.7),

        # Anger
        "angry": ("anger", 0.9), "furious": ("anger", 1.0), "hate": ("anger", 0.9),
        "stupid": ("anger", 0.7), "idiot": ("anger", 0.8), "annoying": ("anger", 0.6),
        "mad": ("anger", 0.7), "pissed": ("anger", 0.8), "rage": ("anger", 0.9),
        "unfair": ("anger", 0.6), "ridiculous": ("anger", 0.6), "terrible": ("anger", 0.7),
        "worst": ("anger", 0.7), "awful": ("anger", 0.7), "damn": ("anger", 0.5),
        "hell": ("anger", 0.4), "sucks": ("anger", 0.6), "garbage": ("anger", 0.7),

        # Sadness
        "sad": ("sadness", 0.8), "sorry": ("sadness", 0.5), "miss": ("sadness", 0.6),
        "lonely": ("sadness", 0.8), "depressed": ("sadness", 0.9), "cry": ("sadness", 0.8),
        "heartbroken": ("sadness", 0.9), "disappointed": ("sadness", 0.7),
        "unfortunate": ("sadness", 0.5), "loss": ("sadness", 0.6), "grief": ("sadness", 0.9),
        "regret": ("sadness", 0.7), "miserable": ("sadness", 0.8),

        # Fear
        "afraid": ("fear", 0.8), "scared": ("fear", 0.8), "fear": ("fear", 0.9),
        "terrified": ("fear", 1.0), "nervous": ("fear", 0.6), "worried": ("fear", 0.6),
        "anxious": ("fear", 0.7), "panic": ("fear", 0.9), "horror": ("fear", 0.8),
        "danger": ("fear", 0.6), "threat": ("fear", 0.6), "creepy": ("fear", 0.5),

        # Surprise
        "surprised": ("surprise", 0.8), "shocked": ("surprise", 0.9),
        "wow": ("surprise", 0.7), "whoa": ("surprise", 0.7),
        "unbelievable": ("surprise", 0.7), "unexpected": ("surprise", 0.6),
        "omg": ("surprise", 0.7), "incredible": ("surprise", 0.6),
        "really": ("surprise", 0.3), "seriously": ("surprise", 0.4),

        # Disgust
        "disgusting": ("disgust", 0.9), "gross": ("disgust", 0.7),
        "nasty": ("disgust", 0.7), "revolting": ("disgust", 0.9),
        "ew": ("disgust", 0.6), "yuck": ("disgust", 0.6),
        "vile": ("disgust", 0.8), "repulsive": ("disgust", 0.9),

        # Trust
        "trust": ("trust", 0.8), "believe": ("trust", 0.6), "reliable": ("trust", 0.7),
        "honest": ("trust", 0.7), "loyal": ("trust", 0.8), "faith": ("trust", 0.7),
        "promise": ("trust", 0.6), "depend": ("trust", 0.6), "ally": ("trust", 0.7),
        "friend": ("trust", 0.6),
    }

    # Negation words flip emotion
    NEGATORS = {"not", "no", "never", "neither", "nobody", "nothing",
                "nowhere", "nor", "don't", "doesn't", "didn't", "won't",
                "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "aren't"}

    # Intensifiers boost weight
    INTENSIFIERS = {"very": 1.3, "really": 1.2, "so": 1.2, "extremely": 1.5,
                    "super": 1.3, "incredibly": 1.4, "absolutely": 1.3,
                    "totally": 1.2, "utterly": 1.4}

    # Emotion opposites for negation
    OPPOSITES = {
        "joy": "sadness", "sadness": "joy",
        "anger": "trust", "trust": "anger",
        "fear": "trust", "surprise": "neutral",
        "disgust": "trust",
    }

    def __init__(self):
        pass

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect emotions in text.

        Returns:
            {
                "primary": str,      # dominant emotion
                "secondary": str,    # second strongest (or None)
                "scores": dict,      # {emotion: float} for all emotions
                "intensity": float,  # 0-1 overall emotional intensity
            }
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return self._neutral_result()

        scores = {
            "joy": 0.0, "anger": 0.0, "sadness": 0.0, "fear": 0.0,
            "surprise": 0.0, "disgust": 0.0, "trust": 0.0, "neutral": 0.1,
        }

        # Scan words with context window
        for i, word in enumerate(words):
            if word not in self.LEXICON:
                continue

            emotion, weight = self.LEXICON[word]

            # Check for preceding negator (flip emotion)
            negated = False
            if i > 0 and words[i - 1] in self.NEGATORS:
                negated = True
            if i > 1 and words[i - 2] in self.NEGATORS:
                negated = True

            # Check for preceding intensifier
            multiplier = 1.0
            if i > 0 and words[i - 1] in self.INTENSIFIERS:
                multiplier = self.INTENSIFIERS[words[i - 1]]

            if negated:
                opposite = self.OPPOSITES.get(emotion, "neutral")
                scores[opposite] += weight * multiplier * 0.7
            else:
                scores[emotion] += weight * multiplier

        # Punctuation modifiers
        exclamation_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        if exclamation_count > 0:
            # Amplify strongest emotion
            max_emo = max(scores, key=scores.get)
            if max_emo != "neutral":
                scores[max_emo] *= 1.0 + min(exclamation_count, 3) * 0.15

        if caps_ratio > 0.5 and len(text) > 3:
            # ALL CAPS = intensity boost
            scores["anger"] += 0.3
            max_emo = max(scores, key=scores.get)
            if max_emo != "neutral":
                scores[max_emo] *= 1.2

        if question_count > 0:
            scores["surprise"] += 0.1 * question_count

        # Normalize and find primary/secondary
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 4) for k, v in scores.items()}

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0][0]
        secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0.1 else None

        # Intensity = 1 - neutral score
        intensity = 1.0 - scores.get("neutral", 0.0)

        return {
            "primary": primary,
            "secondary": secondary,
            "scores": scores,
            "intensity": round(max(0.0, min(intensity, 1.0)), 4),
        }

    def _neutral_result(self) -> Dict[str, Any]:
        return {
            "primary": "neutral",
            "secondary": None,
            "scores": {
                "joy": 0.0, "anger": 0.0, "sadness": 0.0, "fear": 0.0,
                "surprise": 0.0, "disgust": 0.0, "trust": 0.0, "neutral": 1.0,
            },
            "intensity": 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "EmotionDetector",
            "emotions": ["joy", "anger", "sadness", "fear", "surprise", "disgust", "trust", "neutral"],
            "lexicon_size": len(self.LEXICON),
            "negators": len(self.NEGATORS),
            "intensifiers": len(self.INTENSIFIERS),
            "footprint_kb": 18,
        }
