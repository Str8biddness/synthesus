"""
Synthesus 2.0 — ML Swarm Module 2: Sentiment Analyzer
"What TONE is the player using?"

Lightweight sentiment classification for NPC emotional reactions.
Classifies player text into emotional categories that feed directly
into the EmotionStateMachine module.

Sentiment Categories:
  - positive    : friendly, happy, grateful
  - negative    : angry, hostile, frustrated
  - neutral     : matter-of-fact, business-like
  - threatening : combat threats, intimidation
  - pleading    : begging, desperate, sad
  - flirtatious : flattery, charm, romantic intent

Model size: ~40 KB sklearn, ~0.2ms inference, zero GPU.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────────
# Training Data
# ──────────────────────────────────────────────────

_SENTIMENT_TRAINING_DATA = [
    # Positive
    ("hello friend", "positive"), ("you're amazing", "positive"),
    ("thank you so much", "positive"), ("great job", "positive"),
    ("I appreciate your help", "positive"), ("wonderful", "positive"),
    ("you're the best", "positive"), ("I love this place", "positive"),
    ("how kind of you", "positive"), ("brilliant work", "positive"),
    ("that's impressive", "positive"), ("you've been so helpful", "positive"),
    ("I'm grateful", "positive"), ("what a lovely shop", "positive"),
    ("you make me smile", "positive"), ("cheers", "positive"),
    ("bless you", "positive"), ("that's wonderful news", "positive"),
    ("I'm so happy", "positive"), ("perfect", "positive"),

    # Negative
    ("you're useless", "negative"), ("this is terrible", "negative"),
    ("what a waste of time", "negative"), ("you're a liar", "negative"),
    ("I hate this", "negative"), ("that's awful", "negative"),
    ("you're pathetic", "negative"), ("worst shop ever", "negative"),
    ("how dare you", "negative"), ("that's unacceptable", "negative"),
    ("you're cheating me", "negative"), ("I'm angry", "negative"),
    ("this is garbage", "negative"), ("stupid", "negative"),
    ("you're a fool", "negative"), ("disgusting", "negative"),
    ("what a rip off", "negative"), ("I'm furious", "negative"),
    ("you've ruined everything", "negative"), ("go away", "negative"),

    # Neutral
    ("what do you sell", "neutral"), ("I need a sword", "neutral"),
    ("tell me about this town", "neutral"), ("how much is that", "neutral"),
    ("where is the inn", "neutral"), ("who are you", "neutral"),
    ("I'm looking for work", "neutral"), ("any news", "neutral"),
    ("what time is it", "neutral"), ("show me your wares", "neutral"),
    ("I have a question", "neutral"), ("just browsing", "neutral"),
    ("interesting", "neutral"), ("I see", "neutral"),
    ("okay", "neutral"), ("tell me more", "neutral"),
    ("how does that work", "neutral"), ("what happened here", "neutral"),
    ("I understand", "neutral"), ("let me think about it", "neutral"),

    # Threatening
    ("I'll kill you", "threatening"), ("hand over the gold", "threatening"),
    ("give me everything or die", "threatening"), ("prepare to die", "threatening"),
    ("I'm going to destroy this shop", "threatening"), ("fight me coward", "threatening"),
    ("you'll regret this", "threatening"), ("I'll burn this place down", "threatening"),
    ("watch your back", "threatening"), ("don't make me hurt you", "threatening"),
    ("give me what I want or else", "threatening"), ("I challenge you to a duel", "threatening"),
    ("draw your weapon", "threatening"), ("your days are numbered", "threatening"),
    ("I'll make you pay", "threatening"), ("surrender or die", "threatening"),

    # Pleading
    ("please help me", "pleading"), ("I'm begging you", "pleading"),
    ("I have nothing left", "pleading"), ("my family is starving", "pleading"),
    ("I'm desperate", "pleading"), ("please I need this", "pleading"),
    ("have mercy", "pleading"), ("I can't afford it", "pleading"),
    ("please give me a chance", "pleading"), ("I'm so scared", "pleading"),
    ("I don't know what to do", "pleading"), ("I'm lost and alone", "pleading"),
    ("can you spare some food", "pleading"), ("please don't turn me away", "pleading"),
    ("I'll do anything", "pleading"), ("take pity on me", "pleading"),

    # Flirtatious
    ("you have beautiful eyes", "flirtatious"), ("are you single", "flirtatious"),
    ("you're quite handsome", "flirtatious"), ("want to grab a drink", "flirtatious"),
    ("you're very charming", "flirtatious"), ("how about dinner", "flirtatious"),
    ("I find you attractive", "flirtatious"), ("quite the looker aren't you", "flirtatious"),
    ("your smile lights up the room", "flirtatious"), ("is that a blush I see", "flirtatious"),
    ("come here often", "flirtatious"), ("you clean up nicely", "flirtatious"),
]


# ──────────────────────────────────────────────────
# Emotion Mapping (sentiment → EmotionStateMachine triggers)
# ──────────────────────────────────────────────────

SENTIMENT_TO_EMOTION = {
    "positive": "friendly",
    "negative": "angry",
    "neutral": "neutral",
    "threatening": "afraid",
    "pleading": "sad",
    "flirtatious": "embarrassed",
}


class SentimentAnalyzer:
    """
    ML-powered sentiment classifier for player messages.
    
    Feeds into the EmotionStateMachine to determine NPC emotional reactions.
    
    Usage:
        analyzer = SentimentAnalyzer()
        analyzer.train()
        sentiment, confidence = analyzer.predict("You're amazing!")
        # → ("positive", 0.89)
        
        emotion = analyzer.to_emotion("positive")
        # → "friendly"
    """

    LABELS = ["positive", "negative", "neutral", "threatening", "pleading", "flirtatious"]

    def __init__(self, extra_training_data: Optional[List[Tuple[str, str]]] = None):
        self._training_data = list(_SENTIMENT_TRAINING_DATA)
        if extra_training_data:
            self._training_data.extend(extra_training_data)
        self._pipeline: Optional[Pipeline] = None
        self._is_trained = False

    def train(self, verbose: bool = False) -> Dict[str, Any]:
        """Train the sentiment analyzer."""
        texts = [t[0] for t in self._training_data]
        labels = [t[1] for t in self._training_data]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=3000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=3.0,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

        self._pipeline.fit(texts, labels)
        self._is_trained = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(self._pipeline, texts, labels, cv=min(5, len(texts)))

        stats = {
            "samples": len(texts),
            "classes": len(set(labels)),
            "cv_accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
        }
        if verbose:
            print(f"  Trained on {stats['samples']} samples, {stats['classes']} classes")
            print(f"  CV accuracy: {stats['cv_accuracy']:.3f} ± {stats['cv_std']:.3f}")
        return stats

    def predict(self, text: str) -> Tuple[str, float]:
        """Classify sentiment of player text.
        
        Returns: (sentiment_label, confidence)
        """
        if not self._is_trained or self._pipeline is None:
            return "neutral", 0.0

        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        best_idx = np.argmax(proba)
        label = self._pipeline.classes_[best_idx]
        confidence = float(proba[best_idx])
        return label, confidence

    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k sentiment predictions."""
        if not self._is_trained or self._pipeline is None:
            return [("neutral", 0.0)]
        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        top_indices = np.argsort(proba)[::-1][:k]
        return [
            (self._pipeline.classes_[i], float(proba[i]))
            for i in top_indices
        ]

    def to_emotion(self, sentiment: str) -> str:
        """Map a sentiment label to an EmotionStateMachine trigger."""
        return SENTIMENT_TO_EMOTION.get(sentiment, "neutral")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Full analysis: sentiment + emotion mapping + confidence.
        
        Returns:
            {
                "sentiment": str,
                "confidence": float,
                "emotion_trigger": str,
                "top_3": [(label, conf), ...],
            }
        """
        sentiment, confidence = self.predict(text)
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "emotion_trigger": self.to_emotion(sentiment),
            "top_3": self.predict_top_k(text, k=3),
        }

    def save(self, path: str):
        """Save trained model to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._pipeline:
            with open(p / "pipeline.pkl", "wb") as f:
                pickle.dump(self._pipeline, f)
            meta = {
                "labels": list(self._pipeline.classes_),
                "n_samples": len(self._training_data),
                "emotion_mapping": SENTIMENT_TO_EMOTION,
            }
            with open(p / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SentimentAnalyzer":
        """Load trained model from disk."""
        p = Path(path)
        analyzer = cls()
        with open(p / "pipeline.pkl", "rb") as f:
            analyzer._pipeline = pickle.load(f)
        analyzer._is_trained = True
        return analyzer

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_trained": self._is_trained,
            "n_samples": len(self._training_data),
            "n_labels": len(self.LABELS),
            "labels": self.LABELS,
            "emotion_mapping": SENTIMENT_TO_EMOTION,
        }
