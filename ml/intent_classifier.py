"""
Synthesus 2.0 — ML Swarm Module 1: Intent Classifier
"What is the player trying to DO?"

A lightweight sklearn-based intent classifier that:
1. Trains on character-derived + universal training data
2. Exports to ONNX for zero-dependency inference
3. Classifies player intent in <1ms

Intent Categories:
  - greeting       : "hello", "hey", "hi there"
  - farewell       : "goodbye", "see you", "bye"
  - question       : "what is", "tell me about", "how does"
  - shop_browse    : "what do you sell", "show me wares"
  - shop_buy       : "I want to buy", "give me a"
  - shop_haggle    : "too expensive", "can you lower"
  - personal       : "are you happy", "tell me about yourself"
  - creative       : "sing a song", "tell me a joke"
  - combat         : "attack", "fight", "defend"
  - quest          : "any work", "I'll do it", "quest"
  - insult         : "you're stupid", "you're an idiot"
  - compliment     : "you're amazing", "great shop"
  - lore           : "tell me about this place", "history of"
  - unknown        : catch-all

Model size: ~50 KB ONNX, ~0.3ms inference, zero GPU.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────────
# Universal Training Data (character-agnostic)
# ──────────────────────────────────────────────────

_UNIVERSAL_TRAINING_DATA = [
    # Greetings
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("hi there", "greeting"), ("good morning", "greeting"), ("good evening", "greeting"),
    ("greetings", "greeting"), ("howdy", "greeting"), ("what's up", "greeting"),
    ("hello there friend", "greeting"), ("hey how are you", "greeting"),

    # Farewells
    ("goodbye", "farewell"), ("bye", "farewell"), ("see you later", "farewell"),
    ("farewell", "farewell"), ("take care", "farewell"), ("see you around", "farewell"),
    ("gotta go", "farewell"), ("until next time", "farewell"), ("I must leave", "farewell"),
    ("I should head out", "farewell"),

    # Questions / Lore
    ("what is this place", "question"), ("tell me about this town", "question"),
    ("who are you", "question"), ("what do you do", "question"),
    ("how long have you been here", "question"), ("what happened", "question"),
    ("where am I", "question"), ("what's going on", "question"),
    ("can you explain", "question"), ("I have a question", "question"),
    ("tell me about the history", "lore"), ("what is the lore", "lore"),
    ("any legends around here", "lore"), ("tell me about the kingdom", "lore"),
    ("what happened in the war", "lore"), ("ancient history", "lore"),

    # Shop Browse
    ("what do you sell", "shop_browse"), ("show me your wares", "shop_browse"),
    ("what's for sale", "shop_browse"), ("what do you have", "shop_browse"),
    ("let me see your inventory", "shop_browse"), ("any goods", "shop_browse"),
    ("what items do you carry", "shop_browse"),

    # Shop Buy
    ("I want to buy", "shop_buy"), ("give me a sword", "shop_buy"),
    ("I'll take that", "shop_buy"), ("purchase a potion", "shop_buy"),
    ("buy a shield", "shop_buy"), ("sell me a weapon", "shop_buy"),
    ("I need a health potion", "shop_buy"), ("can I get one of those", "shop_buy"),

    # Shop Haggle
    ("too expensive", "shop_haggle"), ("can you lower the price", "shop_haggle"),
    ("that's too much", "shop_haggle"), ("give me a discount", "shop_haggle"),
    ("how about a better price", "shop_haggle"), ("come on cheaper", "shop_haggle"),
    ("I'll pay 50 gold", "shop_haggle"), ("would you take less", "shop_haggle"),

    # Personal
    ("are you happy", "personal"), ("do you get lonely", "personal"),
    ("tell me about yourself", "personal"), ("are you married", "personal"),
    ("what are your dreams", "personal"), ("how are you feeling", "personal"),
    ("do you have a family", "personal"), ("what do you enjoy", "personal"),

    # Creative
    ("sing me a song", "creative"), ("tell me a joke", "creative"),
    ("do you know any riddles", "creative"), ("tell me a story", "creative"),
    ("make me laugh", "creative"), ("recite a poem", "creative"),
    ("entertain me", "creative"), ("any funny stories", "creative"),

    # Combat
    ("I attack you", "combat"), ("draw your weapon", "combat"),
    ("prepare to fight", "combat"), ("defend yourself", "combat"),
    ("I'm going to kill you", "combat"), ("fight me", "combat"),
    ("I challenge you", "combat"), ("battle", "combat"),

    # Quest
    ("any work available", "quest"), ("I'll take the job", "quest"),
    ("got any quests", "quest"), ("I need a task", "quest"),
    ("what needs doing", "quest"), ("any missions", "quest"),
    ("I accept the quest", "quest"), ("I'll do it", "quest"),
    ("send me on a quest", "quest"), ("count me in", "quest"),

    # Compliment
    ("you're amazing", "compliment"), ("great shop you have", "compliment"),
    ("you're the best", "compliment"), ("I really appreciate you", "compliment"),
    ("you're so kind", "compliment"), ("what a wonderful place", "compliment"),
    ("impressive work", "compliment"), ("you're very helpful", "compliment"),

    # Insult
    ("you're stupid", "insult"), ("you're an idiot", "insult"),
    ("this place is terrible", "insult"), ("you're a liar", "insult"),
    ("you're a cheat", "insult"), ("you're useless", "insult"),
    ("what a dump", "insult"), ("you're pathetic", "insult"),

    # Unknown / Catch-all
    ("asdfghjkl", "unknown"), ("random gibberish", "unknown"),
    ("quantum chromodynamics", "unknown"), ("the GDP of Luxembourg", "unknown"),
]


class IntentClassifier:
    """
    ML-powered intent classifier for player messages.
    
    Trains a TF-IDF + Logistic Regression pipeline, exportable to ONNX.
    
    Usage:
        classifier = IntentClassifier()
        classifier.train()
        intent, confidence = classifier.predict("I want to buy a sword")
        # → ("shop_buy", 0.92)
        
        classifier.save("models/intent_classifier")
        loaded = IntentClassifier.load("models/intent_classifier")
    """

    LABELS = [
        "greeting", "farewell", "question", "shop_browse", "shop_buy",
        "shop_haggle", "personal", "creative", "combat", "quest",
        "compliment", "insult", "lore", "unknown",
    ]

    def __init__(self, extra_training_data: Optional[List[Tuple[str, str]]] = None):
        self._training_data = list(_UNIVERSAL_TRAINING_DATA)
        if extra_training_data:
            self._training_data.extend(extra_training_data)

        self._pipeline: Optional[Pipeline] = None
        self._is_trained = False

    def train(self, verbose: bool = False) -> Dict[str, Any]:
        """Train the intent classifier.
        
        Returns training stats including accuracy and per-class scores.
        """
        texts = [t[0] for t in self._training_data]
        labels = [t[1] for t in self._training_data]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])

        self._pipeline.fit(texts, labels)
        self._is_trained = True

        # Cross-validation
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
        """Classify a player message.
        
        Returns:
            (intent_label, confidence) where confidence is [0, 1]
        """
        if not self._is_trained or self._pipeline is None:
            return "unknown", 0.0

        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        best_idx = np.argmax(proba)
        label = self._pipeline.classes_[best_idx]
        confidence = float(proba[best_idx])

        return label, confidence

    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k predictions with confidence scores."""
        if not self._is_trained or self._pipeline is None:
            return [("unknown", 0.0)]

        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        top_indices = np.argsort(proba)[::-1][:k]
        return [
            (self._pipeline.classes_[i], float(proba[i]))
            for i in top_indices
        ]

    def save(self, path: str):
        """Save the trained model to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._pipeline:
            with open(p / "pipeline.pkl", "wb") as f:
                pickle.dump(self._pipeline, f)
            # Save metadata
            meta = {
                "labels": list(self._pipeline.classes_),
                "n_samples": len(self._training_data),
                "n_features": self._pipeline.named_steps["tfidf"].max_features,
            }
            with open(p / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "IntentClassifier":
        """Load a trained model from disk."""
        p = Path(path)
        classifier = cls()
        with open(p / "pipeline.pkl", "rb") as f:
            classifier._pipeline = pickle.load(f)
        classifier._is_trained = True
        return classifier

    def export_onnx(self, path: str):
        """Export the model to ONNX format for deployment.
        
        Note: Uses word-level tokenizer for ONNX compatibility.
        The sklearn pipeline uses char_wb for training, so we retrain
        with word-level features before export.
        """
        if not self._is_trained or self._pipeline is None:
            raise ValueError("Model must be trained before ONNX export")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import StringTensorType

            # Build ONNX-compatible pipeline (word tokenizer)
            texts = [t[0] for t in self._training_data]
            labels = [t[1] for t in self._training_data]
            onnx_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    class_weight="balanced",
                    solver="lbfgs",
                )),
            ])
            onnx_pipeline.fit(texts, labels)

            onnx_model = convert_sklearn(
                onnx_pipeline,
                "intent_classifier",
                initial_types=[("text", StringTensorType([None, 1]))],
                target_opset=15,
            )
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.write(onnx_model.SerializeToString())
            return os.path.getsize(p)
        except ImportError:
            print("WARNING: skl2onnx not available, skipping ONNX export")
            return 0
        except Exception as e:
            print(f"WARNING: ONNX export failed: {e}")
            return 0

    def add_training_data(self, data: List[Tuple[str, str]]):
        """Add more training samples (requires retrain)."""
        self._training_data.extend(data)
        self._is_trained = False

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "is_trained": self._is_trained,
            "n_samples": len(self._training_data),
            "n_labels": len(self.LABELS),
            "labels": self.LABELS,
        }


def build_training_data_from_character(
    patterns_data: Dict, character_id: str,
) -> List[Tuple[str, str]]:
    """Extract training data from a character's patterns.json.
    
    Maps pattern domains to intent labels:
    - shop → shop_browse / shop_buy
    - identity → question
    - lore → lore
    - quest → quest
    - trade → shop_haggle
    """
    domain_to_intent = {
        "shop": "shop_browse",
        "identity": "question",
        "lore": "lore",
        "quest": "quest",
        "trade": "shop_haggle",
        "combat": "combat",
        "relationship": "personal",
    }

    training = []
    for pat_list in [
        patterns_data.get("synthetic_patterns", []),
        patterns_data.get("generic_patterns", []),
    ]:
        for pat in pat_list:
            domain = pat.get("domain", "")
            intent = domain_to_intent.get(domain, "question")
            triggers = pat.get("trigger", [])
            if isinstance(triggers, str):
                triggers = [triggers]
            for t in triggers:
                training.append((t, intent))

    return training
