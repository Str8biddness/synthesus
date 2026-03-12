#!/usr/bin/env python3
"""
BehaviorPredictor — ML Swarm Micro-Model #4
AIVM Synthesus 2.0

Predicts player behavior patterns to enable proactive NPC responses.
Uses lightweight logistic regression on interaction history features.

Predictions:
- Next likely action (buy, sell, leave, fight, ask_question, explore)
- Engagement probability (will the player continue interacting?)
- Escalation risk (is the player getting frustrated?)

Footprint: ~15 KB fitted model, <1ms inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BehaviorPredictor:
    """
    Predicts player behavior from interaction history features.

    Features used:
    - turn_count: number of turns in current conversation
    - avg_msg_length: average message length (words)
    - sentiment_trend: recent sentiment slope (-1 to 1)
    - topic_switches: number of topic changes
    - time_between_msgs: average gap between messages (seconds)
    - question_ratio: fraction of messages that are questions
    """

    ACTIONS = ["buy", "sell", "leave", "fight", "ask_question", "explore", "negotiate", "idle"]
    DEFAULT_WEIGHTS = {
        "buy": np.array([0.1, 0.2, 0.3, -0.1, -0.2, 0.1]),
        "sell": np.array([0.1, 0.1, 0.2, 0.0, -0.1, 0.0]),
        "leave": np.array([-0.3, -0.2, -0.4, 0.3, 0.5, -0.3]),
        "fight": np.array([0.0, -0.1, -0.6, 0.2, 0.0, -0.2]),
        "ask_question": np.array([0.2, 0.3, 0.1, 0.1, -0.1, 0.6]),
        "explore": np.array([0.0, 0.1, 0.1, 0.4, 0.0, 0.2]),
        "negotiate": np.array([0.2, 0.4, 0.2, 0.1, -0.1, 0.3]),
        "idle": np.array([-0.1, -0.3, -0.1, -0.1, 0.6, -0.2]),
    }

    def __init__(self):
        self._weights = {k: v.copy() for k, v in self.DEFAULT_WEIGHTS.items()}

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict next likely action and engagement metrics.

        Args:
            features: dict with keys like turn_count, avg_msg_length,
                      sentiment_trend, topic_switches, time_between_msgs,
                      question_ratio.

        Returns:
            {
                "predicted_action": str,
                "action_probabilities": {action: float, ...},
                "engagement_score": float,  # 0-1
                "escalation_risk": float,   # 0-1
            }
        """
        feat_vec = np.array([
            features.get("turn_count", 0) / 20.0,        # normalize
            features.get("avg_msg_length", 10) / 50.0,
            features.get("sentiment_trend", 0.0),
            features.get("topic_switches", 0) / 5.0,
            features.get("time_between_msgs", 5.0) / 30.0,
            features.get("question_ratio", 0.5),
        ], dtype=np.float32)

        # Compute raw scores for each action
        scores = {}
        for action, w in self._weights.items():
            scores[action] = float(np.dot(w, feat_vec))

        # Softmax to get probabilities
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: round(v / total, 4) for k, v in exp_scores.items()}

        predicted = max(probs, key=probs.get)

        # Engagement = inverse of leave probability
        engagement = 1.0 - probs.get("leave", 0.0)

        # Escalation risk = fight + leave probability
        escalation = probs.get("fight", 0.0) + probs.get("leave", 0.0) * 0.5

        return {
            "predicted_action": predicted,
            "action_probabilities": probs,
            "engagement_score": round(min(engagement, 1.0), 4),
            "escalation_risk": round(min(escalation, 1.0), 4),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "BehaviorPredictor",
            "actions": self.ACTIONS,
            "features": 6,
            "footprint_kb": 15,
        }
