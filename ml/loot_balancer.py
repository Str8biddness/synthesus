#!/usr/bin/env python3
"""
LootBalancer — ML Swarm Micro-Model #5
AIVM Synthesus 2.0

Fair reward distribution system for NPC merchants and quest givers.
Uses weighted scoring to balance loot drops based on player context:
- Player level / progression
- Interaction history (loyalty, quest completion)
- Economy state (inflation, scarcity)
- Character personality (generous vs. stingy merchant)

Footprint: ~8 KB, <1ms inference.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LootBalancer:
    """
    Balances item/reward distribution for NPC interactions.

    Used by merchant NPCs for pricing, quest givers for reward scaling,
    and any character that distributes items or resources.
    """

    # Rarity tiers with base weights
    RARITY_TIERS = {
        "common": {"weight": 0.50, "value_mult": 1.0},
        "uncommon": {"weight": 0.25, "value_mult": 2.0},
        "rare": {"weight": 0.15, "value_mult": 5.0},
        "epic": {"weight": 0.07, "value_mult": 15.0},
        "legendary": {"weight": 0.03, "value_mult": 50.0},
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def calculate_reward_tier(
        self,
        player_level: float = 1.0,
        loyalty_score: float = 0.5,
        quest_difficulty: float = 0.5,
        merchant_generosity: float = 0.5,
        economy_inflation: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Determine reward tier and value modifier.

        Args:
            player_level: 0-1 normalized player progression
            loyalty_score: 0-1 relationship/loyalty with NPC
            quest_difficulty: 0-1 difficulty of completed task
            merchant_generosity: 0-1 NPC personality trait
            economy_inflation: -1 to 1 economy state adjustment

        Returns:
            {"tier": str, "value_modifier": float, "quantity_bonus": int, "probabilities": dict}
        """
        # Adjust weights based on context
        adjusted = {}
        for tier, info in self.RARITY_TIERS.items():
            base_w = info["weight"]

            # Higher level/loyalty/difficulty shift weights toward rarer items
            boost = (player_level * 0.3 + loyalty_score * 0.2 +
                     quest_difficulty * 0.3 + merchant_generosity * 0.2)

            if tier in ("rare", "epic", "legendary"):
                adjusted[tier] = base_w * (1.0 + boost)
            else:
                adjusted[tier] = base_w * (1.0 - boost * 0.3)

        # Normalize
        total = sum(adjusted.values())
        probs = {k: round(v / total, 4) for k, v in adjusted.items()}

        # Roll for tier
        roll = self._rng.random()
        cumulative = 0.0
        selected_tier = "common"
        for tier, prob in probs.items():
            cumulative += prob
            if roll <= cumulative:
                selected_tier = tier
                break

        # Value modifier based on economy and relationship
        value_mod = self.RARITY_TIERS[selected_tier]["value_mult"]
        value_mod *= (1.0 + merchant_generosity * 0.3)
        value_mod *= (1.0 - economy_inflation * 0.2)

        # Quantity bonus for loyal customers
        qty_bonus = 0
        if loyalty_score > 0.7:
            qty_bonus = 1
        if loyalty_score > 0.9:
            qty_bonus = 2

        return {
            "tier": selected_tier,
            "value_modifier": round(value_mod, 2),
            "quantity_bonus": qty_bonus,
            "probabilities": probs,
        }

    def price_adjustment(
        self,
        base_price: float,
        loyalty_score: float = 0.5,
        merchant_generosity: float = 0.5,
        is_buying: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate adjusted price for buy/sell interactions.

        Returns:
            {"adjusted_price": float, "discount_pct": float, "reason": str}
        """
        discount = 0.0

        # Loyalty discount (up to 15%)
        if loyalty_score > 0.3:
            discount += (loyalty_score - 0.3) * 0.2

        # Generosity discount (up to 10%)
        discount += (merchant_generosity - 0.5) * 0.2

        if not is_buying:
            # Selling: merchant pays less
            discount = -abs(discount) * 0.5

        discount = max(-0.3, min(discount, 0.25))  # clamp
        adjusted = base_price * (1.0 - discount)

        reason = "standard pricing"
        if discount > 0.1:
            reason = "loyal customer discount"
        elif discount > 0.05:
            reason = "friendly pricing"
        elif discount < -0.05:
            reason = "tough bargaining"

        return {
            "adjusted_price": round(adjusted, 2),
            "discount_pct": round(discount * 100, 1),
            "reason": reason,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "LootBalancer",
            "rarity_tiers": list(self.RARITY_TIERS.keys()),
            "footprint_kb": 8,
        }
