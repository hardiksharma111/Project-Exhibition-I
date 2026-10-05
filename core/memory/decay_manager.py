"""
Hardik's Research Pillar: Dynamic Relevance Decay & Memory Pruning
Mathematical formulation: S(t) = S0 * exp(-lambda * delta_t)
Preserves high-salience factual anchors while smoothly decaying transient chatter.
"""

import math
import re
import time
from typing import List, Dict, Any, Optional

# Regular expressions to spot high-value factual statements during conversation
FACT_EXTRACTION_PATTERNS = [
    re.compile(r"(patient (has|diagnosed with|presents with) [a-zA-Z0-9\s\-]+)", re.IGNORECASE),
    re.compile(r"((dosage|dose) is \d+(\.\d+)?\s*(mg|ml|mcg|g))", re.IGNORECASE),
    re.compile(r"(the formula (for|is) [a-zA-Z0-9\s\+\-\*\/\^\=]+)", re.IGNORECASE),
    re.compile(r"(the solution is [a-zA-Z0-9\s\.\-]+)", re.IGNORECASE),
    re.compile(r"(result is \d+(\.\d+)?)", re.IGNORECASE),
    re.compile(r"(def [a-zA-Z_]\w*\(.*?\):)", re.IGNORECASE),
    re.compile(r"(\bkey concept is [a-zA-Z0-9\s]+)", re.IGNORECASE)
]


class MemoryDecayManager:
    """
    Implements exponential memory decay and salience-based pruning.
    Hardik Sharma (25BAI10772) - Project Exhibition Review 1.
    """

    def __init__(self, half_life_seconds: float = 3600.0):
        """
        half_life_seconds: Time (in seconds) for unreinforced memory to lose 50% salience.
        lambda = ln(2) / half_life
        """
        self.half_life_seconds = half_life_seconds
        self.decay_lambda = math.log(2.0) / self.half_life_seconds

    def compute_salience(self, initial_importance: float, elapsed_seconds: float) -> float:
        """
        Evaluates S(t) = S0 * exp(-lambda * delta_t).
        """
        salience = initial_importance * math.exp(-self.decay_lambda * max(0.0, elapsed_seconds))
        return round(salience, 4)

    def extract_salient_facts(self, text: str) -> List[str]:
        """
        Scans dialogue text for empirical facts, constants, or formulas to anchor in long-term memory.
        """
        facts = []
        for pattern in FACT_EXTRACTION_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                fact_str = m[0] if isinstance(m, tuple) else m
                fact_str = fact_str.strip()
                if fact_str and len(fact_str) > 6 and fact_str not in facts:
                    facts.append(fact_str)
        return facts

    def prune_and_weight(
        self,
        memory_items: List[Dict[str, Any]],
        pruning_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Recalculates effective salience for a list of memories and prunes below threshold.
        """
        current_time = time.time()
        weighted_items = []

        for item in memory_items:
            created_at = item.get("timestamp", current_time)
            elapsed = current_time - created_at
            initial_weight = item.get("importance", 1.0)
            
            # High-value fact bonus
            if item.get("facts") or item.get("extracted_facts"):
                initial_weight = min(1.0, initial_weight * 1.5)

            effective_salience = self.compute_salience(initial_weight, elapsed)

            if effective_salience >= pruning_threshold:
                augmented = dict(item)
                augmented["effective_salience"] = effective_salience
                augmented["elapsed_seconds"] = round(elapsed, 1)
                weighted_items.append(augmented)

        # Sort by effective salience descending
        weighted_items.sort(key=lambda x: x["effective_salience"], reverse=True)
        return weighted_items
