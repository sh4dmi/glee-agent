"""Decision policies, one per game. Each returns a Decision; none of them calls an LLM."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    action: dict                 # the JSON fields GLEE expects, except "message"
    draft_message: str | None    # a plain message to send with it, or None when the game takes no message
    rationale: str               # one line for the logs
    details: dict = field(default_factory=dict)


def posterior(log_likelihoods: dict) -> dict:
    """Normalise log-likelihoods over a finite set of hypotheses (uniform prior)."""
    import math
    top = max(log_likelihoods.values())
    weights = {k: math.exp(v - top) for k, v in log_likelihoods.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}
