"""Closed-form game theory for the three GLEE games.

Everything here is exact and has no fitted parameters. It is used in two ways: as the rational
baseline the evaluation compares against, and as input features for the opponent models
(for example, how much a perfectly rational responder could secure by rejecting).
"""
from __future__ import annotations

from functools import lru_cache


def proposer(round_number: int) -> int:
    """GLEE alternates proposers: player 1 proposes in odd rounds, player 2 in even rounds."""
    return 1 if round_number % 2 else 2


@lru_cache(maxsize=4096)
def spne_proposer_shares(max_rounds: int, delta_1: float, delta_2: float) -> tuple[float, ...]:
    """Subgame-perfect shares in finite-horizon alternating-offers bargaining (Ståhl 1972).

    Returns s where s[t] (t = 1..T, index 0 unused) is the fraction of the pot the proposer of
    round t keeps in equilibrium. A deal in round t is worth amount * delta_i ** (t - 1) to
    player i, and no deal after round T is worth 0.

    Backward induction: in round T the responder accepts anything, so s[T] = 1. In round t < T
    the responder j (who proposes in round t + 1) accepts x iff x >= delta_j * s[t + 1], so the
    proposer offers exactly that and keeps s[t] = 1 - delta_j * s[t + 1].
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    deltas = {1: delta_1, 2: delta_2}
    s = [0.0] * (max_rounds + 2)
    s[max_rounds] = 1.0
    for t in range(max_rounds - 1, 0, -1):
        s[t] = 1.0 - deltas[proposer(t + 1)] * s[t + 1]
    return tuple(s[: max_rounds + 1])


def spne_reservation_share(round_number: int, max_rounds: int, delta_1: float, delta_2: float) -> float:
    """The smallest share a rational responder accepts in this round (0 in the last round)."""
    if round_number >= max_rounds:
        return 0.0
    s = spne_proposer_shares(max_rounds, delta_1, delta_2)
    responder = 3 - proposer(round_number)
    return (delta_1 if responder == 1 else delta_2) * s[round_number + 1]


def negotiation_spne_price(max_rounds: int, seller_value: float, buyer_value: float) -> float | None:
    """Negotiation has no discounting, so whoever proposes last captures the whole surplus.

    Seller (player 1) proposes in odd rounds. If the last round is the seller's, the equilibrium
    price is the buyer's value; otherwise it is the seller's value. Returns None if there is no
    zone of agreement (buyer_value < seller_value), in which case both should use John.
    """
    if buyer_value < seller_value:
        return None
    return buyer_value if proposer(max_rounds) == 1 else seller_value


def persuasion_threshold(v: float, c: float) -> float:
    """Posterior probability of high quality at which buying at price 1 breaks even.

    Values are in units of the price: a high-quality product is worth v >= 1, a low-quality one
    c <= 1. Buying is worth q * v + (1 - q) * c - 1 >= 0  <=>  q >= (1 - c) / (v - c).
    """
    if v <= c:
        return 1.0
    return min(1.0, max(0.0, (1.0 - c) / (v - c)))


def persuasion_commitment_rate(p: float, v: float, c: float) -> float:
    """Kamenica & Gentzkow (2011) optimal signal: how often to recommend a low-quality product.

    Recommend every high-quality product, and a low-quality one with probability pi chosen so
    that a Bayesian buyer who sees "recommend" is exactly indifferent:
        p / (p + (1 - p) * pi) = q*   =>   pi = p (1 - q*) / ((1 - p) q*).
    If the prior already clears the threshold (p >= q*), recommend everything (pi = 1).
    """
    q = persuasion_threshold(v, c)
    if p >= q:
        return 1.0
    if q <= 0 or p <= 0:
        return 0.0
    return min(1.0, p * (1.0 - q) / ((1.0 - p) * q))
