"""Persuasion: a seller who knows each product's quality sends a signal; the buyer decides to buy.

Prices and values are in units of the fixed price: a high-quality product is worth v >= 1 to the
buyer and a low-quality one c <= 1. A buyer whose belief that the product is high quality is q
should buy iff q >= q* = (1 - c) / (v - c).

Seller ("best_response"): maximise expected sales over the remaining rounds by dynamic programming
over the record the buyer can see, using the fitted model of how LLM buyers react to a
recommendation given that record. The signal is pooled: the same words for every recommended
product, so a recommendation never states a quality it doesn't have.

    state:  (round, n, k) = n products in the visible record, k of them high quality
    V(r, n, k) = E_quality[ max over {recommend, don't} of  P(buy) + E V(r + 1, next state) ]

Non-myopic buyer: the record is every past recommendation and its revealed quality.
Myopic buyers: GLEE shows each new buyer how many products were sold and how many were high.

Buyer ("best_response"): Bayesian. Keep Beta posteriors on how often this seller recommends
low- and high-quality products (priors from the LLM sellers in the GLEE data), turn the signal
into P(high quality | signal), and buy iff it clears q*.

"commitment" (Kamenica & Gentzkow's optimal signal), "honest" and "always" are reference sellers.
"""
from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from ..features import persuasion_features
from ..opponent import OpponentModels
from ..protocol import SELLER_DECLINE, SELLER_RECOMMEND, PersuasionTurn
from ..theory import persuasion_commitment_rate, persuasion_threshold
from . import Decision

PRIOR_STRENGTH = 4.0  # pseudo-observations behind the prior on the seller's behaviour


class PersuasionPolicy:
    def __init__(self, models: OpponentModels, mode: str = "best_response", rng: random.Random | None = None):
        if mode not in ("best_response", "commitment", "honest", "always"):
            raise ValueError(f"unknown persuasion mode {mode!r}")
        self.models, self.mode = models, mode
        self.rng = rng or random.Random(0)

    def decide(self, turn: PersuasionTurn, recommends: bool | None = None) -> Decision:
        """`recommends` is the interpreted seller message (buyer side, free-text games)."""
        return self._seller(turn) if turn.role == "seller" else self._buyer(turn, recommends)

    # ------------------------------------------------------------------------ seller
    def _seller(self, turn: PersuasionTurn) -> Decision:
        if self.mode == "best_response":
            recommend, info = self._seller_dp(turn)
        elif self.mode == "commitment":
            rate = persuasion_commitment_rate(turn.p or 0.5, turn.v or 2.0, turn.c or 0.0)
            recommend, info = bool(turn.high_quality) or self.rng.random() < rate, {"low_recommend_rate": rate}
        elif self.mode == "honest":
            recommend, info = bool(turn.high_quality), {}
        else:
            recommend, info = True, {}
        quality = "high" if turn.high_quality else "low"
        rationale = f"round {turn.round}: {quality} quality, {'recommend' if recommend else 'decline'}"
        if turn.text_messages:
            return Decision({}, SELLER_RECOMMEND if recommend else SELLER_DECLINE, rationale, info)
        return Decision({"decision": "yes" if recommend else "no"}, None, rationale, info)

    @lru_cache(maxsize=64)
    def _tables(self, R: int, p: float, thresholds: tuple, myopic: bool):
        """P(buy) for every (round, n, k) state with and without a recommendation, and the value V."""
        size = R + 2
        r_idx, n_idx, k_idx = np.meshgrid(np.arange(1, R + 1), np.arange(size), np.arange(size), indexing="ij")
        r_f, n_f, k_f = r_idx.ravel(), n_idx.ravel(), k_idx.ravel()
        buy = {}
        for rec in (1, 0):
            probs = [self.models.persuasion_buy.predict_proba(
                persuasion_features(np.full(r_f.shape, rec), k_f, n_f, th, p, myopic, r_f, R))[:, 1]
                for th in thresholds]
            buy[rec] = np.mean(probs, axis=0).reshape(R, size, size)
        V = np.zeros((R + 2, size + 1, size + 1))
        for r in range(R, 0, -1):
            for n in range(min(r, size - 1)):  # the record holds at most r - 1 past rounds
                for k in range(n + 1):
                    V[r, n, k] = p * max(self._option(V, buy, r, n, k, True, rec, myopic) for rec in (1, 0)) + \
                        (1 - p) * max(self._option(V, buy, r, n, k, False, rec, myopic) for rec in (1, 0))
        return V, buy

    def _seller_dp(self, turn: PersuasionTurn):
        R, p = turn.total_rounds, (turn.p if turn.p is not None else 0.5)
        if turn.v is not None and turn.c is not None:
            thresholds = (persuasion_threshold(turn.v, turn.c),)
        else:  # the seller may not know the buyer's values: average over GLEE's values
            thresholds = tuple(persuasion_threshold(v, 0.0) for v in self.models.value_grid_persuasion)
        V, buy = self._tables(R, round(p, 6), thresholds, bool(turn.myopic))
        n0, k0 = self._record(turn)
        options = {rec: self._option(V, buy, turn.round, n0, k0, bool(turn.high_quality), rec, turn.myopic)
                   for rec in (1, 0)}
        recommend = options[1] >= options[0]
        return recommend, {"expected_sales_if_recommend": options[1], "expected_sales_if_decline": options[0],
                           "record": (n0, k0)}

    @staticmethod
    def _option(V, buy, r, n, k, high, rec, myopic) -> float:
        n, k = min(n, buy[1].shape[1] - 1), min(k, buy[1].shape[2] - 1)
        b = buy[rec][r - 1, n, k]
        if myopic:  # the visible record is sales: it only moves when the buyer buys
            n2, k2 = n + 1, k + int(high)
            return b * (1 + V[r + 1, n2, k2]) + (1 - b) * V[r + 1, n, k]
        if rec:      # the visible record is recommendations and their revealed quality
            return b + V[r + 1, n + 1, k + int(high)]
        return b + V[r + 1, n, k]

    def _record(self, turn: PersuasionTurn) -> tuple[int, int]:
        if turn.myopic:
            sold = [h for h in turn.history if h.bought]
            return len(sold), sum(bool(h.high_quality) for h in sold)
        recs = [h for h in turn.history if h.recommended]
        return len(recs), sum(bool(h.high_quality) for h in recs)

    # ------------------------------------------------------------------------- buyer
    def _buyer(self, turn: PersuasionTurn, recommends: bool | None) -> Decision:
        if recommends is None:
            recommends = turn.seller_signal == "yes"
        v = turn.v if turn.v is not None else 2.0
        c = turn.c if turn.c is not None else 0.0
        threshold = persuasion_threshold(v, c)
        belief = self.high_quality_belief(turn, recommends)
        buy = belief >= threshold
        explore = 0.0
        if turn.myopic and recommends and turn.sold is not None and not buy:
            # Myopic buyers only learn about the seller from products that were bought, and every
            # "new buyer" is us: passing forever teaches us nothing. Buying has information value.
            explore = self._exploration_value(turn, v, c)
            buy = explore > 0
        action = {"decision": "yes" if buy else "no"}
        draft = None
        if turn.buyer_messages:
            draft = "I'll take it." if buy else "I'll pass on this one."
        return Decision(action, draft, f"P(high | signal) = {belief:.2f} vs threshold {threshold:.2f}: "
                                       f"{'buy' if buy else 'pass'}",
                        {"belief": belief, "threshold": threshold, "recommends": recommends,
                         "exploration_value": explore})

    def _exploration_value(self, turn: PersuasionTurn, v: float, c: float) -> float:
        """Value of buying now for a myopic seat, as a one-armed Bernoulli bandit over the rounds left.

        theta = P(high | recommended) has a Beta posterior (prior from the LLM-seller rates, updated
        with GLEE's sales statistics). Buying pays v - 1 or c - 1 and reveals one more outcome;
        passing pays 0 and reveals nothing, so once passing is optimal it stays optimal.
        """
        rates = self.models.seller_recommend_rate
        p = turn.p if turn.p is not None else 0.5
        m0 = p * rates["high"] / (p * rates["high"] + (1 - p) * rates["low"])
        a0, b0 = PRIOR_STRENGTH * m0 + turn.sold_high, PRIOR_STRENGTH * (1 - m0) + turn.sold - turn.sold_high
        rec_rate = p * rates["high"] + (1 - p) * rates["low"]
        # the buyer sees sales, not the round: rounds left are at most total - sold
        horizon = max(1, int(round((turn.total_rounds - turn.sold) * rec_rate)))

        @lru_cache(maxsize=None)
        def value(h: int, hi: int, lo: int) -> float:
            if h == 0:
                return 0.0
            q = (a0 + hi) / (a0 + b0 + hi + lo)
            buy = q * (v - 1 + value(h - 1, hi + 1, lo)) + (1 - q) * (c - 1 + value(h - 1, hi, lo + 1))
            return max(0.0, buy)

        q = a0 / (a0 + b0)
        return q * (v - 1 + value(horizon - 1, 1, 0)) + (1 - q) * (c - 1 + value(horizon - 1, 0, 1))

    def high_quality_belief(self, turn: PersuasionTurn, recommends: bool) -> float:
        rates = self.models.seller_recommend_rate
        known = [h for h in turn.history if h.high_quality is not None]
        if turn.p is not None:
            p = turn.p
        else:  # estimate the share of high-quality products from what we've seen (Beta(1, 1) prior)
            p = (1 + sum(h.high_quality for h in known)) / (2 + len(known))

        if turn.myopic and turn.sold is not None and recommends:
            # products sold are (almost all) recommended ones: their quality mix estimates P(high | recommend)
            prior = p * rates["high"] / (p * rates["high"] + (1 - p) * rates["low"])
            return (turn.sold_high + PRIOR_STRENGTH * prior) / (turn.sold + PRIOR_STRENGTH)

        # Beta posteriors on P(recommend | low) and P(recommend | high) from this seller's history
        lows = [h for h in known if not h.high_quality]
        highs = [h for h in known if h.high_quality]
        rec = [self._recommended(h) for h in known]
        low_rec = sum(r for h, r in zip(known, rec) if not h.high_quality)
        high_rec = sum(r for h, r in zip(known, rec) if h.high_quality)
        pi = (PRIOR_STRENGTH * rates["low"] + low_rec) / (PRIOR_STRENGTH + len(lows))
        rho = (PRIOR_STRENGTH * rates["high"] + high_rec) / (PRIOR_STRENGTH + len(highs))
        if recommends:
            num, den = p * rho, p * rho + (1 - p) * pi
        else:
            num, den = p * (1 - rho), p * (1 - rho) + (1 - p) * (1 - pi)
        return num / den if den > 0 else p

    @staticmethod
    def _recommended(record) -> bool:
        if record.recommended is not None:
            return record.recommended
        return record.signal == "yes" or (record.signal not in (None, "no") and not _negative(record.signal))


NEGATIVE = ("not recommend", "don't recommend", "do not recommend", "wouldn't recommend", "would not recommend",
            "low-quality", "low quality", "skip this", "pass on this", "not a good", "can't recommend",
            "cannot recommend")


def _negative(text: str) -> bool:
    text = (text or "").lower()
    return any(phrase in text for phrase in NEGATIVE)


def keyword_recommends(text: str) -> bool:
    """Fallback interpretation of a free-text seller message when no LLM is configured."""
    return not _negative(text)
