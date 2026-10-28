"""Negotiation: a seller and a buyer trade one product; either can walk away to John at any time.

Payoffs are surplus over the John deal: price - seller_value for the seller, buyer_value - price
for the buyer (all in units of GLEE's product_price_order). There is no discounting, so only the
deadline creates pressure.

Default policy ("best_response"): dynamic programming against the fitted models of LLM players,
with the last price on the table as state (binned to 0.1):

    we propose p (s = their last price):
        Vp(t, s) = max_p  a_t(p | s) surplus(p) + (1 - a_t(p | s)) (1 - j_t(p | s)) Vr(t+1, p)
    they propose (s = our last price):
        Vr(t, s) = E_y[ max(surplus(y), 0, Vp(t+1, y)) ]          accept / John / counter

a_t is the probability the opponent accepts, j_t the probability they go to John instead of
countering, and y is drawn from the fitted model of their counteroffers. After the last round
nothing is left. As in bargaining: a receding LOOKAHEAD in long games, prices restricted to what
LLMs in our seat actually propose, no taking back concessions, and a posterior over the
opponent's value when it is hidden.

"spne" (last proposer takes the whole surplus) and "split" (meet in the middle) are reference
policies for the evaluation.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..features import negotiation_features
from ..opponent import NO_HISTORY, OpponentModels, last_offer_bin, round_bucket
from ..protocol import NegotiationTurn
from ..theory import negotiation_spne_price, proposer
from . import Decision, posterior

PRICE_GRID = np.round(np.arange(0.0, 3.0001, 0.02), 2)   # prices we may propose (units of price_order)
STATES = np.round(np.arange(0.0, 3.0001, 0.1), 1)        # the binned "last price"
NONE = len(STATES)                                        # state index for "no price yet"
MIN_SUPPORT = 0.01
LOOKAHEAD = 12


def state_index(price):
    return np.clip(np.round(np.asarray(price, dtype=float) * 10).astype(int), 0, len(STATES) - 1)


class NegotiationPolicy:
    def __init__(self, models: OpponentModels, mode: str = "best_response"):
        if mode not in ("best_response", "spne", "split"):
            raise ValueError(f"unknown negotiation mode {mode!r}")
        self.models, self.mode = models, mode
        self.ratios = models.negotiation_offer.centers   # prices as a ratio to the proposer's own value
        self._grid_state = state_index(PRICE_GRID)
        self._response = lru_cache(maxsize=4096)(self._response_table)
        self._offers = lru_cache(maxsize=4096)(self._offers_table)

    # ---------------------------------------------------------------- opponent models, cached per round
    def _response_table(self, t, T, role, sv, bv, ci, msgs, order):
        """(a, j)[s, i]: rival accepts / goes to John on PRICE_GRID[i], given their last price STATES[s]."""
        rival_role = "buyer" if role == "seller" else "seller"
        last = np.append(STATES, np.nan)
        X = negotiation_features(np.tile(PRICE_GRID, len(last)), rival_role, sv, bv, t, T, ci, msgs, order,
                                 responder_last_price=np.repeat(last, len(PRICE_GRID)))
        shape = (len(last), len(PRICE_GRID))
        a = self.models.negotiation_accept[rival_role].predict_proba(X)[:, 1].reshape(shape)
        j = self.models.negotiation_john[rival_role].predict_proba(X)[:, 1].reshape(shape)
        return a, j

    def _offers_table(self, t, T, proposer_role, proposer_value, other_value, ci):
        """P[s] = distribution of the proposer's price / proposer_value, after being offered STATES[s]."""
        prev = [last_offer_bin(p / proposer_value) for p in STATES] + [NO_HISTORY]
        return np.array([self.models.negotiation_offer.predict(
            proposer_role=proposer_role, ci=bool(ci), one_shot=T == 1, rb=round_bucket(t, T), prev_bin=b,
            other_known=other_value if ci else -1.0) for b in prev])

    def _support(self, t, T, turn: NegotiationTurn, rival_value) -> np.ndarray:
        """S[s, i] = True if LLMs in our seat propose about PRICE_GRID[i] after being offered STATES[s]."""
        P = self._offers(t, T, turn.role, turn.my_value, rival_value, turn.complete_information)
        bins = np.abs((PRICE_GRID / max(turn.my_value, 1e-9))[:, None] - self.ratios[None, :]).argmin(axis=1)
        return P[:, bins] >= MIN_SUPPORT

    def _surplus(self, turn: NegotiationTurn, price):
        return price - turn.seller_value if turn.role == "seller" else turn.buyer_value - price

    @staticmethod
    def _values(turn: NegotiationTurn, rival_value):
        return (turn.seller_value, rival_value) if turn.role == "seller" else (rival_value, turn.buyer_value)

    # ---------------------------------------------------------------- dynamic programming
    def _solve(self, turn: NegotiationTurn, rival_value: float):
        """Value tables Vp, Vr (round -> value per state) for the rounds after the current one."""
        T, role = turn.max_rounds, turn.role
        me = 1 if role == "seller" else 2
        rival_role = "buyer" if role == "seller" else "seller"
        sv, bv = self._values(turn, rival_value)
        end = T if T - turn.round < 2 * LOOKAHEAD else turn.round + LOOKAHEAD - 1
        zeros = np.zeros(len(STATES) + 1)
        surplus = self._surplus(turn, PRICE_GRID)
        their_prices = self.ratios * rival_value
        their_gain = np.maximum(self._surplus(turn, their_prices), 0.0)  # accept, or John at 0
        their_state = state_index(their_prices)
        Vp, Vr = {end + 1: zeros}, {end + 1: zeros}
        for t in range(end, turn.round, -1):
            last_round = t >= T
            if proposer(t) == me:
                a, j = self._response(t, T, role, sv, bv, turn.complete_information, turn.messages_allowed,
                                      turn.price_order)
                cont = 0.0 if last_round else Vr[t + 1][self._grid_state]
                u = a * surplus + (1 - a) * (1 - j) * cont
                ok = self._support(t, T, turn, rival_value) & (surplus >= 0)
                ok = ok | (~ok.any(axis=1, keepdims=True) & (surplus >= 0))
                Vp[t], Vr[t] = np.maximum(np.where(ok, u, -1e9).max(axis=1), 0.0), zeros
            else:
                P = self._offers(t, T, rival_role, rival_value, turn.my_value, turn.complete_information)
                cont = 0.0 if last_round else Vp[t + 1][their_state]
                Vr[t], Vp[t] = P @ np.maximum(their_gain, cont), zeros
        return Vp, Vr

    def _proposal_values(self, turn: NegotiationTurn, rival_value: float) -> np.ndarray:
        """Expected surplus of proposing each PRICE_GRID price now (-1e9 where not allowed)."""
        t, T, role = turn.round, turn.max_rounds, turn.role
        rival_role = "buyer" if role == "seller" else "seller"
        sv, bv = self._values(turn, rival_value)
        _, Vr = self._solve(turn, rival_value)
        last = turn.rival_offers[-1][1] if turn.rival_offers else None
        X = negotiation_features(PRICE_GRID, rival_role, sv, bv, t, T, turn.complete_information,
                                 turn.messages_allowed, turn.price_order,
                                 responder_last_price=np.nan if last is None else last)
        a = self.models.negotiation_accept[rival_role].predict_proba(X)[:, 1]
        j = self.models.negotiation_john[rival_role].predict_proba(X)[:, 1]
        surplus = self._surplus(turn, PRICE_GRID)
        cont = 0.0 if t >= T else Vr[t + 1][self._grid_state]
        u = a * surplus + (1 - a) * (1 - j) * cont
        rational = surplus >= 0
        previous = [p for _, p in turn.my_offers]
        if previous:  # never take back a concession
            rational &= (PRICE_GRID <= min(previous) + 1e-9) if role == "seller" else (PRICE_GRID >= max(previous) - 1e-9)
        support = self._support(t, T, turn, rival_value)[NONE if last is None else int(state_index(last))]
        allowed = rational & support
        if not allowed.any():
            allowed = rational
        return np.where(allowed, u, -1e9)

    def _continuation(self, turn: NegotiationTurn, rival_value: float) -> float:
        """Value of rejecting the current offer and countering next round."""
        if turn.round >= turn.max_rounds:
            return 0.0
        Vp, _ = self._solve(turn, rival_value)
        return float(Vp[turn.round + 1][int(state_index(turn.offer_price))])

    # ---------------------------------------------------------------- beliefs
    def belief(self, turn: NegotiationTurn) -> dict:
        if turn.rival_value is not None:
            return {turn.rival_value: 1.0}
        T, role = turn.max_rounds, turn.role
        rival_role = "buyer" if role == "seller" else "seller"
        width = self.ratios[1] - self.ratios[0]
        timeline = sorted([(r, "me", p) for r, p in turn.my_offers] + [(r, "them", p) for r, p in turn.rival_offers]
                          + ([(turn.round, "them", turn.offer_price)] if turn.offer_price is not None else []))
        loglik = {}
        for cand in self.models.value_grid:
            sv, bv = self._values(turn, cand)
            ll, mine, theirs = 0.0, None, None
            for rnd, who, price in timeline:
                r = min(rnd, T)
                if who == "me":   # rejected, and they chose to counter rather than go to John
                    X = negotiation_features(price, rival_role, sv, bv, r, T, turn.complete_information,
                                             turn.messages_allowed, turn.price_order,
                                             responder_last_price=np.nan if theirs is None else theirs)
                    a = self.models.negotiation_accept[rival_role].predict_proba(X)[0, 1]
                    j = self.models.negotiation_john[rival_role].predict_proba(X)[0, 1]
                    ll += math.log(max(1e-6, (1 - a) * (1 - j)))
                    mine = price
                else:             # their price, as a density in price units
                    P = self._offers(r, T, rival_role, cand, turn.my_value, turn.complete_information)
                    s = NONE if mine is None else int(state_index(mine))
                    k = int(np.abs(self.ratios - price / cand).argmin())
                    ll += math.log(max(1e-6, P[s, k] / (width * cand)))
                    theirs = price
            loglik[cand] = ll
        return posterior(loglik)

    # ---------------------------------------------------------------- decisions
    def decide(self, turn: NegotiationTurn) -> Decision:
        john = "SellToJhon" if turn.role == "seller" else "BuyFromJhon"
        if turn.action == "respond":
            gain = self._surplus(turn, turn.offer_price)
            cont, info = self._continuation_value(turn)
            if gain > 0 and gain >= cont - 1e-12:
                choice = "AcceptOffer"
            elif cont > max(gain, 0.0) + 1e-12 and turn.round < turn.max_rounds:
                choice = "RejectOffer"
            else:
                choice = john
            info.update(gain_now=gain, value_if_counter=cont)
            return Decision({"decision": choice}, None,
                            f"round {turn.round}: offered {turn.offer_price:.3f}, {choice}", info)

        price, info = self._propose(turn)
        amount = price * turn.price_order
        amount = int(round(amount)) if turn.price_order >= 100 else round(amount, 2)
        draft = _proposal_message(turn, amount) if turn.messages_allowed else None
        return Decision({"product_price": amount}, draft, f"round {turn.round}: propose {price:.3f}", info)

    def _continuation_value(self, turn: NegotiationTurn):
        if turn.round >= turn.max_rounds:
            return 0.0, {}
        if self.mode == "best_response":
            belief = self.belief(turn)
            return sum(w * self._continuation(turn, v) for v, w in belief.items()), {"belief": belief}
        target = self._reference_price(turn)
        return max(0.0, self._surplus(turn, target)), {"reference_price": target}

    def _reference_price(self, turn: NegotiationTurn) -> float:
        rival = turn.rival_value if turn.rival_value is not None else float(np.mean(self.models.value_grid))
        sv, bv = self._values(turn, rival)
        if self.mode == "spne":
            p = negotiation_spne_price(turn.max_rounds, sv, bv)
            return turn.my_value if p is None else p
        return (sv + bv) / 2 if bv >= sv else turn.my_value

    def _propose(self, turn: NegotiationTurn):
        if self.mode != "best_response":
            return self._reference_price(turn), {}
        belief = self.belief(turn)
        expected = sum(w * self._proposal_values(turn, v) for v, w in belief.items())
        i = int(np.argmax(expected))
        if expected[i] <= 1e-6:  # no price is expected to produce a deal: ask for break-even
            return turn.my_value, {"belief": belief, "expected_value": 0.0}
        return float(PRICE_GRID[i]), {"expected_value": float(expected[i]), "belief": belief}


def _proposal_message(turn: NegotiationTurn, amount) -> str:
    if turn.role == "seller":
        return f"I can sell it to you for ${amount:,}. It's a fair price for what you're getting."
    return f"I can pay ${amount:,} for it. That's a solid offer and I'm ready to close today."
