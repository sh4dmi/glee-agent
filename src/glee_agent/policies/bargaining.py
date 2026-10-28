"""Bargaining: split a pot over alternating offers while money loses value every round.

Default policy ("best_response"): dynamic programming against the fitted model of LLM opponents.
LLM players react to the last offer on the table, so the state carries it. With the pot
normalised to 1, our discount factor d, and s the last offer made (binned to 10%):

    we propose (s = their last offer to us):
        Vp(t, s) = max_x  A_t(x | s) * (1 - x) * d**(t-1) + (1 - A_t(x | s)) * Vr(t+1, x)
    they propose (s = our last offer to them):
        Vr(t, s) = sum_y  Q_t(y | s) * max(y * d**(t-1), Vp(t+1, y))

x is the share we offer, A_t the probability they accept it and Q_t the distribution of their
counteroffers. Nothing is left after the last round. In long games the recursion looks ahead
LOOKAHEAD rounds and treats the end of that window as the end of the game (a conservative,
receding horizon). Two guards keep the optimiser honest: it only considers offers that LLM
proposers make in the same situation (the response model is unreliable far from its data), and
it never takes back a concession.

When the opponent's discount factor is hidden, a posterior over a grid of candidates is updated
from their play (their offers, and their rejections of ours), and the action maximises the
posterior-expected value.

"spne" and "fair" are reference policies used by the evaluation.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..features import bargaining_features
from ..opponent import NO_HISTORY, OpponentModels, nearest, round_bucket
from ..protocol import BargainingTurn, lower_name
from ..theory import proposer, spne_reservation_share
from . import Decision, posterior

GRID = np.round(np.linspace(0.0, 1.0, 101), 2)       # shares of the pot we may offer, in 1% steps
STATES = np.round(np.arange(0.0, 1.0001, 0.1), 1)     # the binned "last offer"
NONE = len(STATES)                                     # state index for "no offer yet"
MIN_SUPPORT = 0.01   # an offer must carry this much mass (per 5% bin) among LLM proposals in the situation
LOOKAHEAD = 12


def state_index(share):
    return np.clip(np.round(np.asarray(share, dtype=float) * 10).astype(int), 0, len(STATES) - 1)


class BargainingPolicy:
    def __init__(self, models: OpponentModels, mode: str = "best_response"):
        if mode not in ("best_response", "spne", "fair"):
            raise ValueError(f"unknown bargaining mode {mode!r}")
        self.models, self.mode = models, mode
        self.centers = models.bargaining_offer.centers
        self._center_state = state_index(self.centers)
        self._grid_state = state_index(GRID)
        self._grid_bin = np.abs(GRID[:, None] - self.centers[None, :]).argmin(axis=1)
        self._accept = lru_cache(maxsize=4096)(self._accept_table)
        self._counter = lru_cache(maxsize=4096)(self._counter_table)
        self._support = lru_cache(maxsize=4096)(self._support_table)

    # ------------------------------------------------------------------ opponent model, cached per round
    def _accept_table(self, t, T, me, d_me, d_rival, ci, msgs, money) -> np.ndarray:
        """A[s, i] = P(rival accepts GRID[i] in round t | they last offered us STATES[s], or nothing)."""
        demand = np.append(1 - STATES, np.nan)  # what they last asked for themselves
        X = bargaining_features(np.tile(GRID, len(demand)), t, T, d_rival, d_me, me == 1, ci, msgs, money,
                                last_demand=np.repeat(demand, len(GRID)))
        return self.models.bargaining_accept.predict_proba(X)[:, 1].reshape(len(demand), len(GRID))

    def _offers_table(self, t, T, d_proposer, d_responder, ci) -> np.ndarray:
        grid = self.models.delta_grid
        return np.array([self.models.bargaining_offer.predict(
            ci=bool(ci), rb=round_bucket(t, T), prev_bin=prev, delta_proposer=nearest(d_proposer, grid),
            d_resp_known=nearest(d_responder, grid) if ci else -1.0) for prev in list(STATES) + [NO_HISTORY]])

    def _counter_table(self, t, T, d_me, d_rival, ci) -> np.ndarray:
        """Q[s] = distribution of the rival's offer to us in round t, after our last offer STATES[s]."""
        return self._offers_table(t, T, d_rival, d_me, ci)

    def _support_table(self, t, T, d_me, d_rival, ci) -> np.ndarray:
        """S[s, i] = True if LLMs in our seat offer about GRID[i] after being offered STATES[s]."""
        return self._offers_table(t, T, d_me, d_rival, ci)[:, self._grid_bin] >= MIN_SUPPORT

    # ------------------------------------------------------------------ dynamic programming
    def _solve(self, turn: BargainingTurn, d_rival: float):
        """Value tables Vp, Vr (round -> value per state) for the rounds after the current one."""
        T, me, d = turn.max_rounds, turn.me, turn.my_delta
        ci, msgs, money = turn.complete_information, turn.messages_allowed, turn.money
        end = T if T - turn.round < 2 * LOOKAHEAD else turn.round + LOOKAHEAD - 1
        zeros = np.zeros(len(STATES) + 1)
        Vp, Vr = {end + 1: zeros}, {end + 1: zeros}
        for t in range(end, turn.round, -1):
            disc = d ** (t - 1)
            if proposer(t) == me:
                A = self._accept(t, T, me, d, d_rival, ci, msgs, money)
                S = self._support(t, T, d, d_rival, ci)
                u = A * (1 - GRID) * disc + (1 - A) * Vr[t + 1][self._grid_state]
                u = np.where(S | ~S.any(axis=1, keepdims=True), u, -1e9)
                Vp[t], Vr[t] = u.max(axis=1), zeros
            else:
                Q = self._counter(t, T, d, d_rival, ci)
                Vr[t], Vp[t] = Q @ np.maximum(self.centers * disc, Vp[t + 1][self._center_state]), zeros
        return Vp, Vr

    def _proposal_values(self, turn: BargainingTurn, d_rival: float) -> np.ndarray:
        """Expected value of offering each GRID share now (-1e9 where not allowed)."""
        T, me, d, t = turn.max_rounds, turn.me, turn.my_delta, turn.round
        _, Vr = self._solve(turn, d_rival)
        last = self._their_last_offer(turn)
        X = bargaining_features(GRID, t, T, d_rival, d, me == 1, turn.complete_information, turn.messages_allowed,
                                turn.money, last_demand=np.nan if last is None else 1 - last)
        A = self.models.bargaining_accept.predict_proba(X)[:, 1]
        cont = Vr[t + 1][self._grid_state] if t < T else 0.0
        u = A * (1 - GRID) * d ** (t - 1) + (1 - A) * cont
        support = self._support(t, T, d, d_rival, turn.complete_information)
        allowed = support[NONE if last is None else int(state_index(last))]
        floor = max((share for _, share in turn.my_offers), default=0.0)  # never take back a concession
        allowed = allowed & (GRID >= floor - 1e-9)
        if not allowed.any():
            allowed = GRID >= floor - 1e-9
        return np.where(allowed, u, -1e9)

    def _continuation(self, turn: BargainingTurn, d_rival: float, offer: float) -> float:
        """Value of rejecting `offer` now and proposing next round."""
        if turn.round >= turn.max_rounds:
            return 0.0
        Vp, _ = self._solve(turn, d_rival)
        return float(Vp[turn.round + 1][int(state_index(offer))])

    @staticmethod
    def _their_last_offer(turn: BargainingTurn):
        offers = sorted(turn.rival_offers) + ([(turn.round, turn.offer_to_me)] if turn.offer_to_me is not None else [])
        return offers[-1][1] if offers else None

    # ------------------------------------------------------------------ beliefs about the rival
    def belief(self, turn: BargainingTurn) -> dict:
        if turn.rival_delta is not None:
            return {turn.rival_delta: 1.0}
        T, me, d = turn.max_rounds, turn.me, turn.my_delta
        timeline = sorted([(r, "me", s) for r, s in turn.my_offers] + [(r, "them", s) for r, s in turn.rival_offers]
                          + ([(turn.round, "them", turn.offer_to_me)] if turn.offer_to_me is not None else []))
        loglik = {}
        for cand in self.models.delta_grid:
            ll, mine, theirs = 0.0, None, None
            for rnd, who, share in timeline:
                r = min(rnd, T)
                if who == "me":   # they rejected this offer of ours
                    X = bargaining_features(share, r, T, cand, d, me == 1, turn.complete_information,
                                            turn.messages_allowed, turn.money,
                                            last_demand=np.nan if theirs is None else 1 - theirs)
                    ll += math.log(max(1e-6, 1 - self.models.bargaining_accept.predict_proba(X)[0, 1]))
                    mine = share
                else:             # they offered us this, after our last offer
                    Q = self._counter(r, T, d, cand, turn.complete_information)
                    s = NONE if mine is None else int(state_index(mine))
                    ll += math.log(max(1e-6, Q[s, int(np.abs(self.centers - share).argmin())]))
                    theirs = share
            loglik[cand] = ll
        return posterior(loglik)

    # ------------------------------------------------------------------ decisions
    def decide(self, turn: BargainingTurn) -> Decision:
        if self.mode == "best_response":
            share, accept, info = self._best_response(turn)
        elif self.mode == "spne":
            share, accept, info = self._spne(turn)
        else:
            share, accept, info = 0.5, (turn.offer_to_me or 0) >= 0.5 or turn.round >= turn.max_rounds, {}

        if turn.action == "respond":
            decision = "accept" if accept else "reject"
            return Decision({"decision": decision}, None,
                            f"round {turn.round}: offered {turn.offer_to_me:.0%}, {decision}", info)

        rival_amount = int(round(share * turn.money))
        mine = turn.money - rival_amount
        mine = int(mine) if float(mine).is_integer() else mine
        action = {f"{lower_name(turn.my_name)}_gain": mine, f"{lower_name(turn.rival_name)}_gain": rival_amount}
        draft = _proposal_message(turn, rival_amount) if turn.messages_allowed else None
        return Decision(action, draft, f"round {turn.round}: offer rival {share:.0%}", info)

    def _best_response(self, turn: BargainingTurn):
        belief = self.belief(turn)
        if turn.action == "respond":
            cont = sum(w * self._continuation(turn, cand, turn.offer_to_me) for cand, w in belief.items())
            now = turn.offer_to_me * turn.my_delta ** (turn.round - 1)
            return None, now >= cont - 1e-12, {"value_now": now, "value_if_reject": cont, "belief": belief}
        expected = sum(w * self._proposal_values(turn, cand) for cand, w in belief.items())
        i = int(np.argmax(expected))
        return float(GRID[i]), None, {"expected_value": float(expected[i]), "belief": belief}

    def _spne(self, turn: BargainingTurn):
        rival_delta = turn.rival_delta if turn.rival_delta is not None else turn.my_delta
        d1, d2 = (turn.my_delta, rival_delta) if turn.me == 1 else (rival_delta, turn.my_delta)
        reservation = spne_reservation_share(turn.round, turn.max_rounds, d1, d2)
        if turn.action == "respond":
            return None, turn.offer_to_me >= reservation - 1e-9, {"reservation": reservation}
        return float(min(1.0, math.ceil(reservation * 100 - 1e-9) / 100)), None, {"reservation": reservation}


def _proposal_message(turn: BargainingTurn, rival_amount: int) -> str:
    left = turn.max_rounds - turn.round
    pressure = ("Every round we wait, the money is worth less." if turn.my_delta < 1 or (turn.rival_delta or 1) < 1
                else "Let's not drag this out.")
    if left <= 1:
        pressure = "This is our last chance to agree before we both walk away with nothing."
    return f"I'm offering you ${rival_amount:,}. {pressure} Accept and we both lock in our share now."
