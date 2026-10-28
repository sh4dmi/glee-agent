"""A stand-in for an LLM opponent, sampled from models fitted on held-out games.

The evaluation never lets the agent play against the models it plans with. `fit_models` is run
on the 20% test games only, and this player samples its moves from those models: offers from the
proposal histograms, accept/reject from the response classifiers, and for persuasion the
recommendation rates and buying model of the test-split LLMs.
"""
from __future__ import annotations

import json
import random

import numpy as np

from glee_agent.features import bargaining_features, negotiation_features, persuasion_features
from glee_agent.opponent import NO_HISTORY, OpponentModels, last_offer_bin, nearest, round_bucket
from glee_agent.policies.persuasion import keyword_recommends
from glee_agent.protocol import (SELLER_DECLINE, SELLER_RECOMMEND, BargainingTurn, NegotiationTurn, PersuasionTurn,
                                 lower_name, parse_request)
from glee_agent.theory import persuasion_threshold


class LLMProxy:
    def __init__(self, models: OpponentModels, seed: int = 0):
        self.m = models
        self.rng = random.Random(seed)

    def __call__(self, payload: dict) -> str:
        turn = parse_request(payload)
        if isinstance(turn, BargainingTurn):
            return json.dumps(self._bargaining(turn))
        if isinstance(turn, NegotiationTurn):
            return json.dumps(self._negotiation(turn))
        return json.dumps(self._persuasion(turn))

    def _sample(self, probs: np.ndarray, centers: np.ndarray) -> float:
        i = self.rng.choices(range(len(probs)), weights=probs)[0]
        return float(centers[i])

    # ------------------------------------------------------------------------------ bargaining
    def _bargaining(self, t: BargainingTurn) -> dict:
        grid = self.m.delta_grid
        received = sorted(t.rival_offers + ([(t.round, t.offer_to_me)] if t.offer_to_me is not None else []))
        last_received = received[-1][1] if received else None
        if t.action == "respond":
            mine = sorted(t.my_offers)
            demand = 1 - mine[-1][1] if mine else np.nan  # what we (as the proxy) asked for last time
            rival_delta = t.rival_delta if t.rival_delta is not None else 1.0
            X = bargaining_features(t.offer_to_me, t.round, t.max_rounds, t.my_delta, rival_delta, t.me == 2,
                                    t.complete_information, t.messages_allowed, t.money, last_demand=demand)
            p = self.m.bargaining_accept.predict_proba(X)[0, 1]
            return {"decision": "accept" if self.rng.random() < p else "reject"}
        hist = self.m.bargaining_offer
        probs = hist.predict(ci=t.complete_information, rb=round_bucket(t.round, t.max_rounds),
                             prev_bin=NO_HISTORY if last_received is None else last_offer_bin(last_received),
                             delta_proposer=nearest(t.my_delta, grid),
                             d_resp_known=nearest(t.rival_delta, grid) if t.complete_information
                             and t.rival_delta is not None else -1.0)
        share = self._sample(probs, hist.centers)
        rival = int(round(share * t.money))
        out = {f"{lower_name(t.my_name)}_gain": t.money - rival, f"{lower_name(t.rival_name)}_gain": rival}
        for k, v in out.items():
            out[k] = int(v) if float(v).is_integer() else v
        if t.messages_allowed:
            out["message"] = "Here is my offer."
        return out

    # ----------------------------------------------------------------------------- negotiation
    def _negotiation(self, t: NegotiationTurn) -> dict:
        if t.action == "respond":
            mine = [p for _, p in t.my_offers]
            X = negotiation_features(t.offer_price, t.role, t.seller_value if t.seller_value is not None else 1.0,
                                     t.buyer_value if t.buyer_value is not None else 1.0, t.round, t.max_rounds,
                                     t.complete_information, t.messages_allowed, t.price_order,
                                     responder_last_price=mine[-1] if mine else np.nan)
            a = self.m.negotiation_accept[t.role].predict_proba(X)[0, 1]
            j = self.m.negotiation_john[t.role].predict_proba(X)[0, 1]
            r = self.rng.random()
            if r < a:
                return {"decision": "AcceptOffer"}
            if self.rng.random() < j or t.round >= t.max_rounds:
                return {"decision": "SellToJhon" if t.role == "seller" else "BuyFromJhon"}
            return {"decision": "RejectOffer"}
        last = t.rival_offers[-1][1] if t.rival_offers else None
        hist = self.m.negotiation_offer
        probs = hist.predict(proposer_role=t.role, ci=t.complete_information, one_shot=t.max_rounds == 1,
                             rb=round_bucket(t.round, t.max_rounds),
                             prev_bin=NO_HISTORY if last is None else last_offer_bin(last / t.my_value),
                             other_known=t.rival_value if t.complete_information and t.rival_value is not None
                             else -1.0)
        price = self._sample(probs, hist.centers) * t.my_value * t.price_order
        out = {"product_price": int(round(price)) if t.price_order >= 100 else round(price, 2)}
        if t.messages_allowed:
            out["message"] = "This is my price."
        return out

    # ------------------------------------------------------------------------------ persuasion
    def _persuasion(self, t: PersuasionTurn) -> dict:
        if t.role == "seller":
            rate = self.m.seller_recommend_rate["high" if t.high_quality else "low"]
            rec = self.rng.random() < rate
            if t.text_messages:
                return {"message": SELLER_RECOMMEND if rec else SELLER_DECLINE}
            return {"decision": "yes" if rec else "no"}
        rec = (t.seller_signal == "yes") if not t.text_messages else keyword_recommends(t.seller_signal or "")
        if t.myopic:
            count, high = t.sold or 0, t.sold_high or 0
        else:
            recs = [h for h in t.history if (h.signal == "yes") or (h.signal not in (None, "no")
                                                                    and keyword_recommends(h.signal))]
            count, high = len(recs), sum(bool(h.high_quality) for h in recs)
        X = persuasion_features(float(rec), high, count, persuasion_threshold(t.v, t.c), t.p if t.p is not None else 0.5,
                                t.myopic, max(t.round, 1), t.total_rounds)
        buy = self.rng.random() < self.m.persuasion_buy.predict_proba(X)[0, 1]
        out = {"decision": "yes" if buy else "no"}
        if t.buyer_messages:
            out["message"] = "ok"
        return out
