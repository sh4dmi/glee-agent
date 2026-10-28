"""Feature definitions for the opponent models.

The same functions build the training matrix (scripts/train_models.py, from the GLEE logs) and the
inputs at play time (the policies), so the two can never drift apart. Every function is vectorised
over numpy arrays and also works on scalars.
"""
from __future__ import annotations

import numpy as np

from .theory import spne_reservation_share

ROUND_CAP = 12  # GLEE's long games run to 99 rounds; beyond 12 the data is too thin to tell rounds apart


def _cap_rounds(round_number, max_rounds):
    round_number = np.asarray(round_number, dtype=float)
    max_rounds = np.asarray(max_rounds, dtype=float)
    return (np.minimum(round_number, ROUND_CAP),
            np.minimum(max_rounds - round_number, ROUND_CAP),
            (round_number >= max_rounds).astype(float))


# ------------------------------------------------------------------------------------------ bargaining
BARGAINING_FEATURES = [
    "share", "share_below_half", "share_above_half", "at_least_half",
    "spne_gap", "round", "rounds_left", "last_round", "delta_responder", "delta_proposer_known",
    "complete_information", "messages_allowed", "log_money", "gap_to_last_demand",
]


def bargaining_features(share, round_number, max_rounds, delta_responder, delta_proposer,
                        proposer_is_p1, complete_information, messages_allowed, money,
                        last_demand=np.nan) -> np.ndarray:
    """Features for P(responder accepts | an offer giving them `share` of the pot).

    last_demand is the share the responder asked for themselves in the previous round (NaN in
    round 1); LLM responders accept far more often once an offer reaches what they last asked for.
    """
    share = np.atleast_1d(np.asarray(share, dtype=float))
    n = share.shape[0]

    def b(x):
        return np.broadcast_to(np.asarray(x, dtype=float), (n,))

    rnd, left, last = (b(x) for x in _cap_rounds(round_number, max_rounds))
    d_r, d_p = b(delta_responder), b(delta_proposer)
    p1 = b(proposer_is_p1).astype(bool)
    ci = b(complete_information)
    rounds, horizons = b(round_number), b(max_rounds)
    # what a perfectly rational responder could secure by rejecting (exact SPNE continuation)
    reservation = np.array([
        spne_reservation_share(int(t), int(T), dp if is_p1 else dr, dr if is_p1 else dp)
        for t, T, dr, dp, is_p1 in zip(rounds, horizons, d_r, d_p, p1)
    ])
    return np.column_stack([
        share,
        np.minimum(share, 0.5),
        np.maximum(share - 0.5, 0.0),
        (share >= 0.5 - 1e-9).astype(float),  # the 50/50 focal point, as a step so the model stays monotone
        share - reservation,
        rnd, left, last,
        d_r,
        d_p * ci,  # the responder only knows the proposer's inflation under complete information
        ci,
        b(messages_allowed),
        np.log10(b(money)),
        share - b(last_demand),
    ])


# ----------------------------------------------------------------------------------------- negotiation
NEGOTIATION_FEATURES = [
    "responder_gain", "gain_negative", "gain_positive", "no_loss", "price_over_order",
    "responder_value", "proposer_value_known", "surplus_share", "round", "rounds_left", "last_round",
    "one_shot", "complete_information", "messages_allowed", "log_money", "gain_vs_own_last_offer",
]


def negotiation_features(price, responder_role, seller_value, buyer_value, round_number, max_rounds,
                         complete_information, messages_allowed, price_order,
                         responder_last_price=np.nan) -> np.ndarray:
    """Features for the responder's decision on `price` (all values in units of price_order).

    responder_role is "buyer" when the seller proposed and "seller" when the buyer proposed.
    responder_last_price is the price the responder themselves proposed last round (NaN if none).
    """
    price = np.atleast_1d(np.asarray(price, dtype=float))
    n = price.shape[0]

    def b(x):
        return np.broadcast_to(np.asarray(x, dtype=float), (n,))

    is_buyer = b(np.asarray(responder_role) == "buyer")
    vs, vb = b(seller_value), b(buyer_value)
    own = np.where(is_buyer > 0, vb, vs)
    other = np.where(is_buyer > 0, vs, vb)
    # the responder's gain over dealing with John, relative to their own value
    gain = np.where(is_buyer > 0, vb - price, price - vs) / np.maximum(own, 1e-9)
    ci = b(complete_information)
    zopa = np.maximum(vb - vs, 1e-9)
    surplus_share = np.where(is_buyer > 0, vb - price, price - vs) / zopa
    surplus_share = np.clip(surplus_share, -2.0, 3.0) * ci * (vb > vs)
    rnd, left, last = (b(x) for x in _cap_rounds(round_number, max_rounds))
    return np.column_stack([
        gain,
        np.minimum(gain, 0.0),
        np.maximum(gain, 0.0),
        (gain >= -1e-9).astype(float),  # no worse than John: a step, not a spike, to keep monotonicity
        price,
        own,
        other * ci,
        surplus_share,
        rnd, left, last,
        (b(max_rounds) == 1).astype(float),
        ci,
        b(messages_allowed),
        np.log10(b(price_order)),
        # how the offer compares with the responder's own last proposal (positive = better for them)
        np.where(is_buyer > 0, b(responder_last_price) - price, price - b(responder_last_price))
        / np.maximum(own, 1e-9),
    ])


# ------------------------------------------------------------------------------------------ persuasion
PERSUASION_FEATURES = [
    "recommends", "rec_x_track_record", "rec_x_log_history", "rec_x_threshold", "rec_x_prior",
    "rec_x_myopic", "round_fraction", "myopic",
]


def track_record(high_count, count):
    """Laplace-smoothed share of high-quality products in the record the buyer can see."""
    return (np.asarray(high_count, dtype=float) + 1.0) / (np.asarray(count, dtype=float) + 2.0)


def persuasion_features(recommends, high_count, count, threshold, prior, myopic, round_number,
                        total_rounds) -> np.ndarray:
    """Features for P(buyer buys | the seller's signal and the record the buyer has seen).

    Non-myopic buyer: count = products recommended so far, high_count = how many of them were high.
    Myopic buyer (a new buyer every round): count / high_count are GLEE's public sales statistics.
    """
    rec = np.atleast_1d(np.asarray(recommends, dtype=float))
    n = rec.shape[0]

    def b(x):
        return np.broadcast_to(np.asarray(x, dtype=float), (n,))

    my = b(myopic)
    return np.column_stack([
        rec,
        rec * track_record(b(high_count), b(count)),
        rec * np.log1p(b(count)),
        rec * b(threshold),
        rec * b(prior),
        rec * my,
        b(round_number) / b(total_rounds),
        my,
    ])
