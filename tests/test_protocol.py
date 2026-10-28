"""Parse the exact payloads GLEE sends, captured from the simulator."""
import json
import random

import simulator as sim
from glee_agent.protocol import BargainingTurn, NegotiationTurn, PersuasionTurn, parse_request


def capture(game, args, answers_alice, answers_bob, seed=0):
    """Play a game with scripted answers; return every payload each player received."""
    seen = {"Alice": [], "Bob": []}

    def scripted(name, answers):
        it = iter(answers)

        def respond(payload):
            seen[name].append(payload)
            return json.dumps(next(it))
        return respond

    try:
        sim.play(game, args, scripted("Alice", answers_alice), scripted("Bob", answers_bob), rng=random.Random(seed))
    except StopIteration:
        pass
    return seen


def test_bargaining_incomplete_information():
    args = dict(money_to_divide=10000, max_rounds=12, complete_information=False, messages_allowed=True,
                delta_1=0.9, delta_2=0.8)
    seen = capture("bargaining", args,
                   [{"alice_gain": 7000, "bob_gain": 3000, "message": "a"}, {"decision": "reject"},
                    {"alice_gain": 6000, "bob_gain": 4000, "message": "c"}],
                   [{"decision": "reject"}, {"alice_gain": 4500, "bob_gain": 5500, "message": "b"},
                    {"decision": "accept"}])
    t = parse_request(seen["Alice"][1])           # Alice responding in round 2
    assert isinstance(t, BargainingTurn)
    assert (t.action, t.round, t.me, t.money, t.max_rounds) == ("respond", 2, 1, 10000, 12)
    assert t.my_delta == 0.9 and not t.complete_information
    assert abs(t.rival_delta - 0.8) < 1e-9       # recovered from GLEE's inflation update
    assert t.offer_to_me == 0.45 and t.my_offers == [(1, 0.3)]
    t = parse_request(seen["Bob"][2])             # Bob responding in round 3
    assert (t.action, t.round, t.me) == ("respond", 3, 2)
    assert t.my_offers == [(2, 0.45)] and t.rival_offers == [(1, 0.3)] and t.offer_to_me == 0.4


def test_negotiation_long_game_without_round_numbers():
    args = dict(seller_value=1.0, buyer_value=1.5, product_price_order=10000, max_rounds=99,
                complete_information=False, messages_allowed=False)
    seen = capture("negotiation", args,
                   [{"product_price": 14000}, {"decision": "RejectOffer"}],
                   [{"decision": "RejectOffer"}, {"product_price": 11000}])
    t = parse_request(seen["Alice"][1])
    assert isinstance(t, NegotiationTurn)
    assert (t.role, t.action, t.round, t.max_rounds) == ("seller", "respond", 2, 99)
    assert t.seller_value == 1.0 and t.buyer_value is None and t.offer_price == 1.1
    assert t.my_offers == [(1, 1.4)]


def test_negotiation_from_rules_text_only():
    args = dict(seller_value=1.2, buyer_value=1.5, product_price_order=100, max_rounds=10,
                complete_information=True, messages_allowed=True)
    seen = capture("negotiation", args, [{"product_price": 140, "message": "hi"}], [{"decision": "RejectOffer"}])
    payload = seen["Bob"][0]
    payload["game_params"] = {}                   # older GLEE versions send no numbers
    t = parse_request(payload)
    assert (t.role, t.action, t.max_rounds) == ("buyer", "respond", 10)
    assert t.buyer_value == 1.0 and abs(t.seller_value - 1.2 / 1.5) < 1e-9  # in units of our own value
    assert abs(t.offer_price - 140 / 150) < 1e-9


def test_persuasion_buyer_history_and_seller_record():
    args = dict(is_myopic=False, product_price=100, p=0.5, v=2.0, c=0.0, total_rounds=3, is_seller_know_cv=True,
                is_buyer_know_p=True, seller_message_type="binary", allow_buyer_message=False)
    seen = capture("persuasion", args, [{"decision": "yes"}, {"decision": "no"}, {"decision": "yes"}],
                   [{"decision": "yes"}, {"decision": "no"}, {"decision": "no"}], seed=3)
    t = parse_request(seen["Bob"][2])
    assert isinstance(t, PersuasionTurn) and t.role == "buyer" and t.round == 3
    assert [h.signal for h in t.history] == ["yes", "no"] and t.seller_signal == "yes"
    assert [h.bought for h in t.history] == [True, False]
    assert all(h.high_quality is not None for h in t.history)
    s = parse_request(seen["Alice"][2])
    assert s.role == "seller" and s.round == 3 and [h.recommended for h in s.history] == [True, False]
    assert [h.bought for h in s.history] == [True, False]
