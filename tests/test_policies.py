"""Behavioural guarantees that hold whatever the fitted models say."""
import json
import random

import pytest

import simulator as sim
from smoke_games import CONFIGS, seats


@pytest.mark.parametrize("game,args", CONFIGS, ids=[f"{g}-{i}" for i, (g, _) in enumerate(CONFIGS)])
def test_every_seat_plays_a_full_game_without_fallbacks(agent, monkeypatch, game, args):
    import glee_agent.graph as graph

    def fail(payload):
        raise AssertionError("agent fell back to the safe answer")
    monkeypatch.setattr(graph, "fallback", fail)
    for alice, bob in seats(agent.respond).values():
        sim.play(game, args, alice, bob, rng=random.Random(0))


def _games(agent, game, args, n=3):
    logs = []
    for seed in range(n):
        for alice, bob in seats(agent.respond, seed).values():
            logs.append(sim.play(game, args, alice, bob, rng=random.Random(seed)))
    return logs


def test_bargaining_never_takes_back_a_concession(agent):
    args = dict(money_to_divide=10000, max_rounds=12, complete_information=False, messages_allowed=False,
                delta_1=0.9, delta_2=0.95)
    for alice in (True, False):
        def me(payload, _a=alice):
            return agent.respond(payload)
        rnd = sim.random_player(random.Random(1))
        log = sim.play("bargaining", args, me if alice else rnd, rnd if alice else me, rng=random.Random(1))
        name, rival = ("Alice", "bob") if alice else ("Bob", "alice")
        offers = [x[f"{rival}_gain"] for x in log if x.get("player") == name and f"{rival}_gain" in x]
        assert offers == sorted(offers)


def test_negotiation_never_accepts_a_loss(agent):
    args = dict(seller_value=1.2, buyer_value=1.5, product_price_order=100, max_rounds=10,
                complete_information=False, messages_allowed=True)
    for seed in range(3):
        rnd = sim.random_player(random.Random(seed))
        for alice_is_agent in (True, False):
            a, b = (agent.respond, rnd) if alice_is_agent else (rnd, agent.respond)
            log = sim.play("negotiation", args, a, b, rng=random.Random(seed))
            prices = {x["round"]: x["product_price"] for x in log if "product_price" in x}
            me = "Alice" if alice_is_agent else "Bob"
            for x in log:
                if x.get("player") == me and x.get("decision") == "AcceptOffer":
                    p = prices[x["round"]] / 100
                    assert (p >= 1.2) if alice_is_agent else (p <= 1.5)
                if x.get("player") == me and "product_price" in x:
                    p = x["product_price"] / 100
                    assert (p >= 1.2) if alice_is_agent else (p <= 1.5)


def test_persuasion_seller_always_recommends_high_quality(agent):
    args = dict(is_myopic=False, product_price=100, p=0.5, v=2.0, c=0.0, total_rounds=20, is_seller_know_cv=True,
                is_buyer_know_p=True, seller_message_type="binary", allow_buyer_message=False)
    log = sim.play("persuasion", args, agent.respond, sim.random_player(random.Random(0)), rng=random.Random(4))
    quality = {x["round"]: x["round_quality"] for x in log if x["player"] == "Nature"}
    for x in log:
        if x["player"] == "Alice" and quality[x["round"]] == "high-quality":
            assert x["decision"] == "yes"


def test_buyer_ignores_a_seller_who_recommends_everything(agent):
    """A seller who recommends every product, mostly bad ones, should stop being believed."""
    args = dict(is_myopic=False, product_price=100, p=0.2, v=2.0, c=0.0, total_rounds=20, is_seller_know_cv=True,
                is_buyer_know_p=True, seller_message_type="binary", allow_buyer_message=False)
    always = lambda payload: json.dumps({"decision": "yes"})  # noqa: E731
    log = sim.play("persuasion", args, always, agent.respond, rng=random.Random(0))
    late = [x["decision"] for x in log if x["player"] == "Bob" and x["round"] > 10]
    assert late.count("yes") == 0


def test_myopic_buyer_explores_an_honest_seller(agent):
    """With only sales statistics to learn from, never buying would teach the buyer nothing."""
    args = dict(is_myopic=True, product_price=100, p=0.5, v=1.25, c=0.0, total_rounds=20, is_seller_know_cv=False,
                is_buyer_know_p=True, seller_message_type="binary", allow_buyer_message=False)
    honest = lambda payload: json.dumps(  # noqa: E731
        {"decision": "yes" if "product is high-quality" in payload["messages"][-1]["content"] else "no"})
    log = sim.play("persuasion", args, honest, agent.respond, rng=random.Random(0))
    bought = [x for x in log if x["player"] == "the buyer" and x["decision"] == "yes"]
    assert len(bought) >= 5
