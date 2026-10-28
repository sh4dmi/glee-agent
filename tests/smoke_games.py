"""Play every game configuration in the simulator with the agent in each seat (used by tests)."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import simulator as sim  # noqa: E402

CONFIGS = [
    ("bargaining", dict(money_to_divide=10000, max_rounds=12, complete_information=True, messages_allowed=True,
                        delta_1=0.9, delta_2=0.8)),
    ("bargaining", dict(money_to_divide=100, max_rounds=99, complete_information=False, messages_allowed=False,
                        delta_1=0.95, delta_2=1.0)),
    ("bargaining", dict(money_to_divide=1000000, max_rounds=12, complete_information=False, messages_allowed=True,
                        delta_1=1.0, delta_2=1.0)),
    ("negotiation", dict(seller_value=1.0, buyer_value=1.5, product_price_order=10000, max_rounds=10,
                         complete_information=True, messages_allowed=True)),
    ("negotiation", dict(seller_value=1.2, buyer_value=1.5, product_price_order=100, max_rounds=1,
                         complete_information=False, messages_allowed=False)),
    ("negotiation", dict(seller_value=1.0, buyer_value=1.2, product_price_order=1000000, max_rounds=99,
                         complete_information=False, messages_allowed=True)),
    ("negotiation", dict(seller_value=1.5, buyer_value=0.8, product_price_order=100, max_rounds=10,
                         complete_information=True, messages_allowed=True)),
    ("persuasion", dict(is_myopic=False, product_price=100, p=0.5, v=2.0, c=0.0, total_rounds=20,
                        is_seller_know_cv=True, is_buyer_know_p=True, seller_message_type="binary",
                        allow_buyer_message=False)),
    ("persuasion", dict(is_myopic=True, product_price=10000, p=1 / 3, v=1.25, c=0.0, total_rounds=20,
                        is_seller_know_cv=False, is_buyer_know_p=False, seller_message_type="text",
                        allow_buyer_message=True)),
    ("persuasion", dict(is_myopic=False, product_price=10000, p=0.8, v=1.2, c=0.0, total_rounds=20,
                        is_seller_know_cv=False, is_buyer_know_p=False, seller_message_type="text",
                        allow_buyer_message=False)),
]


def seats(agent_respond, seed=1):
    rnd = sim.random_player(random.Random(seed))
    return {"self-play": (agent_respond, agent_respond), "agent as player 1": (agent_respond, rnd),
            "agent as player 2": (rnd, agent_respond)}
