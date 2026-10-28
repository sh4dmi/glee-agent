"""Check an answer against the format GLEE will enforce, and build a safe answer when needed.

GLEE ends the game with an assertion error on a malformed answer, so nothing leaves the server
without passing these checks. They mirror GLEE's own format checkers (is_offer_in_format and
is_decision_in_format in games/*/).
"""
from __future__ import annotations

import json
import re

from .protocol import (SELLER_RECOMMEND, BargainingTurn, NegotiationTurn, PersuasionTurn, lower_name)


def check(turn, answer: dict) -> str | None:
    """None if GLEE would accept `answer` for this turn, otherwise the reason it would not."""
    if isinstance(turn, BargainingTurn):
        if turn.action == "respond":
            return None if answer.get("decision") in ("accept", "reject") else "decision must be accept/reject"
        keys = (f"{lower_name(turn.my_name)}_gain", f"{lower_name(turn.rival_name)}_gain")
        if not all(isinstance(answer.get(k), (int, float)) for k in keys):
            return f"missing numeric {keys}"
        if sum(answer[k] for k in keys) != turn.money:
            return "gains must add up to the pot"
        if min(answer[k] for k in keys) < 0:
            return "negative gain"
        if turn.messages_allowed and not isinstance(answer.get("message"), str):
            return "message required"
        return None
    if isinstance(turn, NegotiationTurn):
        if turn.action == "respond":
            ok = ("AcceptOffer", "RejectOffer", "SellToJhon" if turn.role == "seller" else "BuyFromJhon")
            return None if answer.get("decision") in ok else f"decision must be one of {ok}"
        if not isinstance(answer.get("product_price"), (int, float)) or answer["product_price"] < 0:
            return "product_price must be a non-negative number"
        if turn.messages_allowed and not isinstance(answer.get("message"), str):
            return "message required"
        return None
    if isinstance(turn, PersuasionTurn):
        if turn.role == "seller":
            if turn.text_messages:
                return None if isinstance(answer.get("message"), str) else "message required"
            return None if answer.get("decision") in ("yes", "no") else "decision must be yes/no"
        if answer.get("decision") not in ("yes", "no"):
            return "decision must be yes/no"
        if turn.buyer_messages and not isinstance(answer.get("message"), str):
            return "message required"
        return None
    return "unknown turn type"


def fallback(payload: dict) -> dict:
    """A valid, conservative answer built from the raw request when the agent itself failed.

    Uses only the answer format GLEE put in the system message, so it works even when the
    request could not be parsed.
    """
    messages = payload.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    gp = payload.get("game_params") or {}
    if "Do you accept this offer?" in prompt:
        return {"decision": "accept"}
    gains = re.findall(r'"(\w+)_gain"', system)
    if len(gains) >= 2:
        money = gp.get("money_to_divide")
        if money is None:
            m = re.search(r"divide \$?([\d,\.]+)", prompt + system)
            money = float(m[1].replace(",", "")) if m else 100
        half = int(money // 2)
        return {f"{gains[0]}_gain": money - half, f"{gains[1]}_gain": half, "message": "Let's split it evenly."}
    if "You have three options" in prompt:
        return {"decision": "RejectOffer"}
    if '"product_price"' in system:
        m = re.search(r"worth no (?:less|more) then \$([\d,\.]+)", system)
        value = float(m[1].replace(",", "")) if m else 100
        return {"product_price": value, "message": "This is my offer."}
    if "Send your message" in system:
        return {"message": SELLER_RECOMMEND}
    if "Would you recommend" in system:
        return {"decision": "yes"}
    if '"message"' in system:
        return {"decision": "no", "message": "I'll pass on this one."}
    return {"decision": "no"}


def to_json(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False)
