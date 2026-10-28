"""Parse a GLEE /chat request into a typed observation.

GLEE's HTTP player posts {"messages": [...], "game_params": {...}} on every turn. The message list
is the whole conversation so far: a system message (the rules plus the answer format), then the
game's prompts (role "user") and our earlier answers (role "assistant"). The server keeps no state
between requests; everything the agent knows is rebuilt from this payload.

game_params carries the numbers in recent GLEE versions; for older versions (and as a cross-check)
the same facts are read from the rules text, whose wording comes from GLEE's prompt templates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

Game = Literal["bargaining", "negotiation", "persuasion"]


class ProtocolError(ValueError):
    """The request could not be understood as a GLEE turn."""


# ------------------------------------------------------------------------------------------- helpers
def _num(text: str) -> float:
    return float(text.replace(",", "").replace("$", ""))


def _search(pattern: str, text: str, flags=0) -> Optional[re.Match]:
    return re.search(pattern, text, flags | re.DOTALL)


def _json_objects(text: str) -> list[dict]:
    out = []
    for match in re.finditer(r"\{.*?\}", text or "", re.DOTALL):
        try:
            obj = json.loads(re.sub(r"(?<=\d),(?=\d{3})", "", match.group()))
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def lower_name(name: str) -> str:
    """GLEE's JSON keys use the lower-cased player name with spaces as underscores."""
    return name.lower().replace(" ", "_")


@dataclass
class Turn:
    """Fields shared by every game."""
    messages: list[dict]
    params: dict
    rules: str
    prompt: str          # the latest prompt from the game (last user message)
    my_name: str

    def user_messages(self) -> list[str]:
        return [m.get("content", "") for m in self.messages if m.get("role") == "user"]

    def my_answers(self) -> list[str]:
        return [m.get("content", "") for m in self.messages if m.get("role") == "assistant"]


# ----------------------------------------------------------------------------------------- bargaining
@dataclass
class BargainingTurn(Turn):
    rival_name: str = ""
    action: Literal["propose", "respond"] = "propose"
    round: int = 1
    money: float = 0.0
    max_rounds: int = 99
    complete_information: bool = True
    messages_allowed: bool = True
    me: int = 1                          # 1 if we are GLEE's player 1 (proposes in odd rounds)
    my_delta: float = 1.0
    rival_delta: Optional[float] = None  # None when not known
    my_offers: list[tuple[int, float]] = field(default_factory=list)     # (round, share offered to rival), rejected
    rival_offers: list[tuple[int, float]] = field(default_factory=list)  # (round, share offered to us), rejected
    offer_to_me: Optional[float] = None  # share of the pot offered to us in the current round


def _parse_bargaining(turn: Turn) -> BargainingTurn:
    gp, rules, prompt = turn.params, turn.rules, turn.prompt
    rival = gp.get("rival_name") or (_search(r"You and ([^\n]+?) need to divide", rules) or [None, ""])[1]
    if not rival:
        raise ProtocolError("bargaining: cannot find the other player's name")

    if "Do you accept this offer?" in prompt:
        action = "respond"
    elif "Send your offer to divide" in prompt:
        action = "propose"
    else:
        raise ProtocolError("bargaining: the last prompt is neither an offer request nor an offer")
    rounds = re.findall(r"Round (\d+)", prompt)
    round_number = int(rounds[-1]) if rounds else 1 + sum("Round" in m for m in turn.user_messages()[:-1])

    money = gp.get("money_to_divide")
    if money is None:
        m = _search(r"need to divide ([\d,\.]+)\$", rules)
        if not m:
            raise ProtocolError("bargaining: cannot find the amount to divide")
        money = _num(m[1])
    max_rounds = gp.get("max_rounds")
    if max_rounds is None:
        m = _search(r"You have (\d+) rounds to divide", rules)
        max_rounds = int(m[1]) if m else 99  # GLEE only states the horizon when it is 20 rounds or fewer

    # who we are: player 1 proposes in odd rounds
    me = (1 if round_number % 2 else 2) if action == "propose" else (2 if round_number % 2 else 1)

    # inflation (delta): game_params first, then the rules text
    keys = {1: ("delta_player_1", "delta_1"), 2: ("delta_player_2", "delta_2")}
    def from_params(player):
        for k in keys[player]:
            if gp.get(k) is not None:
                return float(gp[k])
        return None
    my_delta, rival_delta = from_params(me), from_params(3 - me)
    if my_delta is None and gp.get("delta") is not None:
        my_delta = float(gp["delta"])
    complete = gp.get("complete_information")
    if complete is None:
        complete = "You don't know how the inflation" not in rules

    m = _search(r"the money is worth ([\d\.]+)% less for you", rules)
    if my_delta is None:
        my_delta = 1 - _num(m[1]) / 100 if m else 1.0
    if rival_delta is None and complete:
        other = _search(rf"For {re.escape(rival)}, the money is worth ([\d\.]+)% less", rules)
        if other:
            rival_delta = 1 - _num(other[1]) / 100
        elif "Beware of inflation" not in rules:
            rival_delta = 1.0      # GLEE prints no inflation message only when neither player has inflation
        else:
            rival_delta = my_delta  # GLEE omits the rival's rate when it equals ours

    # GLEE's per-round inflation update in offer prompts also states how the rival's money shrinks
    if rival_delta is None:
        rival_delta = _rival_delta_from_updates(turn.user_messages(), rival, my_delta)

    me_key, rival_key = f"{lower_name(turn.my_name)}_gain", f"{lower_name(rival)}_gain"
    my_offers, rival_offers = [], []
    proposals = iter(r for r in range(1, round_number + 1) if (r % 2 == 1) == (me == 1))
    for answer in turn.my_answers():
        for obj in _json_objects(answer):
            if rival_key in obj and me_key in obj:
                my_offers.append((next(proposals, round_number), float(obj[rival_key]) / money))
    offer_to_me = None
    for text in turn.user_messages():
        for block in re.finditer(rf"Round (\d+).*?{re.escape(rival)}'s offer:.*?# {re.escape(turn.my_name)} gain: "
                                 r"([\d,\.]+)", text, re.DOTALL):
            rival_offers.append((int(block[1]), _num(block[2]) / money))
    if action == "respond":
        m = re.findall(rf"# {re.escape(turn.my_name)} gain: ([\d,\.]+)", prompt)
        if not m:
            raise ProtocolError("bargaining: cannot read the offer")
        offer_to_me = _num(m[-1]) / money
        rival_offers = [o for o in rival_offers if o[0] != round_number]

    return BargainingTurn(**vars(turn), rival_name=rival, action=action, round=round_number, money=float(money),
                          max_rounds=int(max_rounds), complete_information=bool(complete),
                          messages_allowed=bool(gp.get("messages_allowed", '"message"' in rules or "message" in
                                                       _system_format(turn))),
                          me=me, my_delta=float(my_delta), rival_delta=rival_delta, my_offers=my_offers,
                          rival_offers=rival_offers, offer_to_me=offer_to_me)


def _rival_delta_from_updates(prompts: list[str], rival: str, my_delta: float) -> Optional[float]:
    """GLEE's "Due to inflation, ..." line in offer prompts also states how the rival's money shrinks.

    It reads "the money <rival> gains is worth Y% less than in the first round" (GLEE capitalises
    the sentence, which lower-cases the name), "... <rival> gains is worth the same ...", or, when
    both lose the same share, only "the money you gain is worth X% less".
    """
    name = re.escape(rival)
    for text in reversed(prompts):
        rnd = re.findall(r"Round (\d+)", text)
        if not rnd or int(rnd[-1]) < 2 or "Due to inflation" not in text:
            continue
        r = int(rnd[-1])
        line = text.split("Due to inflation", 1)[1].split("\n", 1)[0]
        m = re.search(rf"the money {name} gains is worth ([\d\.]+)% less", line, re.IGNORECASE)
        if m:
            return (1 - _num(m[1]) / 100) ** (1 / (r - 1))
        if re.search(rf"the money {name} gains is worth the same", line, re.IGNORECASE):
            return 1.0
        if "the money you gain is worth" in line and not re.search(name, line, re.IGNORECASE):
            return my_delta
    return None


def _system_format(turn: Turn) -> str:
    system = [m.get("content", "") for m in turn.messages if m.get("role") == "system"]
    return system[0] if system else ""


# ---------------------------------------------------------------------------------------- negotiation
@dataclass
class NegotiationTurn(Turn):
    rival_name: str = ""
    role: Literal["seller", "buyer"] = "seller"
    action: Literal["propose", "respond"] = "propose"
    round: int = 1
    max_rounds: int = 99
    price_order: float = 1.0            # all values and prices below are in units of this
    seller_value: Optional[float] = None
    buyer_value: Optional[float] = None
    complete_information: bool = True
    messages_allowed: bool = True
    my_offers: list[tuple[int, float]] = field(default_factory=list)
    rival_offers: list[tuple[int, float]] = field(default_factory=list)
    offer_price: Optional[float] = None

    @property
    def my_value(self) -> float:
        return self.seller_value if self.role == "seller" else self.buyer_value

    @property
    def rival_value(self) -> Optional[float]:
        return self.buyer_value if self.role == "seller" else self.seller_value


def _parse_negotiation(turn: Turn) -> NegotiationTurn:
    gp, rules, prompt = turn.params, turn.rules, turn.prompt
    role = gp.get("role")
    if role not in ("seller", "buyer"):
        role = "seller" if "You are selling" in rules else "buyer"
    if role == "seller":
        rival = (_search(r"\n([^\n]+?) is a potential buyer", rules) or [None, ""])[1]
        own = _search(r"worth no less then \$([\d,\.]+) to you", rules)
    else:
        rival = (_search(r"\n([^\n]+?) is a seller trying", rules) or [None, ""])[1]
        own = _search(r"worth no more then \$([\d,\.]+) to you", rules)
    rival = gp.get("rival_name") or rival.strip()
    other = _search(r"The product is worth \$([\d,\.]+) to ", rules)

    order = float(gp.get("product_price_order") or 0) or None
    sv, bv = gp.get("seller_value"), gp.get("buyer_value")
    if order is None:  # no game_params: measure everything in units of our own value
        order = _num(own[1]) if own else 1.0
    if role == "seller":
        sv = float(sv) if sv is not None else (_num(own[1]) / order if own else None)
        bv = float(bv) if bv is not None else (_num(other[1]) / order if other else None)
    else:
        bv = float(bv) if bv is not None else (_num(own[1]) / order if own else None)
        sv = float(sv) if sv is not None else (_num(other[1]) / order if other else None)
    if (sv if role == "seller" else bv) is None:
        raise ProtocolError("negotiation: cannot find our own value")
    complete = gp.get("complete_information")
    if complete is None:
        complete = other is not None

    max_rounds = gp.get("max_rounds")
    if max_rounds is None:
        m = _search(r"You have (\d+) rounds to close the deal", rules)
        one_shot = "can make a counteroffer" not in rules and "make a counteroffer" not in rules
        max_rounds = int(m[1]) if m else (1 if one_shot else 99)

    if "You have three options" in prompt:
        action = "respond"
    elif re.search(r"Send your (counter)?offer", prompt):
        action = "propose"
    else:
        raise ProtocolError("negotiation: the last prompt is neither an offer request nor an offer")

    my_prices = [float(str(o["product_price"]).replace("$", "").replace(",", ""))
                 for a in turn.my_answers() for o in _json_objects(a) if "product_price" in o]
    rival_prices = [_num(m) for text in turn.user_messages()
                    for m in re.findall(r"The product price will be \$([\d,\.]+)", text)]
    # seller proposes in odd rounds; count proposals to know the round even when GLEE omits "Round N"
    my_first = 1 if role == "seller" else 2
    my_offers = [(my_first + 2 * i, p / order) for i, p in enumerate(my_prices)]
    rival_offers = [(3 - my_first + 2 * i, p / order) for i, p in enumerate(rival_prices)]
    round_number = len(my_prices) + len(rival_prices) + (1 if action == "propose" else 0)
    stated = re.findall(r"Round (\d+)", prompt)
    if stated:
        round_number = int(stated[-1])
    offer_price = None
    if action == "respond":
        offer_price = rival_offers[-1][1] if rival_offers else None
        rival_offers = rival_offers[:-1]
        if offer_price is None:
            raise ProtocolError("negotiation: cannot read the offered price")

    return NegotiationTurn(**vars(turn), rival_name=rival, role=role, action=action, round=round_number,
                           max_rounds=int(max_rounds), price_order=order, seller_value=sv, buyer_value=bv,
                           complete_information=bool(complete),
                           messages_allowed=bool(gp.get("messages_allowed", '"message"' in _system_format(turn))),
                           my_offers=my_offers, rival_offers=rival_offers, offer_price=offer_price)


# ----------------------------------------------------------------------------------------- persuasion
@dataclass
class RoundRecord:
    round: int
    high_quality: Optional[bool] = None
    recommended: Optional[bool] = None
    bought: Optional[bool] = None
    signal: Optional[str] = None  # buyer side: what the seller sent ("yes"/"no" or the text)


@dataclass
class PersuasionTurn(Turn):
    rival_name: str = ""
    role: Literal["seller", "buyer"] = "seller"
    round: int = 1
    total_rounds: int = 20
    price: float = 1.0
    p: Optional[float] = None       # prior probability of high quality (None if unknown to us)
    v: Optional[float] = None       # value of a high-quality product, in units of the price
    c: Optional[float] = None       # value of a low-quality product, in units of the price
    myopic: bool = False
    text_messages: bool = True      # seller sends free text (True) or a yes/no recommendation (False)
    buyer_messages: bool = False    # the buyer must attach a message
    high_quality: Optional[bool] = None   # seller: this round's product
    seller_signal: Optional[str] = None   # buyer: the seller's message, or "yes"/"no" in binary mode
    history: list[RoundRecord] = field(default_factory=list)
    sold: Optional[int] = None            # myopic buyer: GLEE's public sales statistics
    sold_high: Optional[int] = None


SELLER_RECOMMEND = "I recommend this product. It's a good buy for you."
SELLER_DECLINE = "I don't recommend this product."


def _parse_persuasion(turn: Turn) -> PersuasionTurn:
    gp, rules, prompt = turn.params, turn.rules, turn.prompt
    role = gp.get("role")
    if role not in ("seller", "buyer"):
        role = "seller" if "a seller trying to sell products" in rules else "buyer"

    price = gp.get("product_price")
    if price is None:
        m = _search(r"for \$([\d,\.]+) each", rules) or _search(r"fixed price of \$([\d,\.]+)", rules)
        price = _num(m[1]) if m else 1.0
    price = float(price)
    p = gp.get("p")
    if p is None:
        m = _search(r"probability that a product is high-quality is ([\d\.]+)%", rules) or \
            _search(r"has a ([\d\.]+)% chance of being high quality", rules)
        p = _num(m[1]) / 100 if m else None
    v, c = gp.get("v"), gp.get("c")
    if v is None or c is None:
        m = _search(r"A high-quality product is worth \$([\d,\.]+) to .+?, and a low-quality product is worth "
                    r"(?:only )?\$([\d,\.]+)", rules)
        if m:
            v, c = _num(m[1]) / price, _num(m[2]) / price
    myopic = gp.get("is_myopic")
    if myopic is None:
        myopic = "random buyers" in rules or "one of the buyers" in rules
    total = gp.get("total_rounds")
    if total is None:
        m = _search(r"For (\d+) rounds", rules)
        total = int(m[1]) if m else 20
    fmt = _system_format(turn)
    text_messages = gp.get("seller_message_type", "text" if "Send your message" in fmt else "binary") == "text"
    if role == "seller" and "Would you recommend" in fmt:
        text_messages = False
    buyer_messages = bool(gp.get("allow_buyer_message", role == "buyer" and '"message"' in fmt))

    history: list[RoundRecord] = []
    out = PersuasionTurn(**vars(turn), role=role, price=price, p=None if p is None else float(p),
                         v=None if v is None else float(v), c=None if c is None else float(c),
                         myopic=bool(myopic), total_rounds=int(total), text_messages=text_messages,
                         buyer_messages=buyer_messages)

    if role == "seller":
        out.rival_name = "the buyer" if myopic else (_search(r"sell products to ([^\n]+?) for \$", rules) or [None, "the buyer"])[1]
        prompts, answers = turn.user_messages(), turn.my_answers()
        for i, text in enumerate(prompts):
            q = _search(r"This round's product is (high|low)-quality", text)
            outcome = None
            nxt = prompts[i + 1] if i + 1 < len(prompts) else ""
            if "bought the product!" in nxt:
                outcome = True
            elif "didn't bought the product" in nxt:
                outcome = False
            if q:
                rnd = re.findall(r"Round (\d+)", text)
                rec = None
                if i < len(answers):
                    rec = _seller_answer_recommends(answers[i])
                history.append(RoundRecord(int(rnd[-1]) if rnd else len(history) + 1, q[1] == "high", rec, outcome))
        if not history or history[-1].recommended is not None:
            raise ProtocolError("persuasion: no pending product to describe")
        current = history.pop()
        out.round, out.high_quality, out.history = current.round, current.high_quality, history
        return out

    # buyer
    out.rival_name = (_search(r"\n([^\n]+?) is a seller offering products", "\n" + rules) or [None, "the seller"])[1]
    prompts = turn.user_messages()
    for text in prompts:
        # a prompt starts with the outcome of the previous round, then shows the new round
        q = _search(r"The product was (high|low)-quality", text)
        if q and history:
            history[-1].high_quality = q[1] == "high"
            history[-1].bought = "You bought the product" in text
        rnd = re.findall(r"Round (\d+)", text)
        signal = _buyer_signal(text, out.rival_name, out.text_messages)
        if signal is not None:
            history.append(RoundRecord(int(rnd[-1]) if rnd else len(history) + 1, signal=signal))
    if not history or history[-1].high_quality is not None:
        raise ProtocolError("persuasion: no pending product to decide on")
    current = history.pop()
    out.round = current.round
    out.seller_signal = current.signal
    out.history = history
    stats = _search(r"Statistics: .*? (\d+) of the buyers purchased.*?So far, (\d+) high-quality", prompt)
    if stats:
        out.sold, out.sold_high = int(stats[1]), int(stats[2])
    elif "No products have been sold yet" in prompt:
        out.sold, out.sold_high = 0, 0
    if myopic and out.sold is not None:
        out.round = out.sold  # a myopic buyer only knows the sales count, not the round
    return out


def _seller_answer_recommends(answer: str) -> Optional[bool]:
    for obj in _json_objects(answer):
        if obj.get("decision") in ("yes", "no"):
            return obj["decision"] == "yes"
        if "message" in obj:
            return not str(obj["message"]).strip().lower().startswith("i don't recommend")
    return None


def _buyer_signal(text: str, seller: str, text_messages: bool) -> Optional[str]:
    if f"{seller} recommends you to buy" in text:
        return "yes"
    if f"{seller} does not recommend you to buy" in text:
        return "no"
    m = _search(rf"{re.escape(seller)}'s message: (.*?)\n\nWould you like to buy", text)
    if m:
        return m[1].strip()
    return None


# ------------------------------------------------------------------------------------------- entry
def detect_game(params: dict, rules: str) -> Game:
    if "money_to_divide" in params or "need to divide" in rules:
        return "bargaining"
    if "product_price_order" in params or "seller_value" in params or "John" in rules:
        return "negotiation"
    if "is_myopic" in params or "high-quality" in rules:
        return "persuasion"
    raise ProtocolError("cannot tell which game this is")


def parse_request(payload: dict) -> BargainingTurn | NegotiationTurn | PersuasionTurn:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ProtocolError("expected a JSON object with a 'messages' list")
    messages = payload["messages"]
    params = payload.get("game_params") or {}
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not users:
        raise ProtocolError("no prompt from the game yet")
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    rules = params.get("rules") or system
    my_name = params.get("public_name") or (_search(r"You are (?:playing as )?([^\n]+?)[\.,]", rules) or [None, "Player"])[1]
    base = Turn(messages=messages, params=params, rules=rules, prompt=users[-1], my_name=my_name)
    game = detect_game(params, rules)
    if game == "bargaining":
        return _parse_bargaining(base)
    if game == "negotiation":
        return _parse_negotiation(base)
    return _parse_persuasion(base)
