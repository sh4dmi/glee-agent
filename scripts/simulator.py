"""A compact re-implementation of GLEE's three game loops, for offline tests and evaluation.

It reproduces, from GLEE commit 68a33e9 (github.com/eilamshapira/GLEE), everything a player sees:
the rules and format prompts, the per-round prompts, the conversation bookkeeping of GLEE's
HTTPPlayer (system message rewritten every turn, prompts as "user", answers as "assistant", the
chat reset for myopic buyers), the game_params each player receives, and GLEE's format checks.
A player is any function payload -> answer string, so our agent is plugged in exactly as it is
behind the HTTP server.

Game outcomes follow GLEE: bargaining gains shrink by each player's delta per round; negotiation
ends on acceptance, on a John deal, or after the last round; persuasion runs every round.
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable

Respond = Callable[[dict], str]


def pretty_number(x) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def lower(name: str) -> str:
    return name.lower().replace(" ", "_")


class FormatError(AssertionError):
    pass


class Player:
    """GLEE's base Player + HTTPPlayer conversation handling."""

    def __init__(self, respond: Respond, public_name: str, delta=None, player_id: int = 0):
        self.respond, self.public_name, self.delta, self.player_id = respond, public_name, delta, player_id
        self.messages: list[dict] = []
        self.system_message = ""
        self.buffer = ""
        self.rules = ""
        self.req_offer_text = ""
        self.req_decision_text = ""
        self.game_params: dict = {}
        self.last_action_json: dict | None = None
        self.history: list = []

    # --- HTTPPlayer
    def add_message(self, message, role):
        self.messages.append({"role": role, "content": message})

    def set_system_message(self, message):
        self.system_message = message
        if not self.messages:
            self.messages.append({"role": "system", "content": message})
        elif self.messages[0]["role"] == "system":
            self.messages[0]["content"] = message
        else:
            self.messages.insert(0, {"role": "system", "content": message})

    def clean_conv(self):
        self.messages = []
        if self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

    def get_text_answer(self) -> str:
        params = {"public_name": self.public_name, "delta": self.delta, "player_id": self.player_id,
                  "rules": self.rules, "req_offer_text": self.req_offer_text,
                  "req_decision_text": self.req_decision_text}
        params.update(self.game_params)
        answer = self.respond({"messages": [dict(m) for m in self.messages], "decision": False,
                               "game_params": params})
        self.add_message(answer, "assistant")
        return answer

    # --- base Player
    def edit_system_message(self, format_requested):
        self.set_system_message((self.rules.strip() + "\n" + format_requested.strip()).strip())

    def act(self, checker):
        prompt = self.buffer.strip()
        assert prompt, "Prompt is empty."
        self.add_message(prompt, "user")
        response = self.get_text_answer()
        ok, why = checker(response)
        if not ok:
            raise FormatError(f"{self.public_name}: {why}: {response!r}")
        action = re.search(r"\{.*?\}", response, re.DOTALL).group()
        action = re.sub(r"(?<=\d),(?=\d{3})", "", action).replace("$", "")
        data = json.loads(action)
        for k, v in data.items():
            if isinstance(v, str) and v.replace("$", "").replace(".", "", 1).isdigit():
                data[k] = float(v.replace("$", ""))
        self.last_action_json = data
        self.history.append((prompt, response))
        self.buffer = ""

    def end_chat(self):
        self.buffer = ""
        self.clean_conv()

    def set_rules(self, prompt):
        self.rules = prompt
        self.add_message(prompt, "system")


def _first_json(text):
    m = re.search(r"\{.*?\}", text or "", re.DOTALL)
    if not m:
        return None, "No JSON object found in response"
    s = re.sub(r"(?<=\d),(?=\d{3})", "", m.group())
    for attempt in range(2):
        try:
            obj = json.loads(s)
            return (obj, "") if isinstance(obj, dict) else (None, "not a dict")
        except json.JSONDecodeError as e:
            if "$" in s and attempt == 0:
                s = s.replace("$", "")
            else:
                return None, f"JSON parse error: {e.msg}"
    return None, "invalid"


# ============================================================================================ bargaining
def play_bargaining(p1: Player, p2: Player, money_to_divide, max_rounds, complete_information, messages_allowed,
                    delta_1, delta_2, show_inflation_update=True) -> list[dict]:
    p1.delta, p2.delta = delta_1, delta_2
    loss = {1: (1 - delta_1) * 100, 2: (1 - delta_2) * 100}
    base = {"money_to_divide": money_to_divide, "max_rounds": max_rounds,
            "complete_information": complete_information, "messages_allowed": messages_allowed}
    if complete_information:
        p1.game_params = {**base, "delta_player_1": delta_1, "delta_player_2": delta_2}
        p2.game_params = dict(p1.game_params)
    else:
        p1.game_params = {**base, "delta_player_1": delta_1}
        p2.game_params = {**base, "delta_player_2": delta_2}
    players = {1: p1, 2: p2}
    names = [p1.public_name, p2.public_name]

    def inflation_message(i):
        self_loss, other_loss, other = loss[i], loss[3 - i], players[3 - i].public_name
        if not self_loss and not other_loss:
            return ""
        prefix = f"Beware of inflation! With each passing round, the money is worth {pretty_number(self_loss)}% less "
        if complete_information:
            if loss[1] != loss[2]:
                prefix += f"for you. For {other}, the money is worth {pretty_number(other_loss)}% less"
        else:
            prefix += f"for you. You don't know how the inflation effect on {other}"
        return prefix.strip() + "."

    max_msg = (f"You have {max_rounds} rounds to divide the money, or both of you will get nothing!\n"
               if 0 < max_rounds <= 20 else "")
    for i, pl in players.items():
        other = players[3 - i]
        rules = ("Let's play a game.\nIn this game, there are two players. You are playing as {me}.\n"
                 "You and {other} need to divide {money}$ between you.\n"
                 "In each round of the game, one of the players will make a proposal for division and can describe "
                 "their proposal in a few words.\nThe other player will decide whether to accept the offer and finish "
                 "the game, or to proceed to the next round, in which they will be the one making the proposal.\n"
                 "We will relay messages between you and {other}.\n{infl}\n{maxr}").format(
            me=pl.public_name, other=other.public_name, money=pretty_number(money_to_divide),
            infl=inflation_message(i), maxr=max_msg)
        pl.set_rules(rules)
        add1 = " and the message you attached" if messages_allowed else ""
        add2 = f',\n"message": The message you pass to {other.public_name}' if messages_allowed else ""
        pl.req_offer_text = (f"Send your offer to divide {pretty_number(money_to_divide)}${add1} in the JSON format:\n"
                             f'{{"{lower(pl.public_name)}_gain": The part that you will receive in your offer,\n'
                             f'"{lower(other.public_name)}_gain": The part that {other.public_name} will receive in '
                             f"your offer{add2}}}")

    def offer_ok(text):
        a, why = _first_json(text)
        if a is None:
            return False, why
        keys = [f"{lower(n)}_gain" for n in names] + (["message"] if messages_allowed else [])
        if any(k not in a for k in keys):
            return False, "missing keys"
        if not all(isinstance(a[f"{lower(n)}_gain"], (int, float)) for n in names):
            return False, "gain must be number"
        if sum(a[k] for k in a if "_gain" in k) != money_to_divide:
            return False, "sum of gains differs from money_to_divide"
        if messages_allowed and not isinstance(a["message"], str):
            return False, "message must be str"
        return True, ""

    def decision_ok(text):
        a, why = _first_json(text)
        if a is None:
            return False, why
        return (a.get("decision") in ("accept", "reject"), "decision must be accept/reject")

    def inflation_update(r, me, other):
        if not show_inflation_update:
            return ""
        sl, ol = 1 - me.delta ** (r - 1), 1 - other.delta ** (r - 1)
        if not sl and not ol:
            return ""
        u = "Due to inflation, "
        if sl == ol:
            u += "the money you gain is worth " + f"{pretty_number(sl * 100)}% less than in the first round."
        else:
            if sl:
                u += f"the money you gain is worth {pretty_number(sl * 100)}% less than in the first round."
            if ol:
                add = f"the money {other.public_name} gains is worth {pretty_number(ol * 100)}% less than in the first round."
                if sl:
                    add = add.capitalize()
                else:
                    add += " The money you gains is worth the same as in the first round."
                u += f" {add}"
            else:
                u += f" The money {other.public_name} gains is worth the same as in the first round."
        return u + "\n"

    log = []
    for r in range(1, max_rounds + 1):
        prop, rec = (p1, p2) if r % 2 else (p2, p1)
        ask = f"Send your offer to divide ${money_to_divide:,}" + (f" and a message to {rec.public_name}"
                                                                    if messages_allowed else "") + "."
        prop.buffer += f"Round {r}\n".strip() + "\n" + ask
        prop.edit_system_message(prop.req_offer_text)
        prop.act(offer_ok)
        offer = dict(prop.last_action_json)
        log.append({**offer, "player": prop.public_name, "round": r})
        pg, rg = offer[f"{lower(prop.public_name)}_gain"], offer[f"{lower(rec.public_name)}_gain"]
        msg = offer.get("message") if messages_allowed else None
        system = 'Answer with {"decision": "accept"} or {"decision": "reject"}'
        text = (f"\nRound {r}\n{inflation_update(r, rec, prop)}{prop.public_name}'s offer:\n"
                + (f"# {prop.public_name}'s message: {msg} \n" if msg is not None else "")
                + f"# {rec.public_name} gain: {rg} \n# {prop.public_name} gain: {pg} \nDo you accept this offer?")
        rec.edit_system_message(system)
        rec.buffer += text
        rec.act(decision_ok)
        decision = rec.last_action_json["decision"]
        log.append({"decision": decision, "player": rec.public_name, "round": r})
        if decision == "accept":
            prop.buffer += f"{rec.public_name} accepted the offer!"
            rec.buffer += f"You accepted {prop.public_name}'s offer.\n"
            break
        prop.buffer += f"{rec.public_name} rejected your offer from round {r}.\n"
        rec.buffer += f"You have chosen to reject {prop.public_name}'s offer from round {r}.\n"
    for pl in (p1, p2):
        pl.end_chat()
    return log


# =========================================================================================== negotiation
def play_negotiation(p1: Player, p2: Player, max_rounds, seller_value, buyer_value, complete_information,
                     product_price_order, messages_allowed) -> list[dict]:
    p1.final_value, p2.final_value = product_price_order * seller_value, product_price_order * buyer_value

    def complete_msg(other):
        if complete_information:
            return f"The product is worth ${pretty_number(other.final_value)} to {other.public_name}. "
        return f"You don't know the value of the product to {other.public_name}. "

    def seller_next(value, other):
        if max_rounds == 1:
            m = f"If {other} rejects the offer, you will sell the product to another buyer, John, "
        else:
            m = (f"If {other} rejects the offer, he can make a counteroffer to buy your product. You can either accept "
                 f"or reject his counteroffer. If you reject {other}'s counteroffer, you can make a new counteroffer, "
                 "and so on.\n")
            if max_rounds <= 20:
                m += f"You have {max_rounds} rounds to close the deal. However, "
            a = "A" if m.strip()[-1] == "." else "a"
            m += f"{a}t any moment, you can choose to stop the negotiation with {other} and sell the product to " \
                 f"another buyer, John, "
        return m + f"who is willing to buy the product from you for ${pretty_number(value)}."

    def buyer_next(value, other):
        if max_rounds == 1:
            m = "If you reject the offer, you will buy the product from another seller, John, "
        else:
            m = (f"If you reject the offer, you can make a counteroffer to buy the product. {other} can either accept "
                 f"or reject your counteroffer. If {other} rejects your counteroffer, {other} can make a new "
                 "counteroffer, and so on. ")
            if max_rounds <= 20:
                m += f"\nYou have {max_rounds} rounds to close the deal. However, "
            a = "A" if m.strip()[-1] == "." else "a"
            m += f"{a}t any moment, you can choose to stop the negotiation with {other} and buy the product from " \
                 f"another seller, John, "
        return m + f"who is willing to sell the product to you for ${pretty_number(value)}."

    s, b = p1, p2
    s.set_rules(f"You are {s.public_name}. You are selling one product that is worth no less then "
                f"${pretty_number(s.final_value)} to you.\n{b.public_name} is a potential buyer to whom you are offering "
                f"the product. {complete_msg(b)}\nYou will offer {b.public_name} to buy the product at a price of your "
                f"choice. {b.public_name} can either accept or reject the offer.\nYour goal is to earn as much money as "
                f"you can for the product.\n{seller_next(s.final_value, b.public_name)}")
    b.set_rules(f"You are playing as {b.public_name}. You are looking for a product that worth no more then "
                f"${pretty_number(b.final_value)} to you.\n{s.public_name} is a seller trying to sell you the product. "
                f"{complete_msg(s)}\n{s.public_name} will offer you a price to buy the product. You can either accept "
                f"the offer or reject it.\nYour goal is to buy some product but save as much money as you can.\n"
                f"{buyer_next(b.final_value, s.public_name)}")
    if messages_allowed:
        s.req_offer_text = ("Send your {{offer_type}} in the format:\n{\"message\": Your message to " + b.public_name +
                            ", \"product_price\": The price you want to sell the product for}")
        b.req_offer_text = ("Send your counteroffer in the format:\n{\"message\": Your message to " + s.public_name +
                            ", \"product_price\": The price you want to buy the product for}")
    else:
        s.req_offer_text = "Send your {{offer_type}} in the format:\n{\"product_price\": The price you want to sell the product for}"
        b.req_offer_text = "Send your counteroffer in the format:\n{\"product_price\": The price you want to buy the product for}"
    s.john_caption, b.john_caption = "SellToJhon", "BuyFromJhon"
    s.john_text = f"Sell the product to Jhon for ${pretty_number(s.final_value)}"
    b.john_text = f"Buy the product from Jhon for ${pretty_number(b.final_value)}"
    s.act_name, b.act_name = "sell the product", "buy the product"
    base = {"max_rounds": max_rounds, "complete_information": complete_information,
            "product_price_order": product_price_order, "messages_allowed": messages_allowed}
    if complete_information:
        s.game_params = {**base, "role": "seller", "seller_value": seller_value, "buyer_value": buyer_value}
        b.game_params = {**base, "role": "buyer", "seller_value": seller_value, "buyer_value": buyer_value}
    else:
        s.game_params = {**base, "role": "seller", "seller_value": seller_value}
        b.game_params = {**base, "role": "buyer", "buyer_value": buyer_value}

    def offer_ok(text):
        a, why = _first_json(text)
        if a is None:
            return False, why
        need = ["product_price"] + (["message"] if messages_allowed else [])
        if any(k not in a for k in need):
            return False, "missing keys"
        try:
            float(str(a["product_price"]).replace("$", ""))
        except ValueError:
            return False, "product_price not a number"
        return True, ""

    def decision_ok(text):
        a, why = _first_json(text)
        if a is None:
            return False, why
        return (a.get("decision") in ["DealWithJhon", "AcceptOffer", "SellToJhon", "BuyFromJhon", "RejectOffer"],
                "bad decision")

    log = []
    for r in range(1, max_rounds + 1):
        prop, rec = (s, b) if r % 2 else (b, s)
        fmt = prop.req_offer_text.replace("{{offer_type}}", "offer" if r == 1 else "counteroffer")
        prop.buffer += ((f"Round {r}\n" if 1 < max_rounds <= 20 else "")
                        + f"Send your {'counter' if r > 1 else ''}offer to {rec.public_name}. \n")
        prop.edit_system_message(fmt)
        prop.act(offer_ok)
        offer = dict(prop.last_action_json)
        offer["product_price"] = float(str(offer["product_price"]).replace("$", ""))
        log.append({**offer, "player": prop.public_name, "round": r})
        price = offer["product_price"]
        counter = r < max_rounds
        system = 'Answer with {"decision": "AcceptOffer"}, or {"decision": "RejectOffer"}, ' + \
                 f'or {{"decision": "{rec.john_caption}"}}'
        text = ((f"Round {r}\n" if 1 < max_rounds <= 20 else "") + f"{prop.public_name}'s offer:"
                + (f"\n# {prop.public_name}'s message: {offer['message']} \n# {prop.public_name}'s offer:"
                   if messages_allowed else "")
                + f" The product price will be ${pretty_number(price)}. \nYou have three options:\n"
                f"(1) Accept {prop.public_name}'s offer, and {rec.act_name} for ${pretty_number(price)}\n"
                + (f"(2) Reject {prop.public_name}'s offer and proceed to the next round, where you will send "
                   f"{prop.public_name} a counteroffer\n" if counter else f"(2) Reject {prop.public_name}'s offer\n")
                + f"(3) {rec.john_text}\n")
        rec.buffer += text
        rec.edit_system_message(system)
        rec.act(decision_ok)
        d = rec.last_action_json["decision"]
        log.append({"decision": d, "player": rec.public_name, "round": r})
        if d == "AcceptOffer":
            prop.buffer += f"{rec.public_name} accepted your offer."
            rec.buffer += f"You accepted {prop.public_name}'s offer."
            break
        if d == "RejectOffer" and counter:
            prop.buffer += f"{rec.public_name} rejected your offer from round {r}.\n\n"
            rec.buffer += f"You chose to make a counteroffer to {prop.public_name}.\n\n"
            continue
        prop.buffer += f"{rec.public_name} rejected your offer from round {r} and ended the negotiation.\n\n"
        rec.buffer += "ended.\n\n"
        break
    for pl in (s, b):
        pl.end_chat()
    return log


# ============================================================================================ persuasion
def play_persuasion(p1: Player, p2: Player, is_myopic, product_price, p, c, v=0, total_rounds=20,
                    is_seller_know_cv=True, is_buyer_know_p=True, seller_message_type="text",
                    allow_buyer_message=False, rng: random.Random | None = None) -> list[dict]:
    rng = rng or random.Random()
    seller, buyer = p1, p2
    cv_, vv = c * product_price, v * product_price
    if is_myopic:
        buyer.public_name = "the buyer"
    pct = pretty_number(p * 100)
    if is_buyer_know_p:
        buyer_p = (f"Each product in the game has a {p * 100:.2f}% chance of being high quality and a "
                   f"{(1 - p) * 100:.2f}% chance of being low quality.")
        seller_p = f"and {buyer.public_name} knows this"
    else:
        buyer_p = "You do not know what percentage of the products are of high quality."
        seller_p = f"and {buyer.public_name} does not know this"
    if is_seller_know_cv:
        buyer_cv = f"{seller.public_name} knows how much each product worth to you."
        seller_cv = (f"A high-quality product is worth ${pretty_number(vv)} to {buyer.public_name}, and a low-quality "
                     f"product is worth only ${pretty_number(cv_)} to them.")
    else:
        buyer_cv = f"{seller.public_name} does not know how much each product worth to you."
        seller_cv = (f"You do not know how much a high-quality product is worth to {buyer.public_name}, nor do you "
                     "know how much a low-quality product is worth to them.")
    who = "random buyers" if is_myopic else buyer.public_name
    intro = f"You are {seller.public_name}, a seller trying to sell products to {who} for ${product_price} each. "
    intro += "We will relay your messages to the buyers." if is_myopic else \
        f"We will relay your messages to {buyer.public_name}."
    v1 = f"{buyer.public_name}, the buyer" if not is_myopic else "random buyers"
    v2 = buyer.public_name if not is_myopic else "the buyers"
    stats_msg = (" or about any specific previous product, but they have statistics about your selling recoder: how "
                 "many products you have sold, and how many of them are of high quality.") if is_myopic else ""
    if not is_myopic:
        prior = (f"For {total_rounds} rounds, in each round, {seller.public_name} will randomly choose one of the "
                 "products she is selling, reveal its true quality, and then send you a recommendation regarding that "
                 "product. You will decide whether to buy it or pass, in which case the product will be thrown away "
                 "and no one will ever be able to buy it. At the end of the round,")
    else:
        prior = (f"For each buyer, including yourself, {seller.public_name} randomly selects a product and reveals its "
                 f"true quality. Based on the quality of the product, {seller.public_name} then provides you with a "
                 "recommendation regarding that specific product. Each buyer also have access to the seller’s previous "
                 "sales record: how many potential buyers there have been, how many products were sold, and the "
                 "proportion of high-quality products sold.You must decide whether to buy the product or pass on it. "
                 "If you pass, the product will be discarded, and no one will be able to buy it. After making your "
                 "decision, ")
    seller.set_rules(f"{intro} It is known that some of the products are high-quality and some are low-quality.\n"
                     f"The probability that a product is high-quality is {pct}%, {seller_p}.\n{seller_cv}\n"
                     f"For {total_rounds} rounds, you will offer products to {v1}. At the beginning of each round, you "
                     f"will discover whether the product is high-quality or low-quality. {v2.capitalize()} does not know "
                     f"this detail{stats_msg}. You can send any message you wish to {v2}, and based only on this message, "
                     f"they will decide whether to buy the product or not. At the end of the round, {v2} will discover "
                     "the quality of the product.\nThe price is final, no discounts are allowed, and the product is "
                     "non-refundable after purchase.")
    buyer.set_rules(f"Let's play a game.\n{seller.public_name} is a seller offering products for a fixed price of "
                    f"${pretty_number(product_price)}. You are {'one of the buyers' if is_myopic else buyer.public_name + ', the buyer'}.\n"
                    f"{buyer_p}\nA high-quality product is worth ${pretty_number(vv)} to you, and a low-quality product "
                    f"is worth ${pretty_number(cv_)} to you. {buyer_cv}\n{prior} you will find out whether the product "
                    "was of high quality or low quality.")
    rival = "the buyer" if is_myopic else buyer.public_name
    seller.req_offer_text = (f'Send your message in JSON format:\n{{"message": "Your message to {rival}"}}'
                             if seller_message_type == "text" else
                             f'Would you recommend {rival} to buy the product?\\nAnswer in JSON with '
                             '{"decision": "yes"} or {"decision": "no"}')
    buyer.req_offer_text = (f'Answer in JSON format with {{"decision": "yes", "message": "your message to '
                            f'{seller.public_name}"}} or {{"decision": "no", "message": "your message to '
                            f'{seller.public_name}"}}' if allow_buyer_message else
                            'Answer in JSON format with {"decision": "yes"} or {"decision": "no"}')
    base = {"is_myopic": is_myopic, "product_price": product_price, "total_rounds": total_rounds,
            "seller_message_type": seller_message_type, "allow_buyer_message": allow_buyer_message}
    seller.game_params = {**base, "role": "seller", "p": p, **({"c": c, "v": v} if is_seller_know_cv else {})}
    buyer.game_params = {**base, "role": "buyer", **({"p": p} if is_buyer_know_p else {}), "c": c, "v": v}

    def message_ok(text):
        a, why = _first_json(text)
        if a is None:
            return False, why
        return isinstance(a.get("message"), str), "message required"

    def decision_ok(kind):
        def check(text):
            a, why = _first_json(text)
            if a is None:
                return False, why
            need_msg = (kind == "buyer" and allow_buyer_message) or (kind == "seller" and seller_message_type == "text")
            if need_msg and "message" not in a:
                return False, "message required"
            return a.get("decision") in ("yes", "no"), "decision must be yes/no"
        return check

    log, sold, sold_high = [], 0, 0
    for r in range(1, total_rounds + 1):
        high = rng.random() < p
        quality = "high-quality" if high else "low-quality"
        log.append({"round_quality": quality, "product_worth": vv if high else cv_, "player": "Nature", "round": r})
        new_buyer = "A new potential buyer has arrived!\n" if is_myopic else ""
        seller.buffer += (f"Round {r}\nThis round's product is {quality}.\n{new_buyer}"
                          + (f"Send {'the potential buyer' if is_myopic else buyer.public_name} a message to help them "
                             "decide whether to buy the product." if seller_message_type == "text" else
                             f"Decide whether to recommend {'the buyer' if is_myopic else buyer.public_name} to buy "
                             "the product."))
        seller.edit_system_message(seller.req_offer_text)
        seller.act(message_ok if seller_message_type == "text" else decision_ok("seller"))
        content = seller.last_action_json["message" if seller_message_type == "text" else "decision"]
        log.append({**seller.last_action_json, "player": seller.public_name, "round": r})
        if is_myopic:
            intro_r = (f"Statistics: Based on {seller.public_name}'s recommendations, {sold} of the buyers purchased the "
                       f"products offered for them. So far, {sold_high} high-quality product have been sold. You can be "
                       f"the {ordinal(sold + 1)} buyer!\n~" if sold else
                       "Statistics: No products have been sold yet. you can be the first one!\n~")
        else:
            intro_r = f"\n\nRound {r}"
        text = f"{intro_r}\n"
        if seller_message_type == "text":
            text += f"{seller.public_name}'s message: {content}"
        else:
            text += (f"{seller.public_name} recommends you to buy the product." if content == "yes" else
                     f"{seller.public_name} does not recommend you to buy the product.")
        text += "\n\nWould you like to buy the current product? \n"
        buyer.buffer += text
        buyer.edit_system_message(buyer.req_offer_text)
        buyer.act(decision_ok("buyer"))
        log.append({**buyer.last_action_json, "player": buyer.public_name, "round": r})
        bought = buyer.last_action_json["decision"] == "yes"
        worth = pretty_number(vv if high else cv_)
        if bought:
            seller_text = f"{buyer.public_name.capitalize()} bought the product!"
            buyer.buffer += f"You bought the product from {seller.public_name} for ${pretty_number(product_price)}!\n"
            buyer.buffer += f"The product was {quality}. It's worth ${worth} for you."
            sold += 1
            sold_high += int(high)
        else:
            seller_text = f"{buyer.public_name.capitalize()} didn't bought the product."
            buyer.buffer += f"You didn't buy the product from {seller.public_name}.\n"
            buyer.buffer += f"The product was {quality}. It could have been worth ${worth} for you."
        if allow_buyer_message and buyer.last_action_json.get("message"):
            seller.buffer += f"\n{buyer.public_name.capitalize()}'s message: {buyer.last_action_json['message']}\n"
        seller.buffer += seller_text
        if r < total_rounds and is_myopic:
            buyer.end_chat()
            seller.buffer += f"\n{buyer.public_name} left. Waiting for the next buyer.".capitalize()
        seller.buffer += "\n\n"
    for pl in (seller, buyer):
        pl.end_chat()
    return log


GAMES = {"bargaining": play_bargaining, "negotiation": play_negotiation, "persuasion": play_persuasion}


def play(game_type: str, game_args: dict, alice: Respond, bob: Respond, rng: random.Random | None = None):
    """Play one game; `alice` is GLEE's player 1 (proposes first / is the seller)."""
    p1, p2 = Player(alice, "Alice", player_id=0), Player(bob, "Bob", player_id=3)
    args = dict(game_args)
    if game_type == "persuasion":
        args["rng"] = rng
        args.setdefault("v", 0)
    return GAMES[game_type](p1, p2, **args)


def random_player(rng: random.Random | None = None) -> Respond:
    """A player answering at random in GLEE's format, like GLEE's DemoPlayer (for robustness tests)."""
    rng = rng or random.Random(0)

    def respond(payload: dict) -> str:
        system = payload["messages"][0]["content"]
        prompt = payload["messages"][-1]["content"]
        gp = payload["game_params"]
        gains = re.findall(r'"(\w+)_gain"', system)
        if "Do you accept this offer?" in prompt:
            return json.dumps({"decision": rng.choice(["accept", "reject"])})
        if len(gains) >= 2:
            money = gp["money_to_divide"]
            x = rng.randint(0, int(money))
            return json.dumps({f"{gains[0]}_gain": x, f"{gains[1]}_gain": money - x, "message": "deal?"})
        if "You have three options" in prompt:
            john = "SellToJhon" if gp.get("role") == "seller" else "BuyFromJhon"
            return json.dumps({"decision": rng.choice(["AcceptOffer", "RejectOffer", john]), "message": "ok"})
        if '"product_price"' in system:
            return json.dumps({"product_price": round(rng.uniform(0.5, 2) * gp["product_price_order"]),
                               "message": "my price"})
        if "Send your message" in system:
            return json.dumps({"message": "Buy it."})
        return json.dumps({"decision": rng.choice(["yes", "no"]), "message": "hm"})
    return respond
