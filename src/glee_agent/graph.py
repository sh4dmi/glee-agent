"""The agent as a LangGraph pipeline: parse -> interpret -> decide -> compose -> validate.

    parse      rebuild the game state from GLEE's request              (protocol.py)
    interpret  buyer only: does the seller's free-text message recommend? (LLM, or keyword rules)
    decide     pick the action with the game's policy                    (policies/, no LLM)
    compose    attach a message; optionally polished by the LLM, never changing a number
    validate   check against GLEE's format; on any problem, send a safe fallback

The decision is made entirely by the policies. The LLM, when configured, only handles language.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .llm import LanguageModel
from .opponent import OpponentModels
from .policies.bargaining import BargainingPolicy
from .policies.negotiation import NegotiationPolicy
from .policies.persuasion import PersuasionPolicy, keyword_recommends
from .protocol import BargainingTurn, NegotiationTurn, PersuasionTurn, parse_request
from .validate import check, fallback, to_json

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    payload: dict
    turn: Any
    recommends: bool | None
    decision: Any
    answer: dict
    response: str
    error: str


class Agent:
    def __init__(self, models: OpponentModels | None = None, llm: LanguageModel | None = None,
                 modes: dict | None = None):
        self.models = models or OpponentModels.load()
        self.llm = llm or LanguageModel()
        modes = modes or {}
        self.policies = {
            BargainingTurn: BargainingPolicy(self.models, modes.get("bargaining", "best_response")),
            NegotiationTurn: NegotiationPolicy(self.models, modes.get("negotiation", "best_response")),
            PersuasionTurn: PersuasionPolicy(self.models, modes.get("persuasion", "best_response")),
        }
        self.graph = self._build()

    # ----------------------------------------------------------------------------- nodes
    def _parse(self, state: AgentState) -> AgentState:
        return {"turn": parse_request(state["payload"])}

    def _interpret(self, state: AgentState) -> AgentState:
        turn = state["turn"]
        if not (isinstance(turn, PersuasionTurn) and turn.role == "buyer" and turn.text_messages):
            return {"recommends": None}
        verdict = self.llm.recommends(turn.seller_signal or "") if self.llm.enabled else None
        return {"recommends": keyword_recommends(turn.seller_signal or "") if verdict is None else verdict}

    def _decide(self, state: AgentState) -> AgentState:
        turn = state["turn"]
        policy = self.policies[type(turn)]
        if isinstance(turn, PersuasionTurn):
            decision = policy.decide(turn, state.get("recommends"))
        else:
            decision = policy.decide(turn)
        log.info("%s | %s", type(turn).__name__.removesuffix("Turn").lower(), decision.rationale)
        return {"decision": decision}

    def _compose(self, state: AgentState) -> AgentState:
        decision = state["decision"]
        answer = dict(decision.action)
        if decision.draft_message is not None:
            answer["message"] = self.llm.polish(decision.draft_message) if self.llm.enabled else decision.draft_message
        return {"answer": answer}

    def _validate(self, state: AgentState) -> AgentState:
        problem = check(state["turn"], state["answer"])
        if problem:
            log.error("answer failed GLEE's format check (%s): %s", problem, state["answer"])
            return {"response": to_json(fallback(state["payload"])), "error": problem}
        return {"response": to_json(state["answer"])}

    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in [("parse", self._parse), ("interpret", self._interpret), ("decide", self._decide),
                         ("compose", self._compose), ("validate", self._validate)]:
            g.add_node(name, fn)
        g.set_entry_point("parse")
        g.add_edge("parse", "interpret")
        g.add_edge("interpret", "decide")
        g.add_edge("decide", "compose")
        g.add_edge("compose", "validate")
        g.add_edge("validate", END)
        return g.compile()

    # ----------------------------------------------------------------------------- entry
    def respond(self, payload: dict) -> str:
        """The JSON string to return to GLEE. Never raises: falls back to a safe valid answer."""
        try:
            return self.graph.invoke({"payload": payload})["response"]
        except Exception:
            log.exception("agent failed; sending fallback answer")
            return to_json(fallback(payload))
