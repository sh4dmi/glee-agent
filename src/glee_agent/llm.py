"""Optional language model, used only for language: never for a decision.

Two jobs:
  * polish the plain message that goes with an offer, without changing what it commits to;
  * read a free-text message from a seller and say whether it recommends the product.

Configured through environment variables (any OpenAI-compatible endpoint):
  LLM_API_KEY, LLM_BASE_URL (default https://api.openai.com/v1), LLM_MODEL (default gpt-4o-mini).
Without LLM_API_KEY the agent runs fully offline with template messages and keyword rules.
"""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(__name__)

POLISH_PROMPT = """You write the short message a negotiator attaches to an offer in an economic game.
Rewrite the draft below so it is friendly and persuasive, in at most two sentences.
Rules: keep every number exactly as it appears in the draft, add no other numbers, and make no
promises or claims that are not in the draft. Reply with the message only.

Draft: {draft}"""

CLASSIFY_PROMPT = """A seller in a game sent this message about a product to a buyer:

"{text}"

Does the seller recommend buying this product? Answer with one word: yes or no."""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text or "")}


class LanguageModel:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 timeout: float = 20.0):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = None
        if self.api_key:
            from langchain_openai import ChatOpenAI  # imported lazily: not needed offline
            self._client = ChatOpenAI(api_key=self.api_key, base_url=self.base_url, model=self.model,
                                      temperature=0.3, timeout=timeout, max_retries=1)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _ask(self, prompt: str) -> str | None:
        if not self._client:
            return None
        try:
            return str(self._client.invoke(prompt).content).strip()
        except Exception as exc:  # network or provider error: fall back to the template
            log.warning("LLM call failed: %s", exc)
            return None

    def polish(self, draft: str) -> str:
        """A rewritten message, or the draft itself if the rewrite changed any number."""
        text = self._ask(POLISH_PROMPT.format(draft=draft))
        if not text:
            return draft
        text = text.strip().strip('"')
        if len(text) > 400 or not _numbers(text) <= _numbers(draft) or "{" in text:
            log.info("discarded LLM rewrite that changed the offer: %r", text)
            return draft
        return text

    def recommends(self, text: str) -> bool | None:
        answer = self._ask(CLASSIFY_PROMPT.format(text=text))
        if not answer:
            return None
        answer = answer.lower()
        if answer.startswith("yes"):
            return True
        if answer.startswith("no"):
            return False
        return None
