"""Opponent models: how the population of LLM players in GLEE responds and proposes.

Two kinds of model, both fitted by scripts/train_models.py on held-in games only:

* response models, gradient-boosted classifiers with a monotone constraint on the offer
  (more money for the responder never lowers the chance they accept), and
* proposal models, smoothed histograms of the offers LLM proposers make in a given situation.

The fitted models are stored together in one joblib file (models/opponent_models.joblib).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "opponent_models.joblib"


class BackoffHistogram:
    """A conditional histogram P(bin | context) with hierarchical smoothing.

    `levels` lists context columns from coarse to fine, e.g. [[], ["role"], ["role", "round"]].
    The estimate at each level is shrunk toward the level above it:
        p_level = (counts + alpha * p_parent) / (n + alpha),
    so a context seen often is driven by its own data and a rare one falls back to its parent.
    """

    def __init__(self, levels: Sequence[Sequence[str]], edges: np.ndarray, alpha: float = 20.0):
        self.levels = [list(level) for level in levels]
        self.edges = np.asarray(edges, dtype=float)
        self.alpha = alpha
        self.tables: list[dict[tuple, np.ndarray]] = []

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2

    def fit(self, frame: pd.DataFrame, value: str) -> "BackoffHistogram":
        bins = np.clip(np.digitize(frame[value].to_numpy(float), self.edges) - 1, 0, len(self.edges) - 2)
        n_bins = len(self.edges) - 1
        frame = frame.assign(_bin=bins)
        self.tables = []
        for depth, cols in enumerate(self.levels):
            table: dict[tuple, np.ndarray] = {}
            groups = frame.groupby(cols, observed=True) if cols else [((), frame)]
            for key, part in groups:
                key = key if isinstance(key, tuple) else (key,)
                counts = np.bincount(part["_bin"], minlength=n_bins).astype(float)
                parent = self._lookup(depth - 1, dict(zip(cols, key))) if depth else np.full(n_bins, 1 / n_bins)
                table[key] = (counts + self.alpha * parent) / (counts.sum() + self.alpha)
            self.tables.append(table)
        return self

    def _lookup(self, depth: int, context: dict) -> np.ndarray:
        n_bins = len(self.edges) - 1
        probs = np.full(n_bins, 1 / n_bins)
        for d in range(depth + 1):
            key = tuple(context[c] for c in self.levels[d])
            probs = self.tables[d].get(key, probs)
        return probs

    def predict(self, **context) -> np.ndarray:
        """Probability of each bin (aligned with `centers`) in this context."""
        return self._lookup(len(self.levels) - 1, context)


def round_bucket(round_number: int, max_rounds: int) -> str:
    if round_number >= max_rounds:
        return "last"
    return str(min(int(round_number), 6))


NO_HISTORY = -1.0


def last_offer_bin(value, scale: float = 1.0, top: float = 3.0):
    """Context key for "what the other side offered last round": value / scale rounded to 0.1.

    Bargaining uses the share of the pot (scale 1); negotiation uses the price relative to the
    proposer's own value. NaN or None (no previous offer) maps to NO_HISTORY.
    """
    v = np.asarray(value, dtype=float) / scale
    out = np.round(np.clip(v, 0.0, top), 1)
    out = np.where(np.isnan(v), NO_HISTORY, out)
    return out if out.ndim else float(out)


def nearest(value: float, grid: Sequence[float]) -> float:
    grid = np.asarray(grid, dtype=float)
    return float(grid[np.argmin(np.abs(grid - value))])


@dataclass
class OpponentModels:
    """Everything the policies need to predict LLM opponents; loaded once at server start."""

    bargaining_accept: object
    bargaining_offer: BackoffHistogram
    negotiation_accept: dict  # responder role -> classifier
    negotiation_john: dict    # responder role -> classifier for P(go to John | not accept)
    negotiation_offer: BackoffHistogram
    persuasion_buy: object
    # how often LLM sellers recommend low- and high-quality products (binary-signal games)
    seller_recommend_rate: dict = field(default_factory=lambda: {"low": 0.3, "high": 0.99})
    value_grid_persuasion: tuple = (1.2, 1.25, 2.0, 3.0, 4.0)  # GLEE's v values, used when v is hidden
    delta_grid: tuple = (0.8, 0.9, 0.95, 1.0)
    value_grid: tuple = (0.8, 1.0, 1.2, 1.5)
    metrics: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path | str = DEFAULT_MODEL_PATH) -> "OpponentModels":
        return joblib.load(path)
