"""Shared helpers for the training and evaluation scripts."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glee_agent.features import (  # noqa: E402
    bargaining_features, negotiation_features, persuasion_features,
)
from glee_agent.theory import persuasion_threshold  # noqa: E402

DATA = ROOT / "data"
TEST_FRACTION = 0.2


def is_test_game(game_id: str) -> bool:
    """Deterministic 80/20 split by game, so every offer of a game lands on the same side."""
    return int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 100 < TEST_FRACTION * 100


def load(name: str, split: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{name}.parquet")
    if split is not None:
        test = df["game_id"].map(is_test_game)
        df = df[test] if split == "test" else df[~test]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------------- design matrices
def with_history(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Add `prev`: the previous round's offer in the same game (made by the current responder)."""
    df = df.sort_values(["game_id", "round"]).copy()
    prev = df.groupby("game_id")[value].shift(1)
    consecutive = df.groupby("game_id")["round"].diff() == 1
    df["prev"] = prev.where(consecutive)
    return df.reset_index(drop=True)


def bargaining_xy(df: pd.DataFrame):
    X = bargaining_features(df.share_to_responder, df["round"], df.max_rounds, df.delta_responder,
                            df.delta_proposer, df.proposer_is_p1, df.complete_information,
                            df.messages_allowed, df.money, last_demand=1 - df.prev)
    return X, df.accepted.astype(int).to_numpy()


def negotiation_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = with_history(df, "price")
    df = df[(df.price >= 0) & (df.price <= 5) & df.decision.notna()].copy()
    df.loc[df.prev > 5, "prev"] = np.nan
    df["responder"] = np.where(df.proposer_role == "seller", "buyer", "seller")
    df["accepted"] = df.decision == "AcceptOffer"
    df["john"] = df.decision.isin(["BuyFromJhon", "SellToJhon", "DealWithJhon"])
    return df.reset_index(drop=True)


def negotiation_x(df: pd.DataFrame, responder: str):
    return negotiation_features(df.price, responder, df.seller_value, df.buyer_value, df["round"],
                                df.max_rounds, df.complete_information, df.messages_allowed, df.price_order,
                                responder_last_price=df.prev)


def persuasion_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-signal games only, with the record each buyer had seen before deciding."""
    df = df[(df.message_type == "binary") & df.seller_recommends.notna()].copy()
    df = df.sort_values(["game_id", "round"])
    df["seller_recommends"] = df.seller_recommends.astype(bool)
    rec, high, sold = (df.seller_recommends.astype(int), df.high_quality.astype(int), df.bought.astype(int))
    g = df.groupby("game_id")
    # non-myopic buyer: every past recommendation and its revealed quality
    df["rec_count"] = rec.groupby(df.game_id).cumsum() - rec
    df["rec_high"] = (rec * high).groupby(df.game_id).cumsum() - rec * high
    # myopic buyer: GLEE's public statistics (products sold, high-quality products sold)
    df["sold_count"] = sold.groupby(df.game_id).cumsum() - sold
    df["sold_high"] = (sold * high).groupby(df.game_id).cumsum() - sold * high
    df["count"] = np.where(df.is_myopic, df.sold_count, df.rec_count)
    df["high_count"] = np.where(df.is_myopic, df.sold_high, df.rec_high)
    df["threshold"] = [persuasion_threshold(v, c) for v, c in zip(df.v, df.c)]
    df["total_rounds"] = g["round"].transform("max")
    del g
    return df.reset_index(drop=True)


def persuasion_x(df: pd.DataFrame, recommends=None):
    rec = df.seller_recommends if recommends is None else recommends
    return persuasion_features(rec, df.high_count, df["count"], df.threshold, df.p, df.is_myopic,
                               df["round"], df.total_rounds)
