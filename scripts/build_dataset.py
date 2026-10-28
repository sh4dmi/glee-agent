"""Turn the GLEE LLM-vs-LLM game logs into flat tables.

GLEE ships one folder per game (config.json + game.csv). This script reads every folder under
<glee>/Data/llm_vs_llm and writes three parquet files to data/:

  bargaining_offers.parquet   one row per offer, with the responder's accept/reject decision
  negotiation_offers.parquet  one row per price offer, with the responder's decision
  persuasion_rounds.parquet   one row per round: product quality, seller signal, buyer decision

Usage:
    git clone https://github.com/eilamshapira/GLEE.git ../GLEE
    python scripts/build_dataset.py --glee ../GLEE
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = ("bargaining", "negotiation", "persuasion")


def _model(cfg: dict, player: int) -> str:
    args = cfg.get(f"player_{player}_args", {})
    return str(args.get("model_name", cfg.get(f"player_{player}_type", "unknown")))


def _bargaining_rows(game_id: str, cfg: dict, log: pd.DataFrame) -> list[dict]:
    ga = cfg["game_args"]
    names = [cfg["player_1_args"]["public_name"], cfg["player_2_args"]["public_name"]]
    deltas = [float(ga["delta_1"]), float(ga["delta_2"])]
    money = float(ga["money_to_divide"])
    rows = []
    for rnd, turn in log.groupby("round", sort=True):
        offers = turn[turn["decision"].isna()]
        answers = turn[turn["decision"].notna()]
        if offers.empty:
            continue
        offer = offers.iloc[0]
        proposer = 0 if offer["player"] == names[0] else 1
        responder = 1 - proposer
        key = f"{names[responder].lower().replace(' ', '_')}_gain"
        to_responder = float(offer[key])
        decision = answers.iloc[0]["decision"] if not answers.empty else None
        rows.append({
            "game_id": game_id,
            "round": int(rnd),
            "money": money,
            "max_rounds": int(ga["max_rounds"]),
            "complete_information": bool(ga["complete_information"]),
            "messages_allowed": bool(ga["messages_allowed"]),
            "proposer_is_p1": proposer == 0,
            "delta_proposer": deltas[proposer],
            "delta_responder": deltas[responder],
            "model_proposer": _model(cfg, proposer + 1),
            "model_responder": _model(cfg, responder + 1),
            "share_to_responder": to_responder / money,
            "accepted": None if decision is None else decision == "accept",
        })
    return rows


def _negotiation_rows(game_id: str, cfg: dict, log: pd.DataFrame) -> list[dict]:
    ga = cfg["game_args"]
    order = float(ga["product_price_order"])
    names = [cfg["player_1_args"]["public_name"], cfg["player_2_args"]["public_name"]]
    rows = []
    for rnd, turn in log.groupby("round", sort=True):
        offers = turn[turn["decision"].isna()]
        answers = turn[turn["decision"].notna()]
        if offers.empty:
            continue
        offer = offers.iloc[0]
        proposer = 0 if offer["player"] == names[0] else 1
        decision = answers.iloc[0]["decision"] if not answers.empty else None
        rows.append({
            "game_id": game_id,
            "round": int(rnd),
            "price_order": order,
            "seller_value": float(ga["seller_value"]),
            "buyer_value": float(ga["buyer_value"]),
            "max_rounds": int(ga["max_rounds"]),
            "complete_information": bool(ga["complete_information"]),
            "messages_allowed": bool(ga["messages_allowed"]),
            "proposer_role": "seller" if proposer == 0 else "buyer",
            "model_proposer": _model(cfg, proposer + 1),
            "model_responder": _model(cfg, 2 - proposer),
            "price": float(offer["product_price"]) / order,  # in units of price_order
            "decision": decision,
        })
    return rows


def _persuasion_rows(game_id: str, cfg: dict, log: pd.DataFrame) -> list[dict]:
    ga = cfg["game_args"]
    seller = cfg["player_1_args"]["public_name"]
    rows = []
    for rnd, turn in log.groupby("round", sort=True):
        nature = turn[turn["player"] == "Nature"]
        s = turn[turn["player"] == seller]
        b = turn[~turn["player"].isin(["Nature", seller])]
        if nature.empty or b.empty:
            continue
        rows.append({
            "game_id": game_id,
            "round": int(rnd),
            "p": float(ga["p"]),
            "v": float(ga["v"]),
            "c": float(ga["c"]),
            "is_myopic": bool(ga["is_myopic"]),
            "message_type": ga.get("seller_message_type", "text"),
            "seller_knows_cv": bool(ga.get("is_seller_know_cv", True)),
            "model_seller": _model(cfg, 1),
            "model_buyer": _model(cfg, 2),
            "high_quality": nature.iloc[0]["round_quality"] == "high-quality",
            "seller_recommends": (s.iloc[0]["decision"] == "yes") if (not s.empty and "decision" in s
                                                                       and pd.notna(s.iloc[0]["decision"])) else None,
            "seller_message": s.iloc[0]["message"] if (not s.empty and "message" in s) else None,
            "bought": b.iloc[0]["decision"] == "yes",
        })
    return rows


PARSERS = {"bargaining": _bargaining_rows, "negotiation": _negotiation_rows, "persuasion": _persuasion_rows}


def parse_game(args: tuple[str, str]) -> list[dict]:
    game, folder = args
    path = Path(folder)
    try:
        cfg = json.loads((path / "config.json").read_text())
        log = pd.read_csv(path / "game.csv")
        if log.empty:
            return []
        if "decision" not in log:
            log["decision"] = np.nan
        return PARSERS[game](path.name, cfg, log)
    except Exception:  # a handful of logs are truncated or malformed; skip them
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glee", type=Path, required=True, help="path to a clone of the GLEE repository")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = {"bargaining": "bargaining_offers", "negotiation": "negotiation_offers",
             "persuasion": "persuasion_rounds"}
    for game in GAMES:
        folders = sorted(str(p.parent) for p in (args.glee / "Data" / "llm_vs_llm" / game).glob("*/*/*/*/config.json"))
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [r for chunk in pool.map(parse_game, [(game, f) for f in folders], chunksize=200) for r in chunk]
        df = pd.DataFrame(rows)
        out = args.out / f"{names[game]}.parquet"
        df.to_parquet(out, index=False)
        print(f"{game}: {len(folders):,} games -> {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
