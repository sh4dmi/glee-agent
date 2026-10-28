"""Evaluate the agent's policies against simulated LLM opponents on held-out game configurations.

Protocol
  * Configurations: a random sample of games from the 20% test split (never seen in training).
  * Opponent: LLMProxy, sampling from models re-fitted on the test split only, so the agent is
    never scored by the same models it plans with.
  * Each configuration is played with the agent in both seats, with the same random seed for
    every policy (common random numbers), so policies are compared on identical situations.
  * Reference points: simple rule-based policies, the proxy itself in the agent's seat
    ("LLM proxy"), and what the real LLMs earned in those exact test games ("LLM, real games").

    python scripts/evaluate.py --games 200
Writes results/evaluation.csv (one row per game) and results/evaluation_summary.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import ROOT, load
import simulator as sim
from glee_agent.graph import Agent
from glee_agent.llm import LanguageModel
from glee_agent.opponent import OpponentModels
from proxy import LLMProxy
from train_models import fit_models

POLICIES = {
    "bargaining": ["best_response", "spne", "fair", "proxy"],
    "negotiation": ["best_response", "spne", "split", "proxy"],
    "persuasion_seller": ["best_response", "commitment", "honest", "always", "proxy"],
    "persuasion_buyer": ["best_response", "naive", "proxy"],
}

_STATE: dict = {}


# ------------------------------------------------------------------------------------- configurations
def sample_configs(n: int, seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    out = {}

    b = load("bargaining_offers", "test")
    first = b[b["round"] == 1].drop_duplicates("game_id")
    first = first.iloc[rng.choice(len(first), size=min(n, len(first)), replace=False)]
    out["bargaining"] = [{"game_id": r.game_id, "args": dict(
        money_to_divide=float(r.money), max_rounds=int(r.max_rounds), complete_information=bool(r.complete_information),
        messages_allowed=bool(r.messages_allowed), delta_1=float(r.delta_proposer), delta_2=float(r.delta_responder))}
        for r in first.itertuples()]

    ng = load("negotiation_offers", "test").drop_duplicates("game_id")
    ng = ng.iloc[rng.choice(len(ng), size=min(n, len(ng)), replace=False)]
    out["negotiation"] = [{"game_id": r.game_id, "args": dict(
        seller_value=float(r.seller_value), buyer_value=float(r.buyer_value), product_price_order=float(r.price_order),
        max_rounds=int(r.max_rounds), complete_information=bool(r.complete_information),
        messages_allowed=bool(r.messages_allowed))} for r in ng.itertuples()]

    ps = load("persuasion_rounds", "test").drop_duplicates("game_id")
    ps = ps.iloc[rng.choice(len(ps), size=min(n, len(ps)), replace=False)]
    out["persuasion"] = [{"game_id": r.game_id, "args": dict(
        is_myopic=bool(r.is_myopic), product_price=100, p=float(r.p), v=float(r.v), c=float(r.c), total_rounds=20,
        is_seller_know_cv=bool(r.seller_knows_cv), is_buyer_know_p=True, seller_message_type=r.message_type,
        allow_buyer_message=False)} for r in ps.itertuples()]
    return out


def real_outcomes(configs: dict) -> pd.DataFrame:
    """What the actual LLM players earned in the sampled test games, per seat."""
    rows = []
    b = load("bargaining_offers", "test")
    ids = {c["game_id"] for c in configs["bargaining"]}
    for gid, g in b[b.game_id.isin(ids)].groupby("game_id"):
        acc = g[g.accepted]
        p1 = p2 = 0.0
        if len(acc):
            r = acc.iloc[0]
            to_resp = r.share_to_responder
            d_prop, d_resp = r.delta_proposer, r.delta_responder
            mine, theirs = (1 - to_resp) * d_prop ** (r["round"] - 1), to_resp * d_resp ** (r["round"] - 1)
            p1, p2 = (mine, theirs) if r.proposer_is_p1 else (theirs, mine)
        rows += [{"game": "bargaining", "game_id": gid, "seat": 1, "policy": "LLM (real games)", "payoff": p1},
                 {"game": "bargaining", "game_id": gid, "seat": 2, "policy": "LLM (real games)", "payoff": p2}]
    n = load("negotiation_offers", "test")
    ids = {c["game_id"] for c in configs["negotiation"]}
    for gid, g in n[n.game_id.isin(ids)].groupby("game_id"):
        acc = g[g.decision == "AcceptOffer"]
        vs, vb = g.seller_value.iloc[0], g.buyer_value.iloc[0]
        s = b_ = 0.0
        if len(acc):
            price = acc.iloc[0].price
            s, b_ = price - vs, vb - price
        zopa = vb - vs
        rows += [{"game": "negotiation", "game_id": gid, "seat": 1, "policy": "LLM (real games)", "payoff": s,
                  "zopa": zopa},
                 {"game": "negotiation", "game_id": gid, "seat": 2, "policy": "LLM (real games)", "payoff": b_,
                  "zopa": zopa}]
    p = load("persuasion_rounds", "test")
    ids = {c["game_id"] for c in configs["persuasion"]}
    for gid, g in p[p.game_id.isin(ids)].groupby("game_id"):
        gain = np.where(g.bought, np.where(g.high_quality, g.v, g.c) - 1, 0.0)
        rows += [{"game": "persuasion", "game_id": gid, "seat": 1, "policy": "LLM (real games)",
                  "payoff": float(g.bought.mean())},
                 {"game": "persuasion", "game_id": gid, "seat": 2, "policy": "LLM (real games)",
                  "payoff": float(gain.mean())}]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- one game
def _init(eval_models_path: str):
    _STATE["agent_models"] = OpponentModels.load()
    _STATE["eval_models"] = OpponentModels.load(eval_models_path)
    _STATE["agents"] = {}


def _agent(game: str, mode: str):
    key = (game, mode)
    if key not in _STATE["agents"]:
        _STATE["agents"][key] = Agent(models=_STATE["agent_models"], llm=LanguageModel(api_key=None),
                                      modes={game: mode})
    return _STATE["agents"][key].respond


def _naive_buyer(payload: dict) -> str:
    from glee_agent.protocol import parse_request
    from glee_agent.policies.persuasion import keyword_recommends
    t = parse_request(payload)
    rec = t.seller_signal == "yes" if not t.text_messages else keyword_recommends(t.seller_signal or "")
    return json.dumps({"decision": "yes" if rec else "no"})


def play_one(task: tuple) -> dict:
    game, cfg, policy, seat, seed = task
    proxy = LLMProxy(_STATE["eval_models"], seed=seed)
    if policy == "proxy":
        me = LLMProxy(_STATE["eval_models"], seed=seed + 7919)
    elif policy == "naive":
        me = _naive_buyer
    else:
        mode = {"persuasion_seller": "persuasion", "persuasion_buyer": "persuasion"}.get(game, game)
        me = _agent(mode, policy)
    alice, bob = (me, proxy) if seat == 1 else (proxy, me)
    gt = "persuasion" if game.startswith("persuasion") else game
    t0 = time.time()
    log = pd.DataFrame(sim.play(gt, cfg["args"], alice, bob, rng=random.Random(seed)))
    a = cfg["args"]
    row = {"game": gt, "game_id": cfg["game_id"], "seat": seat, "policy": policy, "seconds": time.time() - t0}
    if gt == "bargaining":
        acc = log[log.get("decision") == "accept"] if "decision" in log else log.iloc[0:0]
        payoff, deal_round = 0.0, None
        if len(acc):
            r = int(acc.iloc[0]["round"])
            offer = log[(log["round"] == r) & log["decision"].isna()].iloc[0]
            name = "alice" if seat == 1 else "bob"
            payoff = float(offer[f"{name}_gain"]) / a["money_to_divide"] * a[f"delta_{seat}"] ** (r - 1)
            deal_round = r
        row.update(payoff=payoff, deal=deal_round is not None, rounds=deal_round)
    elif gt == "negotiation":
        acc = log[log.get("decision") == "AcceptOffer"] if "decision" in log else log.iloc[0:0]
        payoff = 0.0
        if len(acc):
            r = int(acc.iloc[0]["round"])
            price = float(log[(log["round"] == r) & log["decision"].isna()].iloc[0]["product_price"]) / a["product_price_order"]
            payoff = price - a["seller_value"] if seat == 1 else a["buyer_value"] - price
        row.update(payoff=payoff, deal=len(acc) > 0, zopa=a["buyer_value"] - a["seller_value"])
    else:
        nature = log[log.player == "Nature"].set_index("round").round_quality
        buyer = log[(log.player != "Nature") & (log.player != "Alice")].set_index("round")
        bought = buyer.decision == "yes"
        high = nature.reindex(bought.index) == "high-quality"
        if seat == 1:
            payoff = float(bought.mean())
        else:
            payoff = float(np.where(bought, np.where(high, a["v"], a["c"]) - 1.0, 0.0).mean())
        row.update(payoff=payoff)
    return row


# ------------------------------------------------------------------------------------------- main
def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for (game, seat), part in df.groupby(["game", "seat"]):
        key = f"{game}/{'player 1' if seat == 1 else 'player 2'}"
        if game == "persuasion":
            key = f"persuasion/{'seller' if seat == 1 else 'buyer'}"
        stats = {}
        for policy, g in part.groupby("policy"):
            x = g.payoff.to_numpy(float)
            if game == "negotiation":  # share of the available surplus, games with a zone of agreement only
                z = g.zopa.to_numpy(float)
                x = x[z > 0] / z[z > 0]
            stats[policy] = {"mean": float(x.mean()), "ci95": float(1.96 * x.std(ddof=1) / np.sqrt(len(x))),
                             "n": int(len(x))}
            if "deal" in g and g.deal.notna().any() and policy != "LLM (real games)":
                stats[policy]["deal_rate"] = float(g.deal.astype(float).mean())
        out[key] = stats
    return out


def adversarial_buyer_check(configs: list[dict], seed: int) -> dict:
    """Buyers against a seller who recommends every product: does skepticism pay off?"""
    always = lambda payload: json.dumps({"message": "Buy it, it's great."}  # noqa: E731
                                        if "Send your message" in payload["messages"][0]["content"]
                                        else {"decision": "yes"})
    out = {}
    for name in ("best_response", "naive"):
        me = _naive_buyer if name == "naive" else _agent("persuasion", "best_response")
        gains = []
        for i, cfg in enumerate(configs):
            log = pd.DataFrame(sim.play("persuasion", cfg["args"], always, me, rng=random.Random(seed + i)))
            a = cfg["args"]
            nature = log[log.player == "Nature"].set_index("round").round_quality
            buyer = log[(log.player != "Nature") & (log.player != "Alice")].set_index("round")
            bought = buyer.decision == "yes"
            high = nature.reindex(bought.index) == "high-quality"
            gains.append(float(np.where(bought, np.where(high, a["v"], a["c"]) - 1.0, 0.0).mean()))
        x = np.array(gains)
        out[name] = {"mean": float(x.mean()), "ci95": float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), "n": len(x)}
    return out


def paired(df: pd.DataFrame) -> dict:
    """Agent minus each other policy on the same configuration and seed (mean and 95% CI)."""
    out = {}
    sim_rows = df[df.policy != "LLM (real games)"]
    for (game, seat), part in sim_rows.groupby(["game", "seat"]):
        key = f"{game}/{('seller' if seat == 1 else 'buyer') if game == 'persuasion' else f'player {seat}'}"
        table = part.pivot_table(index="game_id", columns="policy", values="payoff")
        if game == "negotiation":
            zopa = part.groupby("game_id").zopa.first()
            table = table[zopa > 0].div(zopa[zopa > 0], axis=0)
        out[key] = {}
        for other in table.columns.drop("best_response"):
            diff = (table.best_response - table[other]).dropna()
            out[key][other] = {"mean": float(diff.mean()), "ci95": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                               "n": int(len(diff))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="configurations sampled per game")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true", help="recompute the summary from results/evaluation.csv")
    ap.add_argument("--workers", type=int, default=1, help="processes (each also uses OpenMP threads)")
    args = ap.parse_args()
    if args.summary_only:
        df = pd.read_csv(ROOT / "results" / "evaluation.csv")
        summary = summarise(df)
        summary["paired_difference_agent_minus"] = paired(df)
        _init(str(ROOT / "models" / "evaluator_models.joblib"))
        summary["buyer_vs_always_recommend_seller"] = adversarial_buyer_check(
            sample_configs(args.games, args.seed)["persuasion"], args.seed)
        print(summary["buyer_vs_always_recommend_seller"])
        (ROOT / "results" / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["paired_difference_agent_minus"], indent=1))
        return

    eval_path = ROOT / "models" / "evaluator_models.joblib"
    if not eval_path.exists():
        print("fitting evaluator models on the test split ...")
        names = {"bargaining": "bargaining_offers", "negotiation": "negotiation_offers",
                 "persuasion": "persuasion_rounds"}
        fit_models(**{g: load(f, "test") for g, f in names.items()}).save(eval_path)

    configs = sample_configs(args.games, args.seed)
    tasks = []
    for i in range(args.games):
        for game, policies in POLICIES.items():
            base = "persuasion" if game.startswith("persuasion") else game
            if i >= len(configs[base]):
                continue
            seats = [1] if game == "persuasion_seller" else [2] if game == "persuasion_buyer" else [1, 2]
            for seat in seats:
                for policy in policies:
                    tasks.append((game, configs[base][i], policy, seat, args.seed * 100003 + i * 2 + seat))
    print(f"{len(tasks)} games to play")
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(eval_path),)) as pool:
            results = pool.map(play_one, tasks, chunksize=4)
    else:
        _init(str(eval_path))
        results = map(play_one, tasks)
    for k, row in enumerate(results):
        rows.append(row)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(tasks)} games, {time.time() - t0:.0f}s")
    df = pd.concat([pd.DataFrame(rows), real_outcomes(configs)], ignore_index=True)
    adversarial = adversarial_buyer_check(configs["persuasion"], args.seed)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "evaluation.csv", index=False)
    summary = summarise(df)
    summary["paired_difference_agent_minus"] = paired(df)
    summary["buyer_vs_always_recommend_seller"] = adversarial
    (ROOT / "results" / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    for key, stats in summary.items():
        if key in ("paired_difference_agent_minus", "buyer_vs_always_recommend_seller"):
            continue
        print(f"\n{key}")
        for policy, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean"]):
            deal = f"  deals {s['deal_rate']:.0%}" if "deal_rate" in s else ""
            print(f"  {policy:20s} {s['mean']:+.3f} ± {s['ci95']:.3f}  (n={s['n']}){deal}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
