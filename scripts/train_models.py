"""Fit the opponent models on the training games (80%) and score them on held-out games (20%).

Writes models/opponent_models.joblib and results/model_metrics.json.

    python scripts/train_models.py
"""
from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (ROOT, bargaining_xy, load, negotiation_frame, negotiation_x, persuasion_frame,
                    persuasion_x, with_history)
from glee_agent.features import BARGAINING_FEATURES, NEGOTIATION_FEATURES, PERSUASION_FEATURES
from glee_agent.opponent import BackoffHistogram, OpponentModels, last_offer_bin, round_bucket


def gbm(features: list[str], increasing=(), decreasing=()) -> HistGradientBoostingClassifier:
    mono = [1 if f in increasing else -1 if f in decreasing else 0 for f in features]
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          monotonic_cst=mono, random_state=0)


def logit():
    """Baseline: logistic regression on the same features (missing history imputed, with an indicator)."""
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=3000))


def score(y, p) -> dict:
    return {"n": int(len(y)), "positive_rate": round(float(np.mean(y)), 4),
            "log_loss": round(float(log_loss(y, p)), 4),
            "log_loss_base_rate": round(float(log_loss(y, np.full(len(y), np.mean(y)))), 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4)}


def calibration(y, p, bins=10) -> list[dict]:
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(predicted=("p", "mean"),
                                                                             observed=("y", "mean"),
                                                                             n=("y", "size"))
    return [{k: round(float(v), 4) if k != "n" else int(v) for k, v in row.items()} for row in t.to_dict("records")]


def fit_and_score(name, X_tr, y_tr, X_te, y_te, model, metrics):
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    base = logit().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    metrics[name] = {"gradient_boosting": score(y_te, p), "logistic_baseline": score(y_te, base),
                     "calibration": calibration(y_te, p)}
    print(f"{name:28s} test log-loss {metrics[name]['gradient_boosting']['log_loss']:.4f} "
          f"(logistic {metrics[name]['logistic_baseline']['log_loss']:.4f}, "
          f"base rate {metrics[name]['gradient_boosting']['log_loss_base_rate']:.4f}), "
          f"AUC {metrics[name]['gradient_boosting']['auc']:.3f}")
    return model


def fit_models(bargaining: pd.DataFrame, negotiation: pd.DataFrame, persuasion: pd.DataFrame,
               test: dict | None = None) -> OpponentModels:
    """Fit every opponent model on the given games; score on `test` frames if provided."""
    metrics: dict = {}

    def fit(name, X, y, model, test_xy):
        if test_xy is None:
            return model.fit(X, y)
        return fit_and_score(name, X, y, *test_xy, model, metrics)

    # ---------------------------------------------------------------- bargaining
    b = with_history(bargaining, "share_to_responder")
    te = with_history(test["bargaining"], "share_to_responder") if test else None
    b_accept = fit("bargaining_accept", *bargaining_xy(b),
                   gbm(BARGAINING_FEATURES, increasing={"share", "share_below_half", "share_above_half",
                                                       "at_least_half", "spne_gap", "gap_to_last_demand"}),
                   bargaining_xy(te) if test else None)
    b = b.assign(rb=[round_bucket(r, T) for r, T in zip(b["round"], b.max_rounds)],
                 ci=b.complete_information.astype(bool),
                 prev_bin=last_offer_bin(b.prev),  # what this proposer was offered last round
                 d_resp_known=np.where(b.complete_information, b.delta_responder, -1.0))
    b_offer = BackoffHistogram(
        levels=[[], ["ci"], ["ci", "rb"], ["ci", "rb", "prev_bin"], ["ci", "rb", "prev_bin", "delta_proposer"],
                ["ci", "rb", "prev_bin", "delta_proposer", "d_resp_known"]],
        edges=np.arange(-0.025, 1.026, 0.05)).fit(b, "share_to_responder")

    # ---------------------------------------------------------------- negotiation
    n = negotiation_frame(negotiation)
    nt = negotiation_frame(test["negotiation"]) if test else None
    n_accept, n_john = {}, {}
    gains = {"responder_gain", "gain_negative", "gain_positive", "no_loss", "gain_vs_own_last_offer"}
    for role in ("buyer", "seller"):
        a = n[n.responder == role]
        bt = nt[nt.responder == role] if test else None
        n_accept[role] = fit(f"negotiation_accept_{role}", negotiation_x(a, role), a.accepted.astype(int),
                             gbm(NEGOTIATION_FEATURES, increasing=gains),
                             (negotiation_x(bt, role), bt.accepted.astype(int).to_numpy()) if test else None)
        a = a[~a.accepted]
        bt = bt[~bt.accepted] if test else None
        n_john[role] = fit(f"negotiation_john_{role}", negotiation_x(a, role), a.john.astype(int),
                           gbm(NEGOTIATION_FEATURES, decreasing=gains),
                           (negotiation_x(bt, role), bt.john.astype(int).to_numpy()) if test else None)
    own = np.where(n.proposer_role == "seller", n.seller_value, n.buyer_value)
    n = n.assign(rb=[round_bucket(r, T) for r, T in zip(n["round"], n.max_rounds)],
                 ci=n.complete_information.astype(bool),
                 one_shot=n.max_rounds == 1,
                 prev_bin=last_offer_bin(n.prev / own),
                 ratio=n.price / own,
                 other_known=np.where(n.complete_information,
                                      np.where(n.proposer_role == "seller", n.buyer_value, n.seller_value), -1.0))
    n_offer = BackoffHistogram(
        levels=[[], ["proposer_role"], ["proposer_role", "ci"], ["proposer_role", "ci", "one_shot", "rb"],
                ["proposer_role", "ci", "one_shot", "rb", "prev_bin"],
                ["proposer_role", "ci", "one_shot", "rb", "prev_bin", "other_known"]],
        edges=np.arange(-0.025, 3.026, 0.05)).fit(n, "ratio")

    # ---------------------------------------------------------------- persuasion
    p = persuasion_frame(persuasion)
    pt = persuasion_frame(test["persuasion"]) if test else None
    p_buy = fit("persuasion_buy", persuasion_x(p), p.bought.astype(int),
                gbm(PERSUASION_FEATURES, increasing={"recommends", "rec_x_track_record"}),
                (persuasion_x(pt), pt.bought.astype(int).to_numpy()) if test else None)
    rates = p.groupby("high_quality").seller_recommends.mean()
    rates = {"low": round(float(rates[False]), 4), "high": round(float(rates[True]), 4)}
    metrics["persuasion_seller_recommend_rate"] = rates

    return OpponentModels(bargaining_accept=b_accept, bargaining_offer=b_offer, negotiation_accept=n_accept,
                          negotiation_john=n_john, negotiation_offer=n_offer, persuasion_buy=p_buy,
                          seller_recommend_rate=rates, metrics=metrics,
                          versions={pkg: version(pkg) for pkg in ("scikit-learn", "numpy", "pandas")})


def main() -> None:
    names = {"bargaining": "bargaining_offers", "negotiation": "negotiation_offers", "persuasion": "persuasion_rounds"}
    train = {g: load(f, "train") for g, f in names.items()}
    test = {g: load(f, "test") for g, f in names.items()}
    models = fit_models(**train, test=test)
    models.save()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model_metrics.json").write_text(json.dumps(models.metrics, indent=2))
    print("saved models/opponent_models.joblib and results/model_metrics.json")


if __name__ == "__main__":
    main()
