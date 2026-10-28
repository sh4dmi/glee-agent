# Design

This document explains how the agent decides, why it is built this way, and where its limits are.
The README has the results; the code has the details.

## 1. The setting

[GLEE](https://github.com/eilamshapira/GLEE) (Shapira, Madmon, Reinman, Amoyal, Reichart and Tennenholtz, 2024) runs three two-player games in
natural language. A player is an HTTP server: on every turn GLEE posts the whole conversation so far,
and the player answers with a JSON action (and usually a message to the other side).

| Game | Players | Action | Payoff |
|---|---|---|---|
| Bargaining | Alice and Bob alternate offers to split a pot over at most T rounds | split, or accept/reject | the agreed share, times δᵢ^(t−1) (each player's money loses value every round) |
| Negotiation | a seller and a buyer alternate prices; either can quit to an outside seller/buyer, "John" | price, or accept / reject / go to John | surplus over the John deal: price − v_s for the seller, v_b − price for the buyer |
| Persuasion | a seller who knows each product's quality sends a signal; the buyer buys or passes, for 20 rounds | the signal; buy or pass | seller: number of sales; buyer: value − price for each purchase |

Some parameters can be hidden from a player (the other side's δ or value; whether the buyer knows the
share of good products), and in "myopic" persuasion each round has a new buyer who only sees the seller's
sales statistics.

## 2. The idea: best-respond to how LLMs actually play

Game theory gives exact answers for perfectly rational players: the subgame-perfect equilibrium of
finite alternating-offers bargaining (Ståhl 1972), "last proposer takes the surplus" in negotiation,
Kamenica–Gentzkow optimal signalling in persuasion. They are all in `theory.py`, and they are
the wrong target. The opponents in GLEE are LLMs, and LLMs do not play the equilibrium: they
gravitate to 50/50 splits, accept offers a rational player would reject, reject offers a rational
player would accept, and react to the last offer on the table.

So the agent treats the opponent as a stochastic process to be estimated, and plays the best
response to it:

1. **Data.** GLEE's repository ships ~77,000 LLM-vs-LLM games (13 models: GPT-4o, Claude 3.5/3.7,
   Gemini 1.5/2.0, Llama 3, Mistral, Grok, o3-mini...). `scripts/build_dataset.py` flattens them into
   one row per offer (bargaining, negotiation) or per round (persuasion). Games are split 80/20 by
   game ID; everything below is fitted on the 80% only.
2. **Opponent models** (`opponent.py`, `scripts/train_models.py`):
   * *Response models*: gradient-boosted classifiers for P(accept), and for negotiation also
     P(go to John | not accept). The offer enters with a **monotone constraint**: more money for the
     responder can never lower the predicted chance they accept. Without it the model learns a
     selection artefact (generous offers are made in hard situations, so in raw data they are
     accepted *less*), and an optimiser would exploit it. Features include the exact SPNE
     reservation value (what a rational responder could secure by waiting) and the gap between the
     offer and what the responder asked for in the previous round.
   * *Proposal models*: smoothed conditional histograms of the offers LLMs make, given the round,
     the information structure, the proposer's own parameters and the last offer they received.
     Rare contexts back off to coarser ones (`BackoffHistogram`).
3. **Planning** (`policies/`): dynamic programming over the remaining rounds against those models.
4. **Language** (`llm.py`, optional): an LLM may reword the message that goes with an offer (a
   rewrite that changes any number is discarded) and may read a free-text seller message. It never
   makes a decision. Without an API key the agent runs fully offline with templates.

## 3. Bargaining

State: the round t and the last offer on the table s, binned to 10% (LLMs react strongly to it:
they accept much more often once an offer reaches what they last asked for, and their counteroffers
depend on what they were just offered). With the pot normalised to 1 and our discount factor d:

```
we propose  (s = their last offer to us):
    Vp(t, s) = max_x  A_t(x | s) · (1 − x) · d^(t−1)  +  (1 − A_t(x | s)) · Vr(t+1, x)
they propose (s = our last offer to them):
    Vr(t, s) = Σ_y  Q_t(y | s) · max( y · d^(t−1),  Vp(t+1, y) )
V(T+1, ·) = 0
```

`x` is the share we offer, `A_t` the response model, `Q_t` the proposal model. As a responder we
accept iff the offer is worth at least the value of rejecting and proposing next round.

**Hidden δ.** When the opponent's discount factor is hidden, the agent keeps a posterior over
GLEE's values {0.8, 0.9, 0.95, 1.0}. The likelihood of each candidate is the probability of what
was observed under it: each of our offers they rejected, and each offer they made. The chosen
action maximises posterior-expected value (a QMDP-style approximation: it assumes uncertainty is
resolved after this move). GLEE also leaks the rival's inflation in its per-round "Due to
inflation..." line; the parser reads it when present.

**Guards against exploiting model error.** A planner optimising against a fitted model will find the
places where the model is wrong. Two rules keep it on solid ground:

* *Support constraint*: only offers that LLMs in the same seat and situation actually make
  (at least 1% of the mass in a 5% bin) are considered. Elsewhere the response model is
  extrapolating.
* *No retraction*: the agent never offers less than it already offered. Retracting a concession is
  rarely seen in the data, so its effect is unknown, and it reads badly to a counterpart.

**Long games.** GLEE's long games last up to 99 rounds. The recursion looks 12 rounds ahead and treats
the end of that window as the end of the game (a receding horizon, recomputed every turn); this is
conservative, because it understates the value of waiting.

## 4. Negotiation

The same machinery with prices instead of shares (in units of GLEE's `product_price_order`) and
three responses instead of two:

```
we propose p (s = their last price):
    Vp(t, s) = max_p  a_t(p|s) · surplus(p)  +  (1 − a_t(p|s)) · (1 − j_t(p|s)) · Vr(t+1, p)
they propose (s = our last price):
    Vr(t, s) = E_y[ max( surplus(y), 0, Vp(t+1, y) ) ]        # accept, John, or counter
```

There is no discounting, so only the deadline creates pressure. The agent never proposes or accepts a
price worse than John, and keeps a posterior over the opponent's value when it is hidden (GLEE's values
{0.8, 1.0, 1.2, 1.5}), with the likelihood of their prices taken as a density in price units.

## 5. Persuasion

**Seller.** A buyer believes the product is good with probability q and buys iff
q·v + (1−q)·c ≥ price, i.e. q ≥ q* = (1 − c)/(v − c). Kamenica & Gentzkow's optimal signal recommends
every good product and a bad one with probability π = p(1−q*)/((1−p)q*), which leaves a Bayesian buyer
exactly indifferent. LLM buyers are not Bayesian: in the data they buy 58% of recommended products
that turn out bad, and their trust tracks the seller's visible record. The agent therefore solves a
dynamic program over that record, using the fitted buying model:

```
state (r, n, k): round r, n products in the visible record, k of them good
V(r, n, k) = E_quality[ max over {recommend, don't} of  P(buy | signal, n, k) + E V(r+1, next state) ]
```

For a returning buyer the record is every past recommendation and its revealed quality; for myopic
buyers it is GLEE's public sales statistics. When the seller doesn't know the buyer's values, the
buying probability is averaged over GLEE's values of v.

The signal is **pooled**: every recommendation uses the same sentence, which never claims a quality
the product doesn't have. This is what makes the strategy work (a buyer cannot tell a recommended good
product from a recommended bad one) and it keeps the agent honest in what it states.

**Buyer.** Bayesian: Beta posteriors on how often this seller recommends bad and good products,
with priors from the LLM sellers in the data (they recommend 99% of good products and about 30% of bad
ones), give P(good | signal); the agent buys iff it clears q*. A returning buyer sees every product's
quality, so it learns whether it buys or not. In a myopic seat, though, the only evidence is GLEE's
sales statistics, which move only when someone buys, and every "new buyer" is the agent. A buyer that
passes once learns nothing and keeps passing, so the myopic seat is treated as a one-armed Bernoulli
bandit over the remaining rounds: it buys when the expected payoff *including the information gained*
is positive, even if the immediate expectation is slightly negative.

## 6. Engineering

```
POST /chat ─► parse ─► interpret ─► decide ─► compose ─► validate ─► JSON answer
              protocol.py  llm.py    policies/  llm.py     validate.py
```

* `server.py`: Flask, `X-API-Key` authentication with a constant-time comparison, `/health`.
* `graph.py`: the pipeline as a LangGraph state graph. Every answer is checked against the same
  format rules GLEE enforces (GLEE ends the game on a malformed answer); if anything fails, a
  conservative valid answer is built from the format in the system message.
* The server is stateless: everything is reconstructed from the conversation GLEE sends each turn,
  so restarts and parallel games are safe.
* `scripts/simulator.py` re-implements GLEE's game loop and prompts from its source, so the agent
  can be tested and evaluated offline at the exact interface GLEE uses.

## 7. Limitations

* **The opponent model is a population average** of 2024–25 LLMs with GLEE's own prompts. A
  different model, a different prompt, or an agent built by someone else can behave differently.
  The response models do not condition on message text.
* **The evaluation is simulated.** Opponents are sampled from models fitted on held-out games. This
  is an honest test of generalisation to unseen games, but the simulated opponent shares some
  structural assumptions with the planner (it reacts to the last offer only, not to the full
  history), so the gains are an estimate, not a guarantee against live LLMs.
* **Approximations**: binned states, a 12-round lookahead in long games, and a posterior that is
  treated as fixed within each decision.
* Persuasion models are trained on games with yes/no signals and applied to free-text games through
  a fixed recommendation sentence.
