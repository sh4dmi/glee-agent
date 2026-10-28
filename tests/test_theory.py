import pytest

from glee_agent.theory import (negotiation_spne_price, persuasion_commitment_rate, persuasion_threshold,
                               spne_proposer_shares, spne_reservation_share)


def test_one_round_is_an_ultimatum():
    assert spne_proposer_shares(1, 0.9, 0.9)[1] == 1.0
    assert spne_reservation_share(1, 1, 0.9, 0.9) == 0.0


def test_two_rounds_by_hand():
    # round 2: player 2 keeps everything; round 1: player 2 must get delta_2 * 1
    s = spne_proposer_shares(2, 0.9, 0.8)
    assert s[2] == 1.0
    assert s[1] == pytest.approx(1 - 0.8)


def test_long_horizon_converges_to_rubinstein():
    d1, d2 = 0.9, 0.8
    s = spne_proposer_shares(200, d1, d2)
    assert s[1] == pytest.approx((1 - d2) / (1 - d1 * d2), abs=1e-6)


def test_no_inflation_gives_the_last_proposer_everything():
    # without discounting, whoever proposes last can hold out for the whole pot
    assert spne_proposer_shares(4, 1.0, 1.0)[1] == pytest.approx(0.0)
    assert spne_proposer_shares(5, 1.0, 1.0)[1] == pytest.approx(1.0)


def test_negotiation_spne():
    assert negotiation_spne_price(1, 1.0, 1.5) == 1.5      # seller proposes last
    assert negotiation_spne_price(2, 1.0, 1.5) == 1.0      # buyer proposes last
    assert negotiation_spne_price(3, 1.2, 1.0) is None     # no zone of agreement


def test_persuasion_commitment_makes_the_buyer_indifferent():
    p, v, c = 0.3, 2.0, 0.0
    q = persuasion_threshold(v, c)
    assert q == pytest.approx(0.5)
    pi = persuasion_commitment_rate(p, v, c)
    posterior = p / (p + (1 - p) * pi)
    assert posterior == pytest.approx(q)
    assert persuasion_commitment_rate(0.8, 2.0, 0.0) == 1.0  # prior already convinces the buyer
