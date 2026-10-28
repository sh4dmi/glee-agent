import json

import pytest

from glee_agent.server import create_app

PAYLOAD = {
    "messages": [
        {"role": "system", "content": "Let's play a game.\nIn this game, there are two players. You are playing as "
                                      "Alice.\nYou and Bob need to divide 100$ between you.\n"},
        {"role": "user", "content": "Round 1\nSend your offer to divide $100 and a message to Bob."},
    ],
    "decision": False,
    "game_params": {"public_name": "Alice", "money_to_divide": 100, "max_rounds": 12, "complete_information": True,
                    "messages_allowed": True, "delta_player_1": 0.9, "delta_player_2": 0.9},
}


@pytest.fixture()
def client(agent):
    return create_app(agent=agent, api_key="secret").test_client()


def test_requires_key(client):
    assert client.post("/chat", json=PAYLOAD).status_code == 401
    assert client.post("/chat", json=PAYLOAD, headers={"X-API-Key": "wrong"}).status_code == 401


def test_chat_returns_a_valid_offer(client):
    r = client.post("/chat", json=PAYLOAD, headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    answer = json.loads(r.get_json()["response"])
    assert answer["alice_gain"] + answer["bob_gain"] == 100
    assert isinstance(answer["message"], str)


def test_rejects_malformed_requests(client):
    assert client.post("/chat", json={"nope": 1}, headers={"X-API-Key": "secret"}).status_code == 400


def test_unparseable_turn_still_gets_a_valid_answer(client):
    bad = {"messages": [{"role": "system", "content": '{"alice_gain": ..., "bob_gain": ...}'},
                        {"role": "user", "content": "Send your offer to divide $100"}],
           "game_params": {"money_to_divide": 100}}
    r = client.post("/chat", json=bad, headers={"X-API-Key": "secret"})
    answer = json.loads(r.get_json()["response"])
    assert answer["alice_gain"] + answer["bob_gain"] == 100


def test_health(client):
    assert client.get("/health").get_json()["status"] == "alive"
    assert client.get("/health", headers={"X-API-Key": "secret"}).get_json()["auth"] == "ok"
    assert client.get("/health", headers={"X-API-Key": "x"}).status_code == 401
