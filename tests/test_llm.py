from glee_agent.llm import LanguageModel


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt):
        return type("R", (), {"content": self.reply})()


def model(reply):
    m = LanguageModel(api_key=None)
    m._client = FakeClient(reply)
    return m


def test_polish_keeps_a_rewrite_with_the_same_numbers():
    assert model("Happy to offer you $4,000 today.").polish("I'm offering you $4,000.") == \
        "Happy to offer you $4,000 today."


def test_polish_rejects_a_rewrite_that_changes_the_offer():
    draft = "I'm offering you $4,000."
    assert model("How about $4,500?").polish(draft) == draft
    assert model('{"alice_gain": 1}').polish(draft) == draft


def test_offline_by_default(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert not LanguageModel().enabled


def test_classifier():
    assert model("Yes.").recommends("great product") is True
    assert model("no").recommends("skip it") is False
    assert model("maybe").recommends("hmm") is None
