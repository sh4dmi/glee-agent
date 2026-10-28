import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]


@pytest.fixture(scope="session")
def models():
    from glee_agent.opponent import OpponentModels
    return OpponentModels.load()


@pytest.fixture(scope="session")
def agent(models):
    from glee_agent.graph import Agent
    from glee_agent.llm import LanguageModel
    return Agent(models=models, llm=LanguageModel(api_key=None))
