import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


class FakeTokenizer:
    EOS_ID = 3

    def encode_sentence(self, text, add_sos=False, add_eos=False):
        return [1, 2, 3]

    def decode_sentence(self, ids):
        return "fake output"


class FakeBlockManager:
    num_blocks = 100

    def num_free_blocks(self):
        return 60


class FakeScheduler:
    waiting_requests = [1, 2]
    running_requests = [1]


class FakeEngine:
    def __init__(self, finish_immediately=True):
        self.tok = FakeTokenizer()
        self.block_manager_obj = FakeBlockManager()
        self.scheduler_obj = FakeScheduler()
        self.seq_cnt = 7
        self.results = {}
        self.finish_immediately = finish_immediately
        self.submitted = []

    def submit(self, prompt_token_ids, max_token_to_generate):
        seq_id = len(self.submitted)
        self.submitted.append((prompt_token_ids, max_token_to_generate))
        if self.finish_immediately:
            self.results[seq_id] = ("fake output", True)
        return seq_id


@pytest.fixture(autouse=True)
def reset_state():
    main.is_ready = False
    main.engine = None
    yield


def test_root_ok():
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_503_when_not_ready():
    main.is_ready = False
    response = client.get("/health")
    assert response.status_code == 503


def test_health_returns_200_when_ready():
    main.is_ready = True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_generate_returns_text():
    main.engine = FakeEngine()
    response = client.post("/generate", json={"prompt": "hello", "max_tokens": 5})
    assert response.status_code == 200
    assert response.json()["text"] == "fake output"


def test_generate_passes_max_tokens_to_engine():
    fake = FakeEngine()
    main.engine = fake
    client.post("/generate", json={"prompt": "hello", "max_tokens": 12})
    assert fake.submitted[0][1] == 12


def test_generate_defaults_max_tokens():
    fake = FakeEngine()
    main.engine = fake
    client.post("/generate", json={"prompt": "hello"})
    assert fake.submitted[0][1] == 25


def test_generate_cleans_up_results():
    fake = FakeEngine()
    main.engine = fake
    client.post("/generate", json={"prompt": "hello"})
    assert fake.results == {}


def test_generate_rejects_empty_prompt():
    main.engine = FakeEngine()
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_rejects_oversized_max_tokens():
    main.engine = FakeEngine()
    response = client.post("/generate", json={"prompt": "hello", "max_tokens": 5000})
    assert response.status_code == 422


def test_generate_times_out(monkeypatch):
    monkeypatch.setattr(main, "TIMEOUT", 0.2)
    main.engine = FakeEngine(finish_immediately=False)
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 504


def test_metrics_reports_engine_state():
    main.engine = FakeEngine()
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["free_blocks"] == 60
    assert body["total_blocks"] == 100
    assert body["waiting_seq"] == 2
    assert body["running_seq"] == 1
    assert body["total_requests"] == 7