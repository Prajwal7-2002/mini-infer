"""The server must describe the model it actually serves (clients such as servebench
record this). Regression: /v1/models always listed the 0.5B model."""

from fastapi.testclient import TestClient

from mini_infer import __version__, server


def test_models_and_health_report_the_served_model(monkeypatch):
    monkeypatch.setattr(server, "model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    client = TestClient(server.app)

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b"]

    health = client.get("/health").json()
    assert health == {
        "status": "ok",
        "engine": "mini-infer",
        "version": __version__,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    }


def test_short_model_id():
    assert server.short_model_id("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b"
    assert server.short_model_id("D:\\models\\Qwen2.5-3B-Instruct\\") == "qwen2.5-3b"
    assert server.short_model_id("my-model") == "my-model"
