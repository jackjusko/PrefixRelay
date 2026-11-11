import json
from pathlib import Path

from proxy.config import Config
from proxy.recorder import Recorder


def test_stabilize_and_compaction_are_safe_defaults():
    cfg = Config()
    cfg.raw = {
        "backends": {"provider": {"base_url": "https://api.example/v1"}},
        "routes": [{"name": "model-small", "backend": "provider",
                    "upstream_model": "vendor/model", "context_tokens": 8192}],
    }
    cfg.build_routing()
    route, backend, _ = cfg.resolve("model-small")
    assert cfg.mode == "stabilize"
    assert backend.stabilize is True
    assert backend.compact_enabled is True
    assert route.context_tokens == 8192


def test_explicit_route_context_limit_is_reported_in_health():
    cfg = Config()
    cfg.raw = {
        "backends": {"provider": {"base_url": "https://api.example/v1"}},
        "routes": [{"name": "large", "backend": "provider", "context_tokens": 131072}],
    }
    cfg.build_routing()
    assert cfg.health()["routes"][0]["context_tokens"] == 131072


def test_health_redacts_url_credentials_and_sensitive_query_values():
    cfg = Config()
    cfg.raw = {
        "backends": {"provider": {
            "base_url": "https://user:password@api.example/v1?api_key=private&region=west"}},
        "routes": [{"name": "model", "backend": "provider"}],
    }
    cfg.build_routing()
    health = json.dumps(cfg.health())
    assert "password" not in health and "private" not in health
    assert "region=west" in health


def test_recorder_skips_body_files_by_default_but_keeps_metadata(tmp_path: Path):
    cfg = Config(record_bodies=False, log_response_bodies=False,
                 captures_dir=str(tmp_path / "captures"),
                 bodies_dir=str(tmp_path / "bodies"))
    recorder = Recorder(cfg)
    recorder.begin("request-1", {"authorization": "secret", "x-goog-api-key": "secret-too"},
                   b'{"prompt":"private"}')
    recorder.set_forwarded("request-1", b'{"prompt":"private"}', "main", {"backend": "provider"})
    recorder.finish("request-1", b'{"choices":[{"message":{"content":"private-answer"}}]}', status=200)

    assert list((tmp_path / "bodies").iterdir()) == []
    row = json.loads((tmp_path / "captures" / "index.jsonl").read_text())
    assert row["headers"]["authorization"] == "<redacted>"
    assert row["headers"]["x-goog-api-key"] == "<redacted>"
    assert row["meta"]["backend"] == "provider"
    assert "private" not in json.dumps(row)


def test_recorder_body_capture_requires_explicit_opt_in(tmp_path: Path):
    cfg = Config(record_bodies=True, log_response_bodies=True,
                 captures_dir=str(tmp_path / "captures"),
                 bodies_dir=str(tmp_path / "bodies"))
    recorder = Recorder(cfg)
    recorder.begin("request-2", {}, b'{"prompt":"private"}')
    recorder.set_forwarded("request-2", b'{"prompt":"forwarded"}', "main")
    recorder.finish("request-2", b'{"choices":[]}', status=200)
    assert {p.name for p in (tmp_path / "bodies").iterdir()} == {
        "request-2.recv.json", "request-2.fwd.json", "request-2.resp"}


def test_context_compaction_still_runs_without_a_kv_cache(tmp_path: Path, monkeypatch):
    from proxy import compaction, pipeline

    cfg = Config(state_dir=str(tmp_path / "state"))
    cfg.raw = {
        "backends": {"hosted": {"base_url": "https://api.example/v1",
                                 "stabilize": False, "compact_enabled": True}},
        "routes": [{"name": "hosted-model", "backend": "hosted",
                    "context_tokens": 4096}],
    }
    cfg.build_routing()
    backend = cfg.resolve("hosted-model")[1]
    seen = {}

    def compact(_cfg, body, _backend, meta):
        seen["compacted"] = True
        meta["compact"] = {"stage": "under-threshold"}
        return body

    monkeypatch.setattr(compaction, "compact_if_needed", compact)
    monkeypatch.setattr(pipeline, "_store", None)
    monkeypatch.setattr(pipeline, "_sidecar_rules", None)
    _, route, meta = pipeline.process(cfg, {"model": "hosted-model", "messages": []},
                                     "rid", backend=backend)
    assert route == "main"
    assert meta["stabilize"] == "skipped:hosted"
    assert seen["compacted"] is True


def test_unknown_remote_context_does_not_probe_provider(monkeypatch):
    from proxy import compaction

    class Backend:
        base_url = "https://api.example/v1"
        raw = {}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("remote provider must not receive a private /props probe")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    assert compaction._props_n_ctx(Backend()) == 0
