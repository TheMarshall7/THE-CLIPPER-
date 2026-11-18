"""AI is optional: rules-only by default, API key via settings (PRD §15)."""
import json
import os
import time

import pytest

from engine.ai import llm
from engine.config import AISettings, Settings, secret_hint, set_secret


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text or json.dumps(payload or {})

    def json(self):
        return self._p


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_default_is_rules_only_and_old_modes_migrate():
    assert AISettings().mode == "heuristic"
    for old in ("local", "claude", "bogus"):
        assert AISettings(mode=old).mode == "heuristic"          # Ollama / CLI settings files still load
    assert Settings.model_validate({"ai": {"mode": "local", "ollama_model": "qwen2.5:7b"}}).ai.mode == "heuristic"
    prov, msg = llm.get_provider(AISettings())
    assert prov is None and "Rules only" in msg


def test_api_mode_without_key_falls_back_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prov, msg = llm.get_provider(AISettings(mode="api", provider="anthropic"))
    assert prov is None and "No Anthropic API key" in msg and "rules only" in msg


def test_key_prefix_picks_the_provider():
    from engine.config import detect_provider
    assert detect_provider("AIzaSyD-example-gemini-key") == "gemini"
    assert detect_provider("sk-ant-api03-abcdefgh") == "anthropic"
    assert detect_provider("sk-or-v1-abcdefgh") == "openrouter"       # must beat the generic sk- prefix
    assert detect_provider("sk-proj-abcdefgh") == "openai"
    assert detect_provider("not-a-known-key") is None


def test_gemini_uses_google_openai_endpoint_and_its_own_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-123456")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp(200, {"choices": [{"message": {"content": '{"keep": true}'}}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    prov, msg = llm.get_provider(AISettings(mode="api", provider="gemini", model="gemini-2.5-flash"))
    assert "gemini" in msg and prov.complete_json("s", "p") == {"keep": True}
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer AIza-test-key-123456"
    assert seen["body"]["model"] == "gemini-2.5-flash"


def test_openrouter_uses_its_key_not_openai_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-123456")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    seen = {}
    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=None: (
        seen.update(url=url, auth=headers["Authorization"]) or FakeResp(200, {"choices": [{"message": {"content": '{"a": 1}'}}]})))
    prov, _ = llm.get_provider(AISettings(mode="api", provider="openrouter", model="anthropic/claude-sonnet-5-5"))
    assert prov.complete_json("s", "p") == {"a": 1}
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-or-test-key-123456"


def test_json_mode_rejected_retries_without_it(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-123456")
    bodies = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        if "response_format" in json:
            return FakeResp(400, text='{"error": "response_format type json_object not supported"}')
        return FakeResp(200, {"choices": [{"message": {"content": 'Here: {"keep": false}'}}]})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    prov, _ = llm.get_provider(AISettings(mode="api", provider="openrouter"))
    assert prov.complete_json("s", "p") == {"keep": False}
    assert len(bodies) == 2 and "response_format" not in bodies[1]


def test_custom_endpoint_uses_the_address_the_user_gave(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-test-key-123456")
    seen = {}
    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=None: (
        seen.update(url=url) or FakeResp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})))
    prov, _ = llm.get_provider(AISettings(mode="api", provider="custom", model="m", openai_base_url="http://localhost:1234/v1/"))
    prov.complete_json("s", "p")
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"


def test_connect_with_key_switches_ai_on_and_saves_the_key(tmp_path, monkeypatch):
    import engine.config as cfg
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(cfg, "SECRETS_PATH", tmp_path / "secrets.env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cfg.connect_with_key("AIzaSyD-example-gemini-key") == "gemini"
    s = cfg.load_settings(cfg.CONFIG_PATH)
    assert s.ai.mode == "api" and s.ai.provider == "gemini" and s.ai.model == "gemini-2.5-flash"
    assert "GEMINI_API_KEY=AIzaSyD-example-gemini-key" in cfg.SECRETS_PATH.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        cfg.connect_with_key("unknown-format-key")


def test_anthropic_call_parses_json_and_counts_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        text = 'Sure!\n```json\n{"keep": true, "payoff": 80}\n```'
        return FakeResp(200, {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 120, "output_tokens": 30}})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    prov, _ = llm.get_provider(AISettings(mode="api", provider="anthropic", model="claude-sonnet-5-5"))
    out = prov.complete_json("sys", "prompt")
    assert out == {"keep": True, "payoff": 80}
    assert seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "sk-ant-test-key-1234"
    assert seen["body"]["model"] == "claude-sonnet-5-5" and seen["body"]["system"] == "sys"
    assert (prov.calls, prov.tokens_in, prov.tokens_out) == (1, 120, 30)


def test_openai_compatible_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-98765")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp(200, {"choices": [{"message": {"content": '{"ok": true}'}}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    cfg = AISettings(mode="api", provider="openai", model="some-model", openai_base_url="https://gateway.example/v1/")
    prov, _ = llm.get_provider(cfg)
    assert prov.complete_json("s", "p") == {"ok": True}
    assert seen["url"] == "https://gateway.example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test-key-98765"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_errors_are_plain_language_and_never_leak_the_key(monkeypatch):
    key = "sk-ant-SECRET-abcdef123456"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(401, text="invalid x-api-key " + key))
    prov, _ = llm.get_provider(AISettings(mode="api", provider="anthropic"))
    with pytest.raises(llm.AIError) as e:
        prov.complete_json("s", "p")
    assert "key was rejected" in str(e.value) and key not in str(e.value)


def test_retries_rate_limits_then_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")
    answers = [FakeResp(429, text="slow down"),
               FakeResp(200, {"content": [{"type": "text", "text": '{"a": 1}'}], "usage": {}})]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: answers.pop(0))
    prov, _ = llm.get_provider(AISettings(mode="api", provider="anthropic"))
    assert prov.complete_json("s", "p") == {"a": 1}


def test_a_failed_ai_call_falls_back_to_rules_and_says_so():
    """One bad API call must not kill the batch: that moment is scored by the rules and labelled."""
    from engine.intelligence import moments

    class Broken:
        name, model = "anthropic", "m"

        def complete_json(self, *a):
            raise llm.AIError("Rate limit or quota reached for this API key.")
    sig = {"surprise_hits": 2, "hype_hits": 1, "laugh_hits": 0, "callout_hits": 0, "audio_peak": 0.8, "audio_mean": 0.6,
           "laughter_like": 0.0, "scream_like": 0.1, "silence_ratio": 0.1, "words": 40, "hook_audio": 0.7,
           "hook_question": True, "hook_text": "wait what", "peak_offset_s": 5, "contrast": 0.5, "z_rise_max": 3,
           "repetition": 0}
    cands = [{"peak": 20.0, "start": 10.0, "end": 40.0, "signals": sig}]
    out = moments.evaluate(cands, [], {"streamer": "x", "title": "t", "duration": 100.0}, Settings(), Broken(),
                           log=lambda m: None)
    assert out[0]["ai"]["mode"] == "heuristic (AI call failed)" and out[0]["ai"]["keep"] is True


def test_secret_roundtrip_hides_the_key(tmp_path, monkeypatch):
    f = tmp_path / "secrets.env"
    f.write_text("# ANTHROPIC_API_KEY=\n# TWITCH_CLIENT_ID=\nYOUTUBE_API_KEY=keep-me-123456\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    set_secret("ANTHROPIC_API_KEY", "sk-ant-abcdefgh1234", f)
    text = f.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-abcdefgh1234" in text and "YOUTUBE_API_KEY=keep-me-123456" in text
    assert text.count("ANTHROPIC_API_KEY") == 1
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abcdefgh1234"
    assert secret_hint("ANTHROPIC_API_KEY") == "...1234"
    set_secret("ANTHROPIC_API_KEY", "", f)                         # remove
    assert "sk-ant" not in f.read_text(encoding="utf-8") and "ANTHROPIC_API_KEY" not in os.environ


def test_auto_video_type(tmp_settings):
    from engine.pipeline import Engine

    e = Engine(tmp_settings, log=lambda m: None)
    now = time.time()

    def stream(title, music_fraction, ct="auto"):
        sid = e.db.insert("streams", {"platform": "file", "source_url": f"u-{title}-{music_fraction}", "title": title,
                                      "status": "queued", "metadata_json": {"content_type": ct},
                                      "created_at": now, "updated_at": now})
        e.db.execute("INSERT INTO stream_steps(stream_id, step, status, output_json) VALUES (?,?,?,?)",
                     (sid, "music_scan", "done", json.dumps({"music_fraction": music_fraction})))
        return sid
    assert e.resolve_content_type(stream("GTA 6 MARATHON | Official Trailer", 0.1)) == "game_trailer"
    assert e.resolve_content_type(stream("KAI REACTS to the GTA 6 trailer", 0.6)) == "stream"      # 'reacts' beats 'trailer'
    assert e.resolve_content_type(stream("untitled 4", 0.55)) == "movie_trailer"                        # mostly music
    assert e.resolve_content_type(stream("untitled 5", 0.05)) == "stream"
    assert e.resolve_content_type(stream("GTA 6 Official Trailer", 0.0, ct="stream")) == "stream"  # explicit choice wins
