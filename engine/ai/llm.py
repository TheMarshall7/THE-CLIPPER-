"""AI provider layer (PRD §15). Judgement only; no media processing here.

AI is OPTIONAL:
- heuristic : rules only. No AI, no network, free. (default)
- api       : your own API key - Anthropic (Claude) or any OpenAI-compatible endpoint.

There is no silent fallback: if the configured provider is unusable (no key, bad key, rate limit),
the engine says so and scores with the rules, and every result is labelled with what judged it.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

from engine.config import PROVIDERS, AISettings


class AIError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise AIError(f"Model did not return JSON: {text[:200]}")
    try:
        return json.loads(text[start:end + 1])
    except ValueError as e:
        raise AIError(f"Model returned invalid JSON: {e}") from e


def _friendly(status: int, body: str) -> str:
    low = body.lower()
    if status in (401, 403):
        return "The API key was rejected. Check the key in Settings (and that it has access to this model)."
    if status == 404 or ("model" in low and ("not found" in low or "does not exist" in low)):
        return "The model name was not found for this key. Check the model in Settings."
    if status == 429:
        return "Rate limit or quota reached for this API key."
    if status == 402 or "credit" in low or "billing" in low:
        return "The account has no credit or a billing problem."
    return f"API error {status}: {body[:200]}"


class Provider:
    name = "base"
    model = ""

    def __init__(self):
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def usage_line(self) -> str:
        return f"{self.calls} AI calls, {self.tokens_in:,} tokens in / {self.tokens_out:,} out"

    def _post(self, url: str, headers: dict, body: dict, timeout: int) -> dict:
        last = ""
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as e:
                last = f"Could not reach the API ({type(e).__name__})."
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            last = _friendly(r.status_code, r.text)
            if r.status_code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        raise AIError(last)

    def complete_json(self, system: str, prompt: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, cfg: AISettings):
        super().__init__()
        self.name = "anthropic"
        self.key = os.environ.get(PROVIDERS["anthropic"]["secret"], "")
        self.model = cfg.model
        self.timeout = cfg.timeout_s

    def available(self) -> tuple[bool, str]:
        return (True, "ok") if self.key else (False, "No Anthropic API key set. Add one in Settings.")

    def complete_json(self, system: str, prompt: str) -> dict:
        data = self._post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          {"model": self.model, "max_tokens": 1200, "temperature": 0.2, "system": system,
                           "messages": [{"role": "user", "content": prompt}]}, self.timeout)
        self.calls += 1
        u = data.get("usage", {})
        self.tokens_in += u.get("input_tokens", 0)
        self.tokens_out += u.get("output_tokens", 0)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return extract_json(text)


class OpenAIProvider(Provider):
    """Any host that speaks the OpenAI chat format: Gemini, OpenAI, OpenRouter, or a custom address."""

    def __init__(self, cfg: AISettings):
        super().__init__()
        self.name = cfg.provider
        self.label = PROVIDERS[cfg.provider]["label"]
        self.key = os.environ.get(PROVIDERS[cfg.provider]["secret"], "")
        self.model = cfg.model
        self.base = cfg.base_url()
        self.timeout = cfg.timeout_s

    def available(self) -> tuple[bool, str]:
        return (True, "ok") if self.key else (False, f"No {self.label} API key set. Add one in Settings.")

    def _chat(self, system: str, prompt: str, json_mode: bool) -> dict:
        body = {"model": self.model, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._post(f"{self.base}/chat/completions",
                          {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}, body, self.timeout)

    def complete_json(self, system: str, prompt: str) -> dict:
        try:
            data = self._chat(system, prompt, json_mode=True)
        except AIError as e:
            # some hosts or models reject JSON mode; the prompt already asks for JSON, so retry without it
            if "response_format" not in str(e).lower() and "json" not in str(e).lower():
                raise
            data = self._chat(system, prompt, json_mode=False)
        self.calls += 1
        u = data.get("usage") or {}
        self.tokens_in += u.get("prompt_tokens", 0)
        self.tokens_out += u.get("completion_tokens", 0)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise AIError("Unexpected response shape from the API.") from e
        return extract_json(text or "")


def get_provider(cfg: AISettings) -> tuple[Optional[Provider], str]:
    """Return (provider or None, status message). None means rules only."""
    if cfg.mode != "api":
        return None, "Rules only: no AI is used (add an API key in Settings for smarter picks and hooks)."
    prov: Provider = AnthropicProvider(cfg) if PROVIDERS[cfg.provider]["kind"] == "anthropic" else OpenAIProvider(cfg)
    ok, msg = prov.available()
    if ok:
        return prov, f"AI: {cfg.provider} ({prov.model})"
    return None, f"AI unavailable -> rules only. {msg}"
