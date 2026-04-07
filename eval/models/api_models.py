"""
API model adapters for OpenAI and Google Gemini.

Uses only the standard library (urllib) — no openai or google-generativeai packages required.
Authentication via environment variables:
  - OpenAI: OPENAI_API_KEY (or custom via api_key_env in YAML)
  - Gemini: GEMINI_API_KEY
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from eval.models.base import ModelAdapter, Prediction
from eval.qa_schema import QAItem


_ANS_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    m = _ANS_RE.search(s)
    return m.group(1).upper() if m else None


def _http_post_json(url: str, *, headers: Dict[str, str], body: dict, timeout: int = 120) -> dict:
    """POST JSON and return the parsed response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {error_body[:500]}") from e


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API
# ---------------------------------------------------------------------------

class OpenAIAdapter(ModelAdapter):
    """
    OpenAI-compatible Chat Completions API adapter.

    YAML config:
        adapter: openai
        name: gpt-4o
        model: gpt-4o
        api_base: https://api.openai.com/v1   # optional
        api_key_env: OPENAI_API_KEY            # optional
    """

    def __init__(self, *, name: str, model: str, api_base: str, api_key_env: str,
                 max_tokens: int, temperature: float) -> None:
        self._name = name
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set.")
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return self._name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        content: List[Dict[str, Any]] = []

        # Add frame images
        if frames_data_urls:
            for url in frames_data_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": "You are a strict multiple-choice evaluator. Reply with only one letter: A, B, C, or D."},
            {"role": "user", "content": content},
        ]

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        resp = _http_post_json(
            f"{self._api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            body=body,
        )

        raw_text = ""
        choices = resp.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "")

        return Prediction(answer=_parse_answer_letter(raw_text), raw=raw_text)


# ---------------------------------------------------------------------------
# Google Gemini Generative Language API
# ---------------------------------------------------------------------------

class GeminiAdapter(ModelAdapter):
    """
    Google Gemini API adapter.

    YAML config:
        adapter: gemini
        name: gemini-2.5-pro
        model: gemini-2.5-pro
    """

    def __init__(self, *, name: str, model: str, max_tokens: int, temperature: float) -> None:
        self._name = name
        self._model = model
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return self._name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        parts: List[Dict[str, Any]] = []

        if frames_data_urls:
            for url in frames_data_urls:
                # Parse data URL: data:<mime>;base64,<data>
                if url.startswith("data:") and ";base64," in url:
                    header, b64_data = url.split(",", 1)
                    mime = header.split(";")[0].split(":")[1]
                    parts.append({"inline_data": {"mime_type": mime, "data": b64_data}})

        parts.append({"text": prompt})

        body: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "systemInstruction": {
                "parts": [{"text": "You are a strict multiple-choice evaluator. Reply with only one letter: A, B, C, or D."}],
            },
            "generationConfig": {
                "maxOutputTokens": self._max_tokens,
                "temperature": self._temperature,
            },
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        resp = _http_post_json(url, headers={"Content-Type": "application/json"}, body=body)

        raw_text = ""
        candidates = resp.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts_out = content.get("parts", [])
            if parts_out:
                raw_text = parts_out[0].get("text", "")

        return Prediction(answer=_parse_answer_letter(raw_text), raw=raw_text)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_openai_from_config(cfg: dict) -> OpenAIAdapter:
    return OpenAIAdapter(
        name=str(cfg.get("name", "openai")),
        model=str(cfg.get("model", "")),
        api_base=str(cfg.get("api_base", "https://api.openai.com/v1")),
        api_key_env=str(cfg.get("api_key_env", "OPENAI_API_KEY")),
        max_tokens=int(cfg.get("max_tokens", 16)),
        temperature=float(cfg.get("temperature", 0.0)),
    )


def build_gemini_from_config(cfg: dict) -> GeminiAdapter:
    return GeminiAdapter(
        name=str(cfg.get("name", "gemini")),
        model=str(cfg.get("model", "")),
        max_tokens=int(cfg.get("max_tokens", 16)),
        temperature=float(cfg.get("temperature", 0.0)),
    )
