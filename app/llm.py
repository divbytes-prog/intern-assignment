"""Small, typed JSON interface to the LLM."""
from __future__ import annotations

import json
import os
import re
import time

from openai import OpenAI, RateLimitError
from google import genai
from google.genai import types
from google.genai.errors import APIError


class ModelResponseError(ValueError):
    """A model response was not the JSON object required by the workflow."""


class LLM:
    def __init__(self, api_key: str, model: str, provider: str = "groq"):
        self.provider = provider
        if provider == "gemini":
            self.client = genai.Client(api_key=api_key)
        elif provider == "groq":
            self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1",
                                 max_retries=0, timeout=18.0)
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("AI_PROVIDER must be groq, gemini, or openai")
        self.model = model
        self.fallback_model = os.getenv('GROQ_FALLBACK_MODEL', 'llama-3.1-8b-instant') if provider == 'groq' else None

    def json(self, system: str, user: str) -> dict:
        if self.provider == "gemini":
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model, contents=user,
                        config=types.GenerateContentConfig(
                            system_instruction=system, response_mime_type="application/json", temperature=0,
                        ),
                    )
                    break
                except APIError as exc:
                    if exc.code in {500, 502, 503, 504} and attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    if exc.code != 429 or attempt:
                        raise
                    # Only retry a short per-minute limit; don't wait on daily quota.
                    match = re.search(r"Please retry in ([\d.]+)s", exc.message or "")
                    delay = float(match.group(1)) if match else 60.0
                    if delay > 65:
                        raise
                    time.sleep(max(1.0, delay + 1))
            parsed = json.loads(response.text or "{}")
            if not isinstance(parsed, dict):
                raise ModelResponseError("Model must return a JSON object")
            return parsed
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=messages,
            )
        except RateLimitError:
            # Groq applies per-model limits. A second free model can keep the
            # demo useful during a short burst; never retry the same 429 loop.
            if not self.fallback_model or self.fallback_model == self.model:
                raise
            response = self.client.chat.completions.create(
                model=self.fallback_model, temperature=0, response_format={"type": "json_object"},
                messages=messages,
            )
        parsed = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(parsed, dict):
            raise ModelResponseError("Model must return a JSON object")
        return parsed
