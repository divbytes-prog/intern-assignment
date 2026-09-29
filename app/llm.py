"""Small, typed JSON interface to the LLM."""
from __future__ import annotations

import json

from openai import OpenAI
from google import genai
from google.genai import types


class LLM:
    def __init__(self, api_key: str, model: str, provider: str = "gemini"):
        self.provider = provider
        self.client = genai.Client(api_key=api_key) if provider == "gemini" else OpenAI(api_key=api_key)
        self.model = model

    def json(self, system: str, user: str) -> dict:
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, response_mime_type="application/json", temperature=0,
                ),
            )
            return json.loads(response.text or "{}")
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(response.choices[0].message.content or "{}")
