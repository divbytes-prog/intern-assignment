"""Small, typed JSON interface to the LLM."""
from __future__ import annotations

import json

from openai import OpenAI


class LLM:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def json(self, system: str, user: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(response.choices[0].message.content or "{}")
