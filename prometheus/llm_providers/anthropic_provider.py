import json
import httpx
from .base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = httpx.AsyncClient(timeout=60.0)
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        response = await self.client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        system_prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON."
        content = await self.complete(system_prompt, user_prompt)
        
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content)
