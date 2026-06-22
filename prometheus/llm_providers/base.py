from __future__ import annotations
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict: ...
