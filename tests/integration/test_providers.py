import pytest
import respx
import httpx
from prometheus.llm_providers.openai_compat import OpenAICompatibleProvider
from prometheus.llm_providers.anthropic_provider import AnthropicProvider

@pytest.mark.asyncio
@respx.mock
async def test_groq_provider_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    p = OpenAICompatibleProvider("key", "https://api.groq.com/openai/v1", "model")
    res = await p.complete("sys", "user")
    assert res == "ok"

@pytest.mark.asyncio
@respx.mock
async def test_anthropic_provider_success():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"content": [{"text": "ok"}]})
    )
    p = AnthropicProvider("key", "model")
    res = await p.complete("sys", "user")
    assert res == "ok"

@pytest.mark.asyncio
@respx.mock
async def test_provider_500():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    p = OpenAICompatibleProvider("key", "https://api.groq.com/openai/v1", "model")
    with pytest.raises(httpx.HTTPStatusError):
        await p.complete("sys", "user")
