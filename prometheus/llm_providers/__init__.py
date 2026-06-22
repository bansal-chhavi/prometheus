from .base import BaseLLMProvider
from .openai_compat import OpenAICompatibleProvider
from .anthropic_provider import AnthropicProvider
from ..config import PrometheusConfig, LLMProvider

def create_provider(config: PrometheusConfig) -> BaseLLMProvider:
    if config.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(
            api_key=config.get_api_key(),
            model=config.llm_model,
            temperature=config.llm_temperature
        )
    return OpenAICompatibleProvider(
        api_key=config.get_api_key(),
        base_url=config.get_base_url(),
        model=config.llm_model,
        temperature=config.llm_temperature
    )
