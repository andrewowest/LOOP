from __future__ import annotations

import os
from typing import Any, Dict, Optional, Protocol, runtime_checkable


_PREFIXES_TO_STRIP = ("Response:", "Assistant:", "AI:")
DEFAULT_FALLBACK_RESPONSE = "I'm not sure how to respond."


@runtime_checkable
class LLMEngine(Protocol):
    """Minimal interface every provider adapter must satisfy."""

    def generate(self, prompt: str, max_tokens: int = 128) -> str: ...


class _ChatCompletionEngine:
    """Shared logic for OpenAI-compatible chat-completion clients.

    Both Groq and OpenAI expose `client.chat.completions.create(...)` with
    identical message shapes, so the only thing that varies is which client
    class to instantiate.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.fallback_response = fallback_response

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        for prefix in _PREFIXES_TO_STRIP:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        return text or self.fallback_response


class GroqEngine(_ChatCompletionEngine):
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    ENV_VAR = "GROQ_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "groq package required. Install with: pip install 'loop-core[groq]'"
            ) from exc

        resolved_key = api_key or os.getenv(self.ENV_VAR)
        if not resolved_key:
            raise ValueError(f"{self.ENV_VAR} not set and no api_key provided.")

        super().__init__(
            client=Groq(api_key=resolved_key),
            model=model,
            temperature=temperature,
            fallback_response=fallback_response,
        )


class OpenAIEngine(_ChatCompletionEngine):
    DEFAULT_MODEL = "gpt-4o-mini"
    ENV_VAR = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required. Install with: pip install 'loop-core[openai]'"
            ) from exc

        resolved_key = api_key or os.getenv(self.ENV_VAR)
        if not resolved_key:
            raise ValueError(f"{self.ENV_VAR} not set and no api_key provided.")

        super().__init__(
            client=OpenAI(api_key=resolved_key),
            model=model,
            temperature=temperature,
            fallback_response=fallback_response,
        )


ENGINE_REGISTRY: Dict[str, type] = {
    "groq": GroqEngine,
    "openai": OpenAIEngine,
}


def get_engine(
    provider: str,
    api_key: Optional[str] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> LLMEngine:
    """Construct an LLM engine adapter for the named provider."""
    runtime = runtime or {}
    key = (provider or "groq").lower()
    if key not in ENGINE_REGISTRY:
        supported = ", ".join(sorted(ENGINE_REGISTRY))
        raise ValueError(f"Unsupported provider: {provider!r}. Supported: {supported}.")

    engine_cls = ENGINE_REGISTRY[key]
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "temperature": float(runtime.get("temperature", 0.7)),
    }
    if "model" in runtime:
        kwargs["model"] = runtime["model"]
    if "fallback_response" in runtime:
        kwargs["fallback_response"] = runtime["fallback_response"]
    return engine_cls(**kwargs)
