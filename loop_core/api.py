from __future__ import annotations

from typing import Dict, Optional

import os

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class GroqEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
    ) -> None:
        if Groq is None:
            raise ImportError("groq package required. Install with: pip install groq")

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        generated = response.choices[0].message.content.strip()
        for prefix in ["Response:", "Assistant:", "AI:"]:
            if generated.startswith(prefix):
                generated = generated[len(prefix) :].strip()
                break
        return generated or "I'm not sure how to respond."


class OpenAIEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        generated = response.choices[0].message.content.strip()
        for prefix in ["Response:", "Assistant:", "AI:"]:
            if generated.startswith(prefix):
                generated = generated[len(prefix) :].strip()
                break
        return generated or "I'm not sure how to respond."


ENGINE_REGISTRY = {
    "groq": GroqEngine,
    "openai": OpenAIEngine,
}


def get_engine(provider: str, api_key: Optional[str] = None, runtime: Optional[Dict[str, float]] = None):
    runtime = runtime or {}
    provider = (provider or "groq").lower()
    if provider not in ENGINE_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}")

    engine_cls = ENGINE_REGISTRY[provider]
    kwargs = {
        "api_key": api_key,
        "temperature": float(runtime.get("temperature", 0.7)),
    }

    if provider == "groq":
        kwargs["model"] = runtime.get("model", "llama-3.3-70b-versatile")
    elif provider == "openai":
        kwargs["model"] = runtime.get("model", "gpt-4o-mini")

    return engine_cls(**kwargs)
