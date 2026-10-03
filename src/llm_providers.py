"""
llm_providers.py — provider-agnostic LLM completion client.

Supports:
  - Gemini            (GEMINI_API_KEY)
  - Anthropic         (ANTHROPIC_API_KEY)
  - OpenAI            (OPENAI_API_KEY)
  - Any OpenAI-compatible endpoint
    (LLM_API_KEY + LLM_BASE_URL)

All providers expose the same complete() interface so
reply_generator.py does not need provider-specific logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class GeminiProvider:
    """Native Google Gemini provider."""

    name = "gemini"

    def __init__(self, model):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, system, user, max_tokens=300):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )

        return (response.text or "").strip()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system, user, max_tokens=300):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        return "".join(
            block.text
            for block in resp.content
            if block.type == "text"
        ).strip()


class OpenAICompatibleProvider:
    """
    Handles OpenAI and any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        name="openai",
    ):
        import openai

        kwargs = {}

        if api_key:
            kwargs["api_key"] = api_key

        if base_url:
            kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.name = name

    def complete(self, system, user, max_tokens=300):
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        return (resp.choices[0].message.content or "").strip()


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "openai_compatible": "llama-3.3-70b-versatile",
}


def detect_provider_name():
    """
    Detect the LLM provider.

    Explicit LLM_PROVIDER always takes priority.
    Otherwise provider selection is based on available API keys.
    """

    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if explicit:
        return explicit

    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    if os.environ.get("LLM_API_KEY"):
        return "openai_compatible"

    return None


def get_provider():
    """
    Returns a provider instance exposing:

        complete(system, user, max_tokens)

    Returns None when no LLM credentials are configured,
    which activates mock/template mode.
    """

    provider_name = detect_provider_name()

    if provider_name is None:
        return None

    model = (
        os.environ.get("REPLY_GEN_MODEL")
        or DEFAULT_MODELS.get(provider_name)
    )

    if provider_name == "gemini":
        return GeminiProvider(model=model)

    if provider_name == "anthropic":
        return AnthropicProvider(model=model)

    if provider_name == "openai":
        return OpenAICompatibleProvider(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            name="openai",
        )

    if provider_name == "openai_compatible":
        base_url = os.environ.get("LLM_BASE_URL")

        if not base_url:
            raise ValueError(
                "LLM_PROVIDER=openai_compatible (or LLM_API_KEY set) "
                "requires LLM_BASE_URL to also be set."
            )

        return OpenAICompatibleProvider(
            model=model,
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=base_url,
            name="openai_compatible",
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider_name}'. "
        "Use gemini, anthropic, openai, or openai_compatible."
    )