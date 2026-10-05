"""
llm_providers.py — provider-agnostic LLM completion client, so the reply
generator isn't locked to Anthropic. Supports:

  - Anthropic          (ANTHROPIC_API_KEY)
  - OpenAI             (OPENAI_API_KEY)
  - Any OpenAI-compatible endpoint (LLM_API_KEY + LLM_BASE_URL) --
    this covers Groq, Together, Fireworks, Mistral's API, a local Ollama
    server via its OpenAI-compat shim, etc. -- anything that speaks the
    OpenAI chat-completions wire format.

Auto-detected from environment variables at startup, in this priority order
(first one found wins), or forced explicitly with LLM_PROVIDER=
"anthropic"|"openai"|"openai_compatible":

  1. LLM_PROVIDER explicit override
  2. ANTHROPIC_API_KEY present  -> anthropic
  3. OPENAI_API_KEY present     -> openai
  4. LLM_API_KEY present        -> openai_compatible (requires LLM_BASE_URL too)
  5. none present                -> mock mode (get_provider() returns None)

Kept intentionally thin: one method (`complete`) with the same signature
regardless of provider, so reply_generator.py doesn't need to know which
one is active.
"""

import os


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
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAICompatibleProvider:
    """Handles both real OpenAI and any OpenAI-compatible endpoint -- the
    wire format (chat.completions) is identical; only base_url differs."""

    def __init__(self, model, api_key=None, base_url=None, name="openai"):
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
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "openai_compatible": "llama-3.3-70b-versatile",  # sensible default for e.g. Groq; override with REPLY_GEN_MODEL
}


def detect_provider_name():
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LLM_API_KEY"):
        return "openai_compatible"
    return None


def get_provider():
    """Returns a provider instance with a .complete(system, user, max_tokens)
    method, or None if no credentials are configured (mock mode)."""
    provider_name = detect_provider_name()
    if provider_name is None:
        return None

    model = os.environ.get("REPLY_GEN_MODEL") or DEFAULT_MODELS.get(provider_name)

    if provider_name == "anthropic":
        return AnthropicProvider(model=model)
    if provider_name == "openai":
        return OpenAICompatibleProvider(model=model, api_key=os.environ.get("OPENAI_API_KEY"), name="openai")
    if provider_name == "openai_compatible":
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "LLM_PROVIDER=openai_compatible (or LLM_API_KEY set) requires LLM_BASE_URL "
                "to also be set (e.g. https://api.groq.com/openai/v1)."
            )
        return OpenAICompatibleProvider(
            model=model, api_key=os.environ.get("LLM_API_KEY"), base_url=base_url, name="openai_compatible"
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{provider_name}'. Use anthropic, openai, or openai_compatible.")
