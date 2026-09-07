import os
from typing import Any, Optional

from dotenv import load_dotenv


def _norm_provider(p: Optional[str]) -> str:
    p2 = (p or "").strip().lower()
    return p2 or "groq"


def build_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Any:
    """Build a LlamaIndex LLM instance based on environment.

    Supported providers:
    - groq: uses GROQ_API_KEY
    - selfhost: uses an OpenAI-compatible API via SELFHOST_BASE_URL (/v1)
    """

    # Ensure .env is loaded when running from CLI.
    load_dotenv()

    provider_n = _norm_provider(provider or os.getenv("LLM_PROVIDER"))

    if provider_n == "selfhost":
        from llama_index.llms.openai_like import OpenAILike

        base = (os.getenv("SELFHOST_BASE_URL") or "").strip()
        if not base:
            raise ValueError("LLM_PROVIDER=selfhost requires SELFHOST_BASE_URL")

        # Normalize: accept either https://host or https://host/v1
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        api_key = (os.getenv("SELFHOST_API_KEY") or os.getenv("API_KEY") or "dummy").strip() or "dummy"
        model_id = (model or os.getenv("SELFHOST_MODEL") or os.getenv("MODEL_ID") or "").strip()
        if not model_id:
            raise ValueError("LLM_PROVIDER=selfhost requires SELFHOST_MODEL (or pass --model)")

        # Prefer Chat Completions for maximum compatibility with OpenAI-like servers
        # (e.g. KoboldCpp, llama.cpp llama-server, text-generation-webui public API).
        # LlamaIndex still exposes `.complete()` on chat models; it will route through
        # the chat endpoint under the hood.
        return OpenAILike(
            model=model_id,
            api_base=base,
            api_key=api_key,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            is_chat_model=True,
            is_function_calling_model=False,
        )

    if provider_n == "groq":
        from llama_index.llms.groq import Groq

        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")

        model_id = (model or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        return Groq(
            api_key=api_key,
            model=model_id,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider_n}")
