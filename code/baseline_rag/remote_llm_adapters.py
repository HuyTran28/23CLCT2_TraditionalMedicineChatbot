from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import sys
from pathlib import Path


def _import_chatbot_remote_llm():
    """Import RemoteLLM from chatbot/modules/remote_llm.py.

    The chatbot code is written assuming `chatbot/` is on sys.path (so it can
    `import modules.*`). Baseline scripts often run from repo root or
    `baseline_rag/`, so we add `chatbot/` to sys.path here.
    """

    repo_root = Path(__file__).resolve().parent.parent
    chatbot_dir = repo_root / "chatbot"
    if str(chatbot_dir) not in sys.path:
        sys.path.insert(0, str(chatbot_dir))

    from modules.remote_llm import RemoteLLM  # type: ignore

    return RemoteLLM


# =============================================================================
# LlamaIndex adapter (for Settings.llm)
# =============================================================================

def _llama_index_imports():
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

    return CustomLLM, CompletionResponse, LLMMetadata


class LlamaIndexRemoteLLM(_llama_index_imports()[0]):
    """Adapter that lets baseline code use the existing RemoteLLM client."""

    def __init__(
        self,
        *,
        remote: Any,
        context_window: int = 8192,
        num_output: int = 512,
        model_name: str = "self-hosted-colab",
        temperature: float = 0.0,
    ):
        CustomLLM, _, _ = _llama_index_imports()
        # Call CustomLLM pydantic init
        super(CustomLLM, self).__init__()  # type: ignore[misc]

        self._remote = remote
        self._context_window = int(context_window)
        self._num_output = int(num_output)
        self._model_name = str(model_name)
        self._temperature = float(temperature)

    @classmethod
    def from_env(cls) -> "LlamaIndexRemoteLLM":
        RemoteLLM = _import_chatbot_remote_llm()
        remote = RemoteLLM.from_env()

        # Keep defaults consistent with the remote client.
        context_window = int(os.getenv("LLM_CONTEXT_WINDOW") or "8192")
        num_output = int(os.getenv("LLM_MAX_NEW_TOKENS") or "512")
        temperature = float(os.getenv("LLM_TEMPERATURE") or "0")
        return cls(remote=remote, context_window=context_window, num_output=num_output, temperature=temperature)

    @property
    def metadata(self):
        _, _, LLMMetadata = _llama_index_imports()
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._num_output,
            model_name=self._model_name,
        )

    def complete(self, prompt: str, **kwargs: Any):
        _, CompletionResponse, _ = _llama_index_imports()

        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self._num_output

        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = self._temperature

        resp = self._remote.complete(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
        )
        text = getattr(resp, "text", None)
        return CompletionResponse(text=(text if isinstance(text, str) else str(resp)))

    def stream_complete(self, prompt: str, **kwargs: Any):
        # Remote server is non-streaming; emulate stream.
        yield self.complete(prompt, **kwargs)


# =============================================================================
# LangChain adapter (for RAGAS judge)
# =============================================================================

def _langchain_imports():
    # Prefer langchain-core split packages (modern LangChain). Fall back to legacy
    # `langchain` imports for older installs.
    try:
        from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
        from langchain_core.messages import AIMessage  # type: ignore
        from langchain_core.outputs import ChatGeneration, ChatResult  # type: ignore

        return BaseChatModel, AIMessage, ChatGeneration, ChatResult
    except Exception:  # pragma: no cover
        from langchain.chat_models.base import BaseChatModel  # type: ignore
        # Older LangChain exposed these on langchain.schema.
        from langchain.schema import AIMessage, ChatGeneration, ChatResult  # type: ignore

        return BaseChatModel, AIMessage, ChatGeneration, ChatResult


def _messages_to_prompt(messages: List[Any]) -> str:
    parts: List[str] = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
        content = getattr(m, "content", "")
        parts.append(f"{str(role).upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class _JudgeCfg:
    temperature: float = 0.0
    max_new_tokens: int = 512


class RemoteJudgeChatLLM(_langchain_imports()[0]):
    """Adapter so RAGAS can use the same self-hosted RemoteLLM as judge."""

    def __init__(self, *, remote: Any, cfg: Optional[_JudgeCfg] = None):
        BaseChatModel, *_ = _langchain_imports()
        super(BaseChatModel, self).__init__()  # type: ignore[misc]

        self._remote = remote
        self._cfg = cfg or _JudgeCfg(
            temperature=float(os.getenv("LLM_TEMPERATURE") or "0"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS") or "512"),
        )

    @classmethod
    def from_env(cls, *, temperature: float = 0.0) -> "RemoteJudgeChatLLM":
        RemoteLLM = _import_chatbot_remote_llm()
        remote = RemoteLLM.from_env()
        cfg = _JudgeCfg(
            temperature=float(temperature),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS") or "512"),
        )
        return cls(remote=remote, cfg=cfg)

    @property
    def _llm_type(self) -> str:
        return "remote_colab_chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        base = getattr(self._remote, "_base_url", None)
        return {"base_url": base}

    def _generate(self, messages: List[Any], stop=None, run_manager=None, **kwargs: Any):
        _, AIMessage, ChatGeneration, ChatResult = _langchain_imports()

        prompt = _messages_to_prompt(messages)
        resp = self._remote.complete(
            prompt,
            max_new_tokens=int(self._cfg.max_new_tokens),
            temperature=float(self._cfg.temperature),
        )
        text = getattr(resp, "text", None)
        content = (text if isinstance(text, str) else str(resp)).strip()

        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])
