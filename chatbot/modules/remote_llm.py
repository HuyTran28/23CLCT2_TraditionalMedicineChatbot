from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os


@dataclass(frozen=True)
class RemoteCompletionResponse:
    text: str
    raw: Any = None


class RemoteLLM:
    """Minimal LLM wrapper that matches this repo's usage.

    The router expects an object with a `.complete(prompt)` method returning
    an object with a `.text` attribute.

    Env vars (recommended):
    - LLM_API_BASE: e.g. https://xxxx.ngrok-free.app
    - LLM_API_KEY: optional bearer token; if set, sent as Authorization header
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        default_max_new_tokens: int = 1024,
        default_temperature: float = 0.0,
    ) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        if not self._base_url:
            raise ValueError("RemoteLLM requires a non-empty base_url")

        self._api_key = (api_key or "").strip() or None
        self._timeout_seconds = float(timeout_seconds)
        self._default_max_new_tokens = int(default_max_new_tokens)
        self._default_temperature = float(default_temperature)

    @classmethod
    def from_env(cls) -> "RemoteLLM":
        base = (os.getenv("LLM_API_BASE") or "").strip()
        if not base:
            raise ValueError("LLM_API_BASE is not set")
        key = (os.getenv("LLM_API_KEY") or "").strip() or None
        timeout_s = float(os.getenv("LLM_API_TIMEOUT") or "120")
        return cls(base_url=base, api_key=key, timeout_seconds=timeout_s)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> RemoteCompletionResponse:
        try:
            import requests
        except Exception as e:
            raise RuntimeError("RemoteLLM requires 'requests' to be installed") from e

        payload = {
            "prompt": str(prompt),
            "max_new_tokens": int(max_new_tokens or self._default_max_new_tokens),
            "temperature": float(temperature if temperature is not None else self._default_temperature),
        }

        url = f"{self._base_url}/v1/complete"
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=self._timeout_seconds)
        except Exception as e:
            raise RuntimeError(f"Failed to call remote LLM at {url}: {e}") from e

        if r.status_code >= 400:
            body = (r.text or "").strip()
            raise RuntimeError(f"Remote LLM error {r.status_code} from {url}: {body[:2000]}")

        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Remote LLM returned non-JSON from {url}: {(r.text or '')[:2000]}") from e

        text = data.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Remote LLM response missing 'text': {data}")

        return RemoteCompletionResponse(text=text, raw=data)

    def health(self) -> bool:
        try:
            import requests
        except Exception:
            return False

        url = f"{self._base_url}/health"
        try:
            r = requests.get(url, headers=self._headers(), timeout=10)
            return r.status_code < 400
        except Exception:
            return False
