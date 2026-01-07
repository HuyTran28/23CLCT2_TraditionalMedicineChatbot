from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import time


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
        trust_env: bool = True,
        verify_tls: bool = True,
        max_retries: int = 2,
    ) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        if not self._base_url:
            raise ValueError("RemoteLLM requires a non-empty base_url")

        self._api_key = (api_key or "").strip() or None
        self._timeout_seconds = float(timeout_seconds)
        self._default_max_new_tokens = int(default_max_new_tokens)
        self._default_temperature = float(default_temperature)
        self._trust_env = bool(trust_env)
        self._verify_tls = bool(verify_tls)
        self._max_retries = max(0, int(max_retries))

    @classmethod
    def from_env(cls) -> "RemoteLLM":
        base = (os.getenv("LLM_API_BASE") or "").strip()
        if not base:
            raise ValueError("LLM_API_BASE is not set")
        key = (os.getenv("LLM_API_KEY") or "").strip() or None
        timeout_s = float(os.getenv("LLM_API_TIMEOUT") or "120")
        default_max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS") or "1024")
        default_temperature = float(os.getenv("LLM_TEMPERATURE") or "0")

        # Some Windows setups have HTTPS proxy/AV interception that breaks ngrok TLS.
        trust_env_raw = (os.getenv("LLM_TRUST_ENV") or "1").strip().lower()
        trust_env = trust_env_raw not in {"0", "false", "no", "n"}

        verify_raw = (os.getenv("LLM_TLS_VERIFY") or "1").strip().lower()
        verify_tls = verify_raw not in {"0", "false", "no", "n"}

        max_retries = int(os.getenv("LLM_MAX_RETRIES") or "2")
        return cls(
            base_url=base,
            api_key=key,
            timeout_seconds=timeout_s,
            default_max_new_tokens=default_max_new_tokens,
            default_temperature=default_temperature,
            trust_env=trust_env,
            verify_tls=verify_tls,
            max_retries=max_retries,
        )

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

        # Use a session so we can optionally ignore environment proxy settings.
        session = requests.Session()
        session.trust_env = self._trust_env

        payload = {
            "prompt": str(prompt),
            "max_new_tokens": int(max_new_tokens or self._default_max_new_tokens),
            "temperature": float(temperature if temperature is not None else self._default_temperature),
        }

        url = f"{self._base_url}/v1/complete"

        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                r = session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self._timeout_seconds,
                    verify=self._verify_tls,
                )
                break
            except Exception as e:
                last_exc = e

                # Retry only on common transient network/SSL issues.
                is_retryable = False
                try:
                    from requests.exceptions import SSLError, ConnectionError, Timeout
                except Exception:
                    SSLError = ConnectionError = Timeout = Exception  # type: ignore

                if isinstance(e, (SSLError, ConnectionError, Timeout)):
                    is_retryable = True

                if (attempt >= self._max_retries) or (not is_retryable):
                    hint = ""
                    msg = str(e)
                    if "BAD_RECORD_MAC" in msg or "DECRYPTION_FAILED" in msg:
                        hint = (
                            "\n\nHint: Đây thường là lỗi TLS do mạng/proxy/antivirus can thiệp hoặc tunnel ngrok chập chờn. "
                            "Thử: (1) chạy lại Colab/ngrok để lấy URL mới; (2) đặt LLM_TRUST_ENV=0 để bỏ proxy hệ thống; "
                            "(3) nếu bắt buộc để tiếp tục debug: LLM_TLS_VERIFY=0 (không khuyến nghị dùng lâu dài)."
                        )
                    raise RuntimeError(f"Failed to call remote LLM at {url}: {e}{hint}") from e

                # Backoff a bit before retry.
                time.sleep(1.0 + 0.5 * attempt)
        else:
            # Should never reach here.
            raise RuntimeError(f"Failed to call remote LLM at {url}: {last_exc}")

        if r.status_code >= 400:
            body = (r.text or "").strip()

            if r.status_code == 404:
                hint_404 = (
                    "\n\nHint: Remote trả 404 cho /v1/complete. Thường là LLM_API_BASE đang trỏ sai ngrok URL "
                    "(tunnel đã đổi) hoặc server Colab chưa chạy đúng script. "
                    "Kiểm tra: GET {LLM_API_BASE}/health phải trả 200 JSON, và {LLM_API_BASE}/v1/complete phải tồn tại."
                )
                raise RuntimeError(f"Remote LLM error 404 from {url}: {body[:800]}{hint_404}")

            # ngrok often returns an HTML error page when the tunnel is offline
            # (e.g., Colab runtime stopped or ngrok URL changed).
            content_type = (r.headers.get("content-type") or "").lower()
            is_html = ("text/html" in content_type) or body.lower().startswith("<!doctype html")
            if is_html:
                hint = (
                    "\n\nHint: LLM_API_BASE đang trỏ tới một ngrok tunnel không còn hoạt động (hoặc URL đã đổi). "
                    "Hãy mở Colab, chạy lại cell tạo ngrok URL, cập nhật chatbot/.env (LLM_API_BASE=...). "
                    "Kiểm tra nhanh bằng: GET {LLM_API_BASE}/health phải trả 200 JSON."
                )
                raise RuntimeError(f"Remote LLM error {r.status_code} from {url}: {body[:800]}{hint}")

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
