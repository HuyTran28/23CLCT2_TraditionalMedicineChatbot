import os
import sys
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from openai import OpenAI


def _try_load_dotenv() -> None:
    """Best-effort load of chatbot/.env for local development."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Prefer chatbot/.env if present.
    # This script lives at chatbot/scripts/selfhost_smoke_test.py
    chatbot_dir = Path(__file__).resolve().parents[1]
    env_path = chatbot_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def _normalize_base_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Missing base URL")

    # Allow user to paste either:
    # - https://xxxx.trycloudflare.com
    # - https://xxxx.trycloudflare.com/v1
    # - https://xxxx.trycloudflare.com/v1/
    raw = raw.rstrip("/")
    if not raw.endswith("/v1"):
        raw = raw + "/v1"
    return raw


def _host_for_dns(base_url: str) -> str:
    p = urlparse(base_url)
    return p.hostname or ""


def main() -> int:
    _try_load_dotenv()

    # Prefer explicit env vars, fallback to argv[1]
    raw_base = os.environ.get("SELFHOST_BASE_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    try:
        base_url = _normalize_base_url(raw_base)
    except ValueError:
        print("❌ Missing SELFHOST_BASE_URL")
        print("Pick ONE of the following:")
        print("  1) Pass URL as argument:")
        print("     python chatbot/scripts/selfhost_smoke_test.py https://xxxx.trycloudflare.com")
        print("  2) Set it for the current PowerShell session:")
        print("     $env:SELFHOST_BASE_URL=\"https://xxxx.trycloudflare.com\"")
        print("  3) Or use setx then open a NEW terminal:")
        print("     setx SELFHOST_BASE_URL \"https://xxxx.trycloudflare.com\"")
        print("  4) Or put it into chatbot/.env:")
        print("     SELFHOST_BASE_URL=https://xxxx.trycloudflare.com")
        return 1

    api_key = os.environ.get("SELFHOST_API_KEY") or os.environ.get("API_KEY") or "dummy"
    timeout_s = float(os.environ.get("SELFHOST_TIMEOUT", "30"))

    print("BASE_URL:", base_url)

    # 1) GET /models via requests (simple, good error messages)
    models_url = base_url + "/models"
    try:
        r = requests.get(models_url, timeout=timeout_s)
        print("GET /models status:", r.status_code)
        r.raise_for_status()
        models = r.json()
    except Exception as e:
        print("❌ GET /v1/models failed:", repr(e))
        print("Tip: If you used TryCloudflare, DNS can fail from some networks.")
        host = _host_for_dns(base_url)
        if host:
            print("Host:", host)
        return 2

    # Try to pick a model id
    model_id = os.environ.get("SELFHOST_MODEL") or os.environ.get("MODEL_ID")
    if not model_id:
        try:
            model_id = models["data"][0]["id"]
        except Exception:
            model_id = ""

    print("MODEL_ID:", model_id or "<not detected>")
    print("/v1/models (truncated):")
    print(json.dumps(models, ensure_ascii=False, indent=2)[:1200])

    if not model_id:
        print("❌ Could not detect model id from /v1/models. Set SELFHOST_MODEL and re-run.")
        return 3

    # 2) POST /chat/completions via OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Ping: trả lời 1 câu ngắn."}],
            temperature=0.0,
        )
        print("✅ chat.completions OK:")
        print(resp.choices[0].message.content)
    except Exception as e:
        print("❌ chat.completions failed:", repr(e))
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
