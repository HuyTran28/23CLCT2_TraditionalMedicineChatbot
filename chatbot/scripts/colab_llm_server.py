from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "n"}


def _require_bearer_if_configured(req: Request) -> None:
    expected = (os.getenv("LLM_API_KEY") or "").strip()
    if not expected:
        return

    auth = (req.headers.get("authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")

    token = auth.split(" ", 1)[1].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid API token")


def _load_generator():
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = (
        (os.getenv("HF_MODEL") or "").strip()
        or (os.getenv("MODEL_ID") or "").strip()
        or "Qwen/Qwen2.5-14B-Instruct"
    )
    device_map = (os.getenv("HF_DEVICE_MAP") or "auto").strip() or "auto"
    use_4bit = _env_bool("LOAD_IN_4BIT", False)

    model_kwargs: Dict[str, Any] = {}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",
            )
        except Exception:
            # If bitsandbytes isn't installed, we gracefully fall back to fp16.
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # fp16 is usually the right default on T4; device_map='auto' will shard if needed.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype="auto",
        **model_kwargs,
    )

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
    )

    use_chat_template = _env_bool("USE_CHAT_TEMPLATE", True)

    def build_prompt(user_prompt: str) -> str:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": user_prompt}]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return user_prompt
        return user_prompt

    return model_id, tokenizer, gen, build_prompt


MODEL_ID, _TOKENIZER, _GEN, _BUILD_PROMPT = _load_generator()
_LOCK = threading.Lock()

app = FastAPI()


@app.get("/health")
async def health(req: Request):
    _require_bearer_if_configured(req)
    return JSONResponse({"status": "ok", "model": MODEL_ID})


@app.post("/v1/complete")
async def complete(req: Request):
    _require_bearer_if_configured(req)

    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="Field 'prompt' must be a non-empty string")

    max_new_tokens = payload.get("max_new_tokens", 256)
    temperature = payload.get("temperature", 0.0)

    try:
        max_new_tokens_i = int(max_new_tokens)
        if max_new_tokens_i <= 0:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Field 'max_new_tokens' must be a positive integer")

    try:
        temperature_f = float(temperature)
        if temperature_f < 0:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Field 'temperature' must be a float >= 0")

    do_sample = temperature_f > 0.0
    built = _BUILD_PROMPT(prompt)

    # Serialize generation on a single GPU to avoid OOM spikes.
    with _LOCK:
        try:
            out = _GEN(
                built,
                max_new_tokens=max_new_tokens_i,
                do_sample=do_sample,
                temperature=(temperature_f if do_sample else None),
                return_full_text=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    text = ""
    try:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            text = str(out[0].get("generated_text") or "")
        else:
            text = str(out)
    except Exception:
        text = str(out)

    return JSONResponse({"text": text})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or "8000")
    uvicorn.run(app, host="0.0.0.0", port=port)
