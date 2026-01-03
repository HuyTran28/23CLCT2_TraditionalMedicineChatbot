from __future__ import annotations

import os
import threading
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


def _require_auth(authorization: Optional[str]) -> None:
    expected = (os.getenv("LLM_API_KEY") or "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.0


class CompleteResponse(BaseModel):
    text: str


app = FastAPI(title="Colab LLM Server")

_MODEL_ID = (os.getenv("HF_MODEL") or os.getenv("MODEL_ID") or "Qwen/Qwen2.5-7B-Instruct").strip()

_tokenizer = None
_model = None
_lock = threading.Lock()


def _load_model():
    global _tokenizer, _model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_in_4bit = (os.getenv("LOAD_IN_4BIT") or "").strip().lower() in {"1", "true", "yes", "y"}
    device_map = (os.getenv("HF_DEVICE_MAP") or "auto").strip() or "auto"

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)

    kwargs = {
        "device_map": device_map,
        "torch_dtype": "auto",
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            # bitsandbytes/BNB not available; continue without quant.
            pass

    _model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, **kwargs)


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_id": _MODEL_ID}


@app.post("/v1/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest, authorization: Optional[str] = Header(default=None)) -> CompleteResponse:
    _require_auth(authorization)

    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prompt = req.prompt
    max_new = int(req.max_new_tokens)
    temperature = float(req.temperature)

    with _lock:
        import torch

        inputs = _tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        do_sample = temperature > 0
        gen = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )

        # Decode only generated continuation.
        generated = gen[0]
        input_len = inputs["input_ids"].shape[-1]
        out_ids = generated[input_len:]
        text = _tokenizer.decode(out_ids, skip_special_tokens=True)

    return CompleteResponse(text=text.strip())


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Run the in-memory app directly so this script works regardless of cwd.
    uvicorn.run(app, host=host, port=port, reload=False)
