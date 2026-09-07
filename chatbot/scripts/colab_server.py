import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. Setup Model (HuggingFace)
# ---------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM

app = FastAPI()

# Global LLM instance
llm_instance = None

def init_llm():
    global llm_instance
    if llm_instance is not None:
        return

    model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    print(f"Loading model: {model_name}...")

    # Auto 4-bit config for Colab GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    llm_instance = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=4096,
        max_new_tokens=1024,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
    )
    print("Model loaded successfully!")

# 2. API Definition
# ---------------------------------------------------------
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024

@app.on_event("startup")
async def startup_event():
    init_llm()

@app.get("/")
def root():
    return {"status": "ok", "model": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")}

@app.post("/complete")
def complete(req: CompletionRequest):
    global llm_instance
    if not llm_instance:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    try:
        response = llm_instance.complete(req.prompt)
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Run Server
# ---------------------------------------------------------
if __name__ == "__main__":
    # This script is meant to be run in Colab
    # We will use pyngrok in the notebook cell to expose this port
    uvicorn.run(app, host="0.0.0.0", port=8000)
