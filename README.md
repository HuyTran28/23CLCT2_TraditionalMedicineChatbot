# Traditional Medicine Chatbot (RAG) + OCR utilities

This repo contains:
- A RAG chatbot under `code/chatbot/` that can answer questions using:
  - **Groq** (cloud) via `GROQ_API_KEY`, OR
  - A **self-hosted LLM** running on **Google Colab** exposed via **ngrok** (`LLM_API_BASE`).
- An OCR / PDF conversion pipeline under `code/ocr/`.

## Quick start (chatbot)

### 0) Install dependencies

From the repo root:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 1) Choose an LLM backend (required)

You must choose **one** backend:

#### Option A (self-hosted, recommended for no rate limits): Colab + ngrok

Before you run **either** `query` or the webapp, you must:
1) Start the Colab server (see the **Self-host LLM on Google Colab (T4) + ngrok** section below)
2) Copy the printed `LLM_API_BASE`
3) Set `LLM_API_BASE` locally (repo-root `.env` is recommended)

#### Option B (cloud): Groq

1) Get a Groq key: https://console.groq.com/keys
2) Set `GROQ_API_KEY` locally

### 2) Configure `.env` (recommended)

Copy `.env.example` to `.env` at the repo root and fill in either **Groq** or **Colab/ngrok** values:

```powershell
Copy-Item .env.example .env
```

Notes:
- `.env` is ignored by git (`.gitignore` includes it). Do not commit secrets.

### 3) Run a query from the CLI

From the repo root:

```powershell
py code\chatbot\main.py query --backend disk --persist-dir code\chatbot\vector_data --question "Cây X có tác dụng gì?"
```

LLM selection rules:
- If `LLM_API_BASE` is set, the CLI uses the remote Colab/ngrok LLM.
- Otherwise it uses Groq and requires `GROQ_API_KEY`.

### 4) Start the webapp (FastAPI)

Run from the `code/chatbot` folder:

```powershell
cd code\chatbot
py -m uvicorn webapp:app --reload --host 0.0.0.0 --port 8000
```

Open:
- UI: http://localhost:8000/
- API: `POST http://localhost:8000/api/query` with JSON `{ "question": "..." }`

## Self-host LLM on Google Colab (T4) + ngrok (do this first if using `LLM_API_BASE`)

The remote server is implemented in `code/chatbot/scripts/colab_llm_server.py` and wrapped by `colab_llm_server.ipynb`.

### 1) Get an ngrok auth token

1. Create/login to an ngrok account.
2. Copy your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken

In Colab:
- Add it as a Secret named **`NGROK_TOKEN`** (Notebook UI: “Secrets”).

### 2) Start the Colab server

1. Open `colab_llm_server.ipynb` in Colab.
2. Run all cells.
3. Wait until the notebook prints something like:

```
LLM_API_BASE = https://xxxx.ngrok-free.app
```

The server exposes:
- `GET /health`
- `POST /v1/complete` with JSON `{ "prompt": "...", "max_new_tokens": 1024, "temperature": 0.0 }`

### 3) Use `LLM_API_BASE` locally

Set `LLM_API_BASE` in your repo-root `.env` (recommended) or in PowerShell:

```powershell
$env:LLM_API_BASE = "https://<paste-from-colab>"
# Optional auth, only if you enabled it in Colab:
# $env:LLM_API_KEY = "my-secret-token"
```

`LLM_API_BASE` is simply the ngrok public URL that forwards to the Colab server running on `127.0.0.1:8000`.

## OCR / PDF conversion pipeline

The OCR pipeline lives under `code/ocr/`.

```powershell
cd code\ocr
py -m pip install -r requirements.txt

# Put a PDF somewhere and run:
py main.py --input "path\to\file.pdf" --output .\output --mode auto
```

Optional configuration: edit `code/ocr/config.py` (defaults: `./input` and `./output` relative to `code/ocr`).