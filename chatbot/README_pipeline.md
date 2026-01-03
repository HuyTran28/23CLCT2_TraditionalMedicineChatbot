# Ingestion + Vector DB + Router usage

This README only covers:

1) Run the ingestion pipeline (chunk + extract schema per document with HuggingFace LLM) to JSONL
2) Combine JSONL files by `index_type`
3) Embed and store to the vector database
4) Use the router / chatbot core query engine

## Install

```powershell
cd D:\NLP\Project\23CLCT2_TraditionalMedicineChatbot
\ .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# (optional) server runtime
python -m pip install "uvicorn[standard]"
```

## Remote LLM (Colab + ngrok) quickstart

If you don’t want to run an LLM locally on Windows, you can host the model on **Google Colab** and expose it via **ngrok**. Then, run the rest (vector store / router / webapp) locally and point to the remote LLM.

### 1) On Colab: start the LLM server

In a Colab notebook:

```python
# (Colab) clone + install
!git clone https://github.com/HuyTran28/23CLCT2_TraditionalMedicineChatbot.git
%cd 23CLCT2_TraditionalMedicineChatbot
!pip install -U pip
!pip install -U -r requirements.txt

# (Optional) choose model + auth token
import os
os.environ["HF_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
# os.environ["LLM_API_KEY"] = "your-secret-token"  # optional bearer token

# start server on port 8000
!python chatbot/scripts/colab_llm_server.py
```

### 2) On Colab: expose via ngrok

In another Colab cell:

```python
!pip install -U pyngrok
from pyngrok import ngrok

# If you have an ngrok authtoken, set it once per runtime:
# !ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>

public_url = ngrok.connect(8000, "http")
print("Public URL:", public_url)
```

You will use that URL as `LLM_API_BASE` locally.

### 3) On local machine: point the chatbot to the remote LLM

```powershell
$env:LLM_API_BASE = "https://<your-ngrok-domain>"
# If you set LLM_API_KEY on Colab, set it locally too:
# $env:LLM_API_KEY = "your-secret-token"
```

Quick check (optional):

```powershell
py -c "from modules.remote_llm import RemoteLLM; llm=RemoteLLM.from_env(); print('health', llm.health()); print(llm.complete('Xin chào!').text[:200])"
```

## 1) Extract markdown chunks to JSONL (HuggingFace)

This step calls the configured LLM and writes one JSON object per line to `--jsonl-out`.

Required:
- Pass `--extract`

LLM configuration options:
- Recommended (hybrid): run extraction on Colab GPU (see the notebook in `modules/extractor.ipynb`).
- Local: set `HF_MODEL=<huggingface_model_id>` and optionally `LLM_BACKEND=hf`.

Optional (recommended for this dataset):
- Pass `--enrich-images` to attach image metadata (`images[]`) and store optimized images on disk.
 - Use `--resume` to skip chunks already written to `--jsonl-out` when re-running extraction.

Chunking:
- `--chunk-by book` is the default and is recommended for these documents.

How `split_kind` is chosen (when `--chunk-by book`):
- `MedicinalPlant` -> `split_kind=plants`
- `RemedyRecipe` -> `split_kind=recipes`
- `EndocrineSyndrome` -> `split_kind=syndromes`
- `MedicinalVegetable`, `EmergencyProtocol` -> `split_kind=None` (book-specific splitter decides)

Throttling:
- Use `--rpm` to throttle extraction calls (the pipeline sleeps between calls).
- If running locally, consider smaller models and/or `FORCE_CPU=1`.

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
cd chatbot
python main.py ingest --input data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema MedicinalPlant --chunk-by book --index-type herbs_plants --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/herbs_plants_cay-canh.jsonl
```

Supported schemas in this repo:
- `MedicinalPlant`
- `RemedyRecipe`
- `MedicinalVegetable`
- `EndocrineSyndrome`
- `EmergencyProtocol`

### Final extraction commands (all books)

These commands produce one JSON object per line in `data/processed/*.jsonl` (under the `chatbot` folder).

**Book: Cây cảnh – cây thuốc trong nhà trường (plants)**
- schema: `MedicinalPlant`
- split_kind: `plants`
- index_type: `herbs_plants`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema MedicinalPlant --chunk-by book --index-type herbs_plants --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/herbs_plants_cay-canh.jsonl --resume
```

**Book: Cây cảnh – cây thuốc trong nhà trường (recipes)**
- schema: `RemedyRecipe`
- split_kind: `recipes`
- index_type: `remedies`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema RemedyRecipe --chunk-by book --index-type remedies --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/remedies_cay-canh.jsonl --resume
```

**Book: Cây rau làm thuốc**
- schema: `MedicinalVegetable`
- split_kind: `None`
- index_type: `herbs_vegetables`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cay-rau-lam-thuoc/cay-rau-lam-thuoc.md --schema MedicinalVegetable --chunk-by book --index-type herbs_vegetables --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/herbs_vegetables_cay-rau.jsonl --resume
```

**Book: Cây thuốc, vị thuốc phòng và chữa bệnh nội tiết (syndromes / PHẦN THỨ HAI)**
- schema: `EndocrineSyndrome`
- split_kind: `syndromes`
- index_type: `endocrine_syndromes`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet.md --schema EndocrineSyndrome --chunk-by book --index-type endocrine_syndromes --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/endocrine_syndromes_noi-tiet.jsonl --resume
```

**Book: Cây thuốc, vị thuốc phòng và chữa bệnh nội tiết (plants / PHẦN THỨ BA)**
- schema: `MedicinalPlant`
- split_kind: `plants`
- index_type: `endocrine_plants`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet.md --schema MedicinalPlant --chunk-by book --index-type endocrine_plants --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/endocrine_plants_noi-tiet.jsonl --resume
```

**Book: Cấp cứu & chống độc**
- schema: `EmergencyProtocol`
- split_kind: `None`
- index_type: `emergency`

```powershell
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
python main.py ingest --input data/raw/cc_va_chong_doc_258/cc_va_chong_doc_258.md --schema EmergencyProtocol --chunk-by book --index-type emergency --extract --extract-only --model $env:HF_MODEL --max-output-tokens 1024 --rpm 2 --enrich-images --image-store-dir data/processed/images --jsonl-out data/processed/emergency_cc.jsonl --resume
```

## 2) Combine JSONL by `index_type`

Consolidate multiple extraction outputs into the 4 main index categories. Run these from the **chatbot** directory.

**Combine all plant-related JSONLs into `herbs_all.jsonl`:**
```powershell
cd chatbot
Get-Content data/processed/herbs_plants_*.jsonl, data/processed/herbs_vegetables_*.jsonl, data/processed/endocrine_plants_*.jsonl | Set-Content data/processed/herbs_all.jsonl
```

**Combine all disease/syndrome JSONLs into `diseases_all.jsonl`:**
```powershell
Get-Content data/processed/endocrine_syndromes_*.jsonl, data/processed/emergency_cc_merged.jsonl | Set-Content data/processed/diseases_all.jsonl
```

**Combine all remedy JSONLs into `remedies_all.jsonl`:**
```powershell
Get-Content data/processed/remedies_*.jsonl | Set-Content data/processed/remedies_all.jsonl
```

**Combine all emergency JSONLs into `emergency_all.jsonl`:**
```powershell
Get-Content data/processed/emergency_cc_merged.jsonl | Set-Content data/processed/emergency_all.jsonl
```

## 3) Embed + ingest JSONL to the vector database

Run these from the `chatbot` directory. This step is local (no LLM calls). For Windows, `--backend disk` is the default and recommended.

With `--persist-dir vector_data`, the on-disk layout is:
- `vector_data/herbs/` (SQLite + sharded embeddings)
- `vector_data/diseases/` (SQLite + sharded embeddings)
- `vector_data/remedies/` (SQLite + sharded embeddings)
- `vector_data/emergency/` (SQLite + sharded embeddings)

```powershell
cd chatbot
python main.py ingest --schema MedicinalPlant --index-type herbs --jsonl-out data/processed/herbs_all.jsonl --persist-dir vector_data --embed-model BAAI/bge-m3 --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
python main.py ingest --schema EndocrineSyndrome --index-type diseases --jsonl-out data/processed/diseases_all.jsonl --persist-dir vector_data --embed-model BAAI/bge-m3 --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
python main.py ingest --schema RemedyRecipe --index-type remedies --jsonl-out data/processed/remedies_all.jsonl --persist-dir vector_data --embed-model BAAI/bge-m3 --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
python main.py ingest --schema EmergencyProtocol --index-type emergency --jsonl-out data/processed/emergency_all.jsonl --persist-dir vector_data --embed-model BAAI/bge-m3 --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
```

## 4) Use the router / chatbot core engine

The `query` command loads indices under `--persist-dir` and uses embedding-based routing to choose where to search.

```powershell
$env:LLM_API_BASE = "https://<your-ngrok-domain>"
cd chatbot
python main.py query --persist-dir vector_data --backend disk --question "Cây gì trị ho gà?" --verbose --embed-model BAAI/bge-m3 --device cpu
```

Notes:
- Routing is implemented in `modules/router_engine.py`.
- Retrieval is done by `MedicalVectorStore.query()` against the chosen `index_type`.

## Run the webapp (local)

You can run the local FastAPI webapp for development. From the `chatbot` folder, install `uvicorn` if needed and start the server. Example PowerShell workflow:

```powershell
cd chatbot
# (one-time) install server runtime
python -m pip install -r requirements.txt
python -m pip install "uvicorn[standard]"

# Optional: set environment variables for the vector store
$env:PERSIST_DIR = "vector_data"
$env:EMBED_MODEL = "BAAI/bge-m3"
$env:BACKEND = "disk"

# Start the development server (reload enabled)
python -m uvicorn webapp:app --reload --host 0.0.0.0 --port 8000
```

The webapp exposes endpoints under `http://localhost:8000/` (e.g., the query API at `/api/query`).

## Run the extract_test script (quick extractor smoke-test)

The helper script [scripts/extract_test.py](scripts/extract_test.py) samples markdown chunks, runs a mock extractor (or the real HF extractor), and writes JSONL records to an output file.

Basic (mock extractor — no LLM required):

```powershell
cd chatbot
python scripts\extract_test.py --input data/raw/cay-canh--cay-thuoc-trong-nha-truong --schema MedicinalPlant --sample 5 --out data/processed/test_extracted.jsonl
```

Use the real HF extractor (requires `HF_MODEL` in env or local model availability):

```powershell
python scripts\extract_test.py --input data/raw/cay-canh--cay-thuoc-trong-nha-truong --schema MedicinalPlant --sample 5 --use-llm --rpm 2 --out data/processed/test_extracted.jsonl --enrich-images --image-store-dir data/processed/images
```

Common options:
- `--input`: Markdown file or directory containing `.md` files.
- `--schema`: Pydantic schema name (e.g., `MedicinalPlant`, `RemedyRecipe`, `EndocrineSyndrome`).
- `--sample`: Number of chunks to sample (default 10).
- `--use-llm`: Use the real HF extractor.
- `--out`: Output JSONL path (default `../data/processed/test_extracted.jsonl`).
- `--enrich-images`: Attach and store image metadata; use with `--image-store-dir`.

The script prints a short validation summary and a few sample records after writing the output file.
