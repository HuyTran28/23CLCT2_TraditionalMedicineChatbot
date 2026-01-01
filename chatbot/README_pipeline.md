# Ingestion + Vector DB + Router usage

This README only covers:

1) Run the ingestion pipeline (chunk + extract schema per document with Groq) to JSONL
2) Combine JSONL files by `index_type`
3) Embed and store to the vector database
4) Use the router / chatbot core query engine

## Install

```powershell
cd c:\Users\Admin\Documents\23CLCT2_TraditionalMedicineChatbot\chatbot
py -m pip install -r requirements.txt
```

## 1) Extract markdown chunks to JSONL (Groq)

This step calls Groq and writes one JSON object per line to `--jsonl-out`.

Required:
- Set `GROQ_API_KEY`
- Pass `--extract`

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

Groq rate limits (429 Too Many Requests):
-- Use `--rpm` to keep a safe request rate (the pipeline sleeps between calls).
-- For Groq free plan, start with `--rpm 2`.
-- You can still see 429s even under RPM if you hit the tokens-per-minute (TPM) limit; lower `--rpm` further (e.g. `--rpm 1`).
- If you hit tokens-per-day (TPD) limits (errors like "tokens per day (TPD)" and very large `Retry-After`), switch to a higher-quota model (recommended: `llama-3.1-8b-instant`) or wait for reset.
- Recommended free-tier extraction settings: `--model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120`.
- If Groq returns HTTP 429, the extractor retries automatically.
	- If `Retry-After` header is present, it sleeps exactly that many seconds.
	- Otherwise it uses exponential backoff with jitter.

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema MedicinalPlant --chunk-by book --index-type herbs_plants --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/herbs_plants_cay-canh.jsonl
```

Supported schemas in this repo:
- `MedicinalPlant`
- `RemedyRecipe`
- `MedicinalVegetable`
- `EndocrineSyndrome`
- `EmergencyProtocol`

### Final extraction commands (all books)

These commands produce one JSON object per line in `../data/processed/*.jsonl`.

**Book: Cây cảnh – cây thuốc trong nhà trường (plants)**
- schema: `MedicinalPlant`
- split_kind: `plants`
- index_type: `herbs_plants`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema MedicinalPlant --chunk-by book --index-type herbs_plants --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/herbs_plants_cay-canh.jsonl --resume
```

**Book: Cây cảnh – cây thuốc trong nhà trường (recipes)**
- schema: `RemedyRecipe`
- split_kind: `recipes`
- index_type: `remedies`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-canh--cay-thuoc-trong-nha-truong/cay-canh--cay-thuoc-trong-nha-truong.md --schema RemedyRecipe --chunk-by book --index-type remedies --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/remedies_cay-canh.jsonl --resume
```

**Book: Cây rau làm thuốc**
- schema: `MedicinalVegetable`
- split_kind: `None`
- index_type: `herbs_vegetables`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-rau-lam-thuoc/cay-rau-lam-thuoc.md --schema MedicinalVegetable --chunk-by book --index-type herbs_vegetables --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/herbs_vegetables_cay-rau.jsonl --resume
```

**Book: Cây thuốc, vị thuốc phòng và chữa bệnh nội tiết (syndromes / PHẦN THỨ HAI)**
- schema: `EndocrineSyndrome`
- split_kind: `syndromes`
- index_type: `endocrine_syndromes`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet.md --schema EndocrineSyndrome --chunk-by book --index-type endocrine_syndromes --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/endocrine_syndromes_noi-tiet.jsonl --resume
```

**Book: Cây thuốc, vị thuốc phòng và chữa bệnh nội tiết (plants / PHẦN THỨ BA)**
- schema: `MedicinalPlant`
- split_kind: `plants`
- index_type: `endocrine_plants`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet/cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet.md --schema MedicinalPlant --chunk-by book --index-type endocrine_plants --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/endocrine_plants_noi-tiet.jsonl --resume
```

**Book: Cấp cứu & chống độc**
- schema: `EmergencyProtocol`
- split_kind: `None`
- index_type: `emergency`

```powershell
$env:GROQ_API_KEY = "..."
py main.py ingest --input ../data/raw/cc_va_chong_doc_258/cc_va_chong_doc_258.md --schema EmergencyProtocol --chunk-by book --index-type emergency --extract --extract-only --model llama-3.1-8b-instant --max-output-tokens 1024 --max-retry-after-seconds 120 --rpm 2 --enrich-images --image-store-dir ../data/processed/images --jsonl-out ../data/processed/emergency_cc.jsonl --resume
```

## 2) Combine JSONL by `index_type`

Only combine JSONLs that belong to the same target `--index-type`.

Example: combine all `herbs` JSONLs into one:

```powershell
Get-Content ../data/processed/herbs_plants_*.jsonl | Set-Content ../data/processed/herbs_plants_all.jsonl
Get-Content ../data/processed/herbs_vegetables_*.jsonl | Set-Content ../data/processed/herbs_vegetables_all.jsonl
Get-Content ../data/processed/remedies_*.jsonl | Set-Content ../data/processed/remedies_all.jsonl
Get-Content ../data/processed/endocrine_syndromes_*.jsonl | Set-Content ../data/processed/endocrine_syndromes_all.jsonl
Get-Content ../data/processed/endocrine_plants_*.jsonl | Set-Content ../data/processed/endocrine_plants_all.jsonl
Get-Content ../data/processed/emergency_*.jsonl | Set-Content ../data/processed/emergency_all.jsonl
```

Example: combine all `diseases` JSONLs:

```powershell
Get-Content ../data/processed/diseases_*.jsonl | Set-Content ../data/processed/diseases_all.jsonl
```

Example: combine all `emergency` JSONLs:

```powershell
Get-Content ../data/processed/emergency_*.jsonl | Set-Content ../data/processed/emergency_all.jsonl
```

## 3) Embed + ingest JSONL to the vector database

This step is local (no Groq calls). For Windows, `--backend disk` is the default and recommended.

With `--persist-dir vector_data`, the on-disk layout is:
- `vector_data/herbs/` (SQLite + sharded embeddings)
- `vector_data/diseases/` (SQLite + sharded embeddings)
- `vector_data/emergency/` (SQLite + sharded embeddings)

```powershell
py main.py ingest --schema MedicinalPlant --index-type herbs_plants --jsonl-out ../data/processed/herbs_plants_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
py main.py ingest --schema MedicinalVegetable --index-type herbs_vegetables --jsonl-out ../data/processed/herbs_vegetables_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
py main.py ingest --schema RemedyRecipe --index-type remedies --jsonl-out ../data/processed/remedies_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
py main.py ingest --schema EndocrineSyndrome --index-type endocrine_syndromes --jsonl-out ../data/processed/endocrine_syndromes_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
py main.py ingest --schema MedicinalPlant --index-type endocrine_plants --jsonl-out ../data/processed/endocrine_plants_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
py main.py ingest --schema EmergencyProtocol --index-type emergency --jsonl-out ../data/processed/emergency_all.jsonl --persist-dir vector_data --embed-model intfloat/multilingual-e5-small --device cpu --embed-batch 1 --ingest-batch 8 --backend disk
```

## 4) Use the router / chatbot core engine

The `query` command loads all 3 indices under `--persist-dir` and uses an LLM router to choose where to search.

```powershell
$env:GROQ_API_KEY = "..."
py main.py query --persist-dir vector_data --backend disk --question "Cây gì trị ho gà?"
```

Notes:
- Routing is implemented in `modules/router_engine.py`.
- Retrieval is done by `MedicalVectorStore.query()` against the chosen `index_type`.

## Run the extract_test script (quick extractor smoke-test)

The helper script [scripts/extract_test.py](scripts/extract_test.py) samples markdown chunks, runs a mock extractor (or the real Groq extractor), and writes JSONL records to an output file. Run it from the repository root so imports resolve correctly.

Basic (mock extractor — no GROQ key required):

```powershell
py scripts\extract_test.py --input ../data/raw/cay-canh--cay-thuoc-trong-nha-truong --schema MedicinalPlant --sample 5 --out ../data/processed/test_extracted.jsonl
```

Use the real Groq extractor (requires `GROQ_API_KEY` in env or pass `--groq-key`):

```powershell
$env:GROQ_API_KEY = "..."
py scripts\extract_test.py --input ../data/raw/cay-canh--cay-thuoc-trong-nha-truong --schema MedicinalPlant --sample 5 --use-llm --rpm 2 --out ../data/processed/test_extracted.jsonl --enrich-images --image-store-dir ../data/processed/images
```

Common options:
- `--input`: Markdown file or directory containing `.md` files.
- `--schema`: Pydantic schema name (e.g., `MedicinalPlant`, `RemedyRecipe`, `EndocrineSyndrome`).
- `--sample`: Number of chunks to sample (default 10).
- `--use-llm`: Use the real Groq extractor (requires API key).
- `--groq-key`: Pass a Groq key on the command line (overrides env).
- `--out`: Output JSONL path (default `../data/processed/test_extracted.jsonl`).
- `--enrich-images`: Attach and store image metadata; use with `--image-store-dir`.

The script prints a short validation summary and a few sample records after writing the output file.
