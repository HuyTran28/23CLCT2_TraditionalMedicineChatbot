# Chatbot RAG

Chatbot này dùng pipeline ingest dữ liệu Markdown thành JSONL, tạo embedding/vector index và truy vấn bằng router RAG.

## Cài đặt và cấu hình

Từ thư mục gốc repository:

```powershell
python -m pip install -r chatbot/requirements.txt
Copy-Item chatbot/.env.example chatbot/.env
```

Điền các biến cần thiết vào `chatbot/.env`:

- `GROQ_API_KEY`: dùng cho extraction hoặc trả lời trực tiếp bằng Groq.
- `LLM_API_BASE`: URL server LLM chạy trên Colab/ngrok.
- `LLM_API_KEY`: tùy chọn nếu server remote có bảo vệ bằng token.

Không commit `chatbot/.env`.

## 1. Chuẩn bị dữ liệu

Corpus Markdown dùng chung nằm ở `data/raw/`. Nếu dữ liệu bắt đầu từ PDF, chạy OCR trước theo [hướng dẫn OCR](../ocr/README.md).

Ví dụ extraction cho một tài liệu:

```powershell
python chatbot/main.py ingest `
  --input data/raw/cay-rau-lam-thuoc/cay-rau-lam-thuoc.md `
  --schema MedicinalVegetable `
  --index-type herbs_vegetables `
  --extract --extract-only `
  --jsonl-out data/processed/herbs_vegetables_cay-rau.jsonl `
  --enrich-images --image-store-dir data/processed/images
```

Sau khi có JSONL, ingest vào vector store:

```powershell
python chatbot/main.py ingest `
  --schema MedicinalVegetable `
  --index-type herbs_vegetables `
  --jsonl-out data/processed/herbs_vegetables_cay-rau.jsonl `
  --persist-dir chatbot/vector_data `
  --backend disk
```

Vector index được tạo local tại `chatbot/vector_data/` và không commit.

## 2. Hỏi đáp bằng CLI

```powershell
python chatbot/main.py query `
  --persist-dir chatbot/vector_data `
  --backend disk `
  --question "Cây dứa có tác dụng gì về tiêu hóa?"
```

## 3. Chạy webapp

```powershell
$env:PERSIST_DIR = "chatbot/vector_data"
$env:BACKEND = "disk"
python -m uvicorn chatbot.webapp:app --reload
```

Mở `http://localhost:8000/` sau khi server khởi động.

## LLM trên Colab

Notebook server nằm tại [`notebooks/colab_llm_server.ipynb`](notebooks/colab_llm_server.ipynb). Một số notebook thử nghiệm khác nằm trong cùng thư mục. Sau khi có URL ngrok, đặt `LLM_API_BASE` vào `chatbot/.env` rồi chạy CLI/webapp.

## Tài liệu pipeline

Chi tiết các bước extraction, gộp JSONL và tạo index nằm tại [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Đánh giá

Scripts và kết quả đo được gom tại [`evaluation/`](evaluation/). Xem [evaluation README](evaluation/README.md) để biết lệnh chạy và ý nghĩa từng file kết quả.
