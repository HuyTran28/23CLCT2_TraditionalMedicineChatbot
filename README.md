# Traditional Medicine Tools

Repository này gồm hai tool độc lập:

1. `ocr/`: chuyển PDF sang Markdown/Word, có nhận diện PDF scan và PDF digital.
2. `chatbot/`: xây dựng và truy vấn chatbot RAG cho dữ liệu y học cổ truyền.

## Cấu trúc

```text
ocr/
├── main.py, config.py
├── src/                 # OCR, xử lý Markdown, export Word
├── notebooks/
├── input/               # PDF local, không commit
└── output/              # Kết quả OCR local, không commit

chatbot/
├── main.py, webapp.py
├── modules/, schemas/   # Core RAG và schema dữ liệu
├── scripts/, notebooks/
├── evaluation/
│   ├── baseline_rag/
│   ├── proposed/
│   ├── datasets/
│   └── results/         # Kết quả đo theo baseline/proposed
└── .env.example

data/                    # Corpus dùng chung: raw Markdown và JSONL đã xử lý
```

## Cài đặt

```powershell
python -m pip install -r chatbot/requirements.txt
python -m pip install -r ocr/requirements.txt
```

## Hướng dẫn sử dụng

- [OCR PDF](ocr/README.md)
- [Chatbot RAG](chatbot/README.md)
- [Đánh giá và kết quả đo](chatbot/evaluation/README.md)

## Cấu hình và dữ liệu

Copy `chatbot/.env.example` thành `chatbot/.env`, sau đó điền các biến môi trường cần thiết. Không đưa API key vào source code hoặc commit.

PDF, vector index, embedding, ảnh sinh ra và output chạy local đã được thêm vào `.gitignore`. Các file CSV/notebook trong `chatbot/evaluation/results/` là kết quả đo được sắp xếp để theo dõi và so sánh.
