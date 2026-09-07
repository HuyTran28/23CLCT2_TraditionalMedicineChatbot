# Evaluation

Khu vực này chứa toàn bộ dữ liệu test, script đánh giá và kết quả đo của chatbot RAG.

## Cấu trúc

```text
evaluation/
├── datasets/
│   └── test.csv                 # Bộ câu hỏi và đáp án chuẩn
├── baseline_rag/                # Baseline RAG và script đánh giá
├── proposed/                   # Hệ thống proposed và script đánh giá
└── results/
    ├── baseline/               # CSV/RAGAS/BERTScore của baseline
    ├── proposed/               # CSV/RAGAS/BERTScore của proposed
    └── summary/                # Notebook so sánh tổng hợp
```

## Chạy đánh giá

Chạy từ thư mục gốc repository:

```powershell
python chatbot/evaluation/baseline_rag/evaluate.py
python chatbot/evaluation/baseline_rag/evaluate_bertscore.py
python chatbot/evaluation/proposed/evaluate_proposed.py
python chatbot/evaluation/proposed/evaluate_proposed_bertscore.py
```

Các script có thể cần `GROQ_API_KEY`, `LLM_API_BASE` và vector index đã tạo sẵn tùy phương thức đánh giá.

## Đọc kết quả

- `results/baseline/`: kết quả của baseline RAG.
- `results/proposed/`: kết quả của hệ thống proposed.
- `results/summary/ragas_compare_summary.ipynb`: notebook tổng hợp để so sánh.

CSV kết quả được giữ lại để tái kiểm tra; vector index, cache và dữ liệu trung gian vẫn bị loại khỏi Git bằng `.gitignore`.
