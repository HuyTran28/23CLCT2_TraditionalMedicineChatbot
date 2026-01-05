import os
import time
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset 
from ragas import evaluate, RunConfig
from naive_rag import NaiveMedicalRAG

from remote_llm_adapters import RemoteJudgeChatLLM

from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    answer_correctness
)


RUN_CONFIG_SETTINGS = {
    # Increase parallelism to evaluate multiple items concurrently.
    "max_workers": 1,
    # Keep a reasonable overall timeout per job (seconds).
    "timeout": 600,
    "max_retries": 2,
    "max_wait": 30,
}


def _resolve_run_config() -> RunConfig:
    # Adjust RUN_CONFIG_SETTINGS directly in this file for new limits.
    return RunConfig(**RUN_CONFIG_SETTINGS)

# --- CẤU HÌNH ---
# Danh sách file Markdown đầu vào (moved to chatbot/data/raw)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

if not (os.getenv("LLM_API_BASE") or "").strip():
    raise ValueError("Cần self-host LLM. Hãy set LLM_API_BASE (và LLM_API_KEY nếu có) trước khi chạy evaluate.")

# Always use self-hosted Colab/ngrok judge LLM.
judge_llm = RemoteJudgeChatLLM.from_env(temperature=0.0)

# Chạy Embedding trên CPU (phục vụ RAGAS). Embedding KHÔNG phải LLM.
# Để tránh download model quá lớn, cho phép cấu hình qua env var.
EMBED_MODEL = (os.getenv("EVAL_EMBED_MODEL") or "BAAI/bge-m3").strip()
print(f">>> Đang load model Embedding cho RAGAS: {EMBED_MODEL} ...")

from langchain_community.embeddings import HuggingFaceEmbeddings

eval_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

INPUT_FILES = [
    str(DATA_DIR / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md"),
    str(DATA_DIR / "cay-canh--cay-thuoc-trong-nha-truong" / "cay-canh-cay-thuoc-trong-nha-truong.md"),
    str(DATA_DIR / "cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet" / "cay-thuoc-vi-thuoc-phong-chua-benh-noi-tiet.md"),
    str(DATA_DIR / "cc_va_chong_doc_258" / "cc_va_chong_doc_258.md"),
]

BASELINE_STORAGE_DIR = Path(__file__).resolve().parent / "baseline_storage"


def load_test_data_from_csv(csv_path: Path) -> list[dict]:
    """Load evaluation Q/A pairs from baseline_rag/test.csv.

    Expected Vietnamese columns:
    - Câu hỏi -> question
    - Đáp án -> ground_truth
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file test CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    q_col = "Câu hỏi"
    a_col = "Đáp án"
    if q_col not in df.columns or a_col not in df.columns:
        raise ValueError(
            f"CSV thiếu cột bắt buộc. Cần có '{q_col}' và '{a_col}'. Cột hiện có: {list(df.columns)}"
        )

    rows: list[dict] = []
    for _, r in df.iterrows():
        q = str(r.get(q_col) or "").strip()
        gt = str(r.get(a_col) or "").strip()
        if not q or not gt:
            continue
        rows.append({"question": q, "ground_truth": gt})

    limit_raw = (os.getenv("EVAL_LIMIT") or "").strip()
    if limit_raw:
        try:
            limit = int(limit_raw)
            if limit > 0:
                rows = rows[:limit]
        except Exception:
            pass
    return rows

def main():
    test_csv = Path(__file__).resolve().parent / "test.csv"
    test_data = load_test_data_from_csv(test_csv)
    if not test_data:
        raise ValueError(f"Không có dòng hợp lệ trong {test_csv}")

    print(">>> Đang khởi tạo Baseline Bot...")
    bot = NaiveMedicalRAG(INPUT_FILES, persist_dir=str(BASELINE_STORAGE_DIR))
    
    questions = []
    answers = []
    contexts = []
    ground_truths = [] 

    print(f">>> Đang chạy {len(test_data)} câu hỏi...")
    
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"   Processing: {q}")
        response_obj = bot.query(q, return_full=True)
        
        questions.append(q)
        answers.append(str(response_obj))
        ground_truths.append(gt) # Thêm vào list
        
        # Lấy context
        source_nodes = getattr(response_obj, "source_nodes", None) or []
        retrieved_texts = []
        for node in source_nodes:
            try:
                retrieved_texts.append(node.node.get_content())
            except Exception:
                try:
                    retrieved_texts.append(node.get_content())
                except Exception:
                    continue
        contexts.append(retrieved_texts)

        time.sleep(1)  # Giảm sleep để tăng throughput

    # Tạo Dataset đủ 4 cột: question, answer, contexts, ground_truth
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths 
    }
    dataset = Dataset.from_dict(data_dict)

    print("\n>>> Đang chấm điểm bằng RAGAS...")
    
    metrics_list = [
        context_recall, 
        answer_correctness, 
        faithfulness, 
        answer_relevancy
    ]

    my_run_config = _resolve_run_config()
    print(f">>> RunConfig: max_workers={my_run_config.max_workers}, timeout={my_run_config.timeout}, max_retries={my_run_config.max_retries}, max_wait={my_run_config.max_wait}")

    results = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=judge_llm,
        embeddings=eval_embeddings,
        raise_exceptions=True,
        run_config=my_run_config
    )

    print("\n>>> KẾT QUẢ ĐÁNH GIÁ:")
    print(results)
    
    df_results = results.to_pandas()
    df_results.to_csv("ragas_evaluation.csv", index=False, encoding='utf-8-sig')
    print(">>> Đã lưu báo cáo.")

if __name__ == "__main__":
    main()