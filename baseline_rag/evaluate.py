import os
import gc
import time
import json
import signal
import sys
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
    # Single worker to prevent OOM on T4 GPU
    "max_workers": 1,
    # Generous timeout for slow remote LLM calls (30 min per metric)
    "timeout": 1800,
    # Conservative retries to avoid getting stuck
    "max_retries": 2,
    # Shorter wait between retries
    "max_wait": 30
}

# Results file path
RESULTS_FILE = Path(__file__).resolve().parent / "ragas_evaluation.csv"

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    print("\n>>> Nhận tín hiệu dừng! Sẽ lưu kết quả hiện tại...")
    _shutdown_requested = True


def _clear_gpu_memory():
    """Clear GPU memory cache aggressively to prevent OOM."""
    # Force Python garbage collection first
    for _ in range(3):
        gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Additional cleanup for T4 GPU
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            # Also clear any gradients
            torch.cuda.ipc_collect()
    except (ImportError, AttributeError):
        pass
    
    # Final garbage collection
    for _ in range(2):
        gc.collect()
    
    # Short sleep to allow OS to reclaim memory
    time.sleep(0.3)


def _save_results(all_results: list, filename: Path = RESULTS_FILE):
    """Save current results to CSV."""
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f">>> Đã lưu {len(df)} kết quả vào {filename}")
        return df
    return None


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
    global _shutdown_requested
    
    # Register signal handlers for graceful shutdown
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass  # May fail on some platforms
    
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
        if _shutdown_requested:
            print(">>> Dừng thu thập dữ liệu...")
            break
            
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"   Processing: {q}")
        response_obj = bot.query(q, return_full=True)
        
        questions.append(q)
        answers.append(str(response_obj))
        ground_truths.append(gt) # Thêm vào list
        
        # Lấy context (giới hạn để tránh OOM)
        source_nodes = getattr(response_obj, "source_nodes", None) or []
        retrieved_texts = []
        MAX_CONTEXT_CHUNKS = 4  # Limit context chunks
        MAX_CONTEXT_CHARS = 1200  # Limit chars per chunk
        for node in source_nodes[:MAX_CONTEXT_CHUNKS]:
            try:
                content = node.node.get_content()
            except Exception:
                try:
                    content = node.get_content()
                except Exception:
                    continue
            # Truncate long contexts
            if len(content) > MAX_CONTEXT_CHARS:
                content = content[:MAX_CONTEXT_CHARS] + "..."
            retrieved_texts.append(content)
        contexts.append(retrieved_texts)

        # Clear memory after each query to prevent accumulation
        _clear_gpu_memory()
        time.sleep(0.3)

    # Build records for batch processing
    records = []
    for i in range(len(questions)):
        records.append({
            "question": questions[i],
            "answer": answers[i],
            "contexts": contexts[i],
            "ground_truth": ground_truths[i],
        })

    print(f"\n>>> Thu thập xong {len(records)} câu trả lời hợp lệ.")

    if not records:
        raise ValueError("Không tìm thấy câu trả lời hợp lệ để chấm điểm.")

    # =========================================================================
    # SEQUENTIAL EVALUATION with EARLY STOP SAVE
    # =========================================================================
    print(f"\n>>> Đang chấm điểm bằng RAGAS... (sequential mode)")
    
    metrics_list = [
        context_recall, 
        answer_correctness, 
        faithfulness, 
        answer_relevancy
    ]

    my_run_config = _resolve_run_config()
    print(f">>> RunConfig: max_workers={my_run_config.max_workers}, timeout={my_run_config.timeout}, max_retries={my_run_config.max_retries}, max_wait={my_run_config.max_wait}")

    all_results = []
    total_questions = len(records)

    for idx, record in enumerate(records, start=1):
        # Check for shutdown request (Ctrl+C)
        if _shutdown_requested:
            print(f"\n>>> Đã dừng tại câu {idx}. Đang lưu kết quả...")
            _save_results(all_results)
            break
        
        print(f"\n>>> [{idx}/{total_questions}] {record['question'][:50]}...")
        
        # Clear memory before each evaluation
        _clear_gpu_memory()
        
        single_dataset = Dataset.from_dict(
            {
                "question": [record["question"]],
                "answer": [record["answer"]],
                "contexts": [record["contexts"]],
                "ground_truth": [record["ground_truth"]],
            }
        )
        
        try:
            results = evaluate(
                dataset=single_dataset,
                metrics=metrics_list,
                llm=judge_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False,  # Don't crash on single question failure
                run_config=my_run_config
            )
            print(f">>> Kết quả: {results}")
            all_results.append(results.to_pandas())
            
            # Save incrementally after each successful evaluation
            _save_results(all_results)
            
        except (MemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "oom" in error_str:
                print(f">>> OOM tại câu {idx}! Đang lưu kết quả...")
                _save_results(all_results)
                raise
            print(f">>> Lỗi câu hỏi {idx}: {e}")
            # Continue to next question
        except Exception as e:
            error_msg = str(e)
            print(f">>> Lỗi câu hỏi {idx}: {error_msg[:200]}")
            if "404" in error_msg:
                print("Hint: Remote trả 404 cho /v1/complete. Thường là LLM_API_BASE đang trỏ sai ngrok URL (tunnel đã đổi) hoặc server Colab đã dừng.")
            # Continue to next question
        
        # Clear memory after each evaluation
        _clear_gpu_memory()
        
        # Check shutdown again after evaluation
        if _shutdown_requested:
            print(f"\n>>> Đã dừng sau câu {idx}. Đang lưu kết quả...")
            _save_results(all_results)
            break
        
        # Delay between questions for stability
        time.sleep(0.3)

    if not all_results:
        print(">>> Không có kết quả nào để lưu.")
        return

    print("\n>>> KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP:")
    final_df = pd.concat(all_results, ignore_index=True)
    print(final_df)
    
    # Save final results
    final_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
    print(f">>> Đã lưu báo cáo vào {RESULTS_FILE}")
    
    print(f"\n>>> ✅ Hoàn thành! Đã xử lý {len(final_df)} câu hỏi.")

if __name__ == "__main__":
    main()