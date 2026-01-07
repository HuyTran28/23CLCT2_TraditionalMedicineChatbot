import os
import gc
import time
import re
import json
import signal
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset 
from ragas import evaluate, RunConfig 
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- IMPORT MODULE TỪ PROPOSED MODEL ---

from modules.vector_store import MedicalVectorStore
from modules.router_engine import build_router_query_engine, MedicalStoreQueryEngine 
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.response.schema import Response
from modules.remote_llm import RemoteLLM

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    answer_correctness
)


RUN_CONFIG_SETTINGS = {
    # Single worker to prevent OOM on T4 GPU
    "max_workers": 4,
    # Generous timeout for slow remote LLM calls (30 min per metric)
    "timeout": 1800,
    # Conservative retries to avoid getting stuck
    "max_retries": 2,
    # Shorter wait between retries
    "max_wait": 30
}

# Results file path (match existing repo artifact)
RESULTS_FILE = Path(__file__).resolve().parent / "eval_proposed.csv"

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    print("\n>>> Nhận tín hiệu dừng! Sẽ lưu kết quả hiện tại...")
    _shutdown_requested = True


def _clear_gpu_memory():
    """Clear GPU memory cache aggressively to prevent OOM on T4."""
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


def _resolve_optional_positive_int(env_name: str) -> int | None:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _resolve_positive_int(env_name: str, default: int) -> int:
    override = _resolve_optional_positive_int(env_name)
    return override if override is not None else default


def _resolve_positive_int_with_min(env_name: str, default: int, *, min_value: int) -> int:
    value = _resolve_positive_int(env_name, default)
    return max(int(min_value), int(value))


DEFAULT_MAX_CONTEXT_CHUNKS = 4
DEFAULT_MAX_CONTEXT_CHARS = 1200


def _resolve_max_context_chunks() -> int:
    return _resolve_positive_int_with_min("EVAL_MAX_CONTEXT_CHUNKS", DEFAULT_MAX_CONTEXT_CHUNKS, min_value=1)


def _resolve_max_context_chars() -> int:
    return _resolve_positive_int_with_min("EVAL_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS, min_value=200)


def _resolve_run_config() -> RunConfig:
    # Allow users to override the most common limits via env vars.
    config = dict(RUN_CONFIG_SETTINGS)
    overrides = {
        "max_workers": "RAGAS_MAX_WORKERS",
        "timeout": "RAGAS_TIMEOUT",
        "max_retries": "RAGAS_MAX_RETRIES",
        "max_wait": "RAGAS_MAX_WAIT",
    }
    for attr, env_var in overrides.items():
        value = _resolve_optional_positive_int(env_var)
        if value is not None:
            config[attr] = value
    return RunConfig(**config)


def _save_results(all_results: list, filename: Path = RESULTS_FILE):
    """Save current results to CSV."""
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f">>> Đã lưu {len(df)} kết quả vào {filename}")
        return df
    return None

# ==============================================================================
# HÀM LÀM SẠCH VĂN BẢN (ULTRA-AGGRESSIVE CLEANING)
# ==============================================================================
def clean_response_text(text: str) -> str:
    """
    Loại bỏ triệt để hình ảnh markdown và chuỗi base64 rác.
    """
    if not isinstance(text, str):
        return ""
    
    # Bước 1: Cắt bỏ thẻ ảnh Markdown chuẩn ![...](...)
    # Pattern: Tìm ![...](...) bất kể bên trong là gì
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text, flags=re.DOTALL)

    # Bước 2: Cắt bỏ nhãn "Hình ảnh:" hoặc "Images:" còn sót lại
    text = re.sub(r'(?:Hình ảnh:|Images:)\s*', '', text, flags=re.IGNORECASE)

    # Bước 3: (Quan trọng) Tìm và diệt chuỗi "data:image..." nếu nó bị lọt ra ngoài thẻ
    # Tìm từ "data:image" cho đến hết dấu ngoặc đóng gần nhất hoặc hết dòng
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[^\s\)]+', '', text, flags=re.DOTALL)

    # Bước 4: Xóa khoảng trắng thừa và dòng trống liên tiếp
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    
    # Bước 5: CHỐT CHẶN AN TOÀN
    # Nếu text vẫn quá dài (do rác base64 chưa sạch), cắt đi.
    # 3000 ký tự là quá đủ cho một câu trả lời y tế.
    MAX_LEN = 3000
    if len(text) > MAX_LEN:
        print(f"Text quá dài ({len(text)} chars). Cắt xuống {MAX_LEN} chars...")
        text = text[:MAX_LEN] + "\n...(truncated for safety)"
        
    return text


# ==============================================================================
# PHẦN 1: MONKEY PATCH (VÁ LỖI MẤT CONTEXT)
# ==============================================================================
def patched_query(self, query_str, **kwargs):
    q = self._coerce_query_text(query_str)
    chunks = self._retrieve(q)

    # Tạo source_nodes thủ công
    source_nodes = []
    for ch in chunks:
        node = TextNode(text=ch.text, metadata=ch.metadata, id_=ch.id)
        score = ch.score if ch.score is not None else 0.0
        source_nodes.append(NodeWithScore(node=node, score=score))

    context = self._build_context(chunks)

    if not context:
        return Response(response="Không tìm thấy dữ liệu.", source_nodes=source_nodes)

    answer = self._answer(q, context)
    images_md = self._images_markdown(chunks)
    answer = self._inject_images_before_sources(answer, images_md)

    # Trả về Response có chứa source_nodes
    return Response(response=answer, source_nodes=source_nodes)


# Áp dụng bản vá
print(">>> [Patch] Đang vá lỗi MedicalStoreQueryEngine...")
MedicalStoreQueryEngine._query = patched_query

# ==============================================================================
# --- PHẦN 2: REMOTE-ONLY ---
# Script này chỉ hỗ trợ self-host/remote LLM qua LLM_API_BASE.

# --- Remote judge adapter (LangChain) ---
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import AIMessage  # type: ignore
    from langchain_core.outputs import ChatGeneration, ChatResult  # type: ignore
except Exception:  # pragma: no cover
    from langchain.chat_models.base import BaseChatModel  # type: ignore
    from langchain.schema import AIMessage, ChatGeneration, ChatResult  # type: ignore


def _messages_to_prompt(messages) -> str:
    parts = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
        content = getattr(m, "content", "")
        parts.append(f"{str(role).upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


class RemoteJudgeChatLLM(BaseChatModel):
    def __init__(self, remote: RemoteLLM, *, temperature: float = 0.0):
        super().__init__()
        self._remote = remote
        self._temperature = float(temperature)

    @classmethod
    def from_env(cls, *, temperature: float = 0.0) -> "RemoteJudgeChatLLM":
        return cls(RemoteLLM.from_env(), temperature=temperature)

    @property
    def _llm_type(self) -> str:
        return "remote_colab_chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = _messages_to_prompt(messages)
        # RAGAS metrics generate long JSON (TP/FP/FN lists) - need more tokens
        max_tokens = int(os.getenv("RAGAS_MAX_NEW_TOKENS") or os.getenv("LLM_MAX_NEW_TOKENS") or "2048")
        resp = self._remote.complete(
            prompt,
            max_new_tokens=max_tokens,
            temperature=self._temperature,
        )
        text = getattr(resp, "text", None)
        content = (text if isinstance(text, str) else str(resp)).strip()
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])


# --- CẤU HÌNH ---
PROPOSED_EMBED_MODEL = "BAAI/bge-m3" 
VECTOR_DB_DIR = Path(__file__).resolve().parent / "vector_data"

if not (os.getenv("LLM_API_BASE") or "").strip():
    raise RuntimeError(
        "Thiếu cấu hình self-host LLM. Hãy set biến môi trường LLM_API_BASE (remote-only)."
    )

# Setup Embedding (CPU)
print(">>> Đang load model Embedding đánh giá...")
EVAL_EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "BAAI/bge-m3").strip()
eval_embeddings = HuggingFaceEmbeddings(
    model_name=EVAL_EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Setup Judge LLM (remote-only)
judge_llm = RemoteJudgeChatLLM.from_env(temperature=0.0)


def load_test_data_from_csv(csv_path: Path) -> list[dict]:
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

def setup_proposed_system():
    print(f"--- Đang khởi động Proposed Model ---")
    llm = RemoteLLM.from_env()
    
    vs = MedicalVectorStore(
        persist_dir=VECTOR_DB_DIR,
        embedding_model=PROPOSED_EMBED_MODEL, 
        embed_batch_size=8,
        device="cpu",
        backend="disk"
    )

    router = build_router_query_engine(
        vector_store=vs,
        llm=llm,
        herbs_top_k=3,       
        diseases_top_k=3,    
        emergency_top_k=2,   
        verbose=True
    )
    return router

def main():
    global _shutdown_requested
    
    # Register signal handlers for graceful shutdown
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass  # May fail on some platforms
    
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(f"Chưa thấy folder '{VECTOR_DB_DIR}'!")

    test_csv = Path(__file__).resolve().parent.parent / "baseline_rag" / "test.csv"
    test_data = load_test_data_from_csv(test_csv)
    if not test_data:
        raise ValueError(f"Không có dòng hợp lệ trong {test_csv}")

    query_engine = setup_proposed_system()
    
    records = []

    print(f"\n>>> Đang chạy {len(test_data)} câu hỏi...")
    
    for item in test_data:
        if _shutdown_requested:
            print(">>> Dừng thu thập dữ liệu...")
            break
            
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nProcessing: {q}")
        
        # 1. Gọi Bot
        response_obj = query_engine.query(q)
        raw_ans = str(response_obj)
        
        # 2. Lọc sạch hình ảnh
        clean_ans = clean_response_text(raw_ans)
        
        # In ra độ dài để kiểm tra
        print(f" -> Original len: {len(raw_ans)}")
        print(f" -> Cleaned len : {len(clean_ans)}")
        if len(clean_ans) > 4000:
             print("CẢNH BÁO: Text vẫn còn quá dài!")

        # 3. Lấy Context (giới hạn để prompt chấm điểm không bị quá dài)
        max_ctx_chunks = _resolve_max_context_chunks()
        max_ctx_chars = _resolve_max_context_chars()
        retrieved_ctx = []
        def get_nodes_recursive(resp):
            nodes = []
            if hasattr(resp, 'source_nodes') and resp.source_nodes:
                nodes.extend(resp.source_nodes)
            if hasattr(resp, 'response') and not isinstance(resp.response, str):
                 nodes.extend(get_nodes_recursive(resp.response))
            return nodes

        raw_nodes = get_nodes_recursive(response_obj)

        def _node_score(n) -> float:
            s = getattr(n, "score", None)
            return float(s) if isinstance(s, (int, float)) else 0.0

        seen = set()
        for node in sorted(raw_nodes, key=_node_score, reverse=True):
            content = node.node.get_content() if hasattr(node, 'node') else node.get_content()
            content = str(content or "").strip()
            if not content:
                continue
            key = hash(content)
            if key in seen:
                continue
            seen.add(key)
            if len(content) > max_ctx_chars:
                content = content[:max_ctx_chars] + "..."
            retrieved_ctx.append(content)
            if len(retrieved_ctx) >= max_ctx_chunks:
                break
        
        print(f" -> Retrieved: {len(retrieved_ctx)} chunks context.")

        records.append(
            {
                "question": q,
                "answer": clean_ans,
                "contexts": retrieved_ctx,
                "ground_truth": gt,
            }
        )
        
        # Clear GPU memory after each query to prevent accumulation
        _clear_gpu_memory()
        
        # Avoid hammering the remote endpoint while keeping throughput reasonable.
        time.sleep(0.3)

    print(f"\n>>> Thu thập xong {len(records)} câu trả lời hợp lệ.")

    if not records:
        raise ValueError("Không tìm thấy câu trả lời hợp lệ để chấm điểm.")

    # Debug: print versions to make timeout tuning reproducible.
    try:
        import ragas as _ragas  # type: ignore
        print(f">>> ragas version: {getattr(_ragas, '__version__', 'unknown')}")
    except Exception:
        pass
    try:
        import langchain as _lc  # type: ignore
        print(f">>> langchain version: {getattr(_lc, '__version__', 'unknown')}")
    except Exception:
        pass

    # =========================================================================
    # SEQUENTIAL EVALUATION with EARLY STOP SAVE
    # =========================================================================
    print(f"\n>>> Đang chấm điểm bằng RAGAS... (sequential mode)")
    my_run_config = _resolve_run_config()
    print(
        f">>> RunConfig: max_workers={my_run_config.max_workers}, timeout={my_run_config.timeout}, "
        f"max_retries={my_run_config.max_retries}, max_wait={my_run_config.max_wait}"
    )

    all_results = []
    total_questions = len(records)

    for idx, record in enumerate(records, start=1):
        # Check for shutdown request (Ctrl+C)
        if _shutdown_requested:
            print(f"\n>>> Đã dừng tại câu {idx}. Đang lưu kết quả...")
            _save_results(all_results)
            break
        
        print(f"\n>>> [{idx}/{total_questions}] {record['question'][:50]}...")
        
        # Clear GPU memory before each evaluation
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
                metrics=[context_recall, answer_correctness, faithfulness, answer_relevancy],
                llm=judge_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False,
                run_config=my_run_config,
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
        time.sleep(1.0)

    if not all_results:
        print(">>> Không có kết quả nào để lưu.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save final results
    final_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    print(f">>> Đã lưu kết quả vào: {RESULTS_FILE}")
    
    print(f"\n>>> ✅ Hoàn thành! Đã xử lý {len(final_df)} câu hỏi.")

if __name__ == "__main__":
    main()