import os
import time
import re
import pandas as pd
from pathlib import Path
from datasets import Dataset 
from ragas import evaluate, RunConfig 
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- IMPORT MODULE TỪ PROPOSED MODEL ---

from modules.vector_store import MedicalVectorStore
from modules.router_engine import build_router_query_engine, MedicalStoreQueryEngine 
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.response.schema import Response
from modules.remote_llm import RemoteLLM
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    answer_correctness
)

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
# Script này chỉ hỗ trợ self-host/remote LLM qua LLM_API_BASE (không có fallback Groq).

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
        resp = self._remote.complete(
            prompt,
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS") or "512"),
            temperature=self._temperature,
        )
        text = getattr(resp, "text", None)
        content = (text if isinstance(text, str) else str(resp)).strip()
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])


# --- CẤU HÌNH ---
PROPOSED_EMBED_MODEL = "BAAI/bge-m3" 
VECTOR_DB_DIR = "vector_data"

if not (os.getenv("LLM_API_BASE") or "").strip():
    raise RuntimeError(
        "Thiếu cấu hình self-host LLM. Hãy set biến môi trường LLM_API_BASE (remote-only)."
    )

# Setup Embedding (CPU)
print(">>> Đang load model Embedding đánh giá...")
EVAL_EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "intfloat/multilingual-e5-small")
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

    # Giảm top_k
    router = build_router_query_engine(
        vector_store=vs,
        llm=llm,
        herbs_top_k=2,       
        diseases_top_k=2,    
        emergency_top_k=2,   
        verbose=True
    )
    return router

def main():
    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(f"Chưa thấy folder '{VECTOR_DB_DIR}'!")

    test_csv = Path(__file__).resolve().parent.parent / "baseline_rag" / "test.csv"
    test_data = load_test_data_from_csv(test_csv)
    if not test_data:
        raise ValueError(f"Không có dòng hợp lệ trong {test_csv}")

    query_engine = setup_proposed_system()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = [] 

    print(f"\n>>> Đang chạy {len(test_data)} câu hỏi...")
    
    for item in test_data:
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

        # 3. Lấy Context
        retrieved_ctx = []
        def get_nodes_recursive(resp):
            nodes = []
            if hasattr(resp, 'source_nodes') and resp.source_nodes:
                nodes.extend(resp.source_nodes)
            if hasattr(resp, 'response') and not isinstance(resp.response, str):
                 nodes.extend(get_nodes_recursive(resp.response))
            return nodes

        raw_nodes = get_nodes_recursive(response_obj)
        for node in raw_nodes:
            content = node.node.get_content() if hasattr(node, 'node') else node.get_content()
            if len(content) > 2000:
                content = content[:2000] + "..."
            retrieved_ctx.append(content)
        
        print(f" -> Retrieved: {len(retrieved_ctx)} chunks context.")

        questions.append(q)
        answers.append(clean_ans) 
        contexts.append(retrieved_ctx)
        ground_truths.append(gt)
        
        time.sleep(5)  # Giữ khoảng cách giữa các câu hỏi

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths 
    }
    dataset = Dataset.from_dict(data_dict)

    print("\n>>> Đang chấm điểm bằng RAGAS...")
    
    # RunConfig an toàn
    my_run_config = RunConfig(
        max_workers=1,      
        timeout=180,        
        max_retries=3,      
        max_wait=60         
    )

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[context_recall, answer_correctness, faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False,
            run_config=my_run_config 
        )

        print("\n>>> KẾT QUẢ:")
        print(results)
        
        # Lưu file an toàn (Tránh lỗi Permission Denied)
        df = results.to_pandas()
        base_filename = "evaluation_proposed_final.csv"
        
        df.to_csv(base_filename, index=False, encoding='utf-8-sig')
        print(f">>> Đã lưu kết quả vào: {base_filename}")

    except Exception as e:
        print(f"Lỗi khi chạy đánh giá: {e}")

if __name__ == "__main__":
    main()