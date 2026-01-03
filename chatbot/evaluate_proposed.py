import os
import time
import re
import pandas as pd
from datasets import Dataset 
from ragas import evaluate, RunConfig 
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- IMPORT MODULE TỪ PROPOSED MODEL ---

from modules.vector_store import MedicalVectorStore
from modules.router_engine import build_router_query_engine, MedicalStoreQueryEngine 
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.response.schema import Response
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
# --- PHẦN 2: CẤU HÌNH LLM ĐÁNH GIÁ (HF/remote) ---
# --- CẤU HÌNH ---

# Sửa lại cho khớp với model lúc Ingest (BAAI/bge-m3)
PROPOSED_EMBED_MODEL = "BAAI/bge-m3" 
VECTOR_DB_DIR = "vector_data"

# Setup Embedding (CPU)
print(">>> Đang load model Embedding đánh giá...")
eval_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Setup Judge LLM
def _build_judge_llm():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline
        import torch
    except Exception as e:
        raise RuntimeError("Judge LLM requires transformers + langchain-community") from e

    model_id = (os.getenv("JUDGE_HF_MODEL") or os.getenv("HF_MODEL") or "Qwen/Qwen2.5-0.5B-Instruct").strip()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=gen)


judge_llm = _build_judge_llm()

# Dữ liệu test
TEST_DATA = [
    {
        "question": "đinh lăng có tác dụng gì?",
        "ground_truth": "Rễ đinh lăng làm thuốc bổ, lợi tiểu. Lá chữa cảm sốt, giã nát đắp chữa mụn nhọt, sưng tấy."
    }
]

def setup_proposed_system():
    print(f"--- Đang khởi động Proposed Model ---")
    # Answer LLM: prefer remote Colab/ngrok if configured.
    api_base = (os.getenv("LLM_API_BASE") or "").strip()
    if api_base:
        from modules.remote_llm import RemoteLLM

        llm = RemoteLLM.from_env()
    else:
        backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
        hf_model_id = (os.getenv("HF_MODEL") or "").strip()
        if backend != "hf" and not hf_model_id:
            raise RuntimeError("Set LLM_API_BASE (remote) or HF_MODEL/LLM_BACKEND=hf (local) before running evaluation")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from llama_index.llms.huggingface import HuggingFaceLLM
            import torch
        except Exception as e:
            raise RuntimeError("Local HF answering requires transformers + llama-index-llms-huggingface") from e

        model_id = hf_model_id or "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        llm = HuggingFaceLLM(model=hf_model, tokenizer=tokenizer, temperature=0.0, max_new_tokens=512)
    
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

    query_engine = setup_proposed_system()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = [] 

    print(f"\n>>> Đang chạy {len(TEST_DATA)} câu hỏi...")
    
    for item in TEST_DATA:
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