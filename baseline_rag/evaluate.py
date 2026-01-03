import os
import time
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from langchain_community.embeddings import HuggingFaceEmbeddings
from naive_rag import NaiveMedicalRAG

from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    answer_correctness
)

# --- CẤU HÌNH ---
# Danh sách file Markdown đầu vào (moved to chatbot/data/raw)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _build_judge_llm():
    """RAGAS judge model.

    Uses a local HuggingFace pipeline to avoid external API dependencies.
    """

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

# Chạy Embedding trên CPU
print(">>> Đang load model Embedding...")
eval_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

INPUT_FILES = [
    str(DATA_DIR / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md"),
]

# --- DỮ LIỆU TEST (CẦN BỔ SUNG GROUND TRUTH) ---
TEST_DATA = [
    {
        "question": "Các bài thuốc chữa bệnh đái tháo đường có sử dụng mướp đắng?",
        "ground_truth": "Mướp đắng (khổ qua) có thể dùng tươi, nấu canh hoặc phơi khô sắc nước uống để hỗ trợ hạ đường huyết."
    },
    {
        "question": "Lá lốt có tác dụng gì trong việc chữa đau xương khớp?",
        "ground_truth": "Lá lốt có tác dụng ôn trung, tán hàn, hạ khí, chỉ thống, thường dùng chữa đau nhức xương khớp khi trời lạnh."
    },
    
]

def main():
    print(">>> Đang khởi tạo Baseline Bot...")
    bot = NaiveMedicalRAG(INPUT_FILES)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = [] 

    print(f">>> Đang chạy {len(TEST_DATA)} câu hỏi...")
    
    for item in TEST_DATA:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"   Processing: {q}")
        response_obj = bot.query(q, return_full=True)
        
        questions.append(q)
        answers.append(str(response_obj))
        ground_truths.append(gt) # Thêm vào list
        
        # Lấy context
        retrieved_texts = [node.node.get_content() for node in response_obj.source_nodes]
        contexts.append(retrieved_texts)

        time.sleep(5)  # Giữ khoảng cách giữa các câu hỏi

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

    results = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=judge_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False 
    )

    print("\n>>> KẾT QUẢ ĐÁNH GIÁ:")
    print(results)
    
    df_results = results.to_pandas()
    df_results.to_csv("ragas_evaluation.csv", index=False, encoding='utf-8-sig')
    print(">>> Đã lưu báo cáo.")

if __name__ == "__main__":
    main()