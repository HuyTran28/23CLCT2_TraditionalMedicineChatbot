import os
import pandas as pd
from pathlib import Path
from bert_score import score
import time

# Import class NaiveMedicalRAG 
from naive_rag import NaiveMedicalRAG


# --- CẤU HÌNH DỮ LIỆU ---
ROOT_DIR = Path(__file__).resolve().parent.parent 
DATA_FILES = [
    str(ROOT_DIR / "data" / "raw" / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md")
]

# Bộ câu hỏi kiểm tra kèm đáp án mẫu (Ground Truth)
TEST_DATASET = [
    {
        "id": 1,
        "question": "Ớt được trồng ở đâu?",
        "ground_truth": "Ớt được trồng khắp nơi ở nước ta, từ vùng núi đến đồng bằng."
    },
    {
        "id": 2,
        "question": "Công dụng của rau má là gì?",
        "ground_truth": "Rau má có tác dụng thanh nhiệt, giải độc, chữa sốt, mụn nhọt, vàng da, táo bón."
    },
    {
        "id": 3,
        "question": "Rau mồng tơi có tính chất gì?",
        "ground_truth": "Mồng tơi có vị ngọt, tính lạnh, không độc, giúp nhuận tràng."
    }
]

def run_evaluation():
    print("--- 1. KHỞI TẠO BASELINE RAG ---")
    try:
        # Khởi tạo bot với danh sách file
        bot = NaiveMedicalRAG(file_paths=DATA_FILES)
    except Exception as e:
        print(f"Lỗi khởi tạo bot: {e}")
        return

    print(f"\n--- 2. BẮT ĐẦU CHẠY THỬ NGHIỆM ({len(TEST_DATASET)} câu hỏi) ---")
    
    generated_answers = []
    reference_answers = []
    results = []

    for item in TEST_DATASET:
        q = item["question"]
        ref = item["ground_truth"]
        
        print(f" > Đang hỏi: {q}")
        start_time = time.time()
        
        # Gọi hàm query của Baseline
        # return_full=False để lấy chuỗi text BERTScore
        try:
            ans = bot.query(q, return_full=False)
        except Exception as e:
            ans = "Error generating answer"
            print(f"   Lỗi: {e}")
            
        elapsed = time.time() - start_time
        
        generated_answers.append(ans)
        reference_answers.append(ref)
        
        results.append({
            "ID": item["id"],
            "Question": q,
            "Ground Truth": ref,
            "Baseline Answer": ans,
            "Time (s)": round(elapsed, 2)
        })

    print("\n--- 3. TÍNH ĐIỂM BERTSCORE (Độ tương đồng ngữ nghĩa) ---")
    # Tính toán độ tương đồng giữa câu trả lời của Bot và Đáp án mẫu
    # lang='vi' giúp load model hỗ trợ tiếng Việt 
    try:
        P, R, F1 = score(generated_answers, reference_answers, lang='vi', verbose=True)
        
        # Thêm điểm số vào kết quả
        for i, f1_score in enumerate(F1):
            results[i]["BERT_F1"] = round(f1_score.item(), 4)
            
        avg_f1 = F1.mean().item()
        print(f"\n>>> ĐIỂM SỐ TRUNG BÌNH (Baseline F1): {avg_f1:.4f} <<<")
        
    except Exception as e:
        print(f"Không thể tính BERTScore (có thể do chưa cài thư viện): {e}")
        print("Vẫn sẽ xuất file CSV chứa câu trả lời.")

    # --- 4. XUẤT BÁO CÁO ---
    output_file = "baseline_evaluation_report.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"\nĐã lưu báo cáo chi tiết vào: {output_file}")

if __name__ == "__main__":
    # Đảm bảo biến môi trường API Key đã có 
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = "gsk_..." 
        
    run_evaluation()