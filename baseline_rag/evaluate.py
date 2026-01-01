# evaluate.py
import pandas as pd
import subprocess
import os
import time
import json
from naive_rag import NaiveMedicalRAG
from groq import Groq  # Dùng client native của Groq cho linh hoạt phần Judge

# --- CẤU HÌNH ---
# Danh sách file Markdown đầu vào
INPUT_FILES = [
    "data/raw/cay-rau-lam-thuoc/cay-rau-lam-thuoc.md",
]

TEST_QUESTIONS = [
    "Cây gì trị kiết lỵ?",
    "Các bài thuốc chữa bệnh đái tháo đường có sử dụng mướp đắng?",
    "Lá lốt có tác dụng gì trong việc chữa đau xương khớp?",
]

# Model chấm điểm (Judge)
JUDGE_MODEL = "llama-3.3-70b-versatile" 

# --- HÀM GỌI PROPOSED SYSTEM (ĐÃ COMMENT OUT) ---
def get_proposed_answer(question):
    """
    Hàm này đáng lẽ gọi main.py của nhánh Proposed.
    Hiện tại đang COMMENT OUT logic chạy thật và trả về Placeholder.
    """
    # cmd = [
    #     "python", "main.py", "query",
    #     "--persist-dir", "vector_data",
    #     "--backend", "disk",
    #     "--model", "llama-3.1-8b-instant",
    #     "--question", question
    # ]
    
    # try:
    #     result = subprocess.run(
    #         cmd, 
    #         capture_output=True, 
    #         text=True, 
    #         encoding='utf-8',
    #         check=True
    #     )
    #     return result.stdout.strip()
    # except subprocess.CalledProcessError as e:
    #     return f"ERROR: {e.stderr}"
    
    # --- PLACEHOLDER KHI CHƯA MERGE BRANCH ---
    return "[PENDING] Proposed Model code is on another branch."

# --- HÀM AI JUDGE ---
def evaluate_with_ai(question, baseline_ans, proposed_ans):
    """
    Sử dụng Llama-3-70b để chấm điểm và so sánh 2 câu trả lời.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Prompt cho AI Judge
    system_prompt = """Bạn là một giám khảo y khoa công tâm. Nhiệm vụ của bạn là so sánh hai câu trả lời cho cùng một câu hỏi y học.
    
    Tiêu chí đánh giá:
    1. Độ chính xác (Accuracy): Thông tin có đúng y học không?
    2. Độ đầy đủ (Completeness): Có trả lời hết các ý của câu hỏi không?
    3. Trình bày (Presentation): Dễ đọc, mạch lạc.

    Hãy trả về kết quả dưới dạng JSON (không markdown) với cấu trúc:
    {
        "winner": "Baseline" hoặc "Proposed" hoặc "Tie",
        "reason": "Giải thích ngắn gọn tại sao (tối đa 2 câu)",
        "baseline_score": (thang điểm 1-10),
        "proposed_score": (thang điểm 1-10)
    }
    """
    
    user_prompt = f"""
    Câu hỏi: {question}
    
    --- Câu trả lời Baseline (A):
    {baseline_ans}
    
    --- Câu trả lời Proposed (B):
    {proposed_ans}
    """

    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"} # Bắt buộc trả về JSON
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Lỗi Judge AI: {e}")
        return {
            "winner": "Error", 
            "reason": str(e), 
            "baseline_score": 0, 
            "proposed_score": 0
        }

# --- MAIN LOOP ---
def main():
    # 1. Khởi tạo Baseline Bot
    print(">>> Đang khởi tạo Baseline Bot...")
    # Lưu ý: Đảm bảo class NaiveMedicalRAG đã được update code hỗ trợ CPU/BGE-M3
    baseline_bot = NaiveMedicalRAG(INPUT_FILES)
    
    results = []
    
    print(f">>> Bắt đầu đánh giá {len(TEST_QUESTIONS)} câu hỏi...")
    
    for idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{idx+1}/{len(TEST_QUESTIONS)}] Question: {question}")
        
        # A. Lấy câu trả lời từ Baseline
        start_time = time.time()
        base_ans = baseline_bot.query(question)
        base_time = time.time() - start_time
        print(f"   -> Baseline xong ({base_time:.2f}s)")

        # B. Lấy câu trả lời từ Proposed (Đang giả lập)
        start_time = time.time()
        prop_ans = get_proposed_answer(question)
        prop_time = time.time() - start_time
        print(f"   -> Proposed (Simulated) xong")
        
        # C. AI Judge chấm điểm
        print(f"   -> Đang nhờ {JUDGE_MODEL} chấm điểm...")
        evaluation = evaluate_with_ai(question, base_ans, prop_ans)
        
        # Lưu kết quả
        results.append({
            "Question": question,
            "Baseline Answer": base_ans,
            "Proposed Answer": prop_ans,
            "Winner": evaluation.get("winner"),
            "Explanation": evaluation.get("reason"),
            "Base Score": evaluation.get("baseline_score"),
            "Prop Score": evaluation.get("proposed_score"),
            "Baseline Time (s)": round(base_time, 2),
            "Proposed Time (s)": round(prop_time, 2)
        })

    # 3. Xuất báo cáo
    df = pd.DataFrame(results)
    
    # Lưu CSV
    csv_path = "evaluation_report_with_judge.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n>>> Đã lưu kết quả so sánh vào: {csv_path}")
    
    # Hiển thị bảng so sánh nhanh
    pd.set_option('display.max_columns', None) # Hiển thị hết cột
    print("\n--- PREVIEW ---")
    print(df[["Question", "Winner", "Base Score", "Prop Score"]].head())

if __name__ == "__main__":
    main()