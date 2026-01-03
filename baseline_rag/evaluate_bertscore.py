import os
import pandas as pd
import time
import re
from pathlib import Path
from bert_score import score

# Import class Baseline của bạn
# Đảm bảo file chứa class này tên là 'naive_rag.py' hoặc sửa lại cho đúng tên file bạn đang lưu
try:
    from naive_rag import NaiveMedicalRAG
except ImportError:
    # Fallback nếu bạn lưu tên khác, ví dụ baseline_rag
    try:
        from baseline_rag import NaiveMedicalRAG
    except ImportError:
        raise ImportError("Không tìm thấy file chứa class NaiveMedicalRAG!")

# --- CẤU HÌNH DỮ LIỆU ---
ROOT_DIR = Path(__file__).resolve().parent.parent 
# Đường dẫn đến file Markdown gốc (Dữ liệu nạp vào)
DATA_FILES = [
    str(ROOT_DIR / "data" / "raw" / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md")
]

# Tên file câu hỏi test
TEST_FILE_NAME = "test.csv"

# --- CÁC HÀM HỖ TRỢ ---
def clean_response(text):
    """Làm sạch câu trả lời: Xóa ảnh markdown, xóa khoảng trắng thừa"""
    if not text: return ""
    text = str(text)
    # Loại bỏ thẻ ảnh ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text, flags=re.DOTALL)
    # Loại bỏ khoảng trắng thừa
    return " ".join(text.split())

def load_test_dataset(filename):
    """Đọc file CSV test set"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Không tìm thấy file '{filename}' tại thư mục hiện tại.")
    
    print(f"--- Đang đọc dữ liệu test từ: {filename} ---")
    df = pd.read_csv(filename)
    
    # Kiểm tra cột bắt buộc
    required = ['Câu hỏi', 'Đáp án']
    if not all(col in df.columns for col in required):
        raise ValueError(f"File CSV thiếu cột bắt buộc. Cần có: {required}")
    
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "id": row.get('id', _),
            "difficulty": row.get('Độ khó', 'Unknown'), # Lấy cột độ khó để phân tích
            "question": str(row['Câu hỏi']).strip(),
            "ground_truth": str(row['Đáp án']).strip()
        })
    print(f"-> Đã load thành công {len(dataset)} câu hỏi.")
    return dataset

def run_evaluation():
    # 1. Load dữ liệu test
    try:
        test_dataset = load_test_dataset(TEST_FILE_NAME)
    except Exception as e:
        print(f"Lỗi đọc file test: {e}")
        return

    # 2. Khởi tạo Bot
    print("\n--- 1. KHỞI TẠO BASELINE RAG ---")
    try:
        bot = NaiveMedicalRAG(file_paths=DATA_FILES)
    except Exception as e:
        print(f"Lỗi khởi tạo bot: {e}")
        return

    print(f"\n--- 2. BẮT ĐẦU CHẠY THỬ NGHIỆM ({len(test_dataset)} câu hỏi) ---")
    
    valid_cands = [] # Chứa câu trả lời sạch để tính điểm
    valid_refs = []  # Chứa đáp án mẫu tương ứng
    results = []

    for i, item in enumerate(test_dataset):
        q = item["question"]
        ref = item["ground_truth"]
        diff = item["difficulty"]
        
        print(f"[{i+1}/{len(test_dataset)}] Hỏi ({diff}): {q}")
        start_time = time.time()
        
        try:
            # Query bot
            raw_ans = bot.query(q, return_full=False)
            # Làm sạch (để đảm bảo công bằng với Proposed Model)
            clean_ans = clean_response(raw_ans)
        except Exception as e:
            raw_ans = f"Error: {e}"
            clean_ans = "Error"
            print(f"   Lỗi: {e}")
            
        elapsed = time.time() - start_time
        
        # Lưu vào list để tính BERTScore sau
        valid_cands.append(clean_ans)
        valid_refs.append(ref)
        
        results.append({
            "ID": item["id"],
            "Độ khó": diff,
            "Question": q,
            "Ground Truth": ref,
            "Baseline Raw": raw_ans,
            "Baseline Cleaned": clean_ans,
            "Time (s)": round(elapsed, 2),
            "BERT_F1": 0.0 # Sẽ cập nhật sau
        })

    print("\n--- 3. TÍNH ĐIỂM BERTSCORE ---")
    try:
        # Tính toán hàng loạt
        P, R, F1 = score(valid_cands, valid_refs, lang='vi', verbose=True)
        
        # Cập nhật điểm vào kết quả
        for idx, f1_val in enumerate(F1):
            results[idx]["BERT_F1"] = round(f1_val.item(), 4)
            
        # Tính trung bình
        avg_f1 = F1.mean().item()
        
        # Tạo DataFrame
        df_res = pd.DataFrame(results)
        
        print(f"\n===========================================")
        print(f" KẾT QUẢ BASELINE (NAIVE RAG)")
        print(f" F1 Score Trung Bình: {avg_f1:.4f}")
        print(f"===========================================")
        
        print("\n--- Điểm số theo độ khó ---")
        if "Độ khó" in df_res.columns:
            print(df_res.groupby("Độ khó")["BERT_F1"].mean())

    except Exception as e:
        print(f"Lỗi tính BERTScore: {e}")
        df_res = pd.DataFrame(results)

    # --- 4. XUẤT BÁO CÁO ---
    output_file = "baseline_evaluation_report.csv"
    df_res.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"\nĐã lưu báo cáo chi tiết vào: {output_file}")

if __name__ == "__main__":
    # Đảm bảo biến môi trường API Key đã có 
    if "GROQ_API_KEY" not in os.environ:
        # Điền key của bạn vào đây nếu chưa set trong Environment Variables
        os.environ["GROQ_API_KEY"] = "gsk_..." 
        
    run_evaluation()