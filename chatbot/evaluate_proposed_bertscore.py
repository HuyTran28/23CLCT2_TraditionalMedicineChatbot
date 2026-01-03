# import os
# import pandas as pd
# import time
# import re
# from pathlib import Path
# from dotenv import load_dotenv
# from bert_score import score

# # --- IMPORT MODULES ---
# try:
#     from llama_index.llms.groq import Groq
#     from modules.vector_store import MedicalVectorStore
#     from modules.router_engine import build_router_query_engine
# except ImportError as e:
#     raise ImportError(f"Lỗi import: {e}. Hãy đảm bảo chạy script từ thư mục gốc dự án.")

# load_dotenv()

# # --- CẤU HÌNH ---
# CONFIG = {
#     "persist_dir": "vector_data",
#     "model": "llama-3.3-70b-versatile",
#     "embed_model": "BAAI/bge-m3",
#     "device": "cpu",
#     "backend": "disk",
#     "test_file": "test_set-2.csv"
# }

# # --- HÀM LÀM SẠCH (ĐÃ SỬA) ---
# def clean_response(text):
#     """
#     Chỉ loại bỏ hình ảnh, GIỮ NGUYÊN text 'Không tìm thấy dữ liệu...'
#     """
#     if not text: return ""
#     text = str(text)
    
#     # Loại bỏ thẻ ảnh Markdown ![alt](url)
#     text = re.sub(r"!\[.*?\]\(.*?\)", "", text, flags=re.DOTALL)
    
#     # Loại bỏ khoảng trắng thừa
#     return " ".join(text.split())

# def is_failure_response(text):
#     """Hàm phụ trợ để kiểm tra xem câu trả lời có phải là thất bại không"""
#     text_lower = text.lower()
#     return "không tìm thấy dữ liệu" in text_lower or "tôi không biết" in text_lower

# def load_test_dataset(filepath):
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
#     print(f"--- Đang đọc dữ liệu test từ: {filepath} ---")
#     df = pd.read_csv(filepath)
    
#     required_cols = ['Câu hỏi', 'Đáp án']
#     if not all(col in df.columns for col in required_cols):
#         raise ValueError(f"File CSV cần có các cột: {required_cols}")

#     dataset = []
#     for _, row in df.iterrows():
#         dataset.append({
#             "id": row.get('id', _),
#             "difficulty": row.get('Độ khó', 'Unknown'),
#             "question": str(row['Câu hỏi']).strip(),
#             "ground_truth": str(row['Đáp án']).strip()
#         })
#     print(f"-> Đã load thành công {len(dataset)} câu hỏi.")
#     return dataset

# class ProposedSystemWrapper:
#     def __init__(self):
#         print("--- Đang khởi tạo Proposed System... ---")
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key: raise ValueError("Thiếu GROQ_API_KEY")
        
#         self.llm = Groq(api_key=api_key, model=CONFIG["model"], temperature=0.0)
#         self.vs = MedicalVectorStore(
#             persist_dir=CONFIG["persist_dir"],
#             embedding_model=CONFIG["embed_model"],
#             device=CONFIG["device"],
#             backend=CONFIG["backend"]
#         )
#         self.router = build_router_query_engine(
#             vector_store=self.vs, 
#             llm=self.llm, 
#             herbs_top_k=3, 
#             diseases_top_k=3, 
#             emergency_top_k=2,
#             verbose=False
#         )

#     def query(self, question):
#         response = self.router.query(question)
#         return getattr(response, "response", None) or str(response)

# def run_evaluation():
#     try:
#         test_dataset = load_test_dataset(CONFIG["test_file"])
#         system = ProposedSystemWrapper()
#     except Exception as e:
#         print(f"Lỗi khởi tạo: {e}")
#         return

#     print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ ---")
    
#     results = []
    
#     # Danh sách tạm để gom những câu trả lời "có nội dung" để tính BERTScore một lần cho nhanh
#     valid_indices = []
#     valid_cands = []
#     valid_refs = []

#     for i, item in enumerate(test_dataset):
#         q = item["question"]
#         ref = item["ground_truth"]
#         diff = item["difficulty"]
        
#         print(f"[{i+1}/{len(test_dataset)}] Hỏi: {q}")
        
#         try:
#             raw_ans = system.query(q)
#             clean_ans = clean_response(raw_ans)
#         except Exception as e:
#             raw_ans = f"Error: {e}"
#             clean_ans = "Error"

#         # Logic xử lý điểm số thông minh:
#         # Mặc định gán điểm = 0.0 trước
#         current_f1 = 0.0
        
#         # Nếu câu trả lời KHÔNG PHẢI là "Không tìm thấy", ta mới đưa vào danh sách tính điểm
#         if not is_failure_response(clean_ans) and clean_ans.strip() != "":
#             valid_indices.append(i)
#             valid_cands.append(clean_ans)
#             valid_refs.append(ref)
#             # (Điểm sẽ được cập nhật lại sau khi chạy xong BERTScore)
#         else:
#             # Nếu là "Không tìm thấy", giữ nguyên điểm 0.0
#             pass

#         results.append({
#             "ID": item["id"],
#             "Độ khó": diff,
#             "Question": q,
#             "Ground Truth": ref,
#             "Raw Answer": raw_ans,
#             "Cleaned Answer": clean_ans, # Bây giờ cột này sẽ hiện chữ "Không tìm thấy..."
#             "BERT_F1": current_f1        # Mặc định 0.0, sẽ update nếu valid
#         })

#     # --- TÍNH ĐIỂM BERTSCORE CHO CÁC CÂU HỢP LỆ ---
#     if valid_cands:
#         print(f"\n--- Đang tính BERTScore cho {len(valid_cands)} câu trả lời hợp lệ... ---")
#         try:
#             P, R, F1 = score(valid_cands, valid_refs, lang='vi', verbose=True)
            
#             # Cập nhật điểm ngược lại vào list results
#             for idx, f1_val in enumerate(F1):
#                 original_index = valid_indices[idx]
#                 results[original_index]["BERT_F1"] = round(f1_val.item(), 4)
                
#         except Exception as e:
#             print(f"Lỗi tính điểm: {e}")
#     else:
#         print("\n--- Không có câu trả lời nào hợp lệ để tính điểm (Toàn bộ là Không tìm thấy) ---")

#     # --- TỔNG HỢP VÀ XUẤT BÁO CÁO ---
#     df_res = pd.DataFrame(results)
    
#     print(f"\n===========================================")
#     print(f" KẾT QUẢ ĐÁNH GIÁ CHUNG")
#     print(f" F1 Score Trung Bình: {df_res['BERT_F1'].mean():.4f}")
#     print(f"===========================================")
#     print("\n--- Điểm số theo độ khó ---")
#     print(df_res.groupby("Độ khó")["BERT_F1"].mean())

#     output_file = "evaluation_report_final.csv"
#     df_res.to_csv(output_file, index=False, encoding="utf-8-sig")
#     print(f"\nĐã lưu báo cáo vào: {output_file}")

# if __name__ == "__main__":
#     run_evaluation()

import os
import pandas as pd
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from bert_score import score

# --- IMPORT MODULES ---
try:
    from llama_index.llms.groq import Groq
    from modules.vector_store import MedicalVectorStore
    from modules.router_engine import build_router_query_engine
except ImportError as e:
    raise ImportError(f"Lỗi import: {e}. Hãy đảm bảo chạy script từ thư mục gốc dự án.")

load_dotenv()

# --- CẤU HÌNH (QUAN TRỌNG: PHẢI KHỚP VỚI LỆNH INGEST) ---
CONFIG = {
    "persist_dir": "vector_data",       # Nơi chứa Vector DB
    "model": "llama-3.3-70b-versatile", # Model sinh câu trả lời
    "embed_model": "BAAI/bge-m3",       # <--- Phải khớp với model lúc Ingest
    "device": "cpu",
    "backend": "disk",
    # Use an absolute path relative to repository root (robust when running from project root)
    "test_file": str(Path(__file__).resolve().parent.parent / "baseline_rag" / "test.csv")
}

# --- HÀM LÀM SẠCH ---
def clean_response(text):
    if not text: return ""
    text = str(text)
    # Loại bỏ thẻ ảnh Markdown
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text, flags=re.DOTALL)
    return " ".join(text.split())

def is_failure_response(text):
    text_lower = text.lower()
    return "không tìm thấy dữ liệu" in text_lower or "tôi không biết" in text_lower

def load_test_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    print(f"--- Đang đọc dữ liệu test từ: {filepath} ---")
    df = pd.read_csv(filepath)
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "id": row.get('id', _),
            "difficulty": row.get('Độ khó', 'Unknown'),
            "question": str(row['Câu hỏi']).strip(),
            "ground_truth": str(row['Đáp án']).strip()
        })
    print(f"-> Đã load thành công {len(dataset)} câu hỏi.")
    return dataset

class ProposedSystemWrapper:
    def __init__(self):
        print("--- Đang khởi tạo Proposed System... ---")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("Thiếu GROQ_API_KEY")
        
        # Setup Model
        self.llm = Groq(api_key=api_key, model=CONFIG["model"], temperature=0.0)
        
        # Setup Vector Store (Quan trọng: Load đúng model BAAI)
        print(f"Loading Vector Store với embed model: {CONFIG['embed_model']}...")
        self.vs = MedicalVectorStore(
            persist_dir=CONFIG["persist_dir"],
            embedding_model=CONFIG["embed_model"],
            device=CONFIG["device"],
            backend=CONFIG["backend"]
        )
        
        self.router = build_router_query_engine(
            vector_store=self.vs, 
            llm=self.llm, 
            herbs_top_k=3, 
            diseases_top_k=3, 
            emergency_top_k=2,
            verbose=False
        )

    def query(self, question):
        response = self.router.query(question)
        return getattr(response, "response", None) or str(response)

def run_evaluation():
    try:
        test_dataset = load_test_dataset(CONFIG["test_file"])
        system = ProposedSystemWrapper()
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        return

    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ ---")
    
    results = []
    valid_indices = []
    valid_cands = []
    valid_refs = []

    for i, item in enumerate(test_dataset):
        q = item["question"]
        ref = item["ground_truth"]
        diff = item["difficulty"]
        
        print(f"[{i+1}/{len(test_dataset)}] Hỏi: {q}")
        
        try:
            raw_ans = system.query(q)
            clean_ans = clean_response(raw_ans)
        except Exception as e:
            raw_ans = f"Error: {e}"
            clean_ans = "Error"

        current_f1 = 0.0
        
        # Logic tính điểm: Chỉ tính nếu KHÔNG PHẢI là câu "Không tìm thấy"
        if not is_failure_response(clean_ans) and clean_ans.strip() != "":
            valid_indices.append(i)
            valid_cands.append(clean_ans)
            valid_refs.append(ref)
        
        results.append({
            "ID": item["id"],
            "Độ khó": diff,
            "Question": q,
            "Ground Truth": ref,
            "Raw Answer": raw_ans,
            "Cleaned Answer": clean_ans,
            "BERT_F1": current_f1
        })

    if valid_cands:
        print(f"\n--- Đang tính BERTScore cho {len(valid_cands)} câu trả lời hợp lệ... ---")
        try:
            P, R, F1 = score(valid_cands, valid_refs, lang='vi', verbose=True)
            for idx, f1_val in enumerate(F1):
                original_index = valid_indices[idx]
                results[original_index]["BERT_F1"] = round(f1_val.item(), 4)
        except Exception as e:
            print(f"Lỗi tính điểm: {e}")
    else:
        print("\n!!! CẢNH BÁO: Không có câu trả lời nào hợp lệ. Vui lòng kiểm tra lại bước Ingest !!!")

    # Xuất báo cáo
    df_res = pd.DataFrame(results)
    
    print(f"\n===========================================")
    print(f" KẾT QUẢ ĐÁNH GIÁ CHUNG")
    print(f" F1 Score Trung Bình: {df_res['BERT_F1'].mean():.4f}")
    print(f"===========================================")
    print("\n--- Điểm số theo độ khó ---")
    print(df_res.groupby("Độ khó")["BERT_F1"].mean())

    output_file = "evaluation_report_final.csv"
    df_res.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu báo cáo vào: {output_file}")

if __name__ == "__main__":
    run_evaluation()