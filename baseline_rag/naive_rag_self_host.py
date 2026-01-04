import os
import shutil
import requests
from typing import Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- IMPORT MODULE QUAN TRỌNG ĐỂ TẠO CUSTOM LLM ---
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

# ==============================================================================
# 1. CLASS KẾT NỐI VỚI GOOGLE COLAB (RemoteColabLLM)
# ==============================================================================
class RemoteColabLLM(CustomLLM):
    api_url: str = ""
    context_window: int = 4096
    num_output: int = 512
    model_name: str = "colab-gpu-model"

    def __init__(self, api_url: str, **kwargs):
        super().__init__(api_url=api_url, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Cấu trúc payload khớp với file colab_llm_server.ipynb của bạn
        endpoint = f"{self.api_url}/v1/complete"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": self.num_output,
            "temperature": kwargs.get("temperature", 0.1),
        }
        
        try:
            print(f"📡 Sending request to Colab: {endpoint}...")
            response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            # Xử lý kết quả trả về từ Notebook
            # Dựa vào snippet notebook, server trả về text hoặc JSON
            try:
                data = response.json()
                # Ưu tiên lấy field 'text' hoặc 'content', nếu không có thì lấy cả cục
                text = data.get("text", data.get("content", str(data)))
            except:
                text = response.text
                
            return CompletionResponse(text=text)
            
        except requests.exceptions.RequestException as e:
            return CompletionResponse(text=f"Error connecting to Colab API: {e}")

    def stream_complete(self, prompt: str, **kwargs: Any):
        # Notebook mẫu của bạn dùng /v1/complete (không stream), nên ta giả lập stream
        yield self.complete(prompt, **kwargs)

# ==============================================================================
# 2. CLASS BASELINE RAG (ĐÃ SỬA ĐỂ DÙNG REMOTE LLM)
# ==============================================================================
class NaiveMedicalRAG:
    def __init__(self, file_paths, persist_dir="./baseline_storage", colab_url=None):
        """
        Baseline RAG Hybrid:
        - Embedding: Chạy Local CPU (BAAI/bge-m3) - Vì nó nhẹ.
        - LLM: Chạy Remote GPU (Google Colab) - Vì nó nặng.
        """
        self.persist_dir = persist_dir
        
        # 1. Cấu hình LLM (Kết nối Colab)
        # Ưu tiên URL truyền vào, nếu không thì lấy từ biến môi trường
        api_base = colab_url or os.getenv("LLM_API_BASE")
        
        if not api_base:
            print("⚠️ CẢNH BÁO: Chưa có URL Colab. Hệ thống sẽ không thể trả lời câu hỏi.")
            print("👉 Hãy set biến môi trường LLM_API_BASE hoặc truyền colab_url vào.")
            # Fallback dummy để không crash lúc init, nhưng sẽ lỗi lúc query
            self.llm = None
        else:
            # Xóa dấu / ở cuối nếu có
            api_base = api_base.rstrip("/")
            print(f"--- Đang kết nối với Colab LLM tại: {api_base} ---")
            self.llm = RemoteColabLLM(api_url=api_base)
            Settings.llm = self.llm

        # 2. Cấu hình Embedding (Chạy Local CPU)
        print("--- Đang load model Embedding BAAI/bge-m3 (CPU Mode)... ---")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="cpu", 
            embed_batch_size=4 # Batch nhỏ để nhẹ máy
        )
        Settings.embed_model = self.embed_model
        Settings.context_window = 4096

        # 3. Load hoặc Tạo Index
        if os.path.exists(self.persist_dir):
            print(f"--- [Baseline] Loading Index từ đĩa: {self.persist_dir} ---")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context)
            except Exception as e:
                print(f"Lỗi load index: {e}. Đang tạo lại...")
                if os.path.exists(self.persist_dir):
                    shutil.rmtree(self.persist_dir)
                self.create_index(file_paths)
        else:
            print(f"--- [Baseline] Tạo mới Index... ---")
            self.create_index(file_paths)

        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
        print("--- Baseline RAG đã sẵn sàng ---")

    def create_index(self, file_paths):
        if not file_paths:
            raise ValueError("Danh sách file input đang trống!")
        
        print(f"Đang đọc {len(file_paths)} file tài liệu...")
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        
        print("Đang parse document...")
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        
        print(f"Đã tạo {len(nodes)} nodes. Đang tạo embedding...")
        self.index = VectorStoreIndex(nodes, show_progress=True)
        
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Đã lưu index xuống: {self.persist_dir}")

    def query(self, question: str, return_full=False):
        if not self.llm:
            return "Lỗi: Chưa kết nối được với Colab LLM Server."
            
        response = self.query_engine.query(question)
        if return_full:
            return response
        return str(response)

# ==============================================================================
# 3. CHẠY THỬ
# ==============================================================================
if __name__ == "__main__":
    # --- CẤU HÌNH URL COLAB CỦA BẠN TẠI ĐÂY ---
    # Ví dụ: "https://a1b2-34-123-45-67.ngrok-free.app"
    COLAB_URL = "DÁN_URL_NGROK_CỦA_BẠN_VÀO_ĐÂY" 
    
    # Nếu chạy dòng lệnh thì set: set LLM_API_BASE=https://...
    if COLAB_URL == "DÁN_URL_NGROK_CỦA_BẠN_VÀO_ĐÂY":
        COLAB_URL = os.getenv("LLM_API_BASE")

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    # Sửa đường dẫn file test cho phù hợp
    test_files = [
        str(DATA_DIR / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md")
    ]
    
    # Kiểm tra file tồn tại không để tránh lỗi
    if not os.path.exists(test_files[0]):
        print(f"⚠️ Không tìm thấy file mẫu: {test_files[0]}")
        print("Vui lòng sửa list `test_files` trong code.")
    else:
        try:
            bot = NaiveMedicalRAG(test_files, colab_url=COLAB_URL)
            
            q = "Cây ớt có công dụng gì?"
            print(f"\n❓ Câu hỏi: {q}")
            res = bot.query(q)
            print(f"💡 Câu trả lời từ Colab:\n{res}")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")