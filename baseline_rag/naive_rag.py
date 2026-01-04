import os
import shutil
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Load .env from the project root or current directory
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    load_dotenv = None
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from remote_llm_adapters import LlamaIndexRemoteLLM

class NaiveMedicalRAG:
    def __init__(self, file_paths, persist_dir="./baseline_storage"):
        """
        Baseline RAG (CPU Optimized):
        - Embedding: BAAI/bge-m3 (Run on CPU)
        - Chunking: Hierarchical/Structural (MarkdownNodeParser)
        """
        self.persist_dir = persist_dir

        # Prefer self-hosted Colab/ngrok LLM if configured.
        # Fallback to Groq only when LLM_API_BASE is not set.
        if (os.getenv("LLM_API_BASE") or "").strip():
            self.llm = LlamaIndexRemoteLLM.from_env()
        else:
            if not os.environ.get("GROQ_API_KEY"):
                raise ValueError(
                    "Thiếu cấu hình LLM. Set LLM_API_BASE (self-hosted) hoặc GROQ_API_KEY (Groq fallback)."
                )
            self.llm = Groq(model="llama-3.1-8b-instant")

        # --- CẤU HÌNH CHO CPU ---
        print("--- Đang load model BAAI/bge-m3 (CPU Mode)... ---")
        # device="cpu": Ép chạy trên CPU
        # embed_batch_size=10: Giảm số lượng câu xử lý cùng lúc để đỡ ngốn RAM/CPU
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="cpu", 
            embed_batch_size=10 
        )
        # Setup Global Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.context_window = 8192 # Llama 3 hỗ trợ context dài

        # Load hoặc Create Index
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
            print(f"--- [Baseline] Tạo mới Index trên CPU (Sẽ mất vài phút) ---")
            self.create_index(file_paths)

        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
        print("--- Baseline RAG đã sẵn sàng ---")

    def create_index(self, file_paths):
        if not file_paths:
            raise ValueError("Danh sách file input đang trống!")
            
        print(f"Đang đọc {len(file_paths)} file tài liệu...")
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        
        # Sử dụng Markdown Parsing theo yêu cầu nhóm trưởng
        print("Đang parse document theo cấu trúc Markdown...")
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        
        print(f"Đã tạo {len(nodes)} nodes. Đang tạo embedding (CPU)...")
        print("Vui lòng đợi, quá trình này tốn nhiều thời gian hơn trên GPU...")
        
        # show_progress=True cho biết nó có đang chạy hay bị treo
        self.index = VectorStoreIndex(nodes, show_progress=True)
        
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Đã lưu index xuống: {self.persist_dir}")

    def query(self, question: str, return_full=False):
        """
        return_full=True: Trả về object Response (chứa text + source_nodes) để dùng cho RAGAS
        return_full=False: Chỉ trả về text (string) để in ra màn hình
        """
        response = self.query_engine.query(question)
        if return_full:
            return response
        return str(response)

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    test_files = [
        str(DATA_DIR / "cay-rau-lam-thuoc" / "cay-rau-lam-thuoc.md")
    ]
    try:
        # Nhớ xóa folder baseline_storage cũ trước khi chạy lần đầu code này
        bot = NaiveMedicalRAG(test_files)
        print("Câu trả lời:", bot.query("Ớt được trồng ở đâu?"))
    except Exception as e:
        print(f"Lỗi: {e}")