
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

#API key
os.environ["GROQ_API_KEY"] = ""

PERSIST_DIR = "./baseline_storage"
DATA_FILES = [
    "data/raw/cay-rau-lam-thuoc/cay-rau-lam-thuoc.md",
]

Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
Settings.llm = Groq(model="llama-3.1-8b-instant")

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

_index_instance = None

def get_index():
    global _index_instance
    if _index_instance is not None:
        return _index_instance

    if os.path.exists(PERSIST_DIR):
        print("--- Loading Baseline Index from Disk ---")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        _index_instance = load_index_from_storage(storage_context)
    else:
        print("--- Creating New Baseline Index (This may take time) ---")
        documents = SimpleDirectoryReader(input_files=DATA_FILES).load_data()
        _index_instance = VectorStoreIndex.from_documents(documents)
        _index_instance.storage_context.persist(persist_dir=PERSIST_DIR)
    
    return _index_instance

def query_baseline(question_text: str) -> str:
    index = get_index()
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question_text)
    return str(response)

if __name__ == "__main__":
    print(query_baseline("Mồng tơi có tác dụng gì?"))