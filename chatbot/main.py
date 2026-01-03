import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load env vars from chatbot/.env (for LLM_API_BASE, LLM_API_KEY, etc.)
load_dotenv(Path(__file__).resolve().parent / ".env")

from modules.extractor import MedicalDataExtractor
from modules.ingest_pipeline import (
    iter_markdown_files,
    iter_chunks_from_file,
    extract_chunks_to_jsonl,
    ingest_jsonl_to_vector_store,
)
from modules.vector_store import MedicalVectorStore
from schemas import medical_schemas

def _get_schema_by_name(name: str):
    if not hasattr(medical_schemas, name):
        raise ValueError(f"Unknown schema: {name}")
    return getattr(medical_schemas, name)

def build_arg_parser() -> argparse.ArgumentParser:
    # Default paths relative to this script (chatbot/main.py)
    # In this repo, the dataset lives under chatbot/data/.
    _root = Path(__file__).resolve().parent
    _default_input = _root / "data" / "raw"
    _default_jsonl = _root / "data" / "processed" / "extracted.jsonl"
    _default_images = _root / "data" / "processed" / "images"

    p = argparse.ArgumentParser(description="Traditional medicine chatbot pipeline")
    sub = p.add_subparsers(dest="cmd")

    ingest = sub.add_parser("ingest", help="Extract (optional) and ingest into persistent Chroma")
    ingest.add_argument("--input", default=str(_default_input), help="Markdown file or folder")
    ingest.add_argument("--schema", default="MedicinalPlant", help="Pydantic schema name")
    ingest.add_argument("--chunk-by", default="book", choices=["book", "section", "paragraph"], help="Chunking mode")
    ingest.add_argument(
        "--index-type",
        default="herbs",
        choices=[
            "herbs",
            "diseases",
            "emergency",
            "herbs_plants",
            "herbs_vegetables",
            "remedies",
            "endocrine_syndromes",
            "endocrine_plants",
        ],
        help="Vector index bucket",
    )
    ingest.add_argument("--persist-dir", default="./chroma_data", help="Persistence directory for the selected backend")
    ingest.add_argument("--jsonl-out", default=str(_default_jsonl), help="Extraction cache JSONL")
    ingest.add_argument("--embed-model", default="BAAI/bge-m3", help="HF embedding model")
    ingest.add_argument("--embed-batch", type=int, default=8, help="Embedding batch size (lower uses less RAM)")
    ingest.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Embedding device. Use cpu to avoid GPU VRAM.")
    ingest.add_argument("--ingest-batch", type=int, default=16, help="Vector-store insert batch size")
    ingest.add_argument(
        "--backend",
        "--store",
        dest="backend",
        default="disk",
        choices=["disk", "chroma"],
        help="Vector store backend (alias: --store): disk (portable) or chroma (Colab/Linux)",
    )
    ingest.add_argument("--shard-size", type=int, default=2048, help="Disk-backend shard rows (only for backend=disk)")
    ingest.add_argument("--chroma-prefix", default="traditional_medicine", help="Chroma collection prefix (only for backend=chroma)")
    ingest.add_argument("--extract", action="store_true", help="Run LLM extraction step (runs locally where you execute this command)")
    ingest.add_argument(
        "--model",
        default=os.getenv("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct",
        help=(
            "LLM model for extraction (HuggingFace model id/path). Use HF_MODEL to override."
        ),
    )
    ingest.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Max tokens for extraction output (lower reduces token usage)",
    )
    ingest.add_argument(
        "--max-retry-after-seconds",
        type=float,
        default=120.0,
        help="Max allowed backoff seconds before failing (used only by some backends)",
    )
    ingest.add_argument(
        "--resume",
        action="store_true",
        help="When set, skip chunks already present in the output JSONL instead of starting fresh",
    )
    ingest.add_argument("--rpm", type=float, default=2.0, help="Extraction requests per minute")
    ingest.add_argument(
        "--extract-only",
        action="store_true",
        help="Only run extraction to JSONL (skip embedding + vector ingestion).",
    )

    # Optional enrichment: store images (rendering + provenance)
    ingest.add_argument("--enrich-images", action="store_true", help="Attach image metadata (paths + hashes) to extracted records")
    ingest.add_argument("--image-store-dir", default=str(_default_images), help="Where to store optimized images (e.g., webp)")
    ingest.add_argument("--image-format", default="webp", choices=["webp", "png", "jpg"], help="Output format for stored images")
    ingest.add_argument("--image-quality", type=int, default=80, help="Quality for webp/jpg output (1-100)")

    query = sub.add_parser("query", help="Ask a question; router selects the right index")
    query.add_argument("--persist-dir", default="vector_data", help="Vector index directory (disk backend default)")
    query.add_argument(
        "--backend",
        "--store",
        dest="backend",
        default="disk",
        choices=["disk", "chroma"],
        help="Vector store backend (alias: --store)",
    )
    query.add_argument("--chroma-prefix", default="traditional_medicine", help="Chroma collection prefix (backend=chroma)")
    query.add_argument("--embed-model", default="BAAI/bge-m3", help="HF embedding model (must match ingest)")
    query.add_argument("--embed-batch", type=int, default=8, help="Embedding batch size")
    query.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Embedding device")
    query.add_argument("--shard-size", type=int, default=2048, help="Disk-backend shard rows (backend=disk)")
    query.add_argument("--question", required=True, help="User question")
    query.add_argument(
        "--model",
        default=os.getenv("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct",
        help=(
            "LLM model for answering when running locally (HF_MODEL/LLM_BACKEND=hf). "
            "If using remote, set LLM_API_BASE and this flag is ignored."
        ),
    )
    query.add_argument("--herbs-top-k", type=int, default=3, help="Top-k retrieval for herbs")
    query.add_argument("--diseases-top-k", type=int, default=3, help="Top-k retrieval for diseases")
    query.add_argument("--emergency-top-k", type=int, default=2, help="Top-k retrieval for emergency")
    query.add_argument("--verbose", action="store_true", help="Print router decision process")
    query.add_argument(
        "--no-llm",
        action="store_true",
        help="Retrieval-only: do not call any LLM, just return the most relevant context chunks.",
    )

    return p

def cmd_ingest(args: argparse.Namespace) -> None:
    schema = _get_schema_by_name(args.schema)

    if args.extract_only and not args.extract:
        raise ValueError("--extract-only requires --extract")

    # Optional extraction step (guarded so you don't accidentally trigger paid usage)
    if args.extract:
        extractor = MedicalDataExtractor(
            model=args.model,
            max_tokens=args.max_output_tokens,
            rate_limit_max_retry_after_seconds=args.max_retry_after_seconds,
        )

        for fp in iter_markdown_files(args.input):
            chunks = iter_chunks_from_file(fp, schema, chunk_by=args.chunk_by)
            n_ok = extract_chunks_to_jsonl(
                extractor=extractor,
                chunks=chunks,
                schema=schema,
                out_jsonl_path=args.jsonl_out,
                requests_per_minute=args.rpm,
                enrich_images=args.enrich_images,
                image_store_dir=args.image_store_dir,
                image_prefer_format=args.image_format,
                image_quality=args.image_quality,
                resume=args.resume,
            )
            print(f"Extracted {n_ok} records from {fp}")

        if args.extract_only:
            print(f"Extraction complete. JSONL written to {Path(args.jsonl_out).resolve()}")
            return

    vs = MedicalVectorStore(
        persist_dir=args.persist_dir,
        embedding_model=args.embed_model,
        embed_batch_size=args.embed_batch,
        device=args.device,
        backend=args.backend,
        shard_size=args.shard_size,
        chroma_collection_prefix=args.chroma_prefix,
    )

    if not Path(args.jsonl_out).exists():
        raise FileNotFoundError(
            f"JSONL not found: {args.jsonl_out}. "
            "Run with --extract first to create it, or point --jsonl-out to an existing file."
        )

    # Ingest whatever is already in JSONL (or newly written)
    added = ingest_jsonl_to_vector_store(
        vector_store=vs,
        jsonl_path=args.jsonl_out,
        schema=schema,
        index_type=args.index_type,
        batch_size=args.ingest_batch,
    )
    print(f"Ingested {added} items into {args.index_type} (persisted at {Path(args.persist_dir).resolve()})")


def cmd_query(args: argparse.Namespace) -> None:
    from modules.router_engine import build_router_query_engine

    llm = None
    if not args.no_llm:
        api_base = (os.getenv("LLM_API_BASE") or "").strip()
        if api_base:
            from modules.remote_llm import RemoteLLM

            llm = RemoteLLM.from_env()
        else:
            backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
            hf_model_id = (os.getenv("HF_MODEL") or "").strip()

            if backend == "hf" or hf_model_id:
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                except Exception as e:
                    raise RuntimeError("Self-hosted (HF) backend requires transformers to be installed") from e

                try:
                    from llama_index.llms.huggingface import HuggingFaceLLM
                except Exception as e:
                    raise RuntimeError("Self-hosted (HF) backend requires llama-index-llms-huggingface") from e

                model_id = hf_model_id or args.model
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                import torch

                force_cpu = (os.getenv("FORCE_CPU") or "").strip().lower() in {"1", "true", "yes", "y"}
                device_map = (os.getenv("HF_DEVICE_MAP") or "").strip() or None
                if device_map is None:
                    device_map = "cpu" if (force_cpu or os.name == "nt") else "auto"
                torch_dtype = "auto" if device_map != "cpu" else torch.float32

                hf_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch_dtype)
                llm = HuggingFaceLLM(model=hf_model, tokenizer=tokenizer, temperature=0.0, max_new_tokens=1024)
            else:
                raise ValueError(
                    "query requires an LLM unless --no-llm is set. "
                    "Set LLM_API_BASE (remote Colab/ngrok) or set HF_MODEL/LLM_BACKEND=hf for local self-hosting."
                )

    vs = MedicalVectorStore(
        persist_dir=args.persist_dir,
        embedding_model=args.embed_model,
        embed_batch_size=args.embed_batch,
        device=args.device,
        backend=args.backend,
        shard_size=args.shard_size,
        chroma_collection_prefix=args.chroma_prefix,
    )

    router = build_router_query_engine(
        vector_store=vs,
        llm=llm,
        herbs_top_k=args.herbs_top_k,
        diseases_top_k=args.diseases_top_k,
        emergency_top_k=args.emergency_top_k,
        verbose=args.verbose,
    )

    resp = router.query(args.question)
    # LlamaIndex Response can store text on different attributes across versions.
    text = getattr(resp, "response", None) or getattr(resp, "text", None) or str(resp)
    print(text)


def legacy_demo() -> None:
    from schemas.medical_schemas import MedicinalPlant

    extractor = MedicalDataExtractor()
    plants = extractor.extract_from_file(
        "data\\raw\\cay-canh--cay-thuoc-trong-nha-truong.md",
        MedicinalPlant,
    )
    print(f"Extracted {len(plants)} plants")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.cmd:
        legacy_demo()
    elif args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "query":
        cmd_query(args)