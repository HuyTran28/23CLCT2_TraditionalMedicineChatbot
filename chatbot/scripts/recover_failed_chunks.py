import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure project root is on sys.path so `modules` imports work when running the script.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.book_splitters import split_by_book
from modules.extractor import MedicalDataExtractor, RateLimitPauseRequired
from modules.ingest_pipeline import _sanitize_chunk_text_for_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRef:
    source_path: str
    chunk_index: int

    @property
    def rid(self) -> str:
        return f"{self.source_path}:#{self.chunk_index}"


def find_schema(name: str):
    from schemas import medical_schemas as schemas

    if not hasattr(schemas, name):
        raise ValueError(f"Unknown schema: {name}")
    return getattr(schemas, name)


def _loads_first_json_obj(line: str) -> Optional[Dict[str, Any]]:
    s = (line or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_chunk_index_from_id(rid: str) -> Optional[int]:
    # Expecting: <path>:#<int>
    if not rid:
        return None
    if ":#" not in rid:
        return None
    try:
        return int(rid.rsplit(":#", 1)[1])
    except Exception:
        return None


def iter_failed_chunk_refs(jsonl_path: Path) -> Iterable[ChunkRef]:
    """Yield ChunkRef for records that contain an 'error' key."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = _loads_first_json_obj(line)
            if not rec or "error" not in rec:
                continue
            meta = rec.get("meta") or {}
            source_path = meta.get("source_path") or meta.get("source")
            rid = meta.get("id")
            chunk_index = meta.get("chunk_index")

            if chunk_index is None and rid:
                chunk_index = _parse_chunk_index_from_id(str(rid))
            if source_path is None and rid and ":#" in str(rid):
                source_path = str(rid).rsplit(":#", 1)[0]

            if not source_path or chunk_index is None:
                continue

            try:
                chunk_index = int(chunk_index)
            except Exception:
                continue

            yield ChunkRef(source_path=str(source_path), chunk_index=chunk_index)


def _split_kind_for_schema(schema_name: str) -> Optional[str]:
    if schema_name == "RemedyRecipe":
        return "recipes"
    if schema_name == "MedicinalPlant":
        return "plants"
    if schema_name == "EndocrineSyndrome":
        return "syndromes"
    return None


def load_chunk_text(source_md: Path, schema_name: str, chunk_index: int) -> str:
    content = source_md.read_text(encoding="utf-8")
    split_kind = _split_kind_for_schema(schema_name)
    chunks = split_by_book(str(source_md), content, split_kind=split_kind)
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(
            f"chunk_index out of range: {chunk_index} (chunks={len(chunks)}) for {source_md}"
        )
    return chunks[chunk_index]


def read_existing_ids(base_jsonl: Optional[Path]) -> set[str]:
    if base_jsonl is None or not base_jsonl.exists():
        return set()
    ids: set[str] = set()
    with base_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = _loads_first_json_obj(line)
            if not rec or "data" not in rec:
                continue
            meta = rec.get("meta") or {}
            rid = meta.get("id")
            if rid:
                ids.add(str(rid))
                continue
            sp = meta.get("source_path") or meta.get("source")
            ci = meta.get("chunk_index")
            if sp is not None and ci is not None:
                ids.add(f"{sp}:#{ci}")
    return ids


def merge_jsonl(base_jsonl: Path, recovered_jsonl: Path, out_merged: Path) -> None:
    """Merge data records by id: recovered overrides base."""
    by_id: Dict[str, Dict[str, Any]] = {}

    def _ingest(p: Path):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = _loads_first_json_obj(line)
                if not rec or "data" not in rec:
                    continue
                meta = rec.get("meta") or {}
                rid = meta.get("id") or f"{meta.get('source_path', meta.get('source',''))}:#{meta.get('chunk_index','')}"
                if ":#" not in str(rid):
                    continue
                by_id[str(rid)] = rec

    _ingest(base_jsonl)
    _ingest(recovered_jsonl)

    out_merged.parent.mkdir(parents=True, exist_ok=True)
    with out_merged.open("w", encoding="utf-8") as wf:
        for rid in sorted(by_id.keys()):
            wf.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-extract failed chunks from an error JSONL")
    p.add_argument(
        "--errors-jsonl",
        required=True,
        help="Path to JSONL containing error records (e.g., emergency_cc.errors.jsonl or emergency_cc.jsonl)",
    )
    p.add_argument("--schema", required=True, help="Pydantic schema name (e.g., EmergencyProtocol)")
    p.add_argument(
        "--out",
        required=True,
        help="Output JSONL for recovered data records",
    )
    p.add_argument(
        "--base-jsonl",
        default=None,
        help="Optional existing JSONL with good data; used to skip already-present IDs",
    )
    p.add_argument(
        "--out-merged",
        default=None,
        help="Optional merged JSONL path (base + recovered, recovered wins)",
    )

    p.add_argument("--use-llm", action="store_true", help="Use Groq LLM extraction")
    p.add_argument("--groq-key", default=None, help="Optional Groq key to pass (overrides env)")
    p.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for completion")
    p.add_argument("--rpm", type=float, default=2.0, help="Requests per minute throttle")

    p.add_argument("--limit", type=int, default=0, help="Limit number of failed chunks to process (0 = all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle failed chunks before processing")
    p.add_argument("--seed", type=int, default=0, help="Seed for shuffle")

    return p


def main() -> int:
    args = build_parser().parse_args()

    schema_cls = find_schema(args.schema)
    errors_jsonl = Path(args.errors_jsonl)
    out_path = Path(args.out)
    base_jsonl = Path(args.base_jsonl) if args.base_jsonl else None

    if args.use_llm:
        if not args.groq_key and not (Path(".env").exists() or "GROQ_API_KEY" in os.environ):
            raise RuntimeError("GROQ API key not found. Set GROQ_API_KEY or pass --groq-key")
        extractor = (
            MedicalDataExtractor(api_key=args.groq_key, model=args.model, max_tokens=args.max_tokens)
            if args.groq_key
            else MedicalDataExtractor(model=args.model, max_tokens=args.max_tokens)
        )
    else:
        raise RuntimeError("This script is intended to re-extract using LLM. Pass --use-llm.")

    failed = list(iter_failed_chunk_refs(errors_jsonl))
    # Deduplicate
    uniq: Dict[str, ChunkRef] = {c.rid: c for c in failed}
    failed = list(uniq.values())

    if args.shuffle:
        if args.seed:
            random.seed(args.seed)
        random.shuffle(failed)

    if args.limit and args.limit > 0:
        failed = failed[: int(args.limit)]

    existing_ids = read_existing_ids(base_jsonl)
    to_process = [c for c in failed if c.rid not in existing_ids]

    logger.info(
        f"Found failed={len(failed)} unique chunks; already-have={len(existing_ids)}; to-process={len(to_process)}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0
    with out_path.open("w", encoding="utf-8") as wf:
        for i, ref in enumerate(to_process, 1):
            try:
                source_md = Path(ref.source_path)
                if not source_md.exists():
                    # try relative to project root
                    alt = project_root / ref.source_path
                    if alt.exists():
                        source_md = alt

                chunk_text = load_chunk_text(source_md, args.schema, ref.chunk_index)
                llm_text = _sanitize_chunk_text_for_llm(chunk_text)
                obj = extractor.extract_single(
                    llm_text,
                    schema_cls,
                    context_hint=f"source={source_md.name} chunk={ref.chunk_index}",
                )

                rec = {
                    "data": obj.model_dump(),
                    "meta": {
                        "source_path": str(source_md),
                        "id": ref.rid,
                        "chunk_index": ref.chunk_index,
                        "schema": args.schema,
                        "recovered": True,
                    },
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1

                if args.rpm and args.rpm > 0:
                    import time

                    time.sleep(60.0 / float(args.rpm))

                if i % 5 == 0:
                    logger.info(f"Progress: {i}/{len(to_process)} processed, ok={n_ok}, err={n_err}")

            except RateLimitPauseRequired:
                raise
            except Exception as e:
                n_err += 1
                logger.error(f"Failed to recover {ref.rid}: {e}")

    logger.info(f"Recovered ok={n_ok}, err={n_err}. Output: {out_path}")

    if args.out_merged:
        if base_jsonl is None or not base_jsonl.exists():
            raise ValueError("--out-merged requires --base-jsonl that exists")
        merged_path = Path(args.out_merged)
        merge_jsonl(base_jsonl, out_path, merged_path)
        logger.info(f"Merged JSONL written to: {merged_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
