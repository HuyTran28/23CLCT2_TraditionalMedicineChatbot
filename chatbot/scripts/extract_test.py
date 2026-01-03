import argparse
import json
import logging
import random
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so `modules` imports work when running
# the script via `py scripts\extract_test.py` (sys.path[0] is script dir).
import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.book_splitters import split_by_book

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_schema(name: str):
    from schemas import medical_schemas as schemas

    if not hasattr(schemas, name):
        raise ValueError(f"Unknown schema: {name}")
    return getattr(schemas, name)


def load_chunks_from_path(path: Path, schema_name: str, sample: int, chunk_by: str = "book") -> List[dict]:
    files = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.md"))

    chunks = []
    for fp in files:
        content = fp.read_text(encoding="utf-8")
        split_kind = None
        if schema_name == "MedicinalPlant":
            split_kind = "plants"
        elif schema_name == "EndocrinePatternRecord":
            split_kind = "patterns"
        elif schema_name in {"EndocrineSyndrome", "EndocrineDisease"}:
            split_kind = "syndromes"
        chunks_ext = split_by_book(str(fp), content, split_kind=split_kind)
        for i, ch in enumerate(chunks_ext):
            chunks.append({"source": str(fp), "index": i, "text": ch})

    if not chunks:
        raise ValueError("No chunks found in input path")

    if sample >= len(chunks):
        return chunks
    return random.sample(chunks, sample)


def mock_extract(text: str, schema_cls):
    # Minimal heuristic-filled object: for strings use first 200 chars; lists->[]; numbers->0
    data = {}
    for name, field in schema_cls.model_fields.items():
        t = field.annotation
        if t == str or getattr(t, "__name__", None) == "str":
            data[name] = text.strip()[:200]
        elif t == int or getattr(t, "__name__", None) == "int":
            data[name] = 0
        elif getattr(t, "__origin__", None) is list or getattr(t, "__args__", None) == (str,):
            data[name] = []
        else:
            data[name] = None
    return schema_cls(**data)


def run(args):
    schema_cls = find_schema(args.schema)
    chunks = load_chunks_from_path(Path(args.input), args.schema, sample=args.sample)
    logger.info(f"Loaded {len(chunks)} sample chunks")

    results = []

    if args.use_llm:
        from modules.extractor import MedicalDataExtractor

        extractor = MedicalDataExtractor()

        texts = [c['text'] for c in chunks]
        batch_results = extractor.extract_batch(texts, schema_cls, context_hint=args.context or "", requests_per_minute=args.rpm)
        for ch, obj in zip(chunks, batch_results):
            if args.enrich_images:
                from modules.enrich_images import enrich_record_with_images

                obj = enrich_record_with_images(
                    record=obj,
                    chunk_text=ch['text'],
                    source_markdown_path=ch['source'],
                    store_dir=args.image_store_dir,
                    prefer_format=args.image_format,
                    quality=args.image_quality,
                )
            results.append((ch, obj))

    else:
        for ch in chunks:
            obj = mock_extract(ch['text'], schema_cls)
            if args.enrich_images:
                from modules.enrich_images import enrich_record_with_images

                obj = enrich_record_with_images(
                    record=obj,
                    chunk_text=ch['text'],
                    source_markdown_path=ch['source'],
                    store_dir=args.image_store_dir,
                    prefer_format=args.image_format,
                    quality=args.image_quality,
                )
            results.append((ch, obj))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out_path.open('w', encoding='utf-8') as f:
        for ch, obj in results:
            rec = {'data': obj.model_dump(), 'meta': {'source': ch['source'], 'chunk_index': ch['index'], 'schema': args.schema, 'mock': not args.use_llm}}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_ok += 1

    # Validation summary
    total = len(results)
    print(f"Wrote {n_ok}/{total} records to {out_path}")
    # Print 3 sample records
    for i, (ch, obj) in enumerate(results[:3]):
        print('---')
        print(f"Sample {i+1} - source: {ch['source']}#{ch['index']}")
        print(obj.model_dump_json(indent=2, ensure_ascii=False))


def build_parser():
    # Default paths relative to this script (chatbot/scripts/extract_test.py)
    # In this repo, the dataset lives under chatbot/data/.
    _root = Path(__file__).resolve().parent.parent
    _default_out = _root / "data" / "processed" / "test_extracted.jsonl"
    _default_images = _root / "data" / "processed" / "images"

    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Markdown file or folder')
    p.add_argument('--schema', required=True, help='Pydantic schema name (e.g., MedicinalPlant)')
    p.add_argument('--sample', type=int, default=10, help='Number of chunks to sample')
    p.add_argument('--use-llm', action='store_true', help='Use real HF extractor (requires HF_MODEL or local model availability)')
    p.add_argument('--rpm', type=float, default=2.0, help='Rate limit (requests per minute)')
    p.add_argument('--out', default=str(_default_out), help='Output JSONL')
    p.add_argument('--context', default='', help='Context hint for the extractor')

    p.add_argument('--enrich-images', action='store_true', help='Attach image metadata to each record (when supported by schema)')
    p.add_argument('--image-store-dir', default=str(_default_images), help='Where to store optimized images (webp/png/jpg)')
    p.add_argument('--image-format', default='webp', choices=['webp', 'png', 'jpg'], help='Output format for stored images')
    p.add_argument('--image-quality', type=int, default=80, help='Quality for webp/jpg output (1-100)')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    run(args)
