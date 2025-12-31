import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Type

from pydantic import BaseModel

from modules.extractor import MedicalDataExtractor, RateLimitPauseRequired
from modules.vector_store import MedicalVectorStore
from modules.book_splitters import split_by_book

logger = logging.getLogger(__name__)


_NESTED_IMAGE_RE = re.compile(
    r"!\[\]\(\!\[id:\s*(?P<id>[^\]]+?)\]\((?P<file>[^)]+?)\)\)"
)
_GENERIC_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_BREAK_TAG_RE = re.compile(r"</?break\s*/?>", re.IGNORECASE)
_HEADING_NUMBER_RE = re.compile(r"^(?P<prefix>\s*#{1,6}\s*)?(?P<num>\(?\s*\d+\s*\)?)[\.)]\s+", re.UNICODE)


def _sanitize_chunk_text_for_llm(text: str) -> str:
    """Reduce token waste in chunks sent to the LLM.

    Keeps semantic content while removing/compacting common OCR artifacts:
      - HTML-ish break tags: </break>, <break/>
      - leading numbering: '6. ...' (including after markdown heading hashes)
      - bulky markdown image syntax, replacing with short placeholders
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out_lines: list[str] = []
    for ln in lines:
        if not ln:
            out_lines.append(ln)
            continue

        # Drop standalone break tags anywhere in the line.
        ln = _BREAK_TAG_RE.sub(" ", ln)

        # Remove leading numbering like: '6. ...' or '## 6. ...'
        ln = _HEADING_NUMBER_RE.sub(lambda m: (m.group("prefix") or ""), ln)

        # Compact the dataset's nested image syntax into a tiny placeholder.
        # Example: ![](![id: foo_img_007](foo_img_007.png)) -> [[IMAGE:foo_img_007]]
        def _nested_repl(m: re.Match) -> str:
            image_id = (m.group("id") or "").strip()
            if image_id:
                return f"[[IMAGE:{image_id}]]"
            return "[[IMAGE]]"

        ln = _NESTED_IMAGE_RE.sub(_nested_repl, ln)

        # Generic markdown image syntax -> [[IMAGE]] (keep it short)
        ln = _GENERIC_IMAGE_RE.sub("[[IMAGE]]", ln)

        # Normalize whitespace introduced by replacements.
        ln = re.sub(r"\s{2,}", " ", ln).rstrip()
        out_lines.append(ln)

    # Collapse excessive blank lines (keep at most one blank line).
    cleaned: list[str] = []
    prev_blank = False
    for ln in out_lines:
        blank = not ln.strip()
        if blank and prev_blank:
            continue
        cleaned.append(ln)
        prev_blank = blank

    return "\n".join(cleaned).strip()


@dataclass(frozen=True)
class ChunkRecord:
    source_path: str
    chunk_index: int
    chunk_text: str
    schema_name: str


def iter_markdown_files(input_path: str) -> Iterator[Path]:
    p = Path(input_path)
    if p.is_file():
        yield p
        return
    for fp in sorted(p.rglob("*.md")):
        if fp.is_file():
            yield fp


def iter_chunks_from_file(
    filepath: Path,
    schema: Type[BaseModel],
    chunk_by: str = "book",
) -> Iterator[ChunkRecord]:
    content = filepath.read_text(encoding="utf-8")

    if chunk_by == "book":
        split_kind = None
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "RemedyRecipe":
            split_kind = "recipes"
        elif schema_name == "MedicinalPlant":
            split_kind = "plants"
        elif schema_name == "EndocrineSyndrome":
            split_kind = "syndromes"
        chunks = split_by_book(str(filepath), content, split_kind=split_kind)
    elif chunk_by == "section":
        chunks = [c.strip() for c in content.split("\n#") if c.strip()]
    else:
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

    schema_name = getattr(schema, "__name__", "BaseModel")
    for i, ch in enumerate(chunks):
        yield ChunkRecord(
            source_path=str(filepath),
            chunk_index=i,
            chunk_text=ch,
            schema_name=schema_name,
        )


def extract_chunks_to_jsonl(
    *,
    extractor: MedicalDataExtractor,
    chunks: Iterable[ChunkRecord],
    schema: Type[BaseModel],
    out_jsonl_path: str,
    requests_per_minute: Optional[float] = 30.0,
    enrich_images: bool = False,
    image_store_dir: str = "data/processed/images",
    image_prefer_format: str = "webp",
    image_quality: int = 80,
    resume: bool = False,
) -> int:
    """Extract chunk -> schema JSONL without holding everything in RAM.

    Writes one JSON object per line with keys:
      - data: schema dump
            - meta: source_path, id
    """
    out_path = Path(out_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If not resuming, start fresh.
    if not resume and out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            # If unlink fails on Windows (e.g., file open in editor), we'll truncate via open('w').
            pass

    # Resume support: if JSONL already exists, skip chunks that already have data.
    existing_ids: set[str] = set()
    if resume and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "data" not in rec:
                        continue
                    meta = rec.get("meta") or {}
                    rid = meta.get("id")
                    if not rid:
                        sp = meta.get("source_path", "")
                        ci = meta.get("chunk_index", "")
                        if sp != "" and ci != "":
                            rid = f"{sp}:#{ci}"
                    if rid:
                        existing_ids.add(str(rid))
        except Exception:
            existing_ids = set()

    n_ok = 0
    open_mode = "a" if resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as f:
        for ch in chunks:
            rid = f"{ch.source_path}:#{ch.chunk_index}"
            if rid in existing_ids:
                continue
            try:
                llm_text = _sanitize_chunk_text_for_llm(ch.chunk_text)
                obj = extractor.extract_single(
                    llm_text,
                    schema,
                    context_hint=f"source={Path(ch.source_path).name} chunk={ch.chunk_index}",
                )

                # Throttle between requests (best-effort). Groq client may also retry on 429.
                if requests_per_minute and requests_per_minute > 0:
                    import time

                    time.sleep(60.0 / float(requests_per_minute))

                if enrich_images:
                    from modules.enrich_images import enrich_record_with_images

                    obj = enrich_record_with_images(
                        record=obj,
                        chunk_text=ch.chunk_text,
                        source_markdown_path=ch.source_path,
                        store_dir=image_store_dir,
                        prefer_format=image_prefer_format,
                        quality=image_quality,
                    )

                rec = {
                    "data": obj.model_dump(),
                    "meta": {
                        "source_path": ch.source_path,
                        "id": f"{ch.source_path}:#{ch.chunk_index}",
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                existing_ids.add(rid)
            except Exception as e:
                # If Groq asks for a long wait (e.g., tokens/day exhausted), stop early.
                if isinstance(e, RateLimitPauseRequired):
                    raise

                rec = {
                    "error": str(e),
                    "meta": {
                        "source_path": ch.source_path,
                        "id": f"{ch.source_path}:#{ch.chunk_index}",
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.error(f"Extraction failed for {ch.source_path}#{ch.chunk_index}: {e}")

    return n_ok


def iter_objects_from_jsonl(
    jsonl_path: str,
    schema: Type[BaseModel],
) -> Iterator[BaseModel]:
    p = Path(jsonl_path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "data" not in rec:
                continue
            yield schema(**rec["data"])


def iter_text_records_from_jsonl(
    jsonl_path: str,
    *,
    index_type: str,
) -> Iterator[Tuple[str, Dict[str, Any], str]]:
    """Yield (text, metadata, id) for vector ingestion."""
    p = Path(jsonl_path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "data" not in rec or "meta" not in rec:
                continue

            data = rec["data"]
            meta = rec["meta"]
            # Keep everything useful for citations
            merged_meta = {**meta, **data, "index_type": index_type}

            text = _format_text_from_data(data, index_type=index_type)
            rid = meta.get("id") or f"{meta.get('source_path','')}:#{meta.get('chunk_index','')}"
            yield text, merged_meta, rid


def _format_text_from_data(data: Dict[str, Any], *, index_type: str) -> str:
    if index_type in {"herbs_plants", "herbs"}:
        # MedicinalPlant-like
        plant = data.get("plant_name") or data.get("name") or ""
        other = data.get("other_names") or []
        family = data.get("family") or ""
        treats = data.get("treats") or []
        props = data.get("properties") or ""
        feats = data.get("botanical_features") or data.get("botanical_description") or ""

        parts: list[str] = [
            f"Plant: {plant}",
            f"Other names: {', '.join(other) if isinstance(other, list) else other}",
        ]
        if family:
            parts.append(f"Family: {family}")
        if props:
            parts.append(f"Properties: {props}")
        if feats:
            parts.append(f"Features: {feats}")
        if treats:
            parts.append(f"Treats: {', '.join(treats) if isinstance(treats, list) else treats}")
        return "\n".join(parts).strip()

    if index_type == "herbs_vegetables":
        name = data.get("plant_name") or ""
        sci = data.get("scientific_name") or ""
        family = data.get("family") or ""
        desc = data.get("botanical_description") or ""
        culinary = data.get("culinary_uses") or ""
        props = data.get("medicinal_properties") or ""
        remedies = data.get("remedies") or []
        return (
            f"Vegetable: {name}\n"
            f"Scientific name: {sci}\n"
            f"Family: {family}\n"
            f"Description: {desc}\n"
            f"Culinary uses: {culinary}\n"
            f"Medicinal properties: {props}\n"
            f"Remedies: {', '.join(remedies) if isinstance(remedies, list) else remedies}\n"
        ).strip()

    if index_type == "remedies":
        rname = data.get("recipe_name") or ""
        source = data.get("source_plant") or ""
        ingredients = data.get("ingredients") or []
        steps = data.get("preparation_steps") or []
        usage = data.get("usage_instructions") or ""
        benefits = data.get("health_benefits") or []
        return (
            f"Recipe: {rname}\n"
            f"Source plant: {source}\n"
            f"Ingredients: {', '.join(ingredients) if isinstance(ingredients, list) else ingredients}\n"
            f"Preparation: {', '.join(steps) if isinstance(steps, list) else steps}\n"
            f"Usage: {usage}\n"
            f"Benefits: {', '.join(benefits) if isinstance(benefits, list) else benefits}\n"
        ).strip()

    if index_type in {"endocrine_syndromes", "diseases"}:
        syn = data.get("syndrome_name") or data.get("disease") or ""
        symptoms = data.get("symptoms") or ""
        principle = data.get("treatment_principle") or ""
        prescribed = data.get("prescribed_remedy") or ""
        return (
            f"Syndrome: {syn}\n"
            f"Symptoms: {symptoms}\n"
            f"Treatment principle: {principle}\n"
            f"Prescribed remedy: {prescribed}\n"
        ).strip()

    if index_type == "endocrine_plants":
        plant = data.get("plant_name") or ""
        other = data.get("other_names") or []
        sci = data.get("scientific_name") or ""
        desc = data.get("botanical_description") or ""
        props = data.get("properties_and_dosage") or ""
        apps = data.get("therapeutic_applications") or []

        app_lines: list[str] = []
        if isinstance(apps, list):
            for a in apps:
                if not isinstance(a, dict):
                    continue
                indication = a.get("indication") or ""
                ingredients = a.get("ingredients") or ""
                usage = a.get("usage_instructions") or ""
                if indication or ingredients or usage:
                    app_lines.append(
                        " - "
                        + " | ".join([
                            f"Indication: {indication}" if indication else "",
                            f"Ingredients: {ingredients}" if ingredients else "",
                            f"Usage: {usage}" if usage else "",
                        ]).strip(" |")
                    )

        return (
            f"Endocrine plant: {plant}\n"
            f"Other names: {', '.join(other) if isinstance(other, list) else other}\n"
            f"Scientific name: {sci}\n"
            f"Description: {desc}\n"
            f"Properties/dosage: {props}\n"
            f"Applications:\n" + ("\n".join(app_lines) if app_lines else "(none)")
        ).strip()

    if index_type == "herbs":
        # Backward-compat: already handled above as herbs_plants/herbs.
        return json.dumps(data, ensure_ascii=False)

    if index_type == "diseases":
        # Backward-compat: handled above as endocrine_syndromes/diseases.
        return json.dumps(data, ensure_ascii=False)

    if index_type == "emergency":
        cond = data.get("condition_name") or ""
        signs = data.get("clinical_signs") or []
        steps = data.get("first_aid_steps") or []
        antidote = data.get("specific_antidote") or ""
        return (
            f"Condition: {cond}\n"
            f"Signs: {', '.join(signs) if isinstance(signs, list) else signs}\n"
            f"First aid: {', '.join(steps) if isinstance(steps, list) else steps}\n"
            f"Antidote: {antidote}\n"
        ).strip()

    return json.dumps(data, ensure_ascii=False)


def ingest_jsonl_to_vector_store(
    *,
    vector_store: MedicalVectorStore,
    jsonl_path: str,
    schema: Type[BaseModel],
    index_type: str,
    batch_size: int = 16,
) -> int:
    # schema is unused for ingestion now; kept for backward compatibility with callers
    return vector_store.add_texts_stream(
        index_type=index_type,
        records=iter_text_records_from_jsonl(jsonl_path, index_type=index_type),
        batch_size=batch_size,
    )
