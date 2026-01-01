import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure we can import sibling packages when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.book_splitters import split_by_book


_SECTION_INLINE_RE = re.compile(r"^\s*(?:#{0,6}\s*)?(?P<num>\d{1,4})\s*\.\s+\S+")
_SECTION_STANDALONE_RE = re.compile(r"^\s*(?:#{0,6}\s*)?(?P<num>\d{1,4})\s*\.\s*$")


def _norm_path(s: str) -> str:
    return (s or "").replace("/", "\\").strip().lower()


def _dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen: set[str] = set()
    out: List[Any] = []
    for x in items:
        if x is None:
            continue
        if isinstance(x, str):
            key = x.strip().casefold()
            if not key:
                continue
        else:
            key = json.dumps(x, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _merge_optional_text(values: Iterable[Optional[str]]) -> Optional[str]:
    cleaned: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        cleaned.append(s)
    if not cleaned:
        return None

    # If multiple distinct strings exist, keep the longest + append other distinct ones.
    uniq = _dedupe_preserve_order(cleaned)
    uniq_sorted = sorted(uniq, key=len, reverse=True)
    main = uniq_sorted[0]
    rest = [u for u in uniq if u != main]
    if not rest:
        return main
    return (main + "\n\n" + "\n\n".join(rest)).strip()


def _merge_list(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return _dedupe_preserve_order(out)


def _parse_chunk_index(meta_id: str) -> Optional[int]:
    # expected: <source_path>:#<idx>
    m = re.search(r":#(\d+)\s*$", meta_id or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def build_chunk_index_to_section_number(*, md_path: Path, source_path_in_jsonl: str) -> Dict[int, int]:
    """Reconstruct chunk list and map each chunk_index to its parent section number.

    This relies on the invariant: section boundaries are marked by lines like
    '109.' + next line title OR '109. TITLE' (after our splitter fix). When we
    split a section into multiple sub-chunks, only the first one contains the
    marker; later ones inherit the last seen section number.
    """
    text = md_path.read_text(encoding="utf-8")
    chunks = split_by_book(source_path_in_jsonl, text)

    mapping: Dict[int, int] = {}
    current_section: Optional[int] = None

    for idx, chunk in enumerate(chunks):
        lines = chunk.splitlines()
        head = "\n".join(lines[:10])

        found: Optional[int] = None
        for ln in head.splitlines():
            m = _SECTION_INLINE_RE.match(ln)
            if m:
                found = int(m.group("num"))
                break
            m2 = _SECTION_STANDALONE_RE.match(ln)
            if m2:
                found = int(m2.group("num"))
                break

        if found is not None:
            current_section = found
        if current_section is None:
            # No section marker has been seen yet; treat as section 0 (TOC/frontmatter)
            current_section = 0

        mapping[idx] = current_section

    return mapping


@dataclass
class MergeStats:
    sections_written: int
    sections_missing_data: List[int]
    sections_with_only_errors: List[int]


def merge_emergency_jsonl_by_section(
    *,
    jsonl_in: Path,
    jsonl_out: Path,
    md_path: Path,
    source_path_in_jsonl: str,
) -> MergeStats:
    idx_to_section = build_chunk_index_to_section_number(
        md_path=md_path,
        source_path_in_jsonl=source_path_in_jsonl,
    )

    # section -> list of (data, meta)
    section_data: dict[int, list[tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    section_errors: dict[int, int] = defaultdict(int)

    src_norm = _norm_path(source_path_in_jsonl)

    for line in jsonl_in.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        meta = rec.get("meta") or {}
        sp = meta.get("source_path") or ""
        if _norm_path(sp) != src_norm:
            continue

        meta_id = meta.get("id") or ""
        chunk_index = _parse_chunk_index(str(meta_id))
        if chunk_index is None:
            continue
        section_no = idx_to_section.get(chunk_index)
        if section_no is None:
            continue

        if "data" in rec and isinstance(rec["data"], dict):
            section_data[section_no].append((rec["data"], meta))
        elif "error" in rec:
            section_errors[section_no] += 1

    # Merge
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    sections = sorted(set(idx_to_section.values()))
    # Usually includes 0 for preface/TOC; skip it.
    sections = [s for s in sections if s != 0]

    missing_data: List[int] = []
    only_errors: List[int] = []
    written = 0

    with jsonl_out.open("w", encoding="utf-8") as wf:
        for sec in sections:
            rows = section_data.get(sec) or []
            if not rows:
                if section_errors.get(sec, 0) > 0:
                    only_errors.append(sec)
                else:
                    missing_data.append(sec)
                continue

            datas = [d for d, _m in rows]
            metas = [m for _d, m in rows]

            condition_names = [d.get("condition_name") for d in datas]
            categories = [d.get("category") for d in datas]

            merged: Dict[str, Any] = {
                "condition_name": _merge_optional_text(condition_names) or "",
                "category": Counter([str(c).strip() for c in categories if str(c).strip()]).most_common(1)[0][0]
                if any(str(c).strip() for c in categories)
                else "",
                "clinical_signs": _merge_list(d.get("clinical_signs") for d in datas),
                "diagnostic_tests": _merge_optional_text(d.get("diagnostic_tests") for d in datas),
                "first_aid_steps": _merge_list(d.get("first_aid_steps") for d in datas),
                "professional_treatment": _merge_list(d.get("professional_treatment") for d in datas),
                "medications": _merge_list(d.get("medications") for d in datas),
                "specific_antidote": _merge_optional_text(d.get("specific_antidote") for d in datas),
                "contraindications_warnings": _merge_optional_text(d.get("contraindications_warnings") for d in datas),
                "prevention": _merge_optional_text(d.get("prevention") for d in datas),
                "images": _merge_list(d.get("images") for d in datas),
            }

            # Match the original JSONL meta shape (as in emergency_cc.jsonl):
            #   {"source_path": "...", "id": "...:#<chunk_index>"}
            # Choose a representative chunk index for the section.
            chunk_indices = [
                _parse_chunk_index(str(m.get("id") or ""))
                for m in metas
                if _parse_chunk_index(str(m.get("id") or "")) is not None
            ]
            rep_idx = min(chunk_indices) if chunk_indices else None
            merged_meta: Dict[str, Any] = {
                "source_path": source_path_in_jsonl,
                "id": f"{source_path_in_jsonl}:#{rep_idx if rep_idx is not None else sec}",
            }

            wf.write(json.dumps({"data": merged, "meta": merged_meta}, ensure_ascii=False) + "\n")
            written += 1

    return MergeStats(
        sections_written=written,
        sections_missing_data=missing_data,
        sections_with_only_errors=only_errors,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge emergency JSONL lines back to one-per-section using section numbers from the source markdown."
    )
    ap.add_argument("--jsonl-in", required=True, help="Input JSONL (produced by extraction)")
    ap.add_argument("--jsonl-out", required=True, help="Output merged JSONL")
    ap.add_argument("--md", required=True, help="Source markdown path")
    ap.add_argument(
        "--source-path-in-jsonl",
        default=None,
        help=(
            "Exact meta.source_path value used inside the input JSONL for this markdown. "
            "Example: data\\raw\\cc_va_chong_doc_258\\cc_va_chong_doc_258.md"
        ),
    )

    args = ap.parse_args()

    jsonl_in = Path(args.jsonl_in)
    jsonl_out = Path(args.jsonl_out)
    md_path = Path(args.md)

    if args.source_path_in_jsonl:
        sp = args.source_path_in_jsonl
    else:
        # Best-effort: assume relative path under chatbot/
        sp = str(md_path).replace("/", "\\")

    stats = merge_emergency_jsonl_by_section(
        jsonl_in=jsonl_in,
        jsonl_out=jsonl_out,
        md_path=md_path,
        source_path_in_jsonl=sp,
    )

    print(
        json.dumps(
            {
                "jsonl_in": str(jsonl_in),
                "jsonl_out": str(jsonl_out),
                "md": str(md_path),
                "source_path_in_jsonl": sp,
                "sections_written": stats.sections_written,
                "sections_missing_data": stats.sections_missing_data[:30],
                "sections_with_only_errors": stats.sections_with_only_errors[:30],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
