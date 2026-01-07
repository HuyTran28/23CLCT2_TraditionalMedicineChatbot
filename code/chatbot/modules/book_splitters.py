import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional
from pathlib import Path

def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_empty_edges(lines: List[str]) -> List[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _slice_by_starts(lines: List[str], starts: List[int]) -> List[str]:
    if not starts:
        return []

    starts_sorted = sorted(set(starts))
    out: List[str] = []
    for i, start in enumerate(starts_sorted):
        end = starts_sorted[i + 1] if i + 1 < len(starts_sorted) else len(lines)
        chunk_lines = _strip_empty_edges(lines[start:end])
        if not chunk_lines:
            continue
        out.append("\n".join(chunk_lines).strip())
    return out


def _filter_min_chars(chunks: List[str], min_chars: int) -> List[str]:
    if min_chars <= 0:
        return chunks
    return [c for c in chunks if len(c) >= min_chars]


# ---------------------------------------------------------------------------
# Book-specific splitters
# ---------------------------------------------------------------------------


def _find_line_index(lines: List[str], pattern: re.Pattern) -> Optional[int]:
    for i, ln in enumerate(lines):
        if pattern.match(ln):
            return i
    return None


def split_cay_canh_cay_thuoc_plants(text: str) -> List[str]:
    """Split 'Cây cảnh – cây thuốc' into plant-ish chunks only.

    This source file contains two different content types:
      - Part 1 + Part 2 (section I): plant monographs
      - Part 2 (section II): beverage/recipe instructions ("NƯỚC ...")

    Plant extraction should *exclude* section II; otherwise all recipes get
    appended to the last plant chunk.
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")

    # Cut off before the recipe section: '## II - KĨ THUẬT ...'
    recipe_section_marker = re.compile(r"^\s*##\s*ii\s*[-–—]\s*", re.IGNORECASE)
    recipe_start = _find_line_index(lines, recipe_section_marker)
    if recipe_start is not None:
        lines = lines[:recipe_start]

    # This book has two parts in the same file:
    # - Part 1 uses: '## <n>. CÂY ...'
    # - Part 2 uses: '### <n>. CÂY ...'
    # But Part 1 also contains internal subheadings like '### 1. Cây ...' for variants.
    # To avoid over-splitting, only enable the '### <n>.' rule after the '# Phần hai' marker.
    part2_marker = re.compile(r"^\s*#\s*phần\s+hai\b", re.IGNORECASE)
    part2_start = _find_line_index(lines, part2_marker)

    h2_header = re.compile(r"^\s*##\s*\d+\s*\.\s*(?:CÂY|CAY)\b", re.IGNORECASE)
    h2_to_h4_header = re.compile(
        r"^\s*#{2,4}\s*\d+\s*\.\s*(?:CÂY|CAY)\b",
        re.IGNORECASE,
    )

    starts: List[int] = []
    for i, ln in enumerate(lines):
        if part2_start is None or i < part2_start:
            if h2_header.match(ln):
                starts.append(i)
        else:
            if h2_to_h4_header.match(ln):
                starts.append(i)

    # Fallback: sometimes OCR removes markdown header hashes.
    if not starts:
        plant_header2 = re.compile(r"^\s*\d+\s*\.\s*(?:CÂY|CAY)\b", re.IGNORECASE)
        starts = [i for i, ln in enumerate(lines) if plant_header2.match(ln)]

    chunks = _slice_by_starts(lines, starts)
    return _filter_min_chars(chunks, min_chars=200)


def split_cay_canh_cay_thuoc_recipes(text: str) -> List[str]:
    """Split Part 2 / Section II beverage recipes into one-recipe chunks.

    Observed markers:
      - '## II - KĨ THUẬT LÀM NƯỚC GIẢI KHÁT TỪ TRÁI CÂY'
      - Recipe entries: '### <n>. NƯỚC <NAME>'
      - Subsections: '#### a) ...'
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")

    recipe_section_marker = re.compile(
        r"^\s*##\s*ii\s*[-–—]\s*k[ĩiỹy]\s*thuật\s+làm\s+nước\s+giải\s+khát\b",
        re.IGNORECASE,
    )
    recipe_section_start = _find_line_index(lines, recipe_section_marker)
    if recipe_section_start is None:
        # If the marker is missing (OCR variance), attempt a loose match.
        recipe_section_marker_loose = re.compile(r"^\s*##\s*ii\s*[-–—]\s*", re.IGNORECASE)
        recipe_section_start = _find_line_index(lines, recipe_section_marker_loose)
    if recipe_section_start is None:
        return []

    recipe_lines = lines[recipe_section_start:]

    recipe_header = re.compile(r"^\s*###\s*\d+\s*\.\s*(?:NƯỚC|NUOC)\b", re.IGNORECASE)
    starts = [i for i, ln in enumerate(recipe_lines) if recipe_header.match(ln)]
    chunks = _slice_by_starts(recipe_lines, starts)

    # Some recipes include multiple options in one entry, e.g.:
    #   '### 3. NƯỚC MƠ' then '#### a) ...' and '#### b) ...'
    # We split those into separate chunks so the extractor can output
    # separate RemedyRecipe records.
    out: List[str] = []
    for ch in chunks:
        out.extend(_split_recipe_options(ch))
    # Recipes can be fairly short (especially per-option chunks).
    return _filter_min_chars(out, min_chars=150)


def _split_recipe_options(recipe_chunk: str) -> List[str]:
    """Split a recipe chunk into per-option chunks when option headings exist.

    Supported option markers (OCR variance tolerant):
      - '#### a) ...'
      - '##### a) ...'
      - '#### b) ...'

    If no option markers are detected, returns [recipe_chunk].
    """
    recipe_chunk = _normalize_newlines(recipe_chunk)
    lines = recipe_chunk.split("\n")
    if not lines:
        return []

    # Match headings like: #### a) Quả Mơ muối
    # Allow 3..6 hashes because OCR varies.
    opt_header = re.compile(r"^\s*#{3,6}\s*(?P<label>[a-zA-Z])\)\s*(?P<title>.+?)\s*$")

    opt_starts: List[int] = []
    opt_meta: List[tuple[str, str]] = []  # (label, title)
    for i, ln in enumerate(lines):
        m = opt_header.match(ln)
        if not m:
            continue
        label = (m.group("label") or "").strip().lower()
        title = (m.group("title") or "").strip().rstrip(":")
        if not label:
            continue
        opt_starts.append(i)
        opt_meta.append((label, title))

    if len(opt_starts) < 2:
        # Keep as a single recipe if there aren't clearly multiple options.
        return [recipe_chunk]

    # Identify the main recipe heading line (### <n>. NƯỚC ...)
    recipe_heading_idx = None
    recipe_heading_re = re.compile(r"^\s*###\s*\d+\s*\.\s*(?:NƯỚC|NUOC)\b", re.IGNORECASE)
    for i, ln in enumerate(lines[: min(len(lines), 10)]):
        if recipe_heading_re.match(ln):
            recipe_heading_idx = i
            break
    if recipe_heading_idx is None:
        recipe_heading_idx = 0

    heading_line = lines[recipe_heading_idx]
    intro_lines = lines[recipe_heading_idx + 1 : opt_starts[0]]

    out: List[str] = []
    for j, start in enumerate(opt_starts):
        end = opt_starts[j + 1] if j + 1 < len(opt_starts) else len(lines)

        label, title = opt_meta[j]
        # Help the LLM extractor: make the recipe heading unique per option.
        # Example: '### 3. NƯỚC MƠ - Quả Mơ muối'
        if title:
            new_heading = f"{heading_line.rstrip()} - {title}".strip()
        else:
            new_heading = f"{heading_line.rstrip()} - {label})".strip()

        chunk_lines = [new_heading]
        # Keep shared intro ("2 cách...") so each option has context.
        if any(x.strip() for x in intro_lines):
            chunk_lines.extend(intro_lines)
        # Include the option section itself.
        chunk_lines.extend(lines[start:end])

        out.append("\n".join(chunk_lines).strip())

    return out


def split_cay_canh_cay_thuoc(text: str) -> List[str]:
    """Backward-compatible default splitter for this source: plants only."""
    return split_cay_canh_cay_thuoc_plants(text)


def split_cay_rau_lam_thuoc(text: str) -> List[str]:
    """Split 'Cây rau làm thuốc' into one-vegetable-ish chunks.

    Observed markers:
      - Entries begin with markdown headings like:
        '#### 1) ACTISÔ'
        '#### 2 BẦU'
        '#### (7) BỌ MẨY'

    Note: The OCR/markdown isn't fully consistent: many entries start with '####',
    but some use '##' or '###'. A few entries even lose the leading number.

    We split on headings (##..####) that look like entry titles.
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")

    # 1) Standard entry headers with a number token.
    # Examples seen in file:
    #   '#### 1) ACTISÔ'
    #   '#### 2 BẦU'
    #   '#### (7) BỌ MẨY'
    #   '## 21) CHUỐI'
    numbered_entry_header = re.compile(
        r"^\s*#{2,4}\s*(?:\(?\s*\d+\s*\)?\s*[,\.)]?|\d+\))\s+\S+",
        re.IGNORECASE,
    )

    # 2) Fallback for OCR cases where the number is missing but the title is ALL-CAPS.
    # Example: '## ĐẬU BẮP'
    # Avoid splitting on mixed-case internal headings like '#### Nhiều bộ phận...'
    caps_title_header = re.compile(r"^\s*#{2,4}\s*[A-ZÀ-Ỵ][A-ZÀ-Ỵ\s\-]{2,}$")
    caps_title_blocklist = (
        "CÁC LOÀI",
        "MỤC LỤC",
        "GIỚI THIỆU",
    )

    starts: List[int] = []
    for i, ln in enumerate(lines):
        if numbered_entry_header.match(ln):
            starts.append(i)
            continue
        if caps_title_header.match(ln.strip()):
            up = ln.strip().upper()
            if any(bad in up for bad in caps_title_blocklist):
                continue
            # Avoid extremely long all-caps headings (usually section banners).
            if len(ln.strip()) > 60:
                continue
            starts.append(i)
    chunks = _slice_by_starts(lines, starts)

    # Some entries include multiple distinct variants inside one entry (e.g. "MƯỚP" has
    # "MƯỚP KHÍA" and "MƯỚP HƯƠNG"). For extraction we want separate chunks so the LLM can
    # emit separate MedicinalVegetable records.
    out: List[str] = []
    for ch in chunks:
        out.extend(_split_vegetable_variants(ch))

    # Filter out accidental section headings without content.
    return _filter_min_chars(out, min_chars=250)


def _split_vegetable_variants(entry_chunk: str) -> List[str]:
    """Split a single vegetable entry into multiple chunks when variants exist.

    Heuristic (kept intentionally conservative):
      - Only triggers when we see 2+ ALL-CAPS variant title lines (not markdown headings)
        that contain the base entry name token.
      - Variants must appear within the numbered remedy section block (between '1.' and '2.'
        or later), which is where figure captions typically live.

    Example it should split:
      - '#### 84 MƯỚP' containing 'MƯỚP KHÍA' and 'MƯỚP HƯƠNG'.
    """
    entry_chunk = _normalize_newlines(entry_chunk)
    lines = entry_chunk.split("\n")
    if len(lines) < 10:
        return [entry_chunk]

    # Find the main heading line (first markdown heading in the chunk).
    heading_idx = None
    for i, ln in enumerate(lines[:10]):
        if re.match(r"^\s*#{2,4}\s+", ln):
            heading_idx = i
            break
    if heading_idx is None:
        heading_idx = 0

    heading_line = lines[heading_idx]
    heading_text = _md_heading_text(heading_line)
    # Drop leading numbers like '84 MƯỚP' or '(7) BỌ MẨY'.
    heading_text = re.sub(r"^\(?\s*\d+\s*\)?\s*[\.)]?\s*", "", heading_text).strip()
    base_token = (heading_text.split() or [""])[0].upper()
    if not base_token:
        return [entry_chunk]

    # Identify numbered section boundaries.
    numbered = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*(?P<n>\d{1,3})\s*\.\s+", ln)
        if m:
            try:
                numbered.append((int(m.group("n")), i))
            except Exception:
                continue
    if not numbered:
        return [entry_chunk]

    numbered.sort(key=lambda x: x[1])
    # Consider variant captions in the body from section 1 onwards.
    body_start = numbered[0][1]

    # Candidate variant title lines: all-caps, short, not headings, contains base token.
    cand_idxs: List[int] = []
    cand_titles: List[str] = []
    for i in range(body_start, len(lines)):
        ln = lines[i]
        if ln.lstrip().startswith("#"):
            continue
        s = ln.strip()
        if not s:
            continue
        if len(s) < 4 or len(s) > 40:
            continue
        if not _looks_like_all_caps_title(s):
            continue

        up = s.upper()
        # Must include the base token as a word (e.g., 'MƯỚP' in 'MƯỚP KHÍA').
        if base_token not in up.split():
            continue

        cand_idxs.append(i)
        cand_titles.append(s)

    # Need at least 2 variants, and they must be distinct.
    uniq_titles = []
    for t in cand_titles:
        if t not in uniq_titles:
            uniq_titles.append(t)
    if len(uniq_titles) < 2:
        return [entry_chunk]

    # Restrict to the first occurrence index for each unique title.
    first_idx_by_title: dict[str, int] = {}
    for idx, title in zip(cand_idxs, cand_titles):
        first_idx_by_title.setdefault(title, idx)
    variant_titles = sorted(first_idx_by_title.items(), key=lambda kv: kv[1])

    # Build a map from title->block start/end within the chunk.
    # Expand each title block upward to include an immediately preceding image line.
    img_re = re.compile(r"^\s*!\[\]\(|^\s*!\[[^\]]*\]\(")

    blocks: List[tuple[str, int, int]] = []  # (title, start, end)
    for j, (title, idx) in enumerate(variant_titles):
        start = idx
        # Include a directly preceding image line (common in this dataset).
        k = idx - 1
        while k >= 0:
            prev = lines[k].strip()
            if not prev:
                k -= 1
                continue
            if img_re.search(prev):
                start = k
            break

        end = variant_titles[j + 1][1] if j + 1 < len(variant_titles) else len(lines)
        blocks.append((title, start, end))

    # If we have a clear '2.' section, use it to stop variant blocks (they should live before 2.).
    sec2_idx = None
    for n, idx in numbered:
        if n == 2:
            sec2_idx = idx
            break
    if sec2_idx is not None:
        blocks = [(t, s, min(e, sec2_idx)) for (t, s, e) in blocks if s < sec2_idx]
        if len(blocks) < 2:
            return [entry_chunk]

    # Shared segments:
    # - prefix: from top of chunk until the start of the *first* variant block
    # - between_variants: any non-variant content between the last variant block end and section 2
    # - suffix: from section 2 onward
    first_block_start = blocks[0][1]
    prefix = lines[:first_block_start]

    last_block_end = max(e for _, _, e in blocks)
    between_end = sec2_idx if sec2_idx is not None else last_block_end
    between_variants = lines[last_block_end:between_end] if last_block_end < between_end else []
    suffix = lines[sec2_idx:] if sec2_idx is not None else []

    # Remove the original heading line from prefix; we will replace it.
    prefix_wo_heading = list(prefix)
    if 0 <= heading_idx < len(prefix_wo_heading):
        prefix_wo_heading.pop(heading_idx)

    out: List[str] = []
    for title, start, end in blocks:
        # Make a unique heading for the variant to guide extraction.
        # Example: '#### 84 MƯỚP - MƯỚP KHÍA'
        new_heading = f"{heading_line.rstrip()} - {title.strip()}".strip()
        chunk_lines: List[str] = [new_heading]
        if any(x.strip() for x in prefix_wo_heading):
            chunk_lines.extend(prefix_wo_heading)
        chunk_lines.extend(lines[start:end])
        if any(x.strip() for x in between_variants):
            chunk_lines.extend(between_variants)
        if any(x.strip() for x in suffix):
            chunk_lines.extend(suffix)
        out.append("\n".join(_strip_empty_edges(chunk_lines)).strip())

    return out


def split_noi_tiet(text: str) -> List[str]:
    """Split 'Cây thuốc, vị thuốc ... nội tiết' into extractable chunks.

    This source contains multiple content types:
      - PHẦN THỨ NHẤT: endocrine physiology / theory (not useful for structured extraction)
      - PHẦN THỨ HAI: diseases + Đông y treatment guidance (syndrome-ish)
      - PHẦN THỨ BA: medicinal plants / materia medica entries

    Default behavior returns both PHẦN THỨ HAI and PHẦN THỨ BA chunks.
    """
    return split_noi_tiet_syndromes(text) + split_noi_tiet_plants(text)


def _md_heading_text(s: str) -> str:
    """Normalize a markdown heading line (or heading title) into plain text."""
    # Drop heading hashes and emphasis markers.
    s = re.sub(r"^\s*#{1,6}\s*", "", s)
    s = s.replace("*", "").replace("_", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _is_mostly_uppercase_vi(s: str, threshold: float = 0.75) -> bool:
    """Heuristic: fraction of letters that are uppercase.

    Use Unicode-aware casing instead of brittle Vietnamese codepoint ranges.
    """
    s2 = _md_heading_text(s)
    if not s2:
        return False

    letters = [ch for ch in s2 if ch.isalpha() and ch.lower() != ch.upper()]
    if len(letters) < 3:
        return False
    upper = sum(1 for ch in letters if ch == ch.upper())
    return upper / len(letters) >= threshold


def split_noi_tiet_syndromes(text: str) -> List[str]:
    """Split PHẦN THỨ HAI (bệnh nội tiết + bài thuốc) into one-disease chunks.

    Important: this book also contains many internal numbered subheadings like
    '# 2. Mệt mỏi:'; we avoid splitting on those by requiring ALL-CAPS-ish titles.
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")

    part2_marker = re.compile(r"^\s*#\s*PHẦN\s+THỨ\s+HAI\b", re.IGNORECASE)
    part3_marker = re.compile(r"^\s*#\s*PHẦN\s+THỨ\s+BA\b", re.IGNORECASE)
    part2_start = _find_line_index(lines, part2_marker)
    part3_start = _find_line_index(lines, part3_marker)

    if part2_start is None:
        return []

    body = lines[part2_start : part3_start if part3_start is not None else len(lines)]

    # Disease headers typically look like: '# **1. BỆNH BƯỚU CỔ**'
    # We accept #..### but require the title to be mostly uppercase.
    header = re.compile(
        r"^\s*#{1,3}\s*(?:\*{1,2}\s*)?(?P<num>\d{1,3})\s*[\.)]\s+(?P<title>.+?)\s*$",
        re.IGNORECASE,
    )

    starts: List[int] = []
    for i, ln in enumerate(body):
        m = header.match(ln)
        if not m:
            continue
        title = _md_heading_text(m.group("title"))
        # Avoid very long section banners.
        if len(title) > 110:
            continue
        # Avoid subsection headers like '2. Mệt mỏi:' or '8. Phổi:'
        if not _is_mostly_uppercase_vi(title, threshold=0.80):
            continue
        # Extra guard: short items ending in ':' are often subsections.
        if title.endswith(":") and len(title) < 35:
            continue
        starts.append(i)

    chunks = _slice_by_starts(body, starts)
    return _filter_min_chars(chunks, min_chars=350)


def split_noi_tiet_patterns(text: str) -> List[str]:
    """Split PHẦN THỨ HAI disease chapters into pattern-level chunks.

    Why: disease chapters are long and contain many formulas/tables; extracting a
    single giant JSON object often gets truncated. Pattern-level chunks are
    smaller and map well to `EndocrinePatternRecord`.

    Heuristics:
      - Start from disease chapter chunks (same as split_noi_tiet_syndromes)
      - Within each chapter, split on headings that look like pattern sections:
          '#### *1. Thể ...:*', '#### *2. Chứng ...:*'
        and also keep 'Các bài thuốc kinh nghiệm' sections.
      - Prefix each chunk with the disease title line to preserve context.
    """
    chapters = split_noi_tiet_syndromes(text)
    out: List[str] = []

    # Pattern headings (allow OCR variance):
    #   '#### *1. Thể can khí uất trệ:*'
    #   '#### 2. Thể can hỏa thịnh:'
    pat_header = re.compile(
        r"^\s*#{2,6}\s*(?:\*{0,2}\s*)?(?:\(?\s*\d+\s*\)?\s*[\.)])\s*(Thể|Chứng)\b.*$",
        re.IGNORECASE,
    )
    exp_header = re.compile(r"^\s*#{1,6}\s*.*bài\s+thuốc\s+kinh\s+nghiệm.*$", re.IGNORECASE)

    disease_header = re.compile(
        r"^\s*#{1,3}\s*(?:\*{1,2}\s*)?(?P<num>\d{1,3})\s*[\.)]\s+(?P<title>.+?)\s*$",
        re.IGNORECASE,
    )

    for chapter in chapters:
        lines = _normalize_newlines(chapter).split("\n")
        if not lines:
            continue

        # Find disease title line (best-effort).
        title_line = None
        for ln in lines[: min(60, len(lines))]:
            m = disease_header.match(ln)
            if m:
                title_line = ln.strip()
                break
        if title_line is None:
            # Fallback: first non-empty line.
            for ln in lines:
                if ln.strip():
                    title_line = ln.strip()
                    break
        title_line = title_line or ""

        starts: List[int] = []
        for i, ln in enumerate(lines):
            if pat_header.match(ln) or exp_header.match(ln):
                starts.append(i)

        # If no internal pattern headers, keep the whole chapter as one chunk.
        if not starts:
            out.append(chapter)
            continue

        # Build chunks from starts. Prefix each with disease title.
        starts_sorted = sorted(set(starts))
        for j, start in enumerate(starts_sorted):
            end = starts_sorted[j + 1] if j + 1 < len(starts_sorted) else len(lines)
            chunk_lines = _strip_empty_edges(lines[start:end])
            if not chunk_lines:
                continue
            if title_line:
                chunk_lines = [title_line, "", *chunk_lines]
            out.append("\n".join(chunk_lines).strip())

    # Pattern sections can be shorter than full chapters.
    return _filter_min_chars(out, min_chars=220)


def split_noi_tiet_plants(text: str) -> List[str]:
    """Split PHẦN THỨ BA (cây thuốc / vị thuốc) into one-plant chunks."""
    text = _normalize_newlines(text)
    lines = text.split("\n")

    part3_marker = re.compile(r"^\s*#\s*PHẦN\s+THỨ\s+BA\b", re.IGNORECASE)
    part3_start = _find_line_index(lines, part3_marker)
    if part3_start is None:
        return []

    body = lines[part3_start:]

    # Plant headers are inconsistent in heading level:
    #   '# **1. CÔN BỐ**'
    #   '#### **2. THANH ĐẠI**'
    #   '# 10. SÀI HỒ'
    header = re.compile(
        r"^\s*#{1,4}\s*(?:\*{1,2}\s*)?(?P<num>\d{1,3})\s*(?:[\.)]\s+|\s+)(?P<title>.+?)\s*$",
        re.IGNORECASE,
    )

    starts: List[int] = []
    for i, ln in enumerate(body):
        m = header.match(ln)
        if not m:
            continue
        title = _md_heading_text(m.group("title"))
        if len(title) > 60:
            continue
        # Plant names are typically ALL-CAPS.
        if not _is_mostly_uppercase_vi(title, threshold=0.85):
            continue
        # Avoid false positives like 'BÀI THUỐC ...' if any are numbered.
        up = title.upper()
        if up.startswith("BÀI THUỐC") or up.startswith("CÁC BÀI THUỐC"):
            continue
        starts.append(i)

    chunks = _slice_by_starts(body, starts)
    return _filter_min_chars(chunks, min_chars=300)


def _looks_like_all_caps_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # Remove punctuation/digits; keep letters and spaces.
    letters = re.sub(r"[^A-ZÀ-Ỵ\s]", "", s)
    letters = re.sub(r"\s+", " ", letters).strip()
    if len(letters) < 3:
        return False

    # Heuristic: if the original (without spaces) is mostly uppercase letters/diacritics.
    # Since Vietnamese uses uppercase diacritics too, we treat A-ZÀ-Ỵ as uppercase.
    compact = re.sub(r"\s+", "", s)
    upperish = re.sub(r"[^A-ZÀ-Ỵ]", "", compact)
    return len(upperish) / max(1, len(compact)) >= 0.65


def split_cap_cuu_chong_doc(text: str) -> List[str]:
    """Split 'Cấp cứu & chống độc' into one protocol/condition per chunk.

    Observed marker pattern in body:
      - A standalone number line: '14.'
      - Next non-empty line is condition title in all-caps: 'ĐỘT QUỴ'

    The table of contents earlier uses the same pattern but yields tiny chunks;
    we filter those out by minimum length.
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")

    # Entry headers in this source are inconsistent:
    #   - Table of contents: number-only line, then ALL-CAPS title on next line
    #       1.
    #       RẮN CẮN
    #   - Main body: number and title on the same line
    #       1. RẮN CẮN
    # The old splitter only handled the TOC pattern, which led to extracting trash.

    header_num_only = re.compile(r"^\s*(?P<num>\d{1,3})\s*\.\s*$")
    header_same_line = re.compile(
        r"^\s*(?P<num>\d{1,3})\s*\.\s+(?P<title>.+?)\s*$",
        re.UNICODE,
    )

    # Heuristic to reject TOC entries:
    # real body sections quickly include lowercase Vietnamese (sentences, instructions),
    # while TOC blocks are mostly ALL-CAPS titles and numbers.
    _LOWER_VI_RE = re.compile(r"[a-zà-ỹ]", re.UNICODE)

    def _looks_like_body_after(start_idx: int, *, window: int = 60) -> bool:
        end = min(len(lines), start_idx + max(5, int(window)))
        lower_hits = 0
        for ln in lines[start_idx + 1 : end]:
            s = (ln or "").strip()
            if not s:
                continue
            if header_num_only.match(s):
                continue
            if _looks_like_all_caps_title(s):
                continue
            if "</break>" in s.lower():
                continue
            if _LOWER_VI_RE.search(s):
                lower_hits += 1
                if lower_hits >= 2:
                    return True
        return False

    def _is_dense_numbered_list_behind(idx: int) -> bool:
        """Detect TOC-like runs: many number-only headers in a short window."""
        start = max(0, idx - 20)
        count = 0
        for k in range(start, idx):
            if header_num_only.match(lines[k] or ""):
                count += 1
                if count >= 3:
                    return True
        return False

    starts_same_line: List[int] = []
    starts_num_then_title: List[int] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln is None:
            i += 1
            continue
        # Pattern A: '1. RẮN CẮN' (main body)
        m_same = header_same_line.match(ln)
        if m_same:
            title = (m_same.group("title") or "").strip()
            if _looks_like_all_caps_title(title) and _looks_like_body_after(i, window=60):
                starts_same_line.append(i)
                i += 1
                continue

        # Pattern B: '1.' then next non-empty ALL-CAPS title (TOC or sometimes body)
        if header_num_only.match(ln):
            # Skip dense runs of numbering which are almost always TOC pages.
            if _is_dense_numbered_list_behind(i):
                i += 1
                continue
            j = i + 1
            while j < len(lines) and not (lines[j] or "").strip():
                j += 1
            if j < len(lines) and _looks_like_all_caps_title(lines[j]):
                # For TOC entries, body text typically doesn't appear immediately.
                if _looks_like_body_after(j, window=30):
                    starts_num_then_title.append(i)
                i = j + 1
                continue

        i += 1

    # Combine both header styles (Chapter 1 uses same-line, later chapters often use
    # number-only line + title on next line).
    starts = starts_same_line + starts_num_then_title

    chunks = _slice_by_starts(lines, starts)

    # Remove TOC-like / low-signal chunks.
    # Real condition/protocol sections are typically longer.
    return _filter_min_chars(chunks, min_chars=300)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookSplitter:
    name: str
    matcher: Callable[[str], bool]
    splitter: Callable[[str], List[str]]


_SPLITTERS: List[BookSplitter] = [
    BookSplitter(
        name="cay-canh-cay-thuoc",
        matcher=lambda path: "cay-canh--cay-thuoc" in path.replace("\\", "/").lower(),
        splitter=split_cay_canh_cay_thuoc,
    ),
    BookSplitter(
        name="cay-rau-lam-thuoc",
        matcher=lambda path: "cay-rau-lam-thuoc" in path.replace("\\", "/").lower(),
        splitter=split_cay_rau_lam_thuoc,
    ),
    BookSplitter(
        name="noi-tiet",
        matcher=lambda path: "noi-tiet" in path.replace("\\", "/").lower(),
        splitter=split_noi_tiet,
    ),
    BookSplitter(
        name="cap-cuu-chong-doc",
        matcher=lambda path: "cc_va_chong_doc" in path.replace("\\", "/").lower(),
        splitter=split_cap_cuu_chong_doc,
    ),
]


def split_by_book(filepath: str, text: str, split_kind: Optional[str] = None) -> List[str]:
    """Choose a thin splitter based on filename/book type.

    Args:
        filepath: Used to detect which book/source splitter to use.
        text: Raw markdown.
        split_kind:
            - None / "plants": default behavior for plant-like sources
            - "recipes": for sources that contain recipe subsections
    """
    norm_path = filepath.replace("\\", "/").lower()

    # Special-case: this file contains both plant monographs and recipes.
    if "cay-canh--cay-thuoc" in norm_path:
        if split_kind and split_kind.lower() == "recipes":
            return split_cay_canh_cay_thuoc_recipes(text)
        return split_cay_canh_cay_thuoc_plants(text)

    # Special-case: endocrine book contains syndromes (Part 2) and plant monographs (Part 3).
    if "noi-tiet" in norm_path:
        if split_kind and split_kind.lower() == "plants":
            return split_noi_tiet_plants(text)
        if split_kind and split_kind.lower() == "patterns":
            return split_noi_tiet_patterns(text)
        if split_kind and split_kind.lower() == "syndromes":
            return split_noi_tiet_syndromes(text)
        return split_noi_tiet(text)

    for spec in _SPLITTERS:
        if spec.matcher(norm_path):
            return spec.splitter(text)

    cleaned = _normalize_newlines(text).strip()
    return [cleaned] if cleaned else []

# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Robust path resolution: find 'data' relative to this file's location
    _this_dir = Path(__file__).resolve().parent
    # book_splitters.py is in chatbot/modules/, so data is at ../../data
    example_path = _this_dir.parent.parent / "data" / "raw" / "cc_va_chong_doc_258" / "cc_va_chong_doc_258.md"
    
    split_kind = "recipes"
    if not example_path.exists():
        print(f"Error: Example file not found at {example_path}")
    else:
        with open(example_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_by_book(str(example_path), content, split_kind=split_kind)
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i + 1} ---")
            print(chunk[:500])
            print()
