import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class MarkdownImageRef:
    image_id: Optional[str]
    source_filename: str


# Pattern observed in your OCR markdown:
# ![](![id: cay-canh--..._img_002](cay-canh--..._img_002.png))
_IMAGE_LINE_RE = re.compile(
    r"!\[\]\(\!\[id:\s*(?P<id>[^\]]+?)\]\((?P<file>[^)]+?)\)\)"
)


def iter_image_refs_from_markdown(text: str) -> Iterable[MarkdownImageRef]:
    """Extract image references from the OCR markdown chunk.

    Heuristic:
      - Finds the special nested image syntax used by the dataset
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for i, line in enumerate(lines):
        m = _IMAGE_LINE_RE.search(line)
        if not m:
            continue
        image_id = (m.group("id") or "").strip() or None
        filename = (m.group("file") or "").strip()

        yield MarkdownImageRef(
            image_id=image_id,
            source_filename=filename,
        )
