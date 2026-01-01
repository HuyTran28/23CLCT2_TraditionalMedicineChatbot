import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class StoredImageInfo:
    source_path: Path
    stored_path: Path
    sha256: str
    mime_type: Optional[str]
    width: Optional[int]
    height: Optional[int]
    byte_size: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(md_file_path: Path, referenced_filename: str) -> Optional[Path]:
    """Resolve image filename referenced by markdown.

    This dataset stores PNGs in a sibling `extracted_images/` directory.
    The markdown often references just the filename.
    """
    base_dir = md_file_path.parent
    candidates = [
        base_dir / referenced_filename,
        base_dir / "extracted_images" / referenced_filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def store_image_efficiently(
    *,
    source_path: Path,
    out_dir: Path,
    prefer_format: str = "webp",
    quality: int = 80,
) -> StoredImageInfo:
    """Copy/convert an image to a compact on-disk representation.

    - Default output format: WebP (good compression, widely supported)
    - Output filename uses sha256 prefix for stable dedupe.

    Returns metadata for schema storage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sha256 = _sha256_file(source_path)
    byte_size = source_path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(source_path))

    width = None
    height = None

    prefer_format_norm = (prefer_format or "").strip().lower()
    if prefer_format_norm not in {"webp", "png", "jpg", "jpeg"}:
        prefer_format_norm = "webp"

    # Avoid extra deps unless needed
    from PIL import Image

    with Image.open(source_path) as im:
        width, height = im.size
        # Normalize mode for encoding
        if im.mode not in {"RGB", "RGBA"}:
            im = im.convert("RGB")

        ext = "webp" if prefer_format_norm == "webp" else ("jpg" if prefer_format_norm in {"jpg", "jpeg"} else "png")
        stored_path = out_dir / f"{source_path.stem}_{sha256[:16]}.{ext}"

        if stored_path.exists():
            return StoredImageInfo(
                source_path=source_path,
                stored_path=stored_path,
                sha256=sha256,
                mime_type=mime_type,
                width=width,
                height=height,
                byte_size=byte_size,
            )

        if ext == "webp":
            im.save(stored_path, format="WEBP", quality=int(quality), method=6)
        elif ext == "jpg":
            # JPEG doesn't support alpha; ensure RGB
            if im.mode == "RGBA":
                im = im.convert("RGB")
            im.save(stored_path, format="JPEG", quality=int(quality), optimize=True)
        else:
            im.save(stored_path, format="PNG", optimize=True)

    return StoredImageInfo(
        source_path=source_path,
        stored_path=stored_path,
        sha256=sha256,
        mime_type=mime_type,
        width=width,
        height=height,
        byte_size=byte_size,
    )
