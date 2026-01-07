from pathlib import Path
from pydantic import BaseModel

from modules.image_assets import resolve_image_path, store_image_efficiently
from modules.markdown_images import iter_image_refs_from_markdown

# Robust default for image storage relative to the repository root
_DEFAULT_STORE = str(Path(__file__).resolve().parents[3] / "data" / "processed" / "images")

def enrich_record_with_images(
    *,
    record: BaseModel,
    chunk_text: str,
    source_markdown_path: str,
    store_dir: str = _DEFAULT_STORE,
    prefer_format: str = "webp",
    quality: int = 80,
) -> BaseModel:
    """Attach ImageAsset entries to a schema record when supported.

    Currently targets MedicinalPlant/Misc. plant schemas (adds `images`). If the record type doesn't
    have an `images` field, this is a no-op.
    """
    # Only enrich if the schema supports the field
    if "images" not in getattr(record, "model_fields", {}):
        return record

    md_path = Path(source_markdown_path)
    out_dir = Path(store_dir)

    assets = []

    for ref in iter_image_refs_from_markdown(chunk_text):
        src = resolve_image_path(md_path, ref.source_filename)
        if not src:
            # If we can't resolve the file, skip it.
            continue

        stored = store_image_efficiently(
            source_path=src,
            out_dir=out_dir,
            prefer_format=prefer_format,
            quality=quality,
        )

        # Store workspace-relative-ish paths when possible
        assets.append(
            {
                "stored_path": str(stored.stored_path),
                "sha256": stored.sha256,
                "mime_type": stored.mime_type,
                "width": stored.width,
                "height": stored.height,
                "byte_size": stored.byte_size,
            }
        )

    if not assets:
        return record

    data = record.model_dump()
    data["images"] = assets
    return record.__class__(**data)
