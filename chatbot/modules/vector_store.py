import hashlib
import json
import logging
import os
import sqlite3
import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class _IndexPaths:
    root: Path
    sqlite_path: Path
    state_path: Path

class MedicalVectorStore:
    """
    Manages vector storage and retrieval for the medical knowledge system.
    
    Architecture:
    - Index A: Medicinal plants (MedicinalPlant, MedicinalVegetable)
    - Index B: Diseases & syndromes (EndocrineSyndrome)
    - Index C: Emergency protocols (EmergencyProtocol)
    
    Each index is independently queryable and routable.
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        embedding_model: str = "BAAI/bge-m3",
        embed_batch_size: int = 8,
        device: Optional[str] = None,
        shard_size: int = 2048,
        backend: Literal["disk", "chroma"] = "disk",
        chroma_collection_prefix: str = "traditional_medicine",
        embed_query_prefix: Optional[str] = None,
        embed_text_prefix: Optional[str] = None,
    ):
        """
        Initialize vector store with ChromaDB backend.
        
        Args:
            persist_dir: Where to save ChromaDB data (survives restarts)
            embedding_model: HuggingFace model for embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.shard_size = max(64, int(shard_size))

        self.backend = backend
        self.chroma_collection_prefix = chroma_collection_prefix

        # Embeddings (end-to-end via LlamaIndex HuggingFaceEmbedding)
        self.device = device or "cpu"
        logger.info(f"✓ Embedding device: {self.device}")

        self.embedding_model_name = str(embedding_model)

        # Optional embedding instructions/prefixes.
        # For many BGE retrieval models, using query/passsage prefixes improves alignment.
        # Keep defaults empty to avoid silently changing behavior for existing indices.
        env_qp = (os.getenv("EMBED_QUERY_PREFIX") or "").strip()
        env_tp = (os.getenv("EMBED_TEXT_PREFIX") or "").strip()
        self.embed_query_prefix = (
            (embed_query_prefix if embed_query_prefix is not None else env_qp) or ""
        )
        self.embed_text_prefix = (
            (embed_text_prefix if embed_text_prefix is not None else env_tp) or ""
        )

        self._embedder = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            device=self.device,
            embed_batch_size=self.embed_batch_size,
        )
        logger.info(f"✓ Initialized embeddings (LlamaIndex): {self.embedding_model_name}")
        if self.embed_query_prefix or self.embed_text_prefix:
            logger.info(
                "✓ Embedding prefixes enabled: "
                f"query_prefix='{self.embed_query_prefix}' text_prefix='{self.embed_text_prefix}'"
            )

        self._expected_embedding_dim: Optional[int] = None

        # Storage backend
        self._paths: Dict[str, _IndexPaths] = {}
        self._chroma_client = None
        self._chroma_collections: Dict[str, Any] = {}

        if self.backend == "disk":
            # Disk-backed indices (SQLite + sharded .npy memmaps)
            # Consolidated into 4 main categories for better routing and retrieval.
            index_types = [
                "herbs",      # Plants, vegetables, and medicinal herbs
                "diseases",   # Syndromes, medical conditions, and diseases
                "remedies",   # Recipes and formulas
                "emergency",  # Emergency protocols
            ]
            self._paths = {t: self._init_index_paths(t) for t in index_types}
            for idx in self._paths.values():
                self._init_sqlite(idx.sqlite_path)
                self._init_state(idx.state_path)

        elif self.backend == "chroma":
            # Optional: ChromaDB (works well on Colab/Linux with compatible Python)
            try:
                import chromadb  # type: ignore
            except Exception as e:
                raise ImportError(
                    "backend='chroma' requires chromadb. "
                    "On Windows + Python 3.14 this often fails due to missing native wheels. "
                    "Use backend='disk' locally, or run on Colab (Python 3.10/3.11) and pip install chromadb."
                ) from e

            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            for t in (
                "herbs_plants",
                "herbs_vegetables",
                "remedies",
                "endocrine_syndromes",
            ):
                name = f"{self.chroma_collection_prefix}_{t}"
                self._chroma_collections[t] = self._chroma_client.get_or_create_collection(name=name)

        else:
            raise ValueError("backend must be one of: disk, chroma")
    
    def add_texts(
        self,
        *,
        index_type: str,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
    ) -> int:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")
        if not texts:
            return 0

        if ids is None:
            ids = [self._stable_id(texts[i], metadatas[i]) for i in range(len(texts))]

        if self.backend == "disk":
            idx = self._paths[index_type]
            embeddings = np.asarray(self._embed(texts), dtype=np.float32)
            state = self._read_state(idx.state_path)
            dim = state.get("dim")
            if dim is None:
                state["dim"] = int(embeddings.shape[1])
                dim = state["dim"]
                state["embedding_model"] = self.embedding_model_name
                state["embed_query_prefix"] = self.embed_query_prefix
                state["embed_text_prefix"] = self.embed_text_prefix
                self._write_state(idx.state_path, state)
            if int(dim) != int(embeddings.shape[1]):
                raise ValueError(f"Embedding dim mismatch: index has {dim}, got {embeddings.shape[1]}")

            # Track embedding model used to build this index for better diagnostics.
            em = state.get("embedding_model")
            if not em:
                state["embedding_model"] = self.embedding_model_name
                state["embed_query_prefix"] = self.embed_query_prefix
                state["embed_text_prefix"] = self.embed_text_prefix
                self._write_state(idx.state_path, state)
            elif str(em) != str(self.embedding_model_name):
                raise ValueError(
                    "Embedding model mismatch for index '"
                    + str(index_type)
                    + "': index was built with '"
                    + str(em)
                    + "' but current model is '"
                    + str(self.embedding_model_name)
                    + "'. Re-ingest the index or use the same --embed-model."
                )

            # Ensure prefix/instruction config matches between ingest & query.
            if (state.get("embed_query_prefix") or "") != self.embed_query_prefix or (state.get("embed_text_prefix") or "") != self.embed_text_prefix:
                raise ValueError(
                    "Embedding prefix config mismatch for index '"
                    + str(index_type)
                    + "'. Index was built with embed_query_prefix='"
                    + str(state.get("embed_query_prefix") or "")
                    + "', embed_text_prefix='"
                    + str(state.get("embed_text_prefix") or "")
                    + "' but current config is embed_query_prefix='"
                    + str(self.embed_query_prefix)
                    + "', embed_text_prefix='"
                    + str(self.embed_text_prefix)
                    + "'.\nFix: re-ingest the index after setting EMBED_QUERY_PREFIX/EMBED_TEXT_PREFIX consistently."
                )

            conn = self._sqlite_conn(idx.sqlite_path)
            try:
                for i in range(len(texts)):
                    meta = self._maybe_store_images_in_sqlite(conn, metadatas[i])

                    # IMPORTANT: avoid creating orphan embeddings on re-ingest.
                    # If a doc id already exists, reuse its shard/offset and overwrite the embedding in-place.
                    existing = conn.execute(
                        "SELECT shard, offset FROM docs WHERE id=?",
                        (ids[i],),
                    ).fetchone()
                    if existing:
                        shard, offset = int(existing[0]), int(existing[1])
                        self._write_embedding_at(idx.root, shard, offset, int(dim), embeddings[i])
                    else:
                        shard, offset = self._append_embedding(idx.root, idx.state_path, embeddings[i])

                    conn.execute(
                        "INSERT OR REPLACE INTO docs(id, text, metadata_json, shard, offset) VALUES (?, ?, ?, ?, ?)",
                        (ids[i], texts[i], json.dumps(meta, ensure_ascii=False), int(shard), int(offset)),
                    )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return len(texts)

        # chroma backend
        collection = self._chroma_collections[index_type]
        embeddings = self._embed(texts)
        collection.upsert(
            ids=list(ids),
            documents=list(texts),
            metadatas=list(metadatas),
            embeddings=embeddings,
        )
        return len(texts)

    def add_texts_stream(
        self,
        *,
        index_type: str,
        records: Iterable[Tuple[str, Dict[str, Any], str]],
        batch_size: int = 16,
    ) -> int:
        bs = max(1, int(batch_size))
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []
        total = 0
        for text, meta, rid in records:
            texts.append(text)
            metas.append(meta)
            ids.append(rid)
            if len(texts) >= bs:
                total += self.add_texts(index_type=index_type, texts=texts, metadatas=metas, ids=ids)
                texts.clear(); metas.clear(); ids.clear()
        if texts:
            total += self.add_texts(index_type=index_type, texts=texts, metadatas=metas, ids=ids)
        logger.info(f"✓ Added {total} texts to {index_type} (batch_size={bs})")
        return total

    def query(self, index_type: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.backend == "chroma":
            collection = self._chroma_collections[index_type]
            emb = self._embed_query(query_text)
            res = collection.query(query_embeddings=[emb], n_results=int(top_k))
            out: List[Dict[str, Any]] = []
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)
            for i in range(len(ids)):
                out.append({"id": ids[i], "document": docs[i], "metadata": metas[i], "distance": dists[i]})
            return out

        # disk backend
        idx = self._paths[index_type]
        state = self._read_state(idx.state_path)
        dim = state.get("dim")
        if not dim:
            return []

        # Ensure prefix/instruction config matches index.
        if (state.get("embed_query_prefix") or "") != self.embed_query_prefix or (state.get("embed_text_prefix") or "") != self.embed_text_prefix:
            raise ValueError(
                "Embedding prefix config mismatch for index '"
                + str(index_type)
                + "'. Index embed_query_prefix='"
                + str(state.get("embed_query_prefix") or "")
                + "', embed_text_prefix='"
                + str(state.get("embed_text_prefix") or "")
                + "' but current config is embed_query_prefix='"
                + str(self.embed_query_prefix)
                + "', embed_text_prefix='"
                + str(self.embed_text_prefix)
                + "'.\nFix: set EMBED_QUERY_PREFIX/EMBED_TEXT_PREFIX to match, or re-ingest."
            )

        # state['shard'] is the current shard index being written to.
        # state['offset'] is the next write position within that shard (i.e. number of valid rows in current shard).
        cur_shard = int(state.get("shard") or 0)
        cur_offset = int(state.get("offset") or 0)
        if cur_shard == 0 and cur_offset <= 0:
            return []

        q = np.asarray(self._embed_query(query_text), dtype=np.float32)
        if q.ndim != 1:
            q = q.reshape(-1)
        if int(q.shape[0]) != int(dim):
            em = state.get("embedding_model")
            raise ValueError(
                "Embedding dimension mismatch for index '"
                + str(index_type)
                + "': index dim="
                + str(dim)
                + (" (model='" + str(em) + "')" if em else "")
                + ", query dim="
                + str(int(q.shape[0]))
                + " (model='"
                + str(self.embedding_model_name)
                + "').\n"
                + "Fix: run ingest again with the same --embed-model used to build the index, "
                + "or delete '"
                + str(self.persist_dir / index_type)
                + "' and re-ingest."
            )

        candidates: List[Tuple[float, int, int]] = []  # (score, shard, offset)

        # Brute-force scan across all embedding shards (disk backend).
        # This can be CPU-heavy for large indices; optionally emit progress.
        progress = os.getenv("VECTORSTORE_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}
        shard_paths = sorted(idx.root.glob("embeddings_*.npy"))
        t0 = time.perf_counter()
        if progress:
            logger.info(
                f"[vector_store] scanning {len(shard_paths)} shard(s) for index_type='{index_type}' (top_k={int(top_k)})"
            )

        # Oversample candidates to handle holes (missing shard/offset rows) gracefully.
        cand_mult = int(os.getenv("VECTORSTORE_CANDIDATE_MULT") or "10")
        cand_mult = max(1, cand_mult)

        for i, shard_path in enumerate(shard_paths, start=1):
            shard_idx = int(shard_path.stem.split("_")[-1])
            emb = np.load(shard_path, mmap_mode="r")
            if emb.ndim != 2 or emb.shape[1] != int(dim):
                continue

            # IMPORTANT: shard files are preallocated to shard_size.
            # Unused rows remain zeros and must be excluded, otherwise top-k can hit empty offsets.
            if shard_idx > cur_shard:
                continue
            max_rows = int(emb.shape[0])
            if shard_idx == cur_shard:
                max_rows = min(max_rows, max(0, int(cur_offset)))
            if max_rows <= 0:
                continue

            if progress and (i == 1 or i == len(shard_paths) or i % 5 == 0):
                dt = time.perf_counter() - t0
                logger.info(
                    f"[vector_store] shard {i}/{len(shard_paths)}: {shard_path.name} rows={max_rows} elapsed={dt:.1f}s"
                )

            scores = emb[:max_rows, :] @ q
            k = min(int(top_k) * cand_mult, int(scores.shape[0]))
            if k <= 0:
                continue
            top_idx = np.argpartition(scores, -k)[-k:]
            for off in top_idx.tolist():
                candidates.append((float(scores[off]), shard_idx, int(off)))

        if progress:
            dt = time.perf_counter() - t0
            logger.info(f"[vector_store] scan done in {dt:.2f}s; candidates={len(candidates)}")

        candidates.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict[str, Any]] = []
        conn = self._sqlite_conn(idx.sqlite_path)
        try:
            for score, shard, offset in candidates:
                row = conn.execute(
                    "SELECT id, text, metadata_json FROM docs WHERE shard=? AND offset=?",
                    (int(shard), int(offset)),
                ).fetchone()
                if not row:
                    continue
                out.append(
                    {
                        "id": row[0],
                        "document": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                        "score": score,
                    }
                )
                if len(out) >= int(top_k):
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return out

    def available_index_types(self) -> List[str]:
        """Return index buckets available for the configured backend."""
        if self.backend == "disk":
            return sorted(self._paths.keys())
        return sorted(self._chroma_collections.keys())

    def available_compatible_index_types(self) -> List[str]:
        """Return index buckets compatible with the current embedding model.

        For disk backend, filters indices whose stored embedding dim matches the
        current embedding model's output dim. This avoids runtime matmul crashes
        when mixing indices built with different models.
        """
        if self.backend != "disk":
            return self.available_index_types()

        # Compute expected dim once (cheap and cached after model load).
        if self._expected_embedding_dim is None:
            try:
                self._expected_embedding_dim = int(len(self._embed_query("x")))
            except Exception:
                self._expected_embedding_dim = None

        out: List[str] = []
        for t, idx in self._paths.items():
            state = self._read_state(idx.state_path)
            dim = state.get("dim")
            if not dim or self._expected_embedding_dim is None:
                # If no dim yet, keep it available.
                out.append(t)
                continue
            if int(dim) == int(self._expected_embedding_dim):
                out.append(t)
        return sorted(out)

    def embed_query(self, query_text: str) -> List[float]:
        """Public wrapper for generating a normalized query embedding."""
        return self._embed_query(query_text)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Public wrapper for generating normalized text embeddings."""
        return self._embed(texts)
    
    def _init_index_paths(self, index_type: str) -> _IndexPaths:
        root = self.persist_dir / index_type
        root.mkdir(parents=True, exist_ok=True)
        return _IndexPaths(
            root=root,
            sqlite_path=root / "docs.sqlite",
            state_path=root / "state.json",
        )

    def _sqlite_conn(self, sqlite_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_sqlite(self, sqlite_path: Path) -> None:
        conn = self._sqlite_conn(sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    shard INTEGER NOT NULL,
                    offset INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_shard_offset ON docs(shard, offset)")

            # Optional: store image bytes for later chatbot display.
            # Uses a stable ID (typically source sha256) to dedupe.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    byte_size INTEGER,
                    source_filename TEXT,
                    stored_path TEXT,
                    data BLOB NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_id ON images(id)")
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _maybe_store_images_in_sqlite(self, conn: sqlite3.Connection, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """If metadata contains ImageAsset-like entries, store image bytes in SQLite.

        This keeps your existing architecture intact:
          - Full metadata remains in docs.metadata_json
          - images table stores the actual bytes (BLOB)
          - each image entry gets an optional 'db_id' field for retrieval

        No-ops if there are no images or files are missing.
        """
        if not isinstance(metadata, dict):
            return metadata

        images = metadata.get("images")
        if not isinstance(images, list) or not images:
            return metadata

        # Shallow copy; we only rewrite the images list.
        out = dict(metadata)
        new_images: List[Dict[str, Any]] = []

        for img in images:
            if not isinstance(img, dict):
                continue
            img2 = dict(img)
            db_id = self._upsert_image_from_asset(conn, img2)
            if db_id:
                img2["db_id"] = db_id
            new_images.append(img2)

        if new_images:
            out["images"] = new_images
        return out

    def _guess_mime_type(self, path: Path) -> Optional[str]:
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            return mime
        ext = path.suffix.lower().lstrip(".")
        if ext == "webp":
            return "image/webp"
        if ext in {"jpg", "jpeg"}:
            return "image/jpeg"
        if ext == "png":
            return "image/png"
        return None

    def _upsert_image_from_asset(self, conn: sqlite3.Connection, asset: Dict[str, Any]) -> Optional[str]:
        """Insert image bytes into images table using a stable id.

        Preference order for bytes:
          1) stored_path (optimized output)
          2) source_path (original file)

        ID selection:
          - uses asset['sha256'] when available, otherwise computes sha256 of chosen bytes
        """
        stored_path = asset.get("stored_path")
        source_path = asset.get("source_path")

        p = None
        if isinstance(stored_path, str) and stored_path.strip():
            cand = Path(stored_path)
            if cand.exists() and cand.is_file():
                p = cand
        if p is None and isinstance(source_path, str) and source_path.strip():
            cand = Path(source_path)
            if cand.exists() and cand.is_file():
                p = cand
        if p is None:
            return None

        try:
            data = p.read_bytes()
        except Exception:
            return None

        # Stable ID: prefer source sha256 from pipeline (dedup across formats).
        sha256 = asset.get("sha256")
        if not isinstance(sha256, str) or not sha256.strip():
            sha256 = hashlib.sha256(data).hexdigest()

        # Mime: if we are storing optimized output, ensure mime matches stored file.
        mime_type = self._guess_mime_type(p)
        width = asset.get("width") if isinstance(asset.get("width"), int) else None
        height = asset.get("height") if isinstance(asset.get("height"), int) else None
        byte_size = len(data)
        source_filename = asset.get("source_filename") if isinstance(asset.get("source_filename"), str) else None
        stored_path_s = stored_path if isinstance(stored_path, str) else None

        conn.execute(
            """
            INSERT OR IGNORE INTO images(id, mime_type, width, height, byte_size, source_filename, stored_path, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sha256, mime_type, width, height, int(byte_size), source_filename, stored_path_s, sqlite3.Binary(data)),
        )
        return sha256

    def get_image(self, *, index_type: str, image_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an image from SQLite (disk backend only).

        Returns dict with keys: id, mime_type, width, height, byte_size, data(bytes), source_filename, stored_path
        """
        if self.backend != "disk":
            raise NotImplementedError("get_image is only supported for backend='disk'")

        idx = self._paths[index_type]
        conn = self._sqlite_conn(idx.sqlite_path)
        try:
            row = conn.execute(
                "SELECT id, mime_type, width, height, byte_size, source_filename, stored_path, data FROM images WHERE id=?",
                (str(image_id),),
            ).fetchone()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if not row:
            return None
        return {
            "id": row[0],
            "mime_type": row[1],
            "width": row[2],
            "height": row[3],
            "byte_size": row[4],
            "source_filename": row[5],
            "stored_path": row[6],
            "data": row[7],
        }

    def get_image_data_url(self, *, index_type: str, image_id: str) -> Optional[str]:
        """Return a data URL (data:mime;base64,...) for UI/chatbot rendering."""
        rec = self.get_image(index_type=index_type, image_id=image_id)
        if not rec:
            return None
        mime = rec.get("mime_type") or "application/octet-stream"
        data = rec.get("data")
        if not isinstance(data, (bytes, bytearray)):
            return None
        b64 = base64.b64encode(bytes(data)).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _init_state(self, state_path: Path) -> None:
        if state_path.exists():
            return
        state = {"dim": None, "shard": 0, "offset": 0}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_state(self, state_path: Path) -> Dict[str, Any]:
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _write_state(self, state_path: Path, state: Dict[str, Any]) -> None:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_shard_file(self, root: Path, shard_idx: int, dim: int) -> Path:
        p = root / f"embeddings_{shard_idx}.npy"
        if p.exists():
            return p
        # Preallocate a fixed-size shard; unused rows will remain zeros.
        np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(self.shard_size, int(dim)))
        return p

    def _write_embedding_at(self, root: Path, shard_idx: int, offset: int, dim: int, emb: np.ndarray) -> None:
        shard_path = self._ensure_shard_file(root, int(shard_idx), int(dim))
        mm = np.load(shard_path, mmap_mode="r+")
        if mm.ndim != 2 or int(mm.shape[1]) != int(dim):
            raise ValueError(f"Embedding shard dim mismatch at {shard_path}: expected {dim}, got {mm.shape}")
        if int(offset) < 0 or int(offset) >= int(mm.shape[0]):
            raise IndexError(f"Embedding offset out of range for {shard_path}: offset={offset}")
        mm[int(offset), :] = emb.astype(np.float32, copy=False)
        mm.flush()

    def _append_embedding(self, root: Path, state_path: Path, emb: np.ndarray) -> Tuple[int, int]:
        state = self._read_state(state_path)
        dim = int(state["dim"])
        shard = int(state["shard"])
        offset = int(state["offset"])

        if offset >= self.shard_size:
            shard += 1
            offset = 0

        shard_path = self._ensure_shard_file(root, shard, dim)
        mm = np.load(shard_path, mmap_mode="r+")
        mm[offset, :] = emb.astype(np.float32, copy=False)
        mm.flush()

        state["shard"] = shard
        state["offset"] = offset + 1
        self._write_state(state_path, state)
        return shard, offset

    def _stable_id(self, text: str, metadata: Dict[str, Any]) -> str:
        blob = (text + "\n" + str(sorted(metadata.items()))).encode("utf-8", errors="ignore")
        return hashlib.sha1(blob).hexdigest()

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n <= 0:
            return vec
        return vec / n

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.embed_text_prefix:
            texts = [self.embed_text_prefix + t for t in texts]

        all_embeddings: List[List[float]] = []
        bs = self.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            embs = self._embedder.get_text_embedding_batch(batch)
            for e in embs:
                v = np.asarray(e, dtype=np.float32)
                v = self._l2_normalize(v)
                all_embeddings.append(v.tolist())
        return all_embeddings

    def _embed_query(self, query_text: str) -> List[float]:
        qt = (self.embed_query_prefix + query_text) if self.embed_query_prefix else query_text
        e = self._embedder.get_query_embedding(qt)
        v = np.asarray(e, dtype=np.float32)
        v = self._l2_normalize(v)
        return v.tolist()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # This would be used after extracting documents
    from schemas.medical_schemas import MedicinalPlant
    
    # Initialize vector store
    store = MedicalVectorStore(persist_dir="./medical_knowledge")
    
    # Example: add extracted plants
    sample_plants = [
        # Would come from extractor.extract_from_file(...)
    ]
    
    # Add to appropriate index
    # store.add_documents(sample_plants, index_type="herbs")
    # store.add_documents(diseases, index_type="diseases")
    # store.add_documents(emergencies, index_type="emergency")
    
    # Build router
    # router = store.build_router()
    
    # Query
    # answer = store.query("Bệnh gì cây Bách xù có thể chữa?")
    # print(answer)
