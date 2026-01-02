from __future__ import annotations

import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

# Load environment from .env if present (helps local dev / setx not required)
load_dotenv()
from fastapi.responses import HTMLResponse, JSONResponse, Response

from modules.vector_store import MedicalVectorStore
from modules.router_engine import build_router_query_engine

app = FastAPI()

# Initialize shared resources at import time (one model load)
PERSIST_DIR = os.getenv("PERSIST_DIR", "vector_data")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "1"))
DEVICE = os.getenv("DEVICE", "cpu")
BACKEND = os.getenv("BACKEND", "disk")


def _get_groq_llm():
    from llama_index.llms.groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key, model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.0, max_tokens=1024)


# Build once
_VS = MedicalVectorStore(
    persist_dir=PERSIST_DIR,
    embedding_model=EMBED_MODEL,
    embed_batch_size=EMBED_BATCH,
    device=DEVICE,
    backend=BACKEND,
)

_LLM = None
try:
    _LLM = _get_groq_llm()
except Exception:
    # allow local use without GROQ for diagnostics; queries that require LLM will error later
    _LLM = None

_ROUTER = build_router_query_engine(vector_store=_VS, llm=_LLM, verbose=False)


_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^\)]+)\)")
_IMAGE_HTML_RE = re.compile(r"<img[^>]+src=[\"'](?P<url>[^\"']+)[\"']", re.IGNORECASE)

# When users explicitly ask about a plant's botanical features (đặc điểm thực vật / mô tả thân-lá-hoa-quả),
# we always include images in the response payload so the UI can render them.
_BOTANICAL_FEATURES_Q_RE = re.compile(
    r"(botanical\s*[_-]?\s*features|"
    r"đặc\s*điểm\s*(thực\s*vật|hình\s*thái)|"
    r"mô\s*tả\s*(cây|thân|lá|hoa|quả)|"
    r"hình\s*dáng\s*(cây|thân|lá|hoa|quả)|"
    r"đặc\s*điểm\s*(thân|lá|hoa|quả)|"
    r"(nhìn|trông)\s*(ra\s*sao|như\s*thế\s*nào)|"
    r"(cho\s*(xem|mình)\s*)?(ảnh|hình)\s*(cây|của\s*cây)?|"
    r"hình\s*minh\s*họa)",
    re.IGNORECASE,
)


def _is_botanical_features_question(question: str) -> bool:
    return bool(_BOTANICAL_FEATURES_Q_RE.search(question or ""))


def _workspace_root() -> Path:
    # webapp.py lives in the chatbot/ folder.
    return Path(__file__).resolve().parent


def _coerce_source_path(sp: str) -> Path:
    p = Path(sp)
    if p.is_absolute():
        return p
    return (_workspace_root() / p).resolve()


def _extract_line_hint(chunk_id: Any) -> int | None:
    """Try to recover a 1-based line number from chunk identifiers.

    Observed format: "...md:#53".
    """
    s = str(chunk_id or "")
    m = re.search(r":#(?P<line>\d+)\s*$", s)
    if not m:
        return None
    try:
        n = int(m.group("line"))
        return n if n > 0 else None
    except Exception:
        return None


def _extract_image_paths_from_markdown(md_path: Path, around_line: int | None, *, max_images: int = 3) -> List[Path]:
    if not md_path.exists() or not md_path.is_file():
        return []

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = text.splitlines()
    if not lines:
        return []

    if around_line is None:
        window = lines
    else:
        i = max(0, int(around_line) - 1)  # 1-based -> 0-based
        start = max(0, i - 30)
        end = min(len(lines), i + 30)
        window = lines[start:end]

    candidates: List[str] = []
    for ln in window:
        for rx in (_IMAGE_MD_RE, _IMAGE_HTML_RE):
            for m in rx.finditer(ln):
                url = (m.group("url") or "").strip()
                if not url:
                    continue
                if url.startswith(("http://", "https://", "data:")):
                    continue
                # Strip title part in markdown: (path "title")
                if " " in url:
                    url = url.split(" ", 1)[0].strip()
                candidates.append(url)

    out: List[Path] = []
    seen: set[str] = set()
    for url in candidates:
        norm = url.replace("\\", "/")
        p = Path(norm)
        if not p.is_absolute():
            p = (md_path.parent / p).resolve()
        if not p.exists():
            alt = (md_path.parent / "extracted_images" / Path(norm).name).resolve()
            if alt.exists():
                p = alt
        if not p.exists() or not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= int(max_images):
            break
    return out


def _strip_accents(s: str) -> str:
    # Vietnamese-friendly comparison: remove diacritics for robust substring checks.
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s


_NAME_STOPWORDS = {
    # common Vietnamese words around “what does it look like” queries
    "cay",
    "nhin",
    "trong",
    "ra",
    "sao",
    "nhu",
    "the",
    "nao",
    "giong",
    "dang",
    "hinh",
    "anh",
    "cho",
    "xem",
    "minh",
    "hoa",
    "cua",
    "la",
    "co",
}


def _tokens(s: str) -> List[str]:
    # Keep only letters/digits; split into tokens.
    t = _norm_text(s)
    toks = re.findall(r"[a-z0-9]+", t)
    return [x for x in toks if x and x not in _NAME_STOPWORDS]


def _maybe_filter_chunks_for_images(index_type: str, chunks: List[Any], question: str | None) -> List[Any]:
    """Prefer chunks whose entity name matches the question.

    This avoids returning images from nearby but unrelated chunks.
    """
    if not chunks:
        return chunks
    if not isinstance(question, str) or not question.strip():
        return chunks[:1]

    qn = _norm_text(question)

    # Focus tokens: words immediately after “cây” until the first stopword.
    q_words = re.findall(r"[a-z0-9]+", qn)
    focus_tokens: List[str] = []
    try:
        cay_i = q_words.index("cay")
        for w in q_words[cay_i + 1 :]:
            if w in _NAME_STOPWORDS:
                break
            focus_tokens.append(w)
    except ValueError:
        focus_tokens = []
    # If we extracted something meaningful after “cây”, match on that; else use all tokens.
    qtok = [t for t in (focus_tokens or _tokens(question)) if t]
    qset = set(qtok)

    name_key = "plant_name" if index_type == "herbs" else None
    if not name_key:
        return chunks[:1]

    matched: List[Any] = []
    for ch in chunks:
        meta = getattr(ch, "metadata", None) if hasattr(ch, "metadata") else (ch.get("metadata") if isinstance(ch, dict) else None)
        if not isinstance(meta, dict):
            continue
        pname = meta.get(name_key)
        if not isinstance(pname, str) or not pname.strip():
            continue
        ptok = _tokens(pname)
        if not ptok:
            continue
        # Strict token containment to avoid false positives (e.g., 'oi' matching inside 'voi').
        if set(ptok).issubset(qset):
            matched.append(ch)

    # If we got a direct match, use only those; else fall back to top-1.
    return matched if matched else chunks[:1]


def _gather_images_from_chunks(index_type: str, chunks: List[Any], *, question: str | None = None) -> List[Dict[str, str]]:
    imgs: List[Dict[str, str]] = []
    seen = set()
    chunks = _maybe_filter_chunks_for_images(index_type, chunks, question)
    for ch in chunks:
        # NOTE: Be explicit here. Python's conditional-expression precedence can
        # otherwise drop metadata for non-dict chunk objects (e.g., RetrievedChunk).
        if hasattr(ch, "metadata"):
            meta = getattr(ch, "metadata")
        elif isinstance(ch, dict):
            meta = ch.get("metadata")
        else:
            meta = {}
        if not isinstance(meta, dict):
            continue
        images = meta.get("images") if isinstance(meta.get("images"), list) else None
        if not images:
            # Fallback: parse images from the cited markdown source (near the referenced line).
            sp = meta.get("source_path") or meta.get("source")
            cid = ch.id if hasattr(ch, "id") else ch.get("id") if isinstance(ch, dict) else None
            if isinstance(sp, str) and sp.strip():
                try:
                    md_path = _coerce_source_path(sp)
                    line_hint = _extract_line_hint(cid)
                    found = _extract_image_paths_from_markdown(md_path, line_hint, max_images=3)
                    for p in found:
                        image_id = p.name
                        if image_id in seen:
                            continue
                        seen.add(image_id)
                        url = f"/image/{index_type}/{image_id}?path={quote(str(p))}"
                        imgs.append({"id": image_id, "alt": p.name, "url": url})
                    if found:
                        continue

                    # Secondary fallback: use first image in extracted_images next to markdown.
                    folder = md_path.parent / "extracted_images"
                    if folder.exists() and folder.is_dir():
                        for f in sorted(folder.iterdir()):
                            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                                image_id = f.name
                                if image_id in seen:
                                    continue
                                seen.add(image_id)
                                url = f"/image/{index_type}/{image_id}?path={quote(str(f.resolve()))}"
                                imgs.append({"id": image_id, "alt": f.name, "url": url})
                                break
                        continue
                except Exception:
                    pass
            continue
        for im in images:
            if not isinstance(im, dict):
                continue
            image_id = im.get("db_id") or im.get("sha256")
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            alt = im.get("source_filename") or im.get("image_id") or meta.get("plant_name") or "image"
            # Prefer serving from an explicit stored_path or source_path if present
            stored_path = im.get("stored_path") or im.get("source_path")
            if isinstance(stored_path, str) and stored_path.strip():
                # URL-encode the filesystem path so it can be passed as a query param
                p = quote(stored_path)
                url = f"/image/{index_type}/{image_id}?path={p}"
            else:
                url = f"/image/{index_type}/{image_id}"
            imgs.append({"id": image_id, "alt": str(alt), "url": url})
    return imgs


def query_internal(question: str, include_images: bool = True, verbose: bool = False) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Empty question")
    q = question.strip()

    # Override: for botanical-features questions, always include images.
    if _is_botanical_features_question(q):
        include_images = True

    # Use router's selection logic if available
    sel = getattr(_ROUTER, "_select_index_type", None)
    if sel:
        index_type, ranked = _ROUTER._select_index_type(q)
    else:
        # fallback
        index_type = "herbs"
        ranked = [(index_type, 0.0)]

    engine = None
    engines = getattr(_ROUTER, "_engines", None)
    if engines and index_type in engines:
        engine = engines[index_type]
    else:
        # Try building an ad-hoc engine
        engine = None

    # Retrieve chunks directly and build context
    chunks = []
    if engine is not None and hasattr(engine, "_retrieve"):
        chunks = engine._retrieve(q)
    else:
        # fallback to direct vector store query
        rows = _VS.query(index_type, q, top_k=3)
        for r in rows:
            chunks.append(r)

    # Build answer using engine LLM path if possible
    answer = ""
    if engine is not None and hasattr(engine, "_build_context") and hasattr(engine, "_answer"):
        context = engine._build_context(chunks)
        try:
            answer = engine._answer(q, context)
        except Exception as e:
            # If LLM not configured, fallback to empty answer
            answer = f"(LLM error: {e})"
    else:
        answer = "Không tìm thấy trình trả lời phù hợp trên server."

    images = _gather_images_from_chunks(index_type, chunks, question=q) if include_images else []

    sources = []
    for i, ch in enumerate(chunks, start=1):
        meta = ch.metadata if hasattr(ch, "metadata") else ch.get("metadata") if isinstance(ch, dict) else {}
        sp = (meta.get("source_path") or meta.get("source") or "")
        cid = ch.id if hasattr(ch, "id") else ch.get("id") if isinstance(ch, dict) else f"#{i}"
        sources.append({"id": str(cid), "source": sp})

    return {"question": q, "index": index_type, "ranked": ranked, "answer": answer, "images": images, "sources": sources}


@app.post("/api/query")
async def api_query(req: Request):
    body = await req.json()
    q = body.get("question")
    include_images = bool(body.get("include_images", True))
    if isinstance(q, str) and _is_botanical_features_question(q):
        include_images = True
    try:
        res = query_internal(q, include_images=include_images)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(res)


@app.get("/image/{index_type}/{image_id}")
def image_endpoint(index_type: str, image_id: str, path: str | None = None):
    # If a filesystem path was provided (URL-encoded), try serving that first.
    if path:
        try:
            from urllib.parse import unquote

            base = _workspace_root()
            raw = Path(unquote(path))
            p = raw if raw.is_absolute() else (base / raw).resolve()
            p_res = p.resolve()
            if not str(p_res).startswith(str(base)):
                raise FileNotFoundError()
            if not p.exists() or not p.is_file():
                # fallthrough to sqlite lookup
                raise FileNotFoundError()
            mime = _VS._guess_mime_type(p) or "application/octet-stream"
            return Response(content=p.read_bytes(), media_type=mime)
        except Exception:
            # fallback to DB lookup below
            pass

    rec = _VS.get_image(index_type=index_type, image_id=image_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Image not found")
    data = rec.get("data")
    mime = rec.get("mime_type") or "application/octet-stream"
    return Response(content=bytes(data), media_type=mime)


@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Traditional Medicine Chatbot</title>
            <style>
                body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:20px;}
                textarea{width:100%;height:80px;}
                img{max-width:300px;margin:8px;border:1px solid #ccc;padding:4px}
                #ans{white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;line-height:1.45;}
            </style>
    </head>
    <body>
      <h2>Traditional Medicine Chatbot</h2>
      <p>Ask a question (Vietnamese). Images are shown if available.</p>
      <textarea id="q"></textarea>
      <div>
        <label><input type="checkbox" id="img" checked /> Include images</label>
        <button id="ask">Ask</button>
      </div>
      <h3>Answer</h3>
            <div id="ans"></div>
      <div id="imgs"></div>
      <h4>Sources</h4>
      <ul id="srcs"></ul>

      <script>
        document.getElementById('ask').onclick = async function(){
          const q = document.getElementById('q').value;
          const inc = document.getElementById('img').checked;
          document.getElementById('ans').textContent = '...thinking...';
          document.getElementById('imgs').innerHTML = '';
          document.getElementById('srcs').innerHTML = '';
          try{
            const r = await fetch('/api/query', {method:'POST',headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q, include_images:inc})});
            const j = await r.json();
            document.getElementById('ans').textContent = j.answer;
            for(const im of j.images||[]){
              const el = document.createElement('img'); el.src = im.url; el.alt = im.alt; document.getElementById('imgs').appendChild(el);
            }
            for(const s of j.sources||[]){
              const li = document.createElement('li'); li.textContent = s.source + ' ('+s.id+')'; document.getElementById('srcs').appendChild(li);
            }
          }catch(e){document.getElementById('ans').textContent = 'Error: '+e}
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('webapp:app', host='0.0.0.0', port=8000, reload=False)
