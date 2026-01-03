from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

import numpy as np

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from modules.vector_store import MedicalVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: Optional[float]
    metadata: Dict[str, Any]


class MedicalStoreQueryEngine(BaseQueryEngine):
    """QueryEngine wrapper around `MedicalVectorStore`.

    This lets us plug the existing disk-backed vector store into LlamaIndex's
    RouterQueryEngine.
    """

    def __init__(
        self,
        *,
        vector_store: MedicalVectorStore,
        index_type: str,
        llm: Any,
        similarity_top_k: int = 3,
        system_prompt: Optional[str] = None,
    ):
        self._vector_store = vector_store
        self._index_type = index_type
        self._llm = llm
        self._top_k = int(similarity_top_k)
        self._system_prompt = system_prompt

    def _coerce_query_text(self, query: Any) -> str:
        """Normalize LlamaIndex query inputs.

        Some LlamaIndex versions pass a QueryBundle instead of a raw string.
        """
        if isinstance(query, str):
            return query
        qs = getattr(query, "query_str", None)
        if isinstance(qs, str):
            return qs
        return str(query)

    def _retrieve(self, query: Any) -> List[RetrievedChunk]:
        q = self._coerce_query_text(query)
        rows = self._vector_store.query(self._index_type, q, top_k=self._top_k)
        out: List[RetrievedChunk] = []
        for r in rows:
            out.append(
                RetrievedChunk(
                    id=str(r.get("id") or ""),
                    text=str(r.get("document") or ""),
                    score=(r.get("score") if isinstance(r.get("score"), (int, float)) else None),
                    metadata=(r.get("metadata") if isinstance(r.get("metadata"), dict) else {}),
                )
            )
        return out

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        # Keep prompts small enough for remote LLMs.
        # These defaults are conservative; can be overridden via env.
        max_total = int(os.getenv("LLM_CONTEXT_MAX_CHARS") or "8000")
        max_per_chunk = int(os.getenv("LLM_CONTEXT_MAX_CHARS_PER_CHUNK") or "2500")

        def _condense_for_index(text: str) -> str:
            t = (text or "").strip()
            if not t:
                return t
            # Emergency chunks can be extremely long (lists of signs/steps). Keep only the most useful fields.
            if self._index_type == "emergency":
                # Prefer the "First aid" part; optionally include the condition header.
                lower = t.lower()
                cond = ""
                fa = ""
                ant = ""

                # Condition header (first line if present)
                if lower.startswith("condition:"):
                    first_nl = t.find("\n")
                    cond = t if first_nl < 0 else t[:first_nl]

                # First aid block
                fa_key = "first aid:"
                fa_pos = lower.find(fa_key)
                if fa_pos >= 0:
                    # Stop at Antidote: if present
                    stop_pos = lower.find("antidote:", fa_pos)
                    block = t[fa_pos : (stop_pos if stop_pos >= 0 else len(t))].strip()
                    fa = block

                # Antidote line
                ant_pos = lower.find("antidote:")
                if ant_pos >= 0:
                    ant_line = t[ant_pos:].strip()
                    # Keep just the first line of antidote if it is multi-line
                    nl = ant_line.find("\n")
                    ant = ant_line if nl < 0 else ant_line[:nl]

                parts = [p for p in (cond, fa, ant) if p]
                t = "\n".join(parts).strip() or t

            if len(t) > max_per_chunk:
                t = t[:max_per_chunk].rstrip() + "\n…(cắt bớt)"
            return t

        parts: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            source = ch.metadata.get("source_path") or ch.metadata.get("source") or ""
            idx = ch.metadata.get("chunk_index")
            rid = ch.id
            header_bits = [f"#{i}"]
            if rid:
                header_bits.append(f"id={rid}")
            if source:
                header_bits.append(f"source={source}")
            if idx is not None:
                header_bits.append(f"chunk={idx}")
            if ch.score is not None:
                header_bits.append(f"score={ch.score:.4f}")
            header = " ".join(header_bits)
            text = _condense_for_index(ch.text)
            parts.append(f"{header}\n{text}")

            # Stop once we hit the total context budget.
            joined = "\n\n---\n\n".join(parts)
            if len(joined) >= max_total:
                parts[-1] = (parts[-1][: max(0, max_total - (len(joined) - len(parts[-1])))]).rstrip() + "\n…(cắt bớt)"
                break

        ctx = "\n\n---\n\n".join(parts).strip()
        if len(ctx) > max_total:
            ctx = ctx[:max_total].rstrip() + "\n…(cắt bớt)"
        return ctx

    def _answer(self, query: Any, context: str) -> str:
        query_s = self._coerce_query_text(query)
        if self._llm is None or not hasattr(self._llm, "complete"):
            # Retrieval-only mode: return context directly.
            if not context.strip():
                return "Không tìm thấy dữ liệu phù hợp trong chỉ mục."
            return ("Trích đoạn liên quan:\n\n" + context.strip()).strip()
        def _cleanup_llm_answer(text: str) -> str:
            """Remove common prompt-echo artifacts from the remote LLM output."""
            t = (text or "").strip()
            if not t:
                return t

            # Drop leading instruction bullets that sometimes get echoed.
            # Only removes an initial contiguous block.
            lines = t.splitlines()
            drop_prefix = []
            for line in lines[:12]:
                s = line.strip().lower()
                if not s:
                    drop_prefix.append(line)
                    continue
                is_bullet = s.startswith("-") or s.startswith("•")
                looks_like_rules = any(
                    k in s
                    for k in (
                        "nếu không có thông tin",
                        "không sử dụng",
                        "không nhắc lại",
                        "không trích",
                        "không dùng",
                        "hướng dẫn",
                        "yêu cầu",
                    )
                )
                if is_bullet and looks_like_rules:
                    drop_prefix.append(line)
                    continue
                break

            if drop_prefix:
                t = "\n".join(lines[len(drop_prefix) :]).strip()
            return t

        if self._index_type == "emergency":
            sys = (
                self._system_prompt
                or "Bạn là trợ lý sơ cứu. Trả lời rõ ràng, trọn vẹn, ưu tiên an toàn và dựa trên ngữ cảnh cung cấp. "
                "Nếu ngữ cảnh không đủ, hãy nói rõ không đủ thông tin thay vì bịa."
            )
        else:
            sys = (
                self._system_prompt
                or "Bạn là trợ lý y học cổ truyền. Trả lời ngắn gọn, đúng trọng tâm, dựa trên ngữ cảnh cung cấp. "
                "Nếu ngữ cảnh không đủ, hãy nói rõ không đủ thông tin thay vì bịa."
            )

        extra_rules = (os.getenv("LLM_SYSTEM_RULES") or "").strip()
        if extra_rules:
            sys = (sys.rstrip() + "\n\nQuy tắc:\n" + extra_rules.strip()).strip()

        if self._index_type == "emergency":
            prompt = (
                f"{sys}\n\n"
                f"NGỮ CẢNH (trích từ kho dữ liệu):\n{context}\n\n"
                f"CÂU HỎI: {query_s}\n\n"
                "HƯỚNG DẪN TRẢ LỜI:\n"
                "- CHỈ trả lời nội dung, KHÔNG nhắc lại hướng dẫn này.\n"
                "- Trả lời bằng tiếng Việt.\n"
                "- Trình bày theo các mục ngắn gọn, dạng checklist:\n"
                "  1) Việc cần làm ngay\n"
                "  2) Không nên làm\n"
                "  3) Khi nào cần đi viện/gọi cấp cứu\n"
                "- Giữ câu trả lời gọn (khoảng 8–12 dòng), ưu tiên các bước quan trọng nhất.\n"
                "- Không đưa ra chẩn đoán; ưu tiên khuyến cáo an toàn chung.\n"
                "- Kết thúc bằng 1 dòng 'Nguồn:' liệt kê id hoặc source đã dùng.\n"
            )
        else:
            prompt = (
                f"{sys}\n\n"
                f"NGỮ CẢNH (trích từ kho dữ liệu):\n{context}\n\n"
                f"CÂU HỎI: {query_s}\n\n"
                "YÊU CẦU:\n"
                "- CHỈ trả lời nội dung, KHÔNG nhắc lại các yêu cầu/hướng dẫn.\n"
                "- Trả lời bằng tiếng Việt.\n"
                "- Nếu là câu hỏi về đặc điểm cây, ưu tiên thông tin từ phần 'Features:' (botanical_features).\n"
                "- Nếu có thể, kết thúc bằng 1 dòng 'Nguồn:' liệt kê id hoặc source đã dùng.\n"
            )

        resp = self._llm.complete(prompt)
        text = getattr(resp, "text", None)
        return _cleanup_llm_answer((text or str(resp) or "")).strip()

    def _images_markdown(self, chunks: List[RetrievedChunk], *, max_images: int = 1, max_bytes: int = 250_000) -> str:
        """Build a small Markdown block with embedded images (data URLs).

        This avoids relying on local file paths in the chatbot renderer.
        Only supported for disk backend where images are stored as SQLite BLOBs.
        """
        if getattr(self._vector_store, "backend", None) != "disk":
            return ""

        seen: set[str] = set()
        md_lines: List[str] = []

        for ch in chunks:
            imgs = ch.metadata.get("images") if isinstance(ch.metadata, dict) else None
            if not isinstance(imgs, list):
                continue
            for img in imgs:
                if not isinstance(img, dict):
                    continue
                image_id = img.get("db_id") or img.get("sha256")
                if not isinstance(image_id, str) or not image_id.strip():
                    continue
                if image_id in seen:
                    continue

                try:
                    rec = self._vector_store.get_image(index_type=self._index_type, image_id=image_id)
                except Exception:
                    rec = None
                if not rec:
                    continue

                # Avoid bloating responses too much.
                byte_size = rec.get("byte_size")
                if isinstance(byte_size, int) and byte_size > int(max_bytes):
                    continue

                try:
                    data_url = self._vector_store.get_image_data_url(index_type=self._index_type, image_id=image_id)
                except Exception:
                    data_url = None
                if not data_url:
                    continue

                alt = (
                    img.get("source_filename")
                    or img.get("image_id")
                    or ch.metadata.get("plant_name")
                    or "image"
                )
                alt_s = str(alt)
                md_lines.append(f"![{alt_s}]({data_url})")
                seen.add(image_id)

                if len(md_lines) >= int(max_images):
                    break
            if len(md_lines) >= int(max_images):
                break

        if not md_lines:
            return ""
        return "\n".join(["\nHình ảnh:", *md_lines]).strip()

    def _inject_images_before_sources(self, answer: str, images_md: str) -> str:
        if not images_md:
            return answer

        lines = (answer or "").splitlines()
        # Insert images block before the final 'Nguồn:' line if present.
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().lower().startswith("nguồn:"):
                return "\n".join(lines[:i] + [images_md] + lines[i:]).strip()
        return (answer.rstrip() + "\n" + images_md).strip()

    def query(self, query_str: Any, **kwargs: Any) -> Response:
        # Keep a public sync entrypoint for older LlamaIndex versions.
        return self._query(query_str, **kwargs)

    def _query(self, query_str: Any, **kwargs: Any) -> Response:
        # Newer LlamaIndex BaseQueryEngine uses _query as the abstract method.
        q = self._coerce_query_text(query_str)
        chunks = self._retrieve(q)
        context = self._build_context(chunks)
        if not context:
            return Response(response="Không tìm thấy dữ liệu phù hợp trong chỉ mục.")
        answer = self._answer(q, context)
        images_md = self._images_markdown(chunks)
        answer = self._inject_images_before_sources(answer, images_md)
        return Response(response=answer)

    async def aquery(self, query_str: Any, **kwargs: Any) -> Response:
        # Keep a public async entrypoint for older LlamaIndex versions.
        return await self._aquery(query_str, **kwargs)

    async def _aquery(self, query_str: Any, **kwargs: Any) -> Response:
        # Simple async wrapper; retrieval/LLM call are sync in this project.
        return self._query(query_str, **kwargs)

    def _get_prompt_modules(self) -> List[Any]:
        # No prompt modules are used in this minimal engine.
        return []


@dataclass(frozen=True)
class _RouteSpec:
    index_type: str
    tool_name: str
    description_vi: str


class VietnameseEmbeddingRouterQueryEngine(BaseQueryEngine):
    """Route queries without using an LLM (no translation).

    Uses the same embedding model as the vector store to choose the most relevant
    index bucket based on cosine similarity between the query and per-bucket
    Vietnamese descriptions.
    """

    def __init__(
        self,
        *,
        vector_store: MedicalVectorStore,
        routes: Sequence[_RouteSpec],
        engines: Dict[str, BaseQueryEngine],
        verbose: bool = False,
    ):
        self._vector_store = vector_store
        self._routes = list(routes)
        self._engines = dict(engines)
        self._verbose = bool(verbose)

        # Pre-compute route vectors once.
        route_texts = [r.description_vi for r in self._routes]
        embs = self._vector_store.embed_texts(route_texts)
        self._route_vecs = np.asarray(embs, dtype=np.float32) if embs else np.zeros((0, 0), dtype=np.float32)

    def _coerce_query_text(self, query: Any) -> str:
        if isinstance(query, str):
            return query
        qs = getattr(query, "query_str", None)
        if isinstance(qs, str):
            return qs
        return str(query)

    def _select_index_type(self, query: str) -> Tuple[str, List[Tuple[str, float]]]:
        if not self._routes:
            raise ValueError("No routes configured")
        qv = np.asarray(self._vector_store.embed_query(query), dtype=np.float32)
        if self._route_vecs.size == 0:
            # Fallback: first route.
            return self._routes[0].index_type, [(self._routes[0].index_type, 0.0)]

        scores = self._route_vecs @ qv

        # Small heuristic bias (Vietnamese) to avoid common mis-routing.
        ql = (query or "").lower()
        boost_herbs = 0.0
        boost_emergency = 0.0

        # If user explicitly asks for a plant or vegetable, prefer herbs index.
        if any(k in ql for k in ("cây", "rau", "ăn", "nấu", "luộc", "xào", "canh", "món")):
            boost_herbs += 0.04

        # If user asks about first aid / urgent situations, prefer emergency index.
        if any(
            k in ql
            for k in (
                "sơ cứu",
                "cấp cứu",
                "ngộ độc",
                "bị cắn",
                "rắn cắn",
                "chó cắn",
                "mèo cắn",
                "côn trùng cắn",
                "ong đốt",
                "bỏng",
                "gãy",
                "chảy máu",
                "đuối nước",
                "đột quỵ",
                "ngất",
            )
        ):
            boost_emergency += 0.10

        if boost_herbs:
            for i, r in enumerate(self._routes):
                if r.index_type == "herbs":
                    scores[i] = float(scores[i]) + float(boost_herbs)

        if boost_emergency:
            for i, r in enumerate(self._routes):
                if r.index_type == "emergency":
                    scores[i] = float(scores[i]) + float(boost_emergency)
        order = np.argsort(scores)[::-1]
        ranked: List[Tuple[str, float]] = []
        for idx in order.tolist():
            r = self._routes[idx]
            ranked.append((r.index_type, float(scores[idx])))
        best = ranked[0][0]
        return best, ranked

    def query(self, query_str: Any, **kwargs: Any) -> Response:
        return self._query(query_str, **kwargs)

    def _query(self, query_str: Any, **kwargs: Any) -> Response:
        q = self._coerce_query_text(query_str)
        index_type, ranked = self._select_index_type(q)
        engine = self._engines.get(index_type)
        if engine is None:
            return Response(response=f"Chỉ mục '{index_type}' chưa được khởi tạo.")

        if self._verbose:
            top = ", ".join([f"{t}:{s:.3f}" for t, s in ranked[:5]])
            print(f"[router] Câu hỏi: {q}")
            print(f"[router] Chọn chỉ mục: {index_type}")
            print(f"[router] Điểm tương đồng (top): {top}")

        return engine.query(q)

    async def aquery(self, query_str: Any, **kwargs: Any) -> Response:
        return await self._aquery(query_str, **kwargs)

    async def _aquery(self, query_str: Any, **kwargs: Any) -> Response:
        return self._query(query_str, **kwargs)

    def _get_prompt_modules(self) -> List[Any]:
        return []


def build_router_query_engine(
    *,
    vector_store: MedicalVectorStore,
    llm: Any,
    herbs_top_k: int = 3,
    diseases_top_k: int = 3,
    emergency_top_k: int = 2,
    verbose: bool = True,
) -> BaseQueryEngine:
    """Create a Vietnamese routing engine across available index buckets.

    Routing is embedding-based (no LLM), so it will not translate queries to English.
    """

    available = set(vector_store.available_compatible_index_types())

    engines: Dict[str, BaseQueryEngine] = {}
    if "herbs" in available:
        engines["herbs"] = MedicalStoreQueryEngine(
            vector_store=vector_store,
            index_type="herbs",
            llm=llm,
            similarity_top_k=herbs_top_k,
        )
    if "diseases" in available:
        engines["diseases"] = MedicalStoreQueryEngine(
            vector_store=vector_store,
            index_type="diseases",
            llm=llm,
            similarity_top_k=diseases_top_k,
        )
    if "remedies" in available:
        engines["remedies"] = MedicalStoreQueryEngine(
            vector_store=vector_store,
            index_type="remedies",
            llm=llm,
            similarity_top_k=herbs_top_k,
        )
    if "emergency" in available:
        engines["emergency"] = MedicalStoreQueryEngine(
            vector_store=vector_store,
            index_type="emergency",
            llm=llm,
            similarity_top_k=emergency_top_k,
        )

    # Vietnamese route descriptions; used only for local embedding-based routing.
    routes: List[_RouteSpec] = []
    if "herbs" in engines:
        routes.append(
            _RouteSpec(
                index_type="herbs",
                tool_name="thao_duoc_cay_thuoc",
                description_vi=(
                    "Thảo dược, cây thuốc, cây cảnh và rau làm thuốc: tên cây, đặc điểm thực vật, "
                    "bộ phận dùng, công dụng, chỉ định (trị bệnh gì), cách dùng và lưu ý."
                ),
            )
        )
    if "diseases" in engines:
        routes.append(
            _RouteSpec(
                index_type="diseases",
                tool_name="benh_ly_hoi_chung",
                description_vi=(
                    "Bệnh lý, hội chứng và triệu chứng: các bệnh nội tiết, chuyển hóa, "
                    "triệu chứng lâm sàng, nguyên nhân, nguyên tắc điều trị và hướng dùng thuốc."
                ),
            )
        )
    if "remedies" in engines:
        routes.append(
            _RouteSpec(
                index_type="remedies",
                tool_name="bai_thuoc_cong_thuc",
                description_vi=(
                    "Bài thuốc và công thức: thành phần (vị thuốc), liều lượng, cách sắc, "
                    "cách pha chế, cách dùng, đối tượng sử dụng và các lưu ý khi dùng thuốc."
                ),
            )
        )
    if "emergency" in engines:
        routes.append(
            _RouteSpec(
                index_type="emergency",
                tool_name="cap_cuu_ngo_doc",
                description_vi=(
                    "Cấp cứu và ngộ độc: xử trí khẩn cấp cho rắn cắn, say nắng, sốc, chảy máu, "
                    "bỏng, ngộ độc hóa chất (paraquat) và các kỹ thuật sơ cứu cơ bản."
                ),
            )
        )

    # Keep QueryEngineTool list for compatibility / future extensions (not used for routing now).
    _ = [
        QueryEngineTool(query_engine=engines[r.index_type], metadata=ToolMetadata(name=r.tool_name, description=r.description_vi))
        for r in routes
        if r.index_type in engines
    ]

    return VietnameseEmbeddingRouterQueryEngine(
        vector_store=vector_store,
        routes=routes,
        engines=engines,
        verbose=bool(verbose),
    )
