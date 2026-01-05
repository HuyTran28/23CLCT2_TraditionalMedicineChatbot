import json
import os
import re
import time
import random
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Any, Literal

from pydantic import BaseModel, ValidationError
import logging
from dotenv import load_dotenv

# Load environment variables from repo root (.env)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# LlamaIndex imports (Groq backend)
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.groq import Groq

# Self-host backend
from modules.remote_llm import RemoteLLM

# Book-aware splitting for extraction boundaries
from modules.book_splitters import split_by_book

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)


class RateLimitPauseRequired(RuntimeError):
    """Raised when Groq asks us to wait a long time (e.g., TPD exhaustion)."""


def _strip_code_fences(s: str) -> str:
    s2 = (s or "").strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", s2)
        s2 = re.sub(r"\n```\s*$", "", s2)
    return s2.strip()


def _extract_first_json_object(text: str) -> str:
    """Best-effort: extract the first {...} JSON object from model output."""
    s = _strip_code_fences(text)
    if not s:
        return ""
    start = s.find("{")
    if start < 0:
        return s
    # Greedy to last closing brace; good enough for single-object JSON.
    end = s.rfind("}")
    if end > start:
        return s[start : end + 1].strip()
    return s[start:].strip()


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection for Groq rate limit responses.

    LlamaIndex/Groq may surface different exception types depending on
    dependency versions (httpx, groq SDK, etc.). We intentionally keep this
    heuristic and defensive.
    """
    # Common shapes: exc.status_code, exc.response.status_code
    sc = getattr(exc, "status_code", None)
    if sc == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True

    msg = str(exc) or ""
    msg_low = msg.lower()
    return ("429" in msg_low) or ("too many requests" in msg_low) or ("rate limit" in msg_low)


def _get_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract Retry-After header (seconds) if present."""
    headers: Any = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None

    try:
        ra = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None

    if ra is None:
        msg = str(exc) or ""
        msg_low = msg.lower()
        m = re.search(r"try again in\s+(?:(\d+)\s*m)?\s*([0-9]+(?:\.[0-9]+)?)\s*s", msg_low)
        if m:
            mins = float(m.group(1) or 0)
            secs = float(m.group(2) or 0)
            return mins * 60.0 + secs
        return None
    try:
        return float(str(ra).strip())
    except Exception:
        return None


def _normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _dedupe_consecutive_lines(text: str) -> str:
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    prev = None
    for ln in lines:
        if not ln.strip():
            # keep at most one blank line
            if out and out[-1] == "":
                continue
            out.append("")
            prev = ""
            continue
        if prev is not None and ln.strip() == prev.strip():
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out).strip()


def _dedupe_list(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        norm = _normalize_whitespace(item)
        if not norm:
            continue
        key = norm.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _dedupe_sentences(text: str) -> str:
    """Remove obvious repeated sentence fragments while preserving order."""
    s = _normalize_whitespace(text)
    if not s:
        return s

    # Split on sentence-ish boundaries. This is intentionally simple.
    parts = re.split(r"(?<=[\.!?。؛;])\s+", s)
    seen = set()
    out: list[str] = []
    for p in parts:
        p2 = _normalize_whitespace(p)
        if not p2:
            continue
        key = p2.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(p2)
    return " ".join(out)


def _clean_model_instance(obj: BaseModel) -> BaseModel:
    """Generic cleanup for any Pydantic model: trim whitespace and dedupe lists/strings."""
    data = obj.model_dump()

    def clean(v):
        if isinstance(v, str):
            # remove repeated lines/sentences introduced by OCR or the model
            return _dedupe_sentences(_dedupe_consecutive_lines(v))
        if isinstance(v, list):
            if all(isinstance(x, str) for x in v):
                return _dedupe_list(v)
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(val) for k, val in v.items()}
        return v

    cleaned = clean(data)
    return obj.__class__(**cleaned)


class MedicalDataExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        backend: Literal["auto", "groq", "remote"] = "auto",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry_count: int = 3,
        rate_limit_retries: int = 8,
        rate_limit_base_sleep_seconds: float = 2.0,
        rate_limit_max_sleep_seconds: float = 60.0,
        rate_limit_max_retry_after_seconds: float = 120.0,
    ):
        """
        Initialize the extractor.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            backend: "auto" (prefer self-host if LLM_API_BASE is set), "groq", or "remote"
            model: Groq model name (llama-3.3-70b-versatile is fastest)
            temperature: 0.0 for deterministic medical extraction
            retry_count: How many times to retry on JSON errors
            rate_limit_retries: Max retries when hitting 429 Too Many Requests
            rate_limit_base_sleep_seconds: Base sleep when 429 has no Retry-After header
            rate_limit_max_sleep_seconds: Cap for exponential backoff sleeps
        """
        self.backend = backend
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        api_base = (os.getenv("LLM_API_BASE") or "").strip()
        groq_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()

        if backend == "remote" or (backend == "auto" and api_base):
            if not api_base:
                raise ValueError("LLM_API_BASE environment variable not set (required for remote backend)")
            self.remote = RemoteLLM.from_env()
            self.llm = None
            self.api_key = None
            logger.info("Initialized Remote LLM (self-host): %s", api_base)
        else:
            self.api_key = groq_key or None
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable not set (required for groq backend)")
            self.llm = Groq(
                api_key=self.api_key,
                model=model,
                temperature=self.temperature,
                max_tokens=int(self.max_tokens),
            )
            self.remote = None
        self.retry_count = retry_count
        self.rate_limit_retries = max(0, int(rate_limit_retries))
        self.rate_limit_base_sleep_seconds = max(0.0, float(rate_limit_base_sleep_seconds))
        self.rate_limit_max_sleep_seconds = max(0.0, float(rate_limit_max_sleep_seconds))
        self.rate_limit_max_retry_after_seconds = max(0.0, float(rate_limit_max_retry_after_seconds))
        if self.llm is not None:
            logger.info(f"Initialized Groq LLM: {model}")
    
    def extract_single(
        self,
        text: str,
        schema: Type[ModelT],
        context_hint: str = ""
    ) -> ModelT:
        """
        Extract a single document into the specified Pydantic schema.
        
        Args:
            text: OCR-processed markdown text
            schema: Target Pydantic model class
            context_hint: Additional context (e.g., "This is from book 1: Medicinal Herbs")
        
        Returns:
            Pydantic model instance with validated data
        
        Example:
            plant = extractor.extract_single(
                text="Cây Bách xù...",
                schema=MedicinalPlant,
                context_hint="From chapter on herb identification"
            )
        """

        cleaned_text = _normalize_whitespace(_dedupe_consecutive_lines(text))

        # Make the prompt schema-driven so this extractor works across many schemas.
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        allowed_keys = ", ".join(schema.model_fields.keys())

        prompt_template = f"""
Bạn là chuyên gia trích xuất dữ liệu tiếng Việt từ văn bản OCR (y học/cây thuốc). Nhiệm vụ: tạo JSON đúng schema.

Ngữ cảnh (nếu có): {context_hint}

Văn bản OCR:
{{text}}

Yêu cầu bắt buộc:
1) Chỉ xuất ra MỘT đối tượng JSON hợp lệ (không kèm giải thích, không markdown, không code fence).
2) JSON phải khớp schema bên dưới và có thể được validate.
3) KHÔNG bịa thông tin: chỉ dùng thông tin có trong văn bản OCR.
4) Không tạo thêm key ngoài schema. Danh sách key hợp lệ (top-level): {allowed_keys}
5) Kiểu dữ liệu nghiêm ngặt theo schema:
   - Field kiểu List luôn là mảng JSON (kể cả rỗng: [])
   - Field Optional có thể là null khi không có thông tin
   - Field bắt buộc:
       - Nếu là string: dùng "" khi không có thông tin
       - Nếu là List: dùng [] khi không có thông tin
       - Nếu là object bắt buộc: tạo object tối thiểu theo schema và không bịa nội dung
6) Dọn nhiễu OCR: nếu văn bản có câu/đoạn trùng lặp, chỉ giữ một lần trong nội dung trích xuất.
7) Giữ nguyên thuật ngữ tiếng Việt, không dịch.

Schema JSON:
{schema_json}
"""
        
        json_attempt = 0
        rate_attempt = 0
        while True:
            try:
                if self.remote is not None:
                    prompt = prompt_template.format(text=cleaned_text)
                    resp = self.remote.complete(
                        prompt,
                        max_new_tokens=int(self.max_tokens),
                        temperature=float(self.temperature),
                    )
                    raw = getattr(resp, "text", "")
                    json_str = _extract_first_json_object(str(raw))
                    # Pydantic v2: validate directly from JSON string.
                    result = schema.model_validate_json(json_str)
                else:
                    program = LLMTextCompletionProgram.from_defaults(
                        output_cls=schema,
                        prompt_template_str=prompt_template,
                        llm=self.llm,
                        verbose=False
                    )
                    result = program(text=cleaned_text)

                result = _clean_model_instance(result)
                logger.info("✓ Extraction successful")
                return result
                
            except json.JSONDecodeError as e:
                json_attempt += 1
                logger.warning(f"JSON parsing failed (attempt {json_attempt}): {str(e)[:100]}")
                if json_attempt >= self.retry_count:
                    raise ValueError(f"Failed to extract after {self.retry_count} JSON attempts") from e
                continue

            except ValueError as e:
                # model_validate_json raises ValueError on invalid JSON
                msg = str(e) or ""
                if "json" in msg.lower():
                    json_attempt += 1
                    logger.warning(f"JSON parsing failed (attempt {json_attempt}): {msg[:160]}")
                    if json_attempt >= self.retry_count:
                        raise ValueError(
                            f"Failed to extract after {self.retry_count} JSON attempts"
                        ) from e
                    continue
                raise

            except ValidationError as e:
                # LlamaIndex often wraps JSON parse failures as Pydantic validation errors
                # with type 'json_invalid' when the model outputs truncated/invalid JSON.
                err_types = {str(err.get("type") or "") for err in (e.errors() or []) if isinstance(err, dict)}
                looks_like_json = any(
                    t in {"json_invalid", "json_decode", "value_error.jsondecode"} for t in err_types
                ) or ("Invalid JSON" in str(e))

                if looks_like_json:
                    json_attempt += 1
                    logger.warning(f"✗ JSON parsing failed (attempt {json_attempt}): {str(e)[:160]}")
                    if json_attempt >= self.retry_count:
                        raise ValueError(
                            f"Failed to extract after {self.retry_count} JSON attempts"
                        ) from e
                    continue
                raise
            
            except Exception as e:
                if _is_rate_limit_error(e):
                    rate_attempt += 1
                    if rate_attempt > self.rate_limit_retries:
                        logger.error(f"✗ Rate limited too many times ({rate_attempt}); giving up")
                        raise

                    retry_after = _get_retry_after_seconds(e)
                    # TPD exhaustion often comes with very large Retry-After (tens of minutes).
                    # Don't block the whole run for that long; fail fast with a clear message.
                    if (
                        retry_after is not None
                        and self.rate_limit_max_retry_after_seconds > 0
                        and retry_after > self.rate_limit_max_retry_after_seconds
                    ):
                        msg = str(e)
                        raise RateLimitPauseRequired(
                            "Groq rate limit requires a long wait "
                            f"(Retry-After={retry_after:.0f}s). This usually means TPD (tokens/day) "
                            "or another long-window quota is exhausted. "
                            "Wait for reset or switch to a higher-quota model/plan. "
                            f"Original error: {msg}"
                        ) from e

                    if retry_after is not None and retry_after > 0:
                        sleep_for = retry_after
                    else:
                        # Exponential backoff with jitter.
                        backoff = self.rate_limit_base_sleep_seconds * (2 ** max(0, rate_attempt - 1))
                        sleep_for = min(self.rate_limit_max_sleep_seconds, backoff)
                        sleep_for += random.uniform(0.0, 0.75)

                    logger.warning(
                        "⚠ Groq rate limit (429). "
                        + (f"Retry-After={retry_after}s. " if retry_after is not None else "No Retry-After header. ")
                        + f"Sleeping {sleep_for:.2f}s then retrying (attempt {rate_attempt}/{self.rate_limit_retries})."
                    )
                    time.sleep(max(0.0, sleep_for))
                    continue

                logger.error(f"✗ Extraction error: {str(e)[:200]}")
                raise
    
    def extract_batch(
        self,
        texts: List[str],
        schema: Type[ModelT],
        context_hint: str = "",
        requests_per_minute: Optional[float] = 30.0,
        jitter_seconds: float = 0.25
    ) -> List[ModelT]:
        """
        Extract multiple documents.
        
        Args:
            texts: List of OCR texts
            schema: Target Pydantic schema
            context_hint: Shared context for all documents
        
        Returns:
            List of validated Pydantic objects
        
        Note: Can throttle calls to respect Groq free-tier RPM.
        """
        results = []
        min_interval = None
        if requests_per_minute and requests_per_minute > 0:
            min_interval = 60.0 / float(requests_per_minute)
        last_call = 0.0
        for i, text in enumerate(texts):
            logger.info(f"Extracting {i+1}/{len(texts)}...")
            if min_interval is not None:
                now = time.monotonic()
                elapsed = now - last_call
                if elapsed < min_interval:
                    sleep_for = (min_interval - elapsed) + random.uniform(0.0, max(0.0, float(jitter_seconds)))
                    time.sleep(sleep_for)
            try:
                obj = self.extract_single(text, schema, context_hint)
                results.append(obj)
                last_call = time.monotonic()
            except Exception as e:
                logger.error(f"Failed on item {i}: {e}")
                results.append(None)  # Mark failed extraction
        
        return [r for r in results if r is not None]
    
    def extract_from_file(
        self,
        filepath: str,
        schema: Type[ModelT],
        chunk_by: str = "book"  # "book", "section" or "paragraph"
    ) -> List[ModelT]:
        """
        Load and extract from a markdown file.
        
        Args:
            filepath: Path to .md file (from OCR)
            schema: Target schema
            chunk_by: "section" splits by # headers, "paragraph" by blank lines
        
        Returns:
            List of extracted objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if chunk_by == "book":
            split_kind = None
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "RemedyRecipe":
                split_kind = "recipes"
            elif schema_name == "MedicinalPlant":
                split_kind = "plants"
            elif schema_name == "EndocrinePatternRecord":
                split_kind = "patterns"
            elif schema_name in {"EndocrineSyndrome", "EndocrineDisease"}:
                split_kind = "syndromes"

            chunks = split_by_book(filepath, content, split_kind=split_kind)
        elif chunk_by == "section":
            # Lightweight fallback: split by markdown headers
            chunks = [c.strip() for c in content.split('\n#') if c.strip()]
        else:
            # Split by blank lines
            chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        
        logger.info(f"Loaded {filepath}: {len(chunks)} chunks")
        return self.extract_batch(chunks, schema)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from schemas.medical_schemas import MedicinalPlant
    
    # Initialize
    extractor = MedicalDataExtractor()
    
    # Example OCR text
    sample_text = """
    1. CÂY BÁCH XÙ

(Tên khác : Cốt tía)
Cây Bách xù thân gỗ nhỏ, cao từ 3 – 4 mét, thân tròn hoặc hơi vuông, cành nhỏ. Ở cành non, lá có hình kim, đầu tù ở cành già, lá có dạng vẩy. Lá mọc gần như đối nhau, dày đặc, ở gần giữa lưng lá phình lên thành tuyến hình bầu dục. Nón đực có hình trứng kéo dài, mọc riêng; nón cái có hình cầu. Nón quả có hình hơi tròn, đường kính từ 6 – 8mm, có phủ phần trắng, khi chín có màu nâu, có từ 1 – 4 hạt. Ở trường học cây Bách xù được trồng trong châu hoặc bồn để làm cảnh và để học sinh quan sát các cây thuộc họ Bách. Cành lá dùng để cất tinh dầu. Hạt ép lấy dầu nhờn. Cành, lá, lõi thân được dùng làm thuốc. Cây Bách xù vị cay thơm, tính ấm, có tác dụng tán hàn khu phong, hoat huyết tiêu sưng, giải độc.
    """
    
    # Extract
    plant = extractor.extract_single(
        text=sample_text,
        schema=MedicinalPlant,
        context_hint="From chapter on decorative medicinal plants"
    )
    
    # Output
    print("\n" + "="*60)
    print("EXTRACTED PLANT DATA")
    print("="*60)
    print(plant.model_dump_json(indent=2))
    print("\nValidation: ✓ All fields type-checked by Pydantic")
