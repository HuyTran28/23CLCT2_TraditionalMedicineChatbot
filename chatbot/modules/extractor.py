import json
import os
import re
import time
import random
from typing import Type, TypeVar, List, Optional, Any
from pydantic import BaseModel, ValidationError
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LlamaIndex imports
from llama_index.core.program import LLMTextCompletionProgram

# LLM backend: HuggingFace self-hosted only
from typing import Literal

# Book-aware splitting for extraction boundaries
from modules.book_splitters import split_by_book

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)


LLMBackend = Literal["hf"]


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
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry_count: int = 3,
        rate_limit_retries: int = 8,
        rate_limit_base_sleep_seconds: float = 2.0,
        rate_limit_max_sleep_seconds: float = 60.0,
        rate_limit_max_retry_after_seconds: float = 120.0,
        backend: Optional[LLMBackend] = None,
        load_in_4bit: Optional[bool] = None,
        device_map: Optional[str] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            model:
              - backend="hf": HuggingFace model id/path (e.g., Qwen/Qwen2.5-7B-Instruct)
            temperature: 0.0 for deterministic medical extraction
            retry_count: How many times to retry on JSON errors
            backend: Only "hf" is supported in this repository.
            load_in_4bit: For backend="hf" only. If None, auto-enable on CUDA (non-Windows) when bitsandbytes is available.
                        device_map: For backend="hf" only. If None, defaults to:
                            - "cpu" on Windows (to avoid consuming local GPU)
                            - "auto" elsewhere
        """

        backend_s = (backend or os.getenv("EXTRACTOR_BACKEND") or os.getenv("LLM_BACKEND") or "").strip().lower() or "hf"
        if backend_s != "hf":
            raise ValueError(
                "Only backend='hf' is supported. Run extraction with HuggingFace (e.g., on Colab GPU)."
            )

        self.backend: LLMBackend = "hf"

        # HuggingFace self-hosted backend
        hf_model_id = (os.getenv("HF_MODEL") or model).strip()
        if not hf_model_id:
            raise ValueError("HF model id/path is empty. Set HF_MODEL or pass model=... (backend='hf')")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "HuggingFace backend requested but transformers is missing. Install transformers."
            ) from e

        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                "HuggingFace backend requested but torch is missing. Install torch."
            ) from e

        force_cpu = (os.getenv("FORCE_CPU") or "").strip().lower() in {"1", "true", "yes", "y"}

        # Default device_map behavior:
        # - On Windows, prefer CPU by default (user asked to not consume local GPU).
        # - On Colab/Linux, allow "auto".
        # - Allow override via HF_DEVICE_MAP or explicit device_map parameter.
        env_device_map = (os.getenv("HF_DEVICE_MAP") or "").strip() or None
        if device_map is None:
            device_map = env_device_map
        if device_map is None:
            device_map = "cpu" if (force_cpu or os.name == "nt") else "auto"

        # Auto 4-bit behavior: best effort.
        # - Disable if forcing CPU or on Windows.
        if load_in_4bit is None:
            load_in_4bit = bool(torch.cuda.is_available() and (os.name != "nt") and (not force_cpu))
        if force_cpu:
            load_in_4bit = False

        quantization_config = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as e:
                raise RuntimeError(
                    "load_in_4bit=True requires bitsandbytes + accelerate. "
                    "On Colab: pip install -U bitsandbytes accelerate. "
                    "On Windows native, 4-bit is usually unsupported; set load_in_4bit=False."
                ) from e

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        # torch_dtype: on CPU, prefer float32 for compatibility.
        torch_dtype = "auto" if (device_map != "cpu") else torch.float32

        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )

        # LlamaIndex HF LLM wrapper
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM
        except Exception as e:
            raise RuntimeError(
                "HuggingFaceLLM is not available. Install llama-index-llms-huggingface."
            ) from e

        self.llm = HuggingFaceLLM(
            model=hf_model,
            tokenizer=tokenizer,
            temperature=float(temperature),
            max_new_tokens=int(max_tokens),
        )
        logger.info(f"Initialized HuggingFace LLM (self-hosted): {hf_model_id}")

        self.retry_count = retry_count
    
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
        while True:
            try:
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
                logger.warning(f"✗ JSON parsing failed (attempt {json_attempt}): {str(e)[:100]}")
                if json_attempt >= self.retry_count:
                    raise ValueError(f"Failed to extract after {self.retry_count} JSON attempts") from e
                continue

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
        
        Note: Can throttle calls to avoid overloading the model.
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
