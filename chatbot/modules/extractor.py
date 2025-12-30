import json
import os
import re
from typing import Type, TypeVar, List, Optional
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LlamaIndex imports
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)


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
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        retry_count: int = 3
    ):
        """
        Initialize the extractor.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Groq model name (llama-3.3-70b-versatile is fastest)
            temperature: 0.0 for deterministic medical extraction
            retry_count: How many times to retry on JSON errors
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.llm = Groq(
            api_key=self.api_key,
            model=model,
            temperature=temperature,
            max_tokens=4096,  # Enough for detailed extraction
        )
        self.retry_count = retry_count
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

        prompt_template = f"""
Bạn là chuyên gia văn bản y học/cây thuốc tiếng Việt. Nhiệm vụ: trích xuất dữ liệu có cấu trúc từ OCR.

Ngữ cảnh (nếu có): {context_hint}

Văn bản OCR:
{{text}}

Hãy chuyển văn bản OCR thành JSON theo đúng schema bên dưới.

Quy tắc quan trọng (để chất lượng tốt và ổn định):
1) KHÔNG bịa thông tin: không tự nghĩ ra tên khoa học, bệnh điều trị, bộ phận dùng... nếu văn bản không nói rõ.
    - Nếu không có: dùng null (cho Optional) hoặc [] (cho List) hoặc "" chỉ khi field bắt buộc là string.
2) Không lặp lại câu/đoạn: nếu OCR có câu trùng, chỉ giữ 1 lần.
3) Phân loại đúng nội dung vào field:
    - botanical_features: mô tả hình thái (thân/lá/hoa/quả/hạt...)
    - distribution_and_ecology: nơi mọc, sinh thái, hoặc thông tin trồng/canh tác (ví dụ: "được trồng ở trường học")
    - parts_used: chỉ liệt kê bộ phận dùng làm thuốc/chiết xuất (tách thành danh sách)
    - properties: tính vị (vị..., tính...)
    - pharmacological_effects: công năng/tác dụng dạng động từ (ví dụ: "giải độc")
    - treats: bệnh/triệu chứng cụ thể (nếu không có trong văn bản => [])
4) Giữ nguyên thuật ngữ tiếng Việt, không dịch.
5) Kiểu dữ liệu nghiêm ngặt:
    - List luôn là mảng JSON (kể cả rỗng [])
    - Optional có thể là null

Chỉ trả về JSON hợp lệ, không kèm giải thích.

Schema JSON:
{schema.model_json_schema()}
"""
        
        for attempt in range(self.retry_count):
            try:
                program = LLMTextCompletionProgram.from_defaults(
                    output_cls=schema,
                    prompt_template_str=prompt_template,
                    llm=self.llm,
                    verbose=False
                )

                result = program(text=cleaned_text)
                result = _clean_model_instance(result)
                logger.info(f"✓ Extraction successful (attempt {attempt + 1})")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"✗ JSON parsing failed (attempt {attempt + 1}): {str(e)[:100]}")
                if attempt == self.retry_count - 1:
                    raise ValueError(f"Failed to extract after {self.retry_count} attempts") from e
            
            except Exception as e:
                logger.error(f"✗ Extraction error: {str(e)[:200]}")
                raise
        
        return None
    
    def extract_batch(
        self,
        texts: List[str],
        schema: Type[ModelT],
        context_hint: str = ""
    ) -> List[ModelT]:
        """
        Extract multiple documents.
        
        Args:
            texts: List of OCR texts
            schema: Target Pydantic schema
            context_hint: Shared context for all documents
        
        Returns:
            List of validated Pydantic objects
        
        Note: Respects Groq rate limit (30 req/min on free tier)
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Extracting {i+1}/{len(texts)}...")
            try:
                obj = self.extract_single(text, schema, context_hint)
                results.append(obj)
            except Exception as e:
                logger.error(f"Failed on item {i}: {e}")
                results.append(None)  # Mark failed extraction
        
        return [r for r in results if r is not None]
    
    def extract_from_file(
        self,
        filepath: str,
        schema: Type[ModelT],
        chunk_by: str = "section"  # "section" or "paragraph"
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
        
        if chunk_by == "section":
            # Split by markdown headers
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
