from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic.config import ConfigDict
from typing import List, Optional, Any
from enum import Enum

class ImageAsset(BaseModel):
    """Represents an image attached to a chunk (for rendering).

    Notes:
        - Images are stored efficiently on disk; JSONL stores only metadata + paths.
        - No VLM/captioning is used; plant features come from `botanical_features`.
    """

    model_config = ConfigDict(extra="ignore")

    stored_path: Optional[str] = Field(
        default=None,
        description="Path of the optimized stored image (typically webp)",
    )
    sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the image bytes (stable id for dedupe/caching)",
    )

    db_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional ID for retrieving image bytes from the SQLite 'images' table "
            "(when using disk backend). Typically the same as sha256."
        ),
    )
    mime_type: Optional[str] = Field(default=None, description="MIME type (e.g., image/png)")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    byte_size: Optional[int] = Field(default=None, description="Source image size in bytes")

# ============================================================================
# SCHEMA 1: Medicinal Plants & Herbs (Cây cảnh - Cây thuốc)
# ============================================================================
class PartUsage(BaseModel):
    part: str = Field(..., description="Bộ phận dùng")
    usage_description: Optional[str] = Field(
        default=None,
        description="Cách sử dụng hoặc ghi chú về dịch liệu"
    )


class RemedyApplication(BaseModel):
    """Specific medical application of a plant.

    Used by endocrine-style plant monographs.
    """

    indication: str = Field(..., description="Bệnh hoặc triệu chứng được chữa")
    ingredients: Optional[str] = Field(None, description="Thành phần bài thuốc")
    usage_instructions: Optional[str] = Field(
        default=None,
        description="Cách dùng và liều dùng",
    )

class MedicinalPlant(BaseModel):
    """
    Core schema for extracting medicinal plant entities.
    
    Fields:
        plant_name: Primary name (e.g., "Cây Bách xù")
        other_names: Alternative names
        botanical_features: Physical description
        distribution_and_ecology: Growth habitat and conditions
        parts_used: Plant parts used medicinally
        properties: Taste (vị) and nature (tính)
        pharmacological_effects: General medical actions
        treats: Specific diseases/conditions treated
    """
    
    plant_name: str = Field(
        ...,
        description="Tên chính của cây"
    )
    other_names: List[str] = Field(
        default_factory=list,
        description="Tên gọi khác (có thể nhiều tên).",
    )

    scientific_name: Optional[str] = Field(
        default=None,
        description="Tên khoa học (nếu có)",
    )

    family: Optional[str] = Field(
        default=None,
        description="Họ thực vật (ví dụ: Họ Bách, Họ Cúc, ...)"
    )
    
    botanical_features: Optional[str] = Field(
        default=None,
        description="Mô tả chi tiết: thân, lá, hoa, quả (nếu có)",
    )
    distribution_and_ecology: Optional[str] = Field(
        default=None,
        description="Nơi mọc, mùa hoa quả (nếu có)"
    )
    parts_used: List[PartUsage] = Field(
        default_factory=list,
        description="Danh sách bộ phận dùng kèm cách dùng cụ thể"
    )
    properties: Optional[str] = Field(
        default=None,
        description="Tính vị (e.g., 'Vị cay thơm, tính ấm') (nếu có)",
    )
    properties_and_dosage: Optional[str] = Field(
        default=None,
        description="Tính vị, công năng, liều dùng (một số sách gộp chung) (nếu có)",
    )
    pharmacological_effects: List[str] = Field(
        default_factory=list,
        description="Tác dụng dược lý (e.g., ['thanh nhiệt', 'giải độc'])"
    )
    treats: List[str] = Field(
        default_factory=list,
        description="Danh sách bệnh/triệu chứng cụ thể (nếu có)",
    )

    contraindications_warnings: Optional[str] = Field(
        default=None,
        description="Chống chỉ định/cảnh báo/liều lượng hoặc lưu ý an toàn (nếu có)",
    )

    images: List[ImageAsset] = Field(
        default_factory=list,
        description="Danh sách ảnh liên quan (đường dẫn + metadata). Thường 1 ảnh/cây.",
    )

    therapeutic_applications: List[RemedyApplication] = Field(
        default_factory=list,
        description="Ứng dụng/chỉ định cụ thể (nếu có)",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_plant_fields(cls, data: Any):
        # Accept either dict-like input or already-built objects.
        if not isinstance(data, dict):
            return data

        d = dict(data)

        def _coerce_list_field(key: str):
            v = d.get(key)
            if v is None:
                d[key] = []
                return
            if isinstance(v, list):
                return
            # Sometimes the LLM returns a single object or a single string.
            if isinstance(v, dict):
                d[key] = [v]
                return
            if isinstance(v, str):
                s = v.strip()
                d[key] = [s] if s else []
                return

        # other_names may come as a string from older schema/prompt.
        on = d.get("other_names")
        if isinstance(on, str):
            s = on.strip()
            d["other_names"] = [s] if s else []
        elif on is None:
            d["other_names"] = []

        # LLMs sometimes emit null for list fields; normalize to [].
        _coerce_list_field("pharmacological_effects")
        _coerce_list_field("treats")
        _coerce_list_field("parts_used")
        _coerce_list_field("images")
        _coerce_list_field("therapeutic_applications")

        # LLMs sometimes emit null for nested required-ish strings.
        ta = d.get("therapeutic_applications")
        if isinstance(ta, list):
            cleaned_ta = []
            for item in ta:
                if not isinstance(item, dict):
                    # RemedyApplication requires an object with an indication; drop junk.
                    continue

                item2 = dict(item)

                # Coerce indication to a non-empty string; otherwise drop this entry.
                raw_indication = item2.get("indication")
                if raw_indication is None:
                    continue
                if isinstance(raw_indication, list):
                    raw_indication = ", ".join(str(x).strip() for x in raw_indication if str(x).strip())
                indication = str(raw_indication).strip()
                if not indication:
                    continue
                item2["indication"] = indication

                # Normalize optional-ish strings.
                if item2.get("usage_instructions") is None:
                    item2["usage_instructions"] = ""
                if isinstance(item2.get("ingredients"), list):
                    item2["ingredients"] = ", ".join(
                        str(x).strip() for x in item2["ingredients"] if str(x).strip()
                    )

                cleaned_ta.append(item2)
            d["therapeutic_applications"] = cleaned_ta

        # Some sources use `botanical_description` for the same content.
        # Keep only one canonical field in the schema to avoid duplication.
        bf = d.get("botanical_features")
        bd = d.get("botanical_description")
        if (not bf) and isinstance(bd, str) and bd.strip():
            d["botanical_features"] = bd
        # Drop the alias key so it doesn't leak into model_dump via other paths.
        if "botanical_description" in d:
            d.pop("botanical_description", None)

        p = d.get("properties")
        pd = d.get("properties_and_dosage")
        if (not p) and isinstance(pd, str) and pd.strip():
            d["properties"] = pd
        if (not pd) and isinstance(p, str) and p.strip():
            d["properties_and_dosage"] = p

        return d

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "plant_name": "Cây Bách xù",
                "other_names": ["Cốt tía"],
                "family": "Họ Bách",
                "parts_used": [
                    {"part": "Cành", "usage_description": "Cất tinh dầu"},
                    {"part": "Lá", "usage_description": "Dùng tươi"},
                    {"part": "Hạt", "usage_description": "Ép lấy dầu"},
                ],
                "botanical_features": "Cây gỗ cao 2-3m...",
                "properties": "Vị cay, tính ấm",
                "pharmacological_effects": ["thanh nhiệt"],
                "treats": ["ho gà", "mụn nhọt"],
                "images": [
                    {
                        "stored_path": "data/processed/images/cay-canh--cay-thuoc-trong-nha-truong_img_002.webp",
                        "sha256": "<sha256>",
                    }
                ],
            }
        },
    )




# ============================================================================
# SCHEMA 2: Remedies & Recipes (Bài thuốc, nước uống)
# ============================================================================

class RemedyRecipe(BaseModel):
    """Schema for beverage recipes and home remedies."""
    
    recipe_name: str = Field(
        ...,
        description="Tên bài thuốc (e.g., 'Nước Chanh muối')"
    )
    source_plant: str = Field(
        ...,
        description="Cây chính tạo bài thuốc này"
    )
    ingredients: List[str] = Field(
        ...,
        description="Danh sách nguyên liệu với định lượng"
    )
    preparation_steps: List[str] = Field(
        ...,
        description="Các bước thực hiện, sơ chế"
    )
    usage_instructions: str = Field(
        ...,
        description="Cách dùng, pha chế"
    )
    health_benefits: List[str] = Field(
        default_factory=list,
        description="Công dụng cụ thể"
    )
    indications: Optional[str] = Field(
        default=None,
        description="Chỉ định/đối tượng phù hợp hoặc mục đích dùng (nếu có)",
    )
    dosage: Optional[str] = Field(
        default=None,
        description="Liều dùng/tần suất (nếu có)",
    )
    contraindications_warnings: Optional[str] = Field(
        default=None,
        description="Chống chỉ định/cảnh báo (nếu có)",
    )

    images: List[ImageAsset] = Field(
        default_factory=list,
        description="Ảnh liên quan trong chunk (nếu có)",
    )


class DocumentContent(BaseModel):
    """Wrapper for document extraction."""
    plants: List[MedicinalPlant] = Field(
        default_factory=list,
        description="Danh sách các cây thuốc"
    )
    recipes: List[RemedyRecipe] = Field(
        default_factory=list,
        description="Danh sách các bài thuốc/nước uống"
    )


# ============================================================================
# SCHEMA 3: Medicinal Vegetables (Cây rau làm thuốc)
# ============================================================================

class MedicinalVegetable(BaseModel):
    """Schema for medicinal vegetables with culinary uses."""
    
    plant_name: str = Field(..., description="Tên chính của cây rau")
    other_names: List[str] = Field(default_factory=list, description="Tên gọi khác")
    scientific_name: Optional[str] = Field(None, description="Tên khoa học")
    family: Optional[str] = Field(None, description="Họ thực vật")
    botanical_description: str = Field(..., description="Mô tả đặc điểm")
    culinary_uses: str = Field(..., description="Cách dùng làm thực phẩm")
    medicinal_properties: str = Field(..., description="Tính vị và tác dụng dược lý")
    distribution_and_ecology: Optional[str] = Field(None, description="Nguồn gốc/phân bố/sinh thái")
    chemical_composition: Optional[str] = Field(None, description="Thành phần hoá học (nếu có)")
    parts_used: Optional[str] = Field(None, description="Bộ phận dùng (nếu có)")
    dosage: Optional[str] = Field(None, description="Liều dùng/cách dùng (nếu có)")
    contraindications_warnings: Optional[str] = Field(None, description="Chống chỉ định/cảnh báo (nếu có)")
    remedies: List[str] = Field(
        default_factory=list,
        description="Danh sách bài thuốc hoặc hướng dẫn chữa bệnh"
    )


    images: List[ImageAsset] = Field(
        default_factory=list,
        description="Ảnh liên quan trong chunk (nếu có)",
    )


class VegetableDocumentContent(BaseModel):
    """Root wrapper for vegetable documents."""
    vegetables: List[MedicinalVegetable] = Field(
        ...,
        description="Danh sách rau làm thuốc"
    )


# ============================================================================
# SCHEMA 4: Endocrine Disorders (Cây thuốc, vị thuốc phòng và chữa bệnh nội tiết)
# ============================================================================

class EndocrineSyndrome(BaseModel):
    """Schema for clinical syndromes (Thể bệnh) in TCM."""
    
    syndrome_name: str = Field(..., description="Tên thể bệnh")
    symptoms: str = Field(..., description="Triệu chứng lâm sàng")
    treatment_principle: str = Field(..., description="Pháp điều trị")
    prescribed_remedy: str = Field(..., description="Tên bài thuốc hoặc hướng dẫn")

    images: List[ImageAsset] = Field(
        default_factory=list,
        description="Ảnh liên quan trong chunk (nếu có)",
    )
class EndocrineDocumentContent(BaseModel):
    """Root wrapper for endocrine medicine documents."""
    syndromes: List[EndocrineSyndrome] = Field(
        default_factory=list,
        description="Danh sách thể bệnh và điều trị"
    )
    medicinal_plants: List[MedicinalPlant] = Field(
        default_factory=list,
        description="Danh sách cây thuốc nội tiết"
    )


# ============================================================================
# SCHEMA 5: Emergency & Toxicology (Cấp cứu và Chống độc)
# ============================================================================

class EmergencyProtocol(BaseModel):
    """Schema for medical emergency protocols and antidotes."""
    
    condition_name: str = Field(
        ...,
        description="Tên tình trạng cấp cứu (e.g., 'Rắn cắn')"
    )
    category: str = Field(
        ...,
        description="Phân loại: 'Sơ cứu', 'Hồi sức', hoặc 'Chống độc'"
    )
    clinical_signs: List[str] = Field(
        default_factory=list,
        description="Dấu hiệu, triệu chứng lâm sàng"
    )
    diagnostic_tests: Optional[str] = Field(
        None,
        description="Xét nghiệm cần thiết"
    )
    first_aid_steps: List[str] = Field(
        default_factory=list,
        description="Bước sơ cứu ban đầu"
    )
    professional_treatment: List[str] = Field(
        default_factory=list,
        description="Biện pháp điều trị chuyên sâu"
    )
    medications: List[str] = Field(
        default_factory=list,
        description="Danh sách thuốc điều trị"
    )
    specific_antidote: Optional[str] = Field(
        None,
        description="Thuốc giải độc đặc hiệu"
    )
    contraindications_warnings: Optional[str] = Field(
        None,
        description="Chống chỉ định hoặc cảnh báo"
    )

    prevention: Optional[str] = Field(
        None,
        description="Phòng ngừa/khuyến cáo an toàn (nếu có)",
    )

    images: List[ImageAsset] = Field(
        default_factory=list,
        description="Ảnh liên quan trong chunk (nếu có)",
    )

    @field_validator(
        "clinical_signs",
        "first_aid_steps",
        "professional_treatment",
        "medications",
        "images",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v: Any):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # Sometimes the LLM emits a single string for list fields.
        return [v]

    @field_validator(
        "diagnostic_tests",
        "specific_antidote",
        "contraindications_warnings",
        "prevention",
        mode="before",
    )
    @classmethod
    def _coerce_optional_strings(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        # LLM sometimes emits a list of strings for optional string fields.
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if str(x).strip()]
            return "\n".join(parts) if parts else None
        return str(v).strip() or None


class EmergencyDocumentContent(BaseModel):
    """Root wrapper for emergency documents."""
    protocols: List[EmergencyProtocol] = Field(
        ...,
        description="Danh sách các quy trình y tế cấp cứu"
    )


# ============================================================================
# UNIFIED DOCUMENT TYPE UNION (for flexible routing)
# ============================================================================

DocumentType = DocumentContent | VegetableDocumentContent | EndocrineDocumentContent | EmergencyDocumentContent
