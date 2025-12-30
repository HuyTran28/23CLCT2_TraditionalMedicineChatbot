from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Helper: structured info on what part is used and how
# ---------------------------------------------------------------------------
class PartUsage(BaseModel):
    part: str = Field(..., description="Bộ phận dùng")
    usage_description: Optional[str] = Field(
        default=None,
        description="Cách sử dụng hoặc ghi chú về dịch liệu"
    )
# ============================================================================
# SCHEMA 1: Medicinal Plants & Herbs (Cây cảnh - Cây thuốc)
# ============================================================================

class MedicinalPlant(BaseModel):
    """
    Core schema for extracting medicinal plant entities.
    
    Fields:
        plant_name: Primary name (e.g., "Cây Bách xù")
        other_names: Alternative names
        scientific_name: Latin name if available
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
    other_names: Optional[str] = Field(
        default=None,
        description="Tên gọi khác (một tên, nếu có)"
    )
    scientific_name: Optional[str] = Field(
        default=None,
        description="Tên khoa học (nếu có)"
    )
    botanical_features: str = Field(
        ...,
        description="Mô tả chi tiết: thân, lá, hoa, quả"
    )
    distribution_and_ecology: Optional[str] = Field(
        default=None,
        description="Nơi mọc, mùa hoa quả"
    )
    parts_used: List[PartUsage] = Field(
        default_factory=list,
        description="Danh sách bộ phận dùng kèm cách dùng cụ thể"
    )
    properties: str = Field(
        ...,
        description="Tính vị (e.g., 'Vị cay thơm, tính ấm')"
    )
    pharmacological_effects: List[str] = Field(
        default_factory=list,
        description="Tác dụng dược lý (e.g., ['thanh nhiệt', 'giải độc'])"
    )
    treats: List[str] = Field(
        ...,
        description="Danh sách bệnh/triệu chứng cụ thể"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "plant_name": "Cây Bách xù",
                "other_names": "Cốt tía",
                "parts_used": [
                    {"part": "Cành", "usage_description": "Cất tinh dầu"},
                    {"part": "Lá", "usage_description": "Dùng tươi"},
                    {"part": "Hạt", "usage_description": "Ép lấy dầu"}
                ],
                "botanical_features": "Cây gỗ cao 2-3m...",
                "properties": "Vị cay, tính ấm",
                "pharmacological_effects": ["thanh nhiệt"],
                "treats": ["ho gà", "mụn nhọt"]
            }
        }


