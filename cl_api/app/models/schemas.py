"""
Pydantic schemas for API request / response validation.
"""

from pydantic import BaseModel, Field


class FairValueRequest(BaseModel):
    """Input for a fair-value estimate."""

    project_type: str = Field(
        ...,
        description="Carbon credit project type",
        examples=["REDD+", "Renewable Energy", "Cookstoves", "Methane Avoidance"],
    )
    vintage_year: int = Field(
        ..., ge=2000, le=2030, description="Credit vintage year"
    )
    quality_tier: str | None = Field(
        default=None, description="Quality tier filter (A, B, or C)"
    )
    region: str | None = Field(
        default=None, description="Project region", examples=["Brazil", "India", "Kenya"]
    )


class FairValueResponse(BaseModel):
    """Output of the AI fair-value engine."""

    fair_value: float = Field(..., description="Estimated fair value in EUR per credit")
    currency: str = "EUR"
    confidence_low: float = Field(..., description="Lower bound of confidence range")
    confidence_high: float = Field(..., description="Upper bound of confidence range")
    quality_tier: str = Field(..., description="Quality tier: A, B, or C")
    breakdown: dict[str, float] = Field(
        ...,
        description="Percentage breakdown of price components",
    )
