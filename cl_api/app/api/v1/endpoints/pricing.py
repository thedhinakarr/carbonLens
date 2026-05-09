"""
Pricing endpoints — fair-value estimates for carbon credits.
"""

from fastapi import APIRouter, Query

from app.models.schemas import FairValueRequest, FairValueResponse

router = APIRouter()


@router.post("/fair-value", response_model=FairValueResponse)
async def get_fair_value(request: FairValueRequest):
    """
    Return an AI-generated fair-value estimate for a carbon credit.

    Takes project type, vintage, quality tier, and region as inputs
    and returns a fair-value estimate with confidence range and breakdown.
    """
    # TODO: wire up to ML model inference via cl_ml service
    return FairValueResponse(
        fair_value=12.40,
        currency="EUR",
        confidence_low=10.80,
        confidence_high=14.20,
        quality_tier="A",
        breakdown={
            "base_market_value": 68,
            "co_benefits_premium": 15,
            "vintage_adjustment": -5,
            "policy_sentiment": 12,
        },
    )


@router.get("/eu-ets")
async def get_eu_ets_prices(
    days: int = Query(default=30, le=365, description="Number of days of history"),
):
    """Return historical EU ETS price data."""
    # TODO: query from data pipeline
    return {"prices": [], "days_requested": days}
