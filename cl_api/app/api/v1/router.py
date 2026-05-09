"""
V1 API router — aggregates all endpoint modules.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import pricing

api_router = APIRouter()

api_router.include_router(pricing.router, prefix="/pricing", tags=["pricing"])
