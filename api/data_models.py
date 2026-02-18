"""
FastAPI endpoints for bot detection API.
"""
import logging
from pydantic import BaseModel, Field
from typing import List


logger = logging.getLogger(__name__)


# Request and Response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    is_bot: bool = Field(..., description="Binary prediction: True if bot, False if human")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    text_scores: List[float] = Field(..., description="Individual scores for each text")
    num_texts: int = Field(..., description="Number of texts processed")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    threshold: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
