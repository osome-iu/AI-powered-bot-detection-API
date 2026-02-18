"""
FastAPI application entry point for bot detection API.
"""
import logging
from fastapi import FastAPI, HTTPException
from models.bot_detector import BotDetector
from utils.exceptions import InvalidInputError, InferenceError
from api.data_models import PredictionRequest, PredictionResponse, HealthResponse, ModelInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Bot Detection API",
    description="BERT-based bot detection service",
    version="1.0.0",
)

# Global bot detector instance (lazy loaded)
bot_detector: BotDetector = None


@app.on_event("startup")
async def startup_event():
    """Initialize bot detector on app startup."""
    global bot_detector
    try:
        logger.info("Initializing BotDetector...")
        bot_detector = BotDetector()
        logger.info("BotDetector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BotDetector: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on app shutdown."""
    global bot_detector
    bot_detector = None
    logger.info("BotDetector cleaned up")

# Defining API endpoints:
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict if texts belong to a bot.

    Args:
        request: PredictionRequest containing list of texts

    Returns:
        PredictionResponse with prediction, confidence, and individual scores

    Raises:
        HTTPException: If prediction fails
    """
    try:
        if bot_detector is None:
            raise HTTPException(status_code=500, detail="Bot detector not initialized")

        result = bot_detector.predict(request.texts)
        return PredictionResponse(**result)

    except InvalidInputError as e:
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with status and model loaded status
    """
    model_loaded = bot_detector is not None
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """
    Get current model information.

    Returns:
        ModelInfo with model name, threshold, and device

    Raises:
        HTTPException: If model not initialized
    """
    if bot_detector is None:
        raise HTTPException(status_code=500, detail="Bot detector not initialized")

    return ModelInfo(
        model_name=bot_detector.model_name,
        threshold=bot_detector.threshold
    )


if __name__ == "__main__":
    import uvicorn
    from config.constants import API_HOST, API_PORT, API_RELOAD
    if API_RELOAD:
        uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
    else:
        uvicorn.run(app, host=API_HOST, port=API_PORT, reload=API_RELOAD)
