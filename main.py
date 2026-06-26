"""
FastAPI application entry point for bot detection API.
"""
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from models.bot_detector import BotDetector
from utils.exceptions import InvalidInputError, InferenceError, ModelLoadError
from api.data_models import PredictionRequest, PredictionResponse, HealthResponse, ModelInfo
from config.constants import (
    API_HOST,
    API_KEY,
    API_KEY_HEADER,
    API_PORT,
    API_RELOAD,
    EXPOSE_DOCS,
    EXPOSE_MODEL_INFO,
    REQUIRE_API_KEY,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bot detector instance (lazy loaded)
bot_detector: Optional[BotDetector] = None


def _docs_url(path: str) -> Optional[str]:
    """Disable interactive docs by default in production-style deployments."""
    return path if EXPOSE_DOCS else None


async def require_api_key(x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER)) -> None:
    """Require a static API key when enabled."""
    if not REQUIRE_API_KEY:
        return

    if not API_KEY:
        logger.error("API key protection is enabled but API_KEY is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service authentication is not configured",
        )

    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def _initialize_bot_detector() -> None:
    """Attempt to initialize the global bot detector."""
    global bot_detector
    if bot_detector is not None:
        return

    try:
        logger.info("Initializing BotDetector...")
        bot_detector = BotDetector()
        logger.info("BotDetector initialized successfully")
    except ModelLoadError as e:
        # Keep API process alive; prediction endpoint reports this explicitly.
        bot_detector = None
        logger.error("BotDetector initialization failed: %s", str(e))


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application startup/shutdown without deprecated on_event hooks."""
    _initialize_bot_detector()
    yield
    global bot_detector
    bot_detector = None
    logger.info("BotDetector cleaned up")


# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Bot Detection API",
    description="BERT-based bot detection service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=_docs_url("/docs"),
    redoc_url=_docs_url("/redoc"),
    openapi_url=_docs_url("/openapi.json"),
)

# Defining API endpoints:
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    _: None = Depends(require_api_key),
) -> PredictionResponse:
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
            _initialize_bot_detector()
        if bot_detector is None:
            raise HTTPException(
                status_code=500,
                detail="Bot detector model is not available",
            )

        result = bot_detector.predict(request.texts)
        return PredictionResponse(**result)

    except InvalidInputError as e:
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceError as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=HealthResponse)
async def health_check(_: None = Depends(require_api_key)) -> HealthResponse:
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
async def get_model_info(_: None = Depends(require_api_key)) -> ModelInfo:
    """
    Get current model information.

    Returns:
        ModelInfo with model name, threshold, and device

    Raises:
        HTTPException: If model not initialized
    """
    if not EXPOSE_MODEL_INFO:
        raise HTTPException(status_code=404, detail="Not found")

    if bot_detector is None:
        raise HTTPException(status_code=500, detail="Bot detector not initialized")

    return ModelInfo(
        model_name=bot_detector.model_name,
        threshold=bot_detector.threshold
    )


if __name__ == "__main__":
    import uvicorn
    if API_RELOAD:
        uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
    else:
        uvicorn.run(app, host=API_HOST, port=API_PORT, reload=API_RELOAD)
