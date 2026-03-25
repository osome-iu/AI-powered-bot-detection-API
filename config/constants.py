"""
Configuration constants for the bot detection module.
All parameters can be modified here for easy tuning without code changes.
"""
import os

# Text input constraints
MIN_TEXT_COUNT = 20  # Minimum number of texts required

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "models/mbert_trained")  # Local model path or Hugging Face repo id
BOT_PROBABILITY_THRESHOLD = 0.5  # Threshold for binary classification (0-1)

# Resource management
MAX_CPU_CORES = 4  # Maximum number of CPU cores torch will use for inference
BATCH_SIZE = 32  # Batch size for model processing

# API configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
EXPOSE_DOCS = os.getenv("EXPOSE_DOCS", "false").lower() == "true"
EXPOSE_MODEL_INFO = os.getenv("EXPOSE_MODEL_INFO", "false").lower() == "true"
