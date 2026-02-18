"""
Configuration constants for the bot detection module.
All parameters can be modified here for easy tuning without code changes.
"""

# Text input constraints
MIN_TEXT_COUNT = 10  # Minimum number of texts required

# Model configuration
MODEL_NAME = "data/mbert_trained"  # Path to the trained model
BOT_PROBABILITY_THRESHOLD = 0.5  # Threshold for binary classification (0-1)

# Resource management
MAX_CPU_CORES = 4  # Maximum number of CPU cores torch will use for inference
BATCH_SIZE = 32  # Batch size for model processing

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True
