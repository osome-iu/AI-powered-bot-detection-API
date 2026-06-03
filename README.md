# Bot Detection API

A BERT-based AI-powered bot detection service built with FastAPI. This API analyzes lists of texts and predicts whether they are from a AI-bot or human user.

## Features

- **BERT-based inference**: Uses finetuned mBERT model from HuggingFace for classification
- **Mean probability prediction**: Aggregates individual text scores and applies a configurable threshold
- **Resource management**: Limits CPU core usage for consistent performance
- **Input validation**: Configurable minimum text count for predictions
- **Interactive API docs**: Automatically generated Swagger UI for easy testing

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd AI-powered-bot-detection-API
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All configuration parameters are defined in `config/constants.py`:

- `MIN_TEXT_COUNT` (default: 20): Minimum number of texts required for prediction
- `BOT_PROBABILITY_THRESHOLD` (default: 0.4): Threshold for binary classification (0-1)
- `MODEL_NAME` (default: `trokhymovych/mbert-ai-bot-detector`): Hugging Face repo id or local model path
- `MAX_CPU_CORES` (default: 4): Maximum CPU cores torch will use
- `BATCH_SIZE` (default: 32): Batch size for model processing

Edit these values to customize the API behavior.

The default model is downloaded automatically by Transformers on first startup:

```bash
python main.py
```

You can also override the model at runtime with an environment variable:

```bash
MODEL_NAME=trokhymovych/mbert-ai-bot-detector python main.py
```

To pre-download the model from Hugging Face for offline or repeat deployments:

```bash
huggingface-cli download trokhymovych/mbert-ai-bot-detector --local-dir models/mbert-ai-bot-detector
MODEL_NAME=models/mbert-ai-bot-detector python main.py
```

If `MODEL_NAME` points to a local path (for example `data/mbert_trained`), that directory must exist.

## Running the API

Start the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

- Swagger UI: http://localhost:8000/docs

## API Endpoints

### POST `/predict`

Predict if texts belong to a AI-bot.

**Request:**
```json
{
  "texts": [
    "This is the first text.",
    "This is the second text.",
    "..."
  ]
}
```

**Response:**
```json
{
  "is_bot": false,
  "confidence": 0.2345,
  "text_scores": [0.1234, 0.2456, ...],
  "num_texts": 3
}
```

**Parameters:**
- `texts` (required): List of texts to analyze (minimum 20 texts required)
- `is_bot` (boolean): Binary prediction (True = bot, False = human)
- `confidence` (float): Mean probability score (0-1)
- `text_scores` (array): Individual scores for each text provided
- `num_texts` (integer): Number of texts processed

**Error Responses:**
- `400 Bad Request`: Too few texts (< 20)
- `422 Unprocessable Entity`: Invalid input format
- `500 Internal Server Error`: Model inference failed

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET `/model-info`

Get current model information.

**Response:**
```json
{
  "model_name": "<>",
  "threshold": 0.4,
}
```

## Project Structure

```
/
├── config/
│   ├── __init__.py
│   └── constants.py              # Configuration parameters
├── models/
│   ├── __init__.py
│   └── bot_detector.py           # Core BERT inference logic
├── api/
│   ├── __init__.py
│   └── data_models.py              # Pydantic models for request/response validation
├── utils/
│   ├── __init__.py
│   └── exceptions.py             # Custom exceptions
├── main.py                       # FastAPI app and endpoint definitions
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## How It Works

1. **Input Validation**: Texts are validated (minimum 20 required)
2. **Inference**: Model runs inference on batches of texts
3. **Probability Extraction**: Positive class probabilities are extracted
4. **Aggregation**: Mean probability is calculated across all texts
5. **Classification**: Binary label is determined using configurable threshold
6. **Response**: Results are returned with individual scores and aggregated prediction

## Resource Management

The API limits CPU core usage using PyTorch's `torch.set_num_threads()` to the value specified in `MAX_CPU_CORES`. This prevents resource exhaustion when running multiple inference requests.

## Performance Considerations

- **Batch Processing**: Texts are processed in batches (default: 32) for efficiency
- **CPU Limiting**: Torch is restricted to use only `MAX_CPU_CORES` cores
- **Lazy Loading**: Model is loaded on server startup for faster response times

## Troubleshooting

### Out of Memory
If you encounter memory issues:
- Reduce `BATCH_SIZE` in constants.py
- Reduce `MAX_CPU_CORES` if running multiple instances

### Slow Inference
- Increase `BATCH_SIZE` for better throughput
- Ensure `MAX_CPU_CORES` is set appropriately for your hardware
