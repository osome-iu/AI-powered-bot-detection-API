"""
Core bot detection model using BERT-based HuggingFace models.
"""
import logging
import os
from pathlib import Path
import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from config.constants import (
    MIN_TEXT_COUNT,
    MODEL_NAME,
    BOT_PROBABILITY_THRESHOLD,
    MAX_CPU_CORES,
    BATCH_SIZE,
)
from utils.exceptions import InvalidInputError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)


class BotDetector:
    """
    Bot detection model using BERT-based inference.
    Takes a list of texts and returns bot/not-bot prediction.
    """

    def __init__(self, model_name: str = MODEL_NAME, threshold: float = BOT_PROBABILITY_THRESHOLD):
        """
        Initialize the BotDetector with a HuggingFace model.

        Args:
            model_name: HuggingFace model name or local path
            threshold: Probability threshold for binary classification (0-1)
        """
        self.model_name = model_name
        self.threshold = threshold

        # Set CPU core limit using torch
        torch.set_num_threads(MAX_CPU_CORES)

        try:
            self._validate_model_name(model_name)
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(f"Could not load model {model_name}: {str(e)}")

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        """Raise a clear error when a local model path is configured but missing."""
        model_path = Path(os.path.expanduser(model_name))
        path_hint = (
            model_name.startswith((".", "/", "~"))
            or "\\" in model_name
            or (os.sep in model_name and (model_path.exists() or model_path.parent.exists()))
        )
        if path_hint and not model_path.exists():
            raise ModelLoadError(
                f"Configured model path '{model_name}' does not exist. "
                "Provide a valid local model directory or set MODEL_NAME to a Hugging Face repo id."
            )

    def validate_input(self, texts: List[str]) -> Dict[str, Any]:
        """
        Validate input texts according to constraints.

        Args:
            texts: List of input texts

        Returns:
            Dictionary with validation info including warning message if applicable

        Raises:
            InvalidInputError: If validation fails
        """
        if not isinstance(texts, list):
            raise InvalidInputError("Input must be a list of texts")

        if len(texts) < MIN_TEXT_COUNT:
            raise InvalidInputError(
                f"Minimum {MIN_TEXT_COUNT} texts required, got {len(texts)}"
            )

        return {"is_valid": True}

    def infer(self, texts: List[str]) -> List[float]:
        """
        Run inference on texts and return probability scores.

        Args:
            texts: List of input texts

        Returns:
            List of probability scores (0-1) for positive class (bot)

        Raises:
            InferenceError: If inference fails
        """
        try:
            scores = []

            # Process texts in batches
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i : i + BATCH_SIZE]
                # Run inference using pipeline
                results = self.pipeline(batch_texts)
                # Extract positive class scores
                for result in results:
                    # results return list of dicts with 'label' and 'score'
                    positive_score = next(
                        (item['score'] for item in result if item['label'] in ["LABEL_1"]),
                        result[1]['score'] if len(result) > 1 else result[0]['score']
                    )
                    scores.append(positive_score)
            return scores
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise InferenceError(f"Model inference failed: {str(e)}")

    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Predict if texts belong to a bot based on mean probability.

        Args:
            texts: List of input texts

        Returns:
            Dictionary with:
                - is_bot: Binary prediction (True/False)
                - confidence: Mean probability score (0-1)
                - scores: Individual scores for each text
                - num_texts: Number of texts processed
                - warning: Warning message if applicable
        """
        # Validate input
        self.validate_input(texts)

        # Run inference
        scores = self.infer(texts)

        # Calculate mean confidence
        mean_confidence = sum(scores) / len(scores)

        # Apply threshold for binary prediction
        is_bot = mean_confidence >= self.threshold

        result = {
            "is_bot": is_bot,
            "confidence": round(mean_confidence, 4),
            "text_scores": [round(score, 4) for score in scores],
            "num_texts": len(texts),
        }

        return result
