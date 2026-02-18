"""
Custom exceptions for the bot detection module.
"""


class BotDetectionException(Exception):
    """Base exception for bot detection module."""
    pass


class InvalidInputError(BotDetectionException):
    """Raised when input validation fails (too few/many texts)."""
    pass


class ModelLoadError(BotDetectionException):
    """Raised when model loading fails."""
    pass


class InferenceError(BotDetectionException):
    """Raised when model inference fails."""
    pass
