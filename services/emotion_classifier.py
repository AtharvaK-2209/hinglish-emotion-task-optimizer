"""Emotion Classifier Service.

This service is responsible for emotion classification from multilingual/Hinglish text
using the pretrained model `amaan00z/emotion_xlmr3`.

Note for Phase 1:
Model download, loading, and inference are deferred to Phase 2 as per project requirements.
"""

from typing import Dict, Any, Optional


class EmotionClassifier:
    """Wrapper class for Hugging Face transformer emotion classification model."""

    MODEL_NAME = "amaan00z/emotion_xlmr3"

    def __init__(self, load_on_init: bool = False):
        """Initialize the EmotionClassifier placeholder.

        Args:
            load_on_init: Flag indicating whether model weights should be loaded.
                          Set to False during Phase 1.
        """
        self.is_loaded: bool = False
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        if load_on_init:
            self.load_model()

    def load_model(self) -> None:
        """Load tokenizer and model from Hugging Face hub (Deferred to Phase 2)."""
        # Model loading will be implemented in Phase 2
        raise NotImplementedError(
            f"Model '{self.MODEL_NAME}' loading is deferred to Phase 2."
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotion from input text (Deferred to Phase 2).

        Args:
            text: Input string in English, Hindi, or Hinglish.

        Returns:
            Dict containing predicted emotion and confidence scores.
        """
        raise NotImplementedError(
            "Emotion prediction inference is deferred to Phase 2."
        )

    def get_status(self) -> Dict[str, Any]:
        """Return readiness status of the classifier."""
        return {
            "model_name": self.MODEL_NAME,
            "loaded": self.is_loaded,
            "phase": 1,
            "status": "ready_for_phase_2_integration"
        }
