"""Emotion Classifier Service.

This service performs emotion classification using specialized pretrained models
routed by language/script:
1. English: `SamLowe/roberta-base-go_emotions` (28 fine-grained labels + neutral, multi-label BCE head)
2. Hindi (Devanagari): `vashuag/HindiEmotion` (7 Hindi labels, single-label softmax head)
3. Hinglish (Code-Mixed): `amaan00z/emotion_xlmr3` (10 code-mixed labels, single-label softmax head)

Supports standalone single-model inference (for backward compatibility) as well
as language-routed ensemble inference.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


class EmotionClassifier:
    """Wrapper class for Hugging Face transformer emotion classification model."""

    MODEL_NAME = "amaan00z/emotion_xlmr3"
    MAX_TEXT_LENGTH = 1000  # Character limit to prevent excessive memory/compute usage

    def __init__(
        self,
        model_name: Optional[str] = None,
        load_on_init: bool = False,
        device: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        """Initialize the EmotionClassifier.

        Args:
            model_name: Specific Hugging Face model identifier (defaults to `amaan00z/emotion_xlmr3`).
            load_on_init: Whether to load tokenizer and model immediately.
            device: Explicit device string ('cpu', 'cuda', 'mps') or None for auto-detection.
            task_type: Optional 'multi_label_classification' or 'single_label_classification'.
        """
        self.MODEL_NAME = model_name or self.MODEL_NAME
        self.is_loaded: bool = False
        self.model = None
        self.tokenizer = None
        self.id2label: Dict[int, str] = {}
        self.device = device or self._detect_device()
        self.task_type = task_type

        if load_on_init:
            self.load_model()

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available compute device (MPS, CUDA, or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load_model(self) -> None:
        """Load tokenizer and model weights from Hugging Face hub.

        Loads only once per classifier instance. Reuses existing instances.
        """
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Loading tokenizer for model: %s", self.MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            logger.info("Loading model weights for model: %s", self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

            # Move model to target device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            # Determine task type
            if self.task_type is None:
                if "go_emotions" in self.MODEL_NAME.lower():
                    self.task_type = "multi_label_classification"
                else:
                    self.task_type = getattr(
                        self.model.config, "problem_type", "single_label_classification"
                    )

            # Retrieve actual configured label mapping from model configuration
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                self.id2label = {
                    int(k): str(v) for k, v in self.model.config.id2label.items()
                }
            else:
                num_labels = getattr(self.model.config, "num_labels", 0)
                self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

            self.is_loaded = True
            logger.info(
                "Model %s successfully loaded on device: %s (task: %s) with %d labels",
                self.MODEL_NAME,
                self.device,
                self.task_type,
                len(self.id2label),
            )
        except Exception as exc:
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            self.id2label = {}
            logger.error("Failed to load model %s: %s", self.MODEL_NAME, exc)
            raise RuntimeError(
                f"Failed to load model '{self.MODEL_NAME}'. Error: {exc}"
            ) from exc

    def validate_input(self, text: Any) -> str:
        """Validate input text before inference.

        Args:
            text: Input text candidate.

        Returns:
            Validated string with normalized whitespace.

        Raises:
            TypeError: If input is not a string.
            ValueError: If input is empty, whitespace-only, or exceeds character limit.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Invalid input type: expected str, got {type(text).__name__}."
            )

        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(stripped) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {self.MAX_TEXT_LENGTH} characters "
                f"(received {len(stripped)} characters)."
            )

        return stripped

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotion from input text using the pretrained transformer model.

        Args:
            text: Validated input string.

        Returns:
            Dict containing predicted emotion, scores, model metadata, and device.
        """
        clean_text = self.validate_input(text)

        # Lazy-load model if not already loaded
        if not self.is_loaded:
            self.load_model()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                f"Model '{self.MODEL_NAME}' is not loaded. Cannot run inference."
            )

        try:
            import torch

            # Tokenize input text (truncation enabled within model max sequence length)
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )

            # Move tensors to the target device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference in evaluation mode without gradient calculation
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                if self.task_type == "multi_label_classification":
                    # Multi-label BCE head: independent sigmoid per class
                    probabilities = torch.sigmoid(logits).squeeze(0)
                else:
                    # Single-label CrossEntropy head: normalized softmax across classes
                    probabilities = torch.softmax(logits, dim=-1).squeeze(0)

            # Extract scores for all labels
            all_scores: List[Dict[str, Any]] = []
            for idx, prob in enumerate(probabilities):
                raw_lbl = self.id2label.get(idx, f"LABEL_{idx}")
                score_val = float(prob.item())
                all_scores.append({
                    "label": raw_lbl.strip().lower(),
                    "raw_label": raw_lbl,
                    "score": round(score_val, 6),
                })

            # Sort scores descending by confidence
            all_scores.sort(key=lambda x: x["score"], reverse=True)

            top_prediction = all_scores[0]

            res_payload = {
                "label": top_prediction["label"],
                "raw_label": top_prediction["raw_label"],
                "score": top_prediction["score"],
                "all_scores": all_scores,
                "model_id": self.MODEL_NAME,
                "device": self.device,
            }

            if self.task_type == "multi_label_classification":
                res_payload["is_multilabel"] = True
                res_payload["active_labels"] = [
                    s for s in all_scores if s["score"] >= 0.5
                ]
                res_payload["conversion_method"] = "argmax(sigmoid(logits))"

            return res_payload

        except Exception as exc:
            logger.error("Inference failure for model %s: %s", self.MODEL_NAME, exc)
            raise RuntimeError(
                f"Inference failed on model '{self.MODEL_NAME}': {exc}"
            ) from exc

    def get_status(self) -> Dict[str, Any]:
        """Return readiness status and configuration of the classifier."""
        return {
            "model_name": self.MODEL_NAME,
            "loaded": self.is_loaded,
            "device": self.device,
            "task_type": self.task_type,
            "configured_labels": list(self.id2label.values()) if self.is_loaded else [],
            "max_text_length": self.MAX_TEXT_LENGTH,
            "phase": 4,
            "status": "ready" if self.is_loaded else "not_loaded",
        }


class LanguageRoutedEmotionClassifier(EmotionClassifier):
    """Ensemble Emotion Classifier routing input text to specialized models by language/script."""

    MODELS = {
        "English": {
            "model_id": "SamLowe/roberta-base-go_emotions",
            "task_type": "multi_label_classification",
            "description": "RoBERTa fine-tuned on GoEmotions (28 fine-grained labels including neutral)",
        },
        "Hindi": {
            "model_id": "vashuag/HindiEmotion",
            "task_type": "single_label_classification",
            "description": "IndicBERT fine-tuned for Devanagari Hindi (7 labels)",
        },
        "Hinglish": {
            "model_id": "amaan00z/emotion_xlmr3",
            "task_type": "single_label_classification",
            "description": "XLM-RoBERTa fine-tuned on code-mixed Hinglish (10 labels)",
        },
    }

    # Lexical markers for romanized Hindi (excluding common English homographs like 'the', 'to', 'me', 'so', 'is')
    HINGLISH_MARKERS: Set[str] = {
        "aaj", "kal", "parson", "kya", "kyun", "kaise", "kahan", "kab", "kaun", "kis", "kisko",
        "main", "mujhe", "mera", "meri", "mere", "hum", "humein", "hamara", "hamari", "hamare",
        "tu", "tum", "tumhe", "tumhara", "tumhari", "aap", "aapko", "aapka", "aapki",
        "yeh", "woh", "unhe", "unka", "unki", "unke", "iska", "iske", "iski", "inko", "isko",
        "hai", "hain", "hoon", "tha", "thi", "hoga", "hogi", "honge",
        "kar", "karo", "karna", "karke", "karta", "karti", "karte", "karenge", "karunga", "karungi",
        "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "aaya", "aayi", "aaye", "diya", "diye", "liya", "liye",
        "yaar", "bhai", "bhaiya", "didi", "arre", "achha", "accha", "shukriya",
        "bahut", "bohot", "jyada", "zyada", "thoda", "thodi", "kuch", "koi",
        "lekin", "magar", "toh", "bhi", "nahi", "nhi", "mat",
        "tension", "darr", "maza", "mazza", "yakeen", "dimag", "kharab",
        "baat", "kaam", "subah", "shaam", "pata", "lag", "laga", "lagi", "baje", "hona", "hone",
    }

    def __init__(self, load_on_init: bool = False, device: Optional[str] = None):
        """Initialize the ensemble router with lazy-loaded sub-classifiers."""
        super().__init__(model_name="language_routed_ensemble", load_on_init=False, device=device)
        self.sub_classifiers: Dict[str, EmotionClassifier] = {}
        if load_on_init:
            self.load_all_models()

    def get_sub_classifier(self, language: str) -> EmotionClassifier:
        """Get or lazily instantiate and load a sub-classifier for a target language."""
        if language not in self.MODELS:
            language = "English"

        if language not in self.sub_classifiers:
            model_info = self.MODELS[language]
            logger.info("Initializing sub-classifier for %s (%s)...", language, model_info["model_id"])
            classifier = EmotionClassifier(
                model_name=model_info["model_id"],
                device=self.device,
                task_type=model_info["task_type"],
                load_on_init=True,
            )
            self.sub_classifiers[language] = classifier

        return self.sub_classifiers[language]

    def load_all_models(self) -> None:
        """Explicitly load all 3 ensemble models."""
        for lang in self.MODELS:
            self.get_sub_classifier(lang)
        self.is_loaded = True

    def route_text(self, text: str) -> Dict[str, str]:
        """Determine language and corresponding model route based on script and lexical markers.

        Args:
            text: Validated input text.

        Returns:
            Dict containing 'language', 'model_id', and 'reason'.
        """
        has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
        has_latin = any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text)

        # 1. Pure Devanagari script -> Hindi
        if has_devanagari and not has_latin:
            return {
                "language": "Hindi",
                "model_id": self.MODELS["Hindi"]["model_id"],
                "reason": "Pure Devanagari script detected",
            }

        # 2. Mixed Devanagari & Latin script -> Hinglish
        if has_devanagari and has_latin:
            return {
                "language": "Hinglish",
                "model_id": self.MODELS["Hinglish"]["model_id"],
                "reason": "Mixed Devanagari and Latin script detected (code-mixed)",
            }

        # 3. Pure Latin script: Check for romanized Hindi vocabulary markers
        words = set(re.findall(r"[a-zA-Z]+", text.lower()))
        hinglish_overlap = words & self.HINGLISH_MARKERS

        if hinglish_overlap:
            sample_words = ", ".join(sorted(hinglish_overlap)[:3])
            return {
                "language": "Hinglish",
                "model_id": self.MODELS["Hinglish"]["model_id"],
                "reason": f"Romanized Hindi marker(s) detected: [{sample_words}]",
            }

        # 4. Standard Latin English
        return {
            "language": "English",
            "model_id": self.MODELS["English"]["model_id"],
            "reason": "Standard English text detected",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Route input text to the specialized model and return formatted prediction."""
        clean_text = self.validate_input(text)
        route = self.route_text(clean_text)
        target_lang = route["language"]

        # Delegate to specialized model
        sub_classifier = self.get_sub_classifier(target_lang)
        result = sub_classifier.predict(clean_text)

        # Augment with routing metadata
        result["routed_language"] = target_lang
        result["routing_reason"] = route["reason"]
        result["ensemble_router"] = "LanguageRoutedEmotionClassifier"
        result["available_routes"] = list(self.MODELS.keys())

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return readiness status and configuration of the routed ensemble."""
        loaded_langs = list(self.sub_classifiers.keys())
        return {
            "model_name": "LanguageRoutedEmotionClassifier",
            "loaded": len(loaded_langs) > 0,
            "device": self.device,
            "loaded_models": {
                lang: self.sub_classifiers[lang].MODEL_NAME for lang in loaded_langs
            },
            "ensemble_catalog": self.MODELS,
            "max_text_length": self.MAX_TEXT_LENGTH,
            "phase": 4,
            "status": "ready" if len(loaded_langs) > 0 else "standby",
        }
