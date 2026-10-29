"""EmoSync: Emotion-Aware Productivity Assistant.

Main Flask Application Entry Point.
Integration of Language-Routed Emotion Classifier, NLTK Processor,
Rule-based Recommendation Engine, and RESTful API endpoints.
"""

import os
import logging
from typing import Optional
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from services.emotion_classifier import EmotionClassifier, LanguageRoutedEmotionClassifier
from services.nltk_processor import NLTKProcessor
from services.recommendation_engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(
    emotion_classifier: Optional[EmotionClassifier] = None,
    nltk_processor: Optional[NLTKProcessor] = None,
    recommendation_engine: Optional[RecommendationEngine] = None,
) -> Flask:
    """Application factory for Flask app.

    Args:
        emotion_classifier: Optional custom or mocked EmotionClassifier instance.
        nltk_processor: Optional custom or mocked NLTKProcessor instance.
        recommendation_engine: Optional custom or mocked RecommendationEngine instance.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    CORS(app)

    # Initialize or inject modular service instances
    # Default: do not load heavy weights on initialization; lazy-load on first inference
    classifier = emotion_classifier or LanguageRoutedEmotionClassifier(load_on_init=False)
    processor = nltk_processor or NLTKProcessor()
    engine = recommendation_engine or RecommendationEngine()

    # Store references on app config for inspection and testing
    app.services = {
        "emotion_classifier": classifier,
        "nltk_processor": processor,
        "recommendation_engine": engine,
    }

    @app.route("/", methods=["GET"])
    def index():
        """Render the frontend homepage dashboard."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint to verify backend status and service availability."""
        return jsonify({
            "status": "healthy",
            "system_status": "Operational",
            "phase": 4,
            "project": "EmoSync — Emotion-Aware Productivity Assistant",
            "application": "EmoSync — Emotion-Aware Productivity Assistant",
            "services": {
                "emotion_classifier": classifier.get_status(),
                "nltk_processor": processor.get_status(),
                "recommendation_engine": engine.get_status(),
            },
        }), 200

    @app.route("/api/analyze", methods=["POST"])
    def analyze_text():
        """Analyze text endpoint.

        Performs:
          1. JSON input validation.
          2. Emotion classification on original unmodified text.
          3. Linguistic and token statistics via NLTK.
          4. Rule-based task recommendations based on detected emotion.

        Returns:
            JSON response with emotion, text statistics, and recommendations.
        """
        # 1. Validate JSON presence
        if not request.is_json:
            return jsonify({
                "status": "error",
                "error": "Request body must be valid JSON with 'Content-Type: application/json'.",
            }), 400

        data = request.get_json(silent=True)
        if data is None or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "error": "Malformed JSON payload in request body.",
            }), 400

        # 2. Validate 'text' field existence
        if "text" not in data:
            return jsonify({
                "status": "error",
                "error": "Missing required field 'text' in JSON payload.",
            }), 400

        raw_text = data["text"]

        # 3. Validate 'text' type
        if not isinstance(raw_text, str):
            return jsonify({
                "status": "error",
                "error": f"Invalid 'text' field type: expected string, got {type(raw_text).__name__}.",
            }), 400

        # 4. Validate non-empty / non-whitespace
        stripped_text = raw_text.strip()
        if not stripped_text:
            return jsonify({
                "status": "error",
                "error": "Input text cannot be empty or contain only whitespace.",
            }), 400

        # 5. Validate length against classifier limit
        max_length = getattr(classifier, "MAX_TEXT_LENGTH", 1000)
        if len(stripped_text) > max_length:
            return jsonify({
                "status": "error",
                "error": f"Input text exceeds maximum allowed length of {max_length} characters.",
            }), 400

        try:
            # 6. Step A: Emotion classification (unmodified original text)
            prediction = classifier.predict(raw_text)

            # 7. Step B: NLTK text processing and metrics
            text_stats = processor.analyze_text(raw_text)

            # 8. Step C: Generate productivity recommendations
            detected_label = prediction.get("label", "neutral")
            confidence_score = prediction.get("score", 0.0)
            recommendations_data = engine.get_recommendation_details(
                emotion=detected_label,
                confidence=confidence_score,
            )

            # 9. Format structured response
            response_payload = {
                "status": "success",
                "data": {
                    "original_text": raw_text,
                    "emotion": {
                        "predicted_label": detected_label,
                        "raw_label": prediction.get("raw_label", detected_label),
                        "score": round(confidence_score, 4),
                        "ranked_scores": prediction.get("all_scores", []),
                        "model_id": prediction.get("model_id", "amaan00z/emotion_xlmr3"),
                        "routed_language": prediction.get("routed_language", text_stats.get("detected_script", "Unknown")),
                        "routing_reason": prediction.get("routing_reason", "Auto-routed by language and script"),
                        "is_multilabel": prediction.get("is_multilabel", False),
                        "active_labels": prediction.get("active_labels", []),
                        "device": prediction.get("device", "cpu"),
                    },
                    "text_statistics": {
                        "tokens": text_stats.get("tokens", []),
                        "token_count": text_stats.get("token_count", 0),
                        "words": text_stats.get("words", []),
                        "word_count": text_stats.get("word_count", 0),
                        "char_count": text_stats.get("char_count", 0),
                        "char_count_no_spaces": text_stats.get("char_count_no_spaces", 0),
                        "sentence_count": text_stats.get("sentence_count", 1),
                        "avg_word_length": text_stats.get("avg_word_length", 0.0),
                        "vocabulary_size": text_stats.get("vocabulary_size", 0),
                        "top_words": text_stats.get("top_words", []),
                        "detected_script": text_stats.get("detected_script", "unknown"),
                        "language_hint": text_stats.get("language_hint", "Other"),
                    },
                    "recommendations": {
                        "emotion": recommendations_data.get("emotion", detected_label),
                        "is_supported": recommendations_data.get("is_supported", True),
                        "mapped_strategy": recommendations_data.get("mapped_strategy", detected_label),
                        "primary_strategy": recommendations_data.get("primary_strategy", ""),
                        "summary": recommendations_data.get("summary", ""),
                        "suggestions": recommendations_data.get("suggestions", []),
                    },
                },
            }

            return jsonify(response_payload), 200

        except Exception as exc:
            logger.error("Error processing analysis request: %s", exc, exc_info=True)
            return jsonify({
                "status": "error",
                "error": "An error occurred while processing the text analysis request.",
            }), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
