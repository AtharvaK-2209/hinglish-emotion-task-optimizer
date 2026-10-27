"""AI-Powered Task Optimizer: Text-Based Hinglish Emotion Analysis.

Main Flask Application Entry Point.
Phase 1: Architecture setup, routing, and modular service scaffolding.
"""

import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS

from services.emotion_classifier import EmotionClassifier
from services.nltk_processor import NLTKProcessor
from services.recommendation_engine import RecommendationEngine


def create_app() -> Flask:
    """Application factory for Flask app."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates"
    )
    CORS(app)

    # Initialize modular service placeholders (without loading heavy DL weights)
    emotion_classifier = EmotionClassifier(load_on_init=False)
    nltk_processor = NLTKProcessor()
    recommendation_engine = RecommendationEngine()

    @app.route("/", methods=["GET"])
    def index():
        """Render the frontend homepage."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint to verify backend status and service separation."""
        return jsonify({
            "status": "healthy",
            "phase": 1,
            "project": "AI-Powered Task Optimizer: Text-Based Hinglish Emotion Analysis",
            "services": {
                "emotion_classifier": emotion_classifier.get_status(),
                "nltk_processor": nltk_processor.get_status(),
                "recommendation_engine": recommendation_engine.get_status()
            }
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
