"""Unit tests for Phase 1 minimal Flask application and modular services."""

import pytest
from app import create_app
from services.emotion_classifier import EmotionClassifier
from services.nltk_processor import NLTKProcessor
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def app():
    """Create and configure a Flask app instance for testing."""
    test_app = create_app()
    test_app.config.update({"TESTING": True})
    return test_app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_index_route(client):
    """Verify that root route '/' renders the landing page with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"EmoSync" in response.data


def test_health_check_route(client):
    """Verify that '/api/health' returns JSON with healthy status and service entries."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["phase"] in [1, 4]
    assert "services" in data
    assert "emotion_classifier" in data["services"]
    assert "nltk_processor" in data["services"]
    assert "recommendation_engine" in data["services"]


def test_emotion_classifier_unloaded_state():
    """Verify EmotionClassifier initializes without loading heavy transformer weights when load_on_init=False."""
    classifier = EmotionClassifier(load_on_init=False)
    status = classifier.get_status()
    assert status["loaded"] is False
    assert status["model_name"] == "amaan00z/emotion_xlmr3"

    # Fast offline input validation tests (does not trigger model download)
    with pytest.raises(ValueError):
        classifier.validate_input("")

    with pytest.raises(ValueError):
        classifier.validate_input("   ")

    with pytest.raises(TypeError):
        classifier.validate_input(12345)


def test_nltk_processor_phase1_state():
    """Verify NLTKProcessor initializes and exposes status."""
    processor = NLTKProcessor()
    status = processor.get_status()
    assert status["processor"] == "NLTK"
    assert status["ready"] is True


def test_recommendation_engine_phase1_state():
    """Verify RecommendationEngine initializes and exposes status."""
    engine = RecommendationEngine()
    status = engine.get_status()
    assert status["engine"] == "RecommendationEngine"
    assert status["ready"] is True
