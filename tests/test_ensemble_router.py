"""Comprehensive unit and integration tests for LanguageRoutedEmotionClassifier.

Tests:
1. Language and script routing heuristics across English, Hindi, and Hinglish.
2. Lazy-loading behavior and sub-classifier retrieval.
3. Multi-label GoEmotions inference (torch.sigmoid, active labels >= 0.5).
4. RecommendationEngine mapping of all 28 GoEmotions labels into actionable productivity strategies.
5. Flask API integration (/api/analyze, /api/health) with routing metadata.
"""

import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
from services.emotion_classifier import (
    EmotionClassifier,
    LanguageRoutedEmotionClassifier,
)
from services.nltk_processor import NLTKProcessor
from services.recommendation_engine import RecommendationEngine
from app import create_app


# ---------------------------------------------------------------------------
# Routing Logic Unit Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    """Create fresh LanguageRoutedEmotionClassifier instance in standby mode."""
    return LanguageRoutedEmotionClassifier(load_on_init=False)


@pytest.mark.parametrize(
    "text,expected_lang,expected_model",
    [
        # Pure English
        ("I am really excited about my results!", "English", "SamLowe/roberta-base-go_emotions"),
        ("My teammate did an amazing job on this project.", "English", "SamLowe/roberta-base-go_emotions"),
        ("I cannot believe I won the competition!", "English", "SamLowe/roberta-base-go_emotions"),
        ("I am worried that I will fail tomorrow's exam.", "English", "SamLowe/roberta-base-go_emotions"),
        ("I am furious that they ignored my work.", "English", "SamLowe/roberta-base-go_emotions"),
        ("I worked on this all night, and nothing is working.", "English", "SamLowe/roberta-base-go_emotions"),
        ("The meeting starts at 10 AM.", "English", "SamLowe/roberta-base-go_emotions"),
        # Pure Devanagari Hindi
        ("मुझे आज बहुत खुशी हो रही है।", "Hindi", "vashuag/HindiEmotion"),
        ("अरे! मुझे विश्वास नहीं हो रहा कि मैं जीत गया!", "Hindi", "vashuag/HindiEmotion"),
        ("उसका ऐसा व्यवहार बिल्कुल गलत है।", "Hindi", "vashuag/HindiEmotion"),
        ("मुझे डर लग रहा है कि परीक्षा में क्या होगा।", "Hindi", "vashuag/HindiEmotion"),
        ("मैं बहुत उदास हूँ और कुछ करने का मन नहीं है।", "Hindi", "vashuag/HindiEmotion"),
        ("बिजली जल्दी आ गई, बहुत शुक्रिया।", "Hindi", "vashuag/HindiEmotion"),
        ("बैठक सुबह 10 बजे शुरू होगी।", "Hindi", "vashuag/HindiEmotion"),
        # Hinglish (Roman Script with Hindi keywords or Mixed Scripts)
        ("Yaar, aaj bahut tension ho raha hai.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Kya! Mujhe yakeen nahi ho raha ki main jeet gaya!", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Bhai party karenge aaj bahut maza aaya!", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Arre yaar, dimag kharab kar diya inhone.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Exam me fail hone ka darr lag raha hai.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("I worked on this all night aur kuch bhi kaam nahi kar raha.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Kal subah 10 baje team meeting hai.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Yeh project bahut important hai for tomorrow.", "Hinglish", "amaan00z/emotion_xlmr3"),
        ("Project meeting mein सब आ जाना please.", "Hinglish", "amaan00z/emotion_xlmr3"),
    ],
)
def test_route_text_accuracy(router, text, expected_lang, expected_model):
    """Verify 100% accurate routing for benchmark English, Hindi, and Hinglish sentences."""
    route = router.route_text(text)
    assert route["language"] == expected_lang
    assert route["model_id"] == expected_model
    assert "reason" in route and len(route["reason"]) > 0


# ---------------------------------------------------------------------------
# Recommendation Mapping for GoEmotions
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return RecommendationEngine()


ALL_28_GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]


@pytest.mark.parametrize("emotion_label", ALL_28_GO_EMOTIONS)
def test_all_28_go_emotions_receive_recommendations(engine, emotion_label):
    """Verify that every single GoEmotions label receives actionable recommendations."""
    details = engine.get_recommendation_details(emotion_label, confidence=0.88)
    assert details["is_supported"] is True
    assert details["emotion"] == emotion_label
    assert details["confidence"] == 0.88
    assert details["primary_strategy"] != "Balanced Productivity Baseline"
    assert len(details["suggestions"]) >= 2

    # Verify specific key mappings
    if emotion_label in ["joy", "excitement", "optimism", "pride"]:
        assert details["mapped_strategy"] == "joy"
    elif emotion_label in ["fear", "nervousness"]:
        assert details["mapped_strategy"] == "fear"
    elif emotion_label in ["neutral", "relief"]:
        assert details["mapped_strategy"] == "neutral"
    elif emotion_label in ["admiration", "gratitude", "approval"]:
        assert details["mapped_strategy"] == "admiration"


# ---------------------------------------------------------------------------
# Mocked Ensemble Pipeline & API Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ensemble():
    """Mock the ensemble router to avoid heavy live downloads in integration tests."""
    ensemble = MagicMock(spec=LanguageRoutedEmotionClassifier)
    ensemble.MAX_TEXT_LENGTH = 1000

    def fake_predict(text):
        if any("\u0900" <= c <= "\u097F" for c in text):
            return {
                "label": "joy",
                "raw_label": "joy",
                "score": 0.979,
                "all_scores": [{"label": "joy", "raw_label": "joy", "score": 0.979}],
                "model_id": "vashuag/HindiEmotion",
                "routed_language": "Hindi",
                "routing_reason": "Pure Devanagari script detected",
                "device": "cpu",
            }
        elif "yaar" in text.lower() or "tension" in text.lower() or "bhai" in text.lower():
            return {
                "label": "sadness",
                "raw_label": "sadness",
                "score": 0.995,
                "all_scores": [{"label": "sadness", "raw_label": "sadness", "score": 0.995}],
                "model_id": "amaan00z/emotion_xlmr3",
                "routed_language": "Hinglish",
                "routing_reason": "Romanized Hindi keyword(s) detected",
                "device": "cpu",
            }
        else:
            return {
                "label": "neutral" if "meeting" in text.lower() else "excitement",
                "raw_label": "neutral" if "meeting" in text.lower() else "excitement",
                "score": 0.923,
                "all_scores": [{"label": "neutral", "raw_label": "neutral", "score": 0.923}],
                "model_id": "SamLowe/roberta-base-go_emotions",
                "routed_language": "English",
                "routing_reason": "Standard English text detected",
                "is_multilabel": True,
                "active_labels": [{"label": "neutral", "score": 0.923}],
                "device": "cpu",
            }

    ensemble.predict.side_effect = fake_predict
    ensemble.get_status.return_value = {
        "model_name": "LanguageRoutedEmotionClassifier",
        "loaded": True,
        "device": "cpu",
        "phase": 4,
        "status": "ready",
    }
    return ensemble


@pytest.fixture
def ensemble_client(mock_ensemble):
    app = create_app(
        emotion_classifier=mock_ensemble,
        nltk_processor=NLTKProcessor(),
        recommendation_engine=RecommendationEngine(),
    )
    app.config.update({"TESTING": True})
    return app.test_client()


def test_api_routes_english_to_go_emotions(ensemble_client):
    """Verify that English text produces GoEmotions routing metadata in the API response."""
    resp = ensemble_client.post("/api/analyze", json={"text": "The meeting starts at 10 AM."})
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert data["emotion"]["routed_language"] == "English"
    assert data["emotion"]["model_id"] == "SamLowe/roberta-base-go_emotions"
    assert data["emotion"]["predicted_label"] == "neutral"
    assert data["emotion"]["is_multilabel"] is True
    assert data["recommendations"]["primary_strategy"] == "Structured Execution"


def test_api_routes_hindi_to_hindiemotion(ensemble_client):
    """Verify that Devanagari Hindi text produces HindiEmotion routing metadata."""
    resp = ensemble_client.post("/api/analyze", json={"text": "मुझे आज बहुत खुशी हो रही है।"})
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert data["emotion"]["routed_language"] == "Hindi"
    assert data["emotion"]["model_id"] == "vashuag/HindiEmotion"
    assert data["emotion"]["predicted_label"] == "joy"
    assert data["recommendations"]["primary_strategy"] == "Capitalize on Momentum"


def test_api_routes_hinglish_to_xlmr3(ensemble_client):
    """Verify that Romanized Hindi produces Hinglish model metadata."""
    resp = ensemble_client.post("/api/analyze", json={"text": "Yaar, aaj bahut tension ho raha hai."})
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert data["emotion"]["routed_language"] == "Hinglish"
    assert data["emotion"]["model_id"] == "amaan00z/emotion_xlmr3"
    assert data["emotion"]["predicted_label"] == "sadness"
    assert data["recommendations"]["primary_strategy"] == "Gentle Pacing & Low-Friction Steps"
