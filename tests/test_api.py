"""Unit and integration tests for Phase 4 Flask API (/api/analyze).

Tests input validation, mock model inference, NLTK metrics integration,
recommendation generation, and error handling.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from services.emotion_classifier import EmotionClassifier
from services.nltk_processor import NLTKProcessor
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def mock_classifier():
    """Create a mocked EmotionClassifier that returns deterministic predictions."""
    classifier = MagicMock(spec=EmotionClassifier)
    classifier.MAX_TEXT_LENGTH = 1000
    classifier.is_loaded = True
    classifier.get_status.return_value = {
        "model_name": "amaan00z/emotion_xlmr3",
        "loaded": True,
        "device": "cpu",
        "configured_labels": RecommendationEngine.SUPPORTED_LABELS,
        "phase": 4,
        "status": "ready",
    }

    def fake_predict(text):
        # Deterministic predictions based on text content
        if "खुशी" in text or "excited" in text:
            label = "joy"
            score = 0.999
        elif "tension" in text or "sad" in text:
            label = "sadness"
            score = 0.985
        elif "gussa" in text or "angry" in text:
            label = "anger"
            score = 0.970
        else:
            label = "neutral"
            score = 0.850

        return {
            "label": label,
            "raw_label": label,
            "score": score,
            "all_scores": [
                {"label": label, "raw_label": label, "score": score},
                {"label": "neutral" if label != "neutral" else "joy", "raw_label": "other", "score": round(1 - score, 4)},
            ],
            "model_id": "amaan00z/emotion_xlmr3",
            "device": "cpu",
        }

    classifier.predict.side_effect = fake_predict
    return classifier


@pytest.fixture
def client(mock_classifier):
    """Create a Flask test client with the mocked classifier and real NLTK/recommendation engines."""
    test_app = create_app(
        emotion_classifier=mock_classifier,
        nltk_processor=NLTKProcessor(),
        recommendation_engine=RecommendationEngine(),
    )
    test_app.config.update({"TESTING": True})
    return test_app.test_client()


# ---------------------------------------------------------------------------
# Successful API Analysis Tests
# ---------------------------------------------------------------------------

def test_analyze_english_sentence(client, mock_classifier):
    """Verify analysis of a normal English sentence."""
    text = "I am really excited about my results!"
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    assert response.is_json
    payload = response.get_json()

    assert payload["status"] == "success"
    data = payload["data"]

    # Verify original text was passed unmodified
    assert data["original_text"] == text
    mock_classifier.predict.assert_called_with(text)

    # Verify emotion prediction
    assert data["emotion"]["predicted_label"] == "joy"
    assert data["emotion"]["score"] > 0.9
    assert isinstance(data["emotion"]["ranked_scores"], list)

    # Verify NLTK statistics
    assert data["text_statistics"]["word_count"] == 7
    assert data["text_statistics"]["token_count"] == 8
    assert data["text_statistics"]["detected_script"] == "latin"
    assert "English" in data["text_statistics"]["language_hint"]

    # Verify recommendations
    assert data["recommendations"]["emotion"] == "joy"
    assert data["recommendations"]["is_supported"] is True
    assert len(data["recommendations"]["suggestions"]) >= 2


def test_analyze_hindi_devanagari_sentence(client, mock_classifier):
    """Verify analysis of a Hindi sentence written in Devanagari script."""
    text = "मुझे आज बहुत खुशी हो रही है।"
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    payload = response.get_json()
    data = payload["data"]

    assert data["original_text"] == text
    assert data["emotion"]["predicted_label"] == "joy"
    assert data["text_statistics"]["detected_script"] == "devanagari"
    assert "Hindi" in data["text_statistics"]["language_hint"]
    assert data["text_statistics"]["word_count"] == 7


def test_analyze_hinglish_sentence(client, mock_classifier):
    """Verify analysis of a Hinglish sentence in Roman script."""
    text = "Yaar, aaj bahut tension ho raha hai."
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    payload = response.get_json()
    data = payload["data"]

    assert data["original_text"] == text
    assert data["emotion"]["predicted_label"] == "sadness"
    assert data["recommendations"]["emotion"] == "sadness"
    assert "Micro-Steps" in [s["title"] for s in data["recommendations"]["suggestions"]][0] or len(data["recommendations"]["suggestions"]) > 0


# ---------------------------------------------------------------------------
# Validation & Error Handling Tests
# ---------------------------------------------------------------------------

def test_analyze_non_json_request(client):
    """Verify 400 error when request is not JSON."""
    response = client.post("/api/analyze", data="Plain text string", content_type="text/plain")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_missing_text_key(client):
    """Verify 400 error when 'text' field is omitted."""
    response = client.post("/api/analyze", json={"message": "hello"})
    assert response.status_code == 400
    assert "Missing required field 'text'" in response.get_json()["error"]


@pytest.mark.parametrize("invalid_text", ["", "   ", " \n\t  "])
def test_analyze_empty_and_whitespace_text(client, invalid_text):
    """Verify 400 error on empty or whitespace-only text."""
    response = client.post("/api/analyze", json={"text": invalid_text})
    assert response.status_code == 400
    assert "cannot be empty or contain only whitespace" in response.get_json()["error"]


@pytest.mark.parametrize("bad_type", [12345, 99.9, True, False, ["sample"], {"sub": "dict"}])
def test_analyze_invalid_types(client, bad_type):
    """Verify 400 error when 'text' field is non-string."""
    response = client.post("/api/analyze", json={"text": bad_type})
    assert response.status_code == 400
    assert "expected string" in response.get_json()["error"]


def test_analyze_exceeding_max_length(client):
    """Verify 400 error when text exceeds character length limit."""
    long_text = "a" * 1001
    response = client.post("/api/analyze", json={"text": long_text})
    assert response.status_code == 400
    assert "exceeds maximum allowed length" in response.get_json()["error"]


def test_analyze_internal_error_handling(client, mock_classifier):
    """Verify 500 error gracefully returned when inference fails without leaking stack traces."""
    mock_classifier.predict.side_effect = RuntimeError("Inference tensor device failure")

    response = client.post("/api/analyze", json={"text": "Trigger exception"})
    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "An error occurred" in data["error"]
    assert "Inference tensor device failure" not in data["error"]  # Do not leak internal trace
