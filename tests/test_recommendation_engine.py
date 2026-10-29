"""Unit tests for Phase 4 Recommendation Engine.

Tests mapping of all 10 supported emotion labels, fallback handling for unknown
labels, structure of recommendations, and status reporting.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def engine():
    """Create fresh instance of RecommendationEngine."""
    return RecommendationEngine()


def test_recommendation_engine_initialization(engine):
    """Verify initialization and status reporting."""
    status = engine.get_status()
    assert status["engine"] == "RecommendationEngine"
    assert status["ready"] is True
    assert status["phase"] == 4
    assert status["status"] == "ready"
    assert len(status["supported_labels"]) == 10
    assert status["fallback_available"] is True


@pytest.mark.parametrize("label", RecommendationEngine.SUPPORTED_LABELS)
def test_all_supported_labels_have_recommendations(engine, label):
    """Verify that every supported model label returns valid recommendations."""
    recommendations = engine.get_recommendations(label, confidence=0.95)
    assert isinstance(recommendations, list)
    assert len(recommendations) >= 2

    for rec in recommendations:
        assert "title" in rec and isinstance(rec["title"], str) and rec["title"]
        assert "action" in rec and isinstance(rec["action"], str) and rec["action"]
        assert "category" in rec and isinstance(rec["category"], str) and rec["category"]

    details = engine.get_recommendation_details(label, confidence=0.95)
    assert details["is_supported"] is True
    assert details["emotion"] == label
    assert details["confidence"] == 0.95
    assert isinstance(details["primary_strategy"], str) and details["primary_strategy"]
    assert isinstance(details["summary"], str) and details["summary"]
    assert len(details["suggestions"]) == len(recommendations)


@pytest.mark.parametrize("unknown_label", ["confused", "ecstatic", "bored", "random_123", "", None])
def test_unknown_label_fallback(engine, unknown_label):
    """Verify that unsupported labels gracefully use generic fallback without errors."""
    recommendations = engine.get_recommendations(unknown_label)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    details = engine.get_recommendation_details(unknown_label)
    assert details["is_supported"] is False
    assert details["primary_strategy"] == "Balanced Productivity Baseline"
    assert len(details["suggestions"]) == len(recommendations)
    for rec in details["suggestions"]:
        assert "title" in rec
        assert "action" in rec
