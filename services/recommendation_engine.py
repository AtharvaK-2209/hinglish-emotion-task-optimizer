"""Recommendation Engine Service.

This service generates supportive, actionable, and nonjudgmental task scheduling
and productivity recommendations based on emotion classification labels detected
by the pretrained emotion models.

Supports:
1. Core 10 emotion labels directly supported by the baseline and Hindi models.
2. Comprehensive mapping for all 28 fine-grained GoEmotions labels into core strategy buckets.
3. Safe generic fallback for unknown or unsupported labels.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Engine mapping emotion classification outputs to task recommendations."""

    # Explicitly supported core labels matching the baseline model configuration
    SUPPORTED_LABELS: List[str] = [
        "anger",
        "fear",
        "disgust",
        "sadness",
        "disapproval",
        "surprise",
        "admiration",
        "joy",
        "love",
        "neutral",
    ]

    # Pre-defined recommendation rules for every supported emotion
    RECOMMENDATION_CATALOG: Dict[str, Dict[str, Any]] = {
        "joy": {
            "primary_strategy": "Capitalize on Momentum",
            "summary": "High enthusiasm and positive momentum. An ideal time for ambitious, creative, or collaborative work.",
            "suggestions": [
                {
                    "title": "Tackle Demanding Tasks",
                    "action": "Dive into challenging, creative, or strategic tasks while energy and focus are elevated.",
                    "category": "Deep Work",
                },
                {
                    "title": "Capture Forward Ideas",
                    "action": "Brainstorm or outline future project milestones while in an expansive, open mindset.",
                    "category": "Planning",
                },
                {
                    "title": "Acknowledge Progress",
                    "action": "Record recent milestones or celebrate the progress made so far with your team.",
                    "category": "Reflection",
                },
            ],
        },
        "love": {
            "primary_strategy": "Sustained Engagement",
            "summary": "Warm, appreciative, and engaged state. Excellent for teamwork, mentoring, and passion projects.",
            "suggestions": [
                {
                    "title": "Focus on Meaningful Tasks",
                    "action": "Channel your current affinity into passion projects or core user-facing features.",
                    "category": "Engagement",
                },
                {
                    "title": "Collaborative Sharing",
                    "action": "Share constructive feedback, assist a teammate, or improve shared documentation.",
                    "category": "Collaboration",
                },
                {
                    "title": "Protect Sustainable Pace",
                    "action": "Set clear boundaries to prevent over-commitment despite high enthusiasm.",
                    "category": "Pacing",
                },
            ],
        },
        "admiration": {
            "primary_strategy": "Knowledge Capture & Reinforcement",
            "summary": "Inspiration and respect for good work. Prime opportunity to adopt best practices and raise quality.",
            "suggestions": [
                {
                    "title": "Study Effective Patterns",
                    "action": "Analyze what made the observed solution work well and document key takeaways.",
                    "category": "Learning",
                },
                {
                    "title": "Apply High Standards",
                    "action": "Incorporate observed quality patterns into your current tasks and review checklists.",
                    "category": "Quality",
                },
                {
                    "title": "Express Appreciation",
                    "action": "Leave thoughtful feedback or acknowledge peers who contributed to this milestone.",
                    "category": "Teamwork",
                },
            ],
        },
        "neutral": {
            "primary_strategy": "Structured Execution",
            "summary": "Balanced, objective focus. Excellent for systematic execution of scheduled backlog priorities.",
            "suggestions": [
                {
                    "title": "Sequential Task Execution",
                    "action": "Work through your prioritized task backlog one item at a time without multitasking.",
                    "category": "Execution",
                },
                {
                    "title": "Use Structured Time Blocks",
                    "action": "Set a 25-minute Pomodoro timer to maintain steady, predictable momentum.",
                    "category": "Time Management",
                },
                {
                    "title": "Review Daily Milestones",
                    "action": "Verify deadlines, organize deliverables, and clear routine administrative tasks.",
                    "category": "Organization",
                },
            ],
        },
        "sadness": {
            "primary_strategy": "Gentle Pacing & Low-Friction Steps",
            "summary": "Lower energy or potential fatigue. Focus on reducing cognitive load and securing small wins.",
            "suggestions": [
                {
                    "title": "Break Tasks into Micro-Steps",
                    "action": "Divide larger deliverables into small 10-15 minute increments to lower resistance.",
                    "category": "Pacing",
                },
                {
                    "title": "Shift to Low-Stakes Activities",
                    "action": "Focus on routine maintenance, documentation formatting, or light administrative tasks.",
                    "category": "Task Selection",
                },
                {
                    "title": "Take a Restorative Pause",
                    "action": "Hydrate, step away from screens for a few minutes, or take a short walk.",
                    "category": "Wellness",
                },
            ],
        },
        "anger": {
            "primary_strategy": "De-escalation & Channeled Focus",
            "summary": "High arousal or frustration with blockers. Prioritize emotional reset before taking critical actions.",
            "suggestions": [
                {
                    "title": "Step Back Temporarily",
                    "action": "Pause for 5-10 minutes before sending messages or merging complex changes.",
                    "category": "De-escalation",
                },
                {
                    "title": "Channel Energy into Independent Tasks",
                    "action": "Apply focused energy to code refactoring, organizing files, or writing diagnostic notes.",
                    "category": "Constructive Focus",
                },
                {
                    "title": "Isolate the Root Blocker",
                    "action": "Write down the exact obstacle objectively to separate facts from emotional friction.",
                    "category": "Problem Solving",
                },
            ],
        },
        "fear": {
            "primary_strategy": "Uncertainty Reduction & Verification",
            "summary": "Apprehension or worry about deadlines and risks. Focus on gaining clarity and concrete control.",
            "suggestions": [
                {
                    "title": "List Knowns and Unknowns",
                    "action": "Write down what is confirmed versus what is uncertain to remove ambiguity.",
                    "category": "Clarity",
                },
                {
                    "title": "Focus on the Immediate Next Action",
                    "action": "Ignore the entire project scale for a moment and complete only the single next step.",
                    "category": "Action",
                },
                {
                    "title": "Seek Early Feedback",
                    "action": "Consult documentation or review current plans with a peer to validate direction.",
                    "category": "Validation",
                },
            ],
        },
        "disapproval": {
            "primary_strategy": "Constructive Action & Checklist Review",
            "summary": "Critical perspective on quality or approach. Convert critique into tangible, step-by-step improvements.",
            "suggestions": [
                {
                    "title": "Convert Critique to Items",
                    "action": "Translate points of dissatisfaction into concrete, measurable checklist criteria.",
                    "category": "Actionable Planning",
                },
                {
                    "title": "Prioritize High-Impact Refinements",
                    "action": "Address the single most critical quality gap before attempting widespread changes.",
                    "category": "Prioritization",
                },
                {
                    "title": "Establish Clear Acceptance Criteria",
                    "action": "Define objective test criteria to ensure future revisions meet the desired standard.",
                    "category": "Standards",
                },
            ],
        },
        "surprise": {
            "primary_strategy": "Diagnostic Pause & Fact-Finding",
            "summary": "Unexpected results or sudden changes. Pause to investigate root causes before reacting.",
            "suggestions": [
                {
                    "title": "Inspect Changes and Logs",
                    "action": "Examine recent logs, diffs, or requirements to understand what actually changed.",
                    "category": "Diagnosis",
                },
                {
                    "title": "Avoid Hasty Edits",
                    "action": "Formulate a concrete hypothesis and verify it before trying exploratory fixes.",
                    "category": "Methodical Testing",
                },
                {
                    "title": "Realign the Plan",
                    "action": "Update task priorities and timelines based on the new information discovered.",
                    "category": "Adaptation",
                },
            ],
        },
        "disgust": {
            "primary_strategy": "Task Rotation & Simplification",
            "summary": "Fatigue or aversion toward a tedious blocker. Shift perspective to overcome friction.",
            "suggestions": [
                {
                    "title": "Rotate Tasks Temporarily",
                    "action": "Switch to an alternative task or topic to refresh cognitive engagement.",
                    "category": "Task Rotation",
                },
                {
                    "title": "Define Minimum Viable Scope",
                    "action": "Set a strict scope threshold to finish the tedious task without over-engineering.",
                    "category": "Scoping",
                },
                {
                    "title": "Automate Repetitive Friction",
                    "action": "Identify whether scripting, snippets, or tools can eliminate repetitive chores.",
                    "category": "Efficiency",
                },
            ],
        },
    }

    # Comprehensive mapping for all 28 fine-grained GoEmotions labels into core strategy buckets
    GO_EMOTIONS_STRATEGY_MAP: Dict[str, str] = {
        # Joy / Momentum & Expansion
        "joy": "joy",
        "excitement": "joy",
        "optimism": "joy",
        "pride": "joy",
        "amusement": "joy",
        "desire": "joy",
        # Love / Collaboration & Mentorship
        "love": "love",
        "caring": "love",
        # Admiration / Quality & Knowledge Capture
        "admiration": "admiration",
        "gratitude": "admiration",
        "approval": "admiration",
        # Neutral / Structured Execution
        "neutral": "neutral",
        "relief": "neutral",
        "realization": "neutral",
        # Sadness / Gentle Pacing & Micro-Steps
        "sadness": "sadness",
        "disappointment": "sadness",
        "grief": "sadness",
        "remorse": "sadness",
        # Anger / De-escalation & Channeled Focus
        "anger": "anger",
        "annoyance": "anger",
        # Disapproval / Constructive Review
        "disapproval": "disapproval",
        # Disgust / Task Rotation & Simplification
        "disgust": "disgust",
        # Fear / Uncertainty Reduction & Verification
        "fear": "fear",
        "nervousness": "fear",
        "embarrassment": "fear",
        # Surprise / Diagnostic Pause & Fact-Finding
        "surprise": "surprise",
        "curiosity": "surprise",
        "confusion": "surprise",
    }

    # Safe generic fallback recommendations for unknown, unsupported, or empty labels
    GENERIC_FALLBACK: Dict[str, Any] = {
        "primary_strategy": "Balanced Productivity Baseline",
        "summary": "Structured, mindful approach suited for general workflow organization and task execution.",
        "suggestions": [
            {
                "title": "Identify One High Priority",
                "action": "Select your single most important deliverable and dedicate the next 25 minutes to it.",
                "category": "Focus",
            },
            {
                "title": "Take Scheduled Short Breaks",
                "action": "Rest for 5 minutes after focused intervals to sustain mental clarity and prevent fatigue.",
                "category": "Pacing",
            },
            {
                "title": "Review and Adjust Backlog",
                "action": "Assess progress at the end of the session and organize the backlog for tomorrow.",
                "category": "Review",
            },
        ],
    }

    def __init__(self):
        """Initialize recommendation engine."""
        self.is_ready: bool = True

    def get_recommendations(
        self, emotion: str, confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate task recommendations based on detected emotion label.

        Args:
            emotion: Detected emotion label (e.g. 'joy', 'sadness', 'anger', 'neutral').
            confidence: Optional confidence score of the emotion detection.

        Returns:
            List of suggested task handling strategies.
        """
        details = self.get_recommendation_details(emotion, confidence)
        return details["suggestions"]

    def get_recommendation_details(
        self, emotion: str, confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate full recommendation payload including strategy and summary.

        Maps both core labels and GoEmotions extended labels to tailored strategies,
        falling back gracefully for truly unknown labels without errors.

        Args:
            emotion: Detected emotion label string.
            confidence: Optional model score as a float.

        Returns:
            Dict containing emotion, primary_strategy, summary, suggestions, and mapped strategy.
        """
        normalized_label = emotion.strip().lower() if isinstance(emotion, str) else ""

        # 1. Direct core catalog lookup
        if normalized_label in self.RECOMMENDATION_CATALOG:
            catalog_entry = self.RECOMMENDATION_CATALOG[normalized_label]
            is_supported = True
            strategy = catalog_entry["primary_strategy"]
            summary = catalog_entry["summary"]
            suggestions = [dict(s) for s in catalog_entry["suggestions"]]
            mapped_strategy = normalized_label

        # 2. GoEmotions extended taxonomy mapping
        elif normalized_label in self.GO_EMOTIONS_STRATEGY_MAP:
            target_key = self.GO_EMOTIONS_STRATEGY_MAP[normalized_label]
            catalog_entry = self.RECOMMENDATION_CATALOG[target_key]
            is_supported = True
            strategy = catalog_entry["primary_strategy"]
            summary = catalog_entry["summary"]
            suggestions = [dict(s) for s in catalog_entry["suggestions"]]
            mapped_strategy = target_key

        # 3. Graceful generic fallback
        else:
            is_supported = False
            strategy = self.GENERIC_FALLBACK["primary_strategy"]
            summary = self.GENERIC_FALLBACK["summary"]
            suggestions = [dict(s) for s in self.GENERIC_FALLBACK["suggestions"]]
            mapped_strategy = "generic_fallback"
            logger.info("Unrecognized emotion label '%s', using generic fallback.", emotion)

        return {
            "emotion": normalized_label if is_supported else (str(emotion) if emotion else "unknown"),
            "is_supported": is_supported,
            "confidence": confidence,
            "mapped_strategy": mapped_strategy,
            "primary_strategy": strategy,
            "summary": summary,
            "suggestions": suggestions,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return readiness status and configuration of the recommendation engine."""
        return {
            "engine": "RecommendationEngine",
            "ready": self.is_ready,
            "phase": 4,
            "status": "ready",
            "supported_labels": list(self.SUPPORTED_LABELS),
            "extended_labels_count": len(self.GO_EMOTIONS_STRATEGY_MAP),
            "fallback_available": True,
        }
