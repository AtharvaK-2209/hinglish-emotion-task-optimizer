"""Standalone test script for Phase 2 Pretrained Emotion Classifier.

Tests the `amaan00z/emotion_xlmr3` model across English, Hindi (Devanagari),
Hinglish, indirect emotion, and neutral statements, as well as edge/error cases.

Usage:
    python3 tests/test_emotion_model.py
    # Or with pytest:
    pytest tests/test_emotion_model.py -s
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.emotion_classifier import EmotionClassifier


TEST_CASES = [
    {
        "category": "English",
        "text": "I am really excited about my results!",
    },
    {
        "category": "Hindi (Devanagari)",
        "text": "मुझे आज बहुत खुशी हो रही है।",
    },
    {
        "category": "Hinglish",
        "text": "Yaar, aaj bahut tension ho raha hai.",
    },
    {
        "category": "Indirect emotion",
        "text": "I worked on this all night, and nothing is working.",
    },
    {
        "category": "Neutral text",
        "text": "The meeting starts at 10 AM.",
    },
]

INVALID_CASES = [
    {"name": "Empty string", "input": "", "expected_err": ValueError},
    {"name": "Whitespace only", "input": "   \n\t  ", "expected_err": ValueError},
    {"name": "NoneType input", "input": None, "expected_err": TypeError},
    {"name": "Integer input", "input": 12345, "expected_err": TypeError},
    {"name": "List input", "input": ["Hello"], "expected_err": TypeError},
]


def run_standalone_tests():
    """Execute all tests and display formatted results."""
    print("=" * 80)
    print("PHASE 2: Pretrained Emotion Classification Verification")
    print("Model: amaan00z/emotion_xlmr3")
    print("=" * 80)

    # 1. Initialize classifier and load model
    print("\n[Step 1] Initializing EmotionClassifier and loading model...")
    try:
        classifier = EmotionClassifier(load_on_init=True)
    except Exception as exc:
        print(f"❌ Failed to initialize/load model: {exc}")
        return False

    status = classifier.get_status()
    print(f"✔ Device: {status['device']}")
    print(f"✔ Model loaded: {status['loaded']}")
    print(f"✔ Actual configured labels ({len(status['configured_labels'])} total):")
    for idx, lbl in enumerate(status["configured_labels"]):
        print(f"    [{idx}] {lbl}")

    # 2. Test valid sentences (English, Hindi, Hinglish, Indirect, Neutral)
    print("\n[Step 2] Testing Required Test Sentences...")
    print("-" * 80)

    for case in TEST_CASES:
        category = case["category"]
        text = case["text"]
        print(f"\nCategory: [{category}]")
        print(f"Input   : \"{text}\"")

        try:
            res = classifier.predict(text)
            print(f"Result  : Top Label = '{res['label']}' | Score = {res['score']:.4f}")
            print("Ranked label distribution:")
            for rank, item in enumerate(res["all_scores"][:5], 1):
                print(f"   {rank}. {item['label']:<12}: {item['score']:.4f}")
        except Exception as exc:
            print(f"❌ Error during prediction: {exc}")
            return False

    # 3. Test instance reuse (verify model is not reloaded)
    print("\n[Step 3] Verifying Model Instance Reuse...")
    first_model_id = id(classifier.model)
    classifier.predict("Testing model instance reuse.")
    second_model_id = id(classifier.model)
    assert first_model_id == second_model_id, "Model was reloaded instead of reused!"
    print(f"✔ Same model object reused across calls (Object ID: {first_model_id}).")

    # 4. Test error handling and edge cases
    print("\n[Step 4] Testing Input Validation & Error Handling...")
    print("-" * 80)

    for err_case in INVALID_CASES:
        name = err_case["name"]
        val = err_case["input"]
        expected_err = err_case["expected_err"]
        try:
            classifier.predict(val)  # type: ignore
            print(f"❌ Expected {expected_err.__name__} for {name}, but call succeeded!")
            return False
        except expected_err as e:
            print(f"✔ Correctly rejected {name} ({expected_err.__name__}): {e}")
        except Exception as e:
            print(f"❌ Unexpected exception type for {name}: {type(e).__name__}: {e}")
            return False

    print("\n" + "=" * 80)
    print("✔ All Phase 2 emotion classifier checks completed successfully.")
    print("=" * 80)
    return True


# Pytest-compatible tests
def test_valid_sentences_inference():
    """Pytest test checking that all valid test cases return non-empty labels and scores."""
    classifier = EmotionClassifier(load_on_init=True)
    for case in TEST_CASES:
        res = classifier.predict(case["text"])
        assert isinstance(res["label"], str) and len(res["label"]) > 0
        assert isinstance(res["score"], float)
        assert 0.0 <= res["score"] <= 1.0
        assert len(res["all_scores"]) > 0
        assert res["model_id"] == "amaan00z/emotion_xlmr3"


def test_input_validation_errors():
    """Pytest test checking empty, whitespace, and type errors."""
    classifier = EmotionClassifier(load_on_init=False)
    for err_case in INVALID_CASES:
        try:
            classifier.predict(err_case["input"])  # type: ignore
            assert False, f"Expected {err_case['expected_err']} for {err_case['name']}"
        except err_case["expected_err"]:
            pass


def test_model_reuse():
    """Pytest test ensuring the same model instance is reused."""
    classifier = EmotionClassifier(load_on_init=True)
    m1 = id(classifier.model)
    classifier.predict("First test sentence.")
    m2 = id(classifier.model)
    assert m1 == m2


if __name__ == "__main__":
    success = run_standalone_tests()
    sys.exit(0 if success else 1)
