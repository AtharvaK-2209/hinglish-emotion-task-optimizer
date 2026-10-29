"""Unit and integration tests for Phase 3 NLTK Processor.

Tests input validation, multilingual tokenization (English, Hindi Devanagari,
and Romanized Hinglish), linguistic statistics, and edge/error cases.

Usage:
    python3 tests/test_nltk_processor.py
    # Or with pytest:
    pytest tests/test_nltk_processor.py -v
"""

import sys
import os
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.nltk_processor import NLTKProcessor


@pytest.fixture
def processor():
    """Create a fresh instance of NLTKProcessor."""
    return NLTKProcessor()


# ---------------------------------------------------------------------------
# Test Cases Data
# ---------------------------------------------------------------------------

SAMPLE_CASES = [
    {
        "category": "English",
        "text": "I am really excited about my results!",
        "expected_tokens": ["I", "am", "really", "excited", "about", "my", "results", "!"],
        "expected_words": ["I", "am", "really", "excited", "about", "my", "results"],
        "expected_word_count": 7,
        "expected_token_count": 8,
        "expected_script": "latin",
        "expected_hint": "English / Hinglish (Roman Script)",
    },
    {
        "category": "Hindi (Devanagari)",
        "text": "मुझे आज बहुत खुशी हो रही है।",
        "expected_tokens": ["मुझे", "आज", "बहुत", "खुशी", "हो", "रही", "है", "।"],
        "expected_words": ["मुझे", "आज", "बहुत", "खुशी", "हो", "रही", "है"],
        "expected_word_count": 7,
        "expected_token_count": 8,
        "expected_script": "devanagari",
        "expected_hint": "Hindi (Devanagari)",
    },
    {
        "category": "Hinglish (Roman Script)",
        "text": "Yaar, aaj bahut tension ho raha hai.",
        "expected_tokens": ["Yaar", ",", "aaj", "bahut", "tension", "ho", "raha", "hai", "."],
        "expected_words": ["Yaar", "aaj", "bahut", "tension", "ho", "raha", "hai"],
        "expected_word_count": 7,
        "expected_token_count": 9,
        "expected_script": "latin",
        "expected_hint": "English / Hinglish (Roman Script)",
    },
]

INVALID_CASES = [
    {"name": "Empty string", "input": "", "expected_err": ValueError},
    {"name": "Whitespace only (spaces)", "input": "    ", "expected_err": ValueError},
    {"name": "Whitespace only (mixed)", "input": " \t\n \r ", "expected_err": ValueError},
    {"name": "NoneType input", "input": None, "expected_err": TypeError},
    {"name": "Integer input", "input": 12345, "expected_err": TypeError},
    {"name": "Float input", "input": 99.9, "expected_err": TypeError},
    {"name": "List input", "input": ["Hello", "world"], "expected_err": TypeError},
    {"name": "Dict input", "input": {"text": "hello"}, "expected_err": TypeError},
]


# ---------------------------------------------------------------------------
# Pytest Test Functions
# ---------------------------------------------------------------------------

def test_processor_initialization_and_status(processor):
    """Verify processor initialization and metadata status reporting."""
    status = processor.get_status()
    assert status["processor"] == "NLTK"
    assert status["ready"] is True
    assert status["phase"] == 3
    assert status["status"] == "ready"
    assert "tokenizer" in status
    assert "punkt_available" in status
    assert isinstance(status["supported_scripts"], list)


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=lambda c: c["category"])
def test_sample_tokenization(processor, case):
    """Verify tokenization accuracy across English, Hindi, and Hinglish."""
    tokens = processor.tokenize(case["text"])
    assert tokens == case["expected_tokens"]


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=lambda c: c["category"])
def test_sample_text_analysis(processor, case):
    """Verify text analysis statistics for English, Hindi, and Hinglish."""
    analysis = processor.analyze_text(case["text"])

    assert analysis["original_text"] == case["text"]
    assert analysis["tokens"] == case["expected_tokens"]
    assert analysis["words"] == case["expected_words"]
    assert analysis["word_count"] == case["expected_word_count"]
    assert analysis["token_count"] == case["expected_token_count"]
    assert analysis["detected_script"] == case["expected_script"]
    assert analysis["language_hint"] == case["expected_hint"]

    # Character count checks
    assert analysis["char_count"] == len(case["text"])
    assert analysis["char_count_no_spaces"] > 0
    assert analysis["char_count_no_spaces"] < analysis["char_count"]

    # Basic statistics
    assert analysis["sentence_count"] >= 1
    assert analysis["avg_word_length"] > 0
    assert analysis["vocabulary_size"] > 0
    assert isinstance(analysis["top_words"], list)
    assert len(analysis["top_words"]) > 0


def test_mixed_script_handling(processor):
    """Verify handling of code-mixed Hindi (Devanagari) and Roman script."""
    text = "Aaj mujhe बहुत टेंशन ho raha hai!"
    analysis = processor.analyze_text(text)

    assert analysis["detected_script"] == "mixed"
    assert analysis["language_hint"] == "Mixed (Devanagari & Roman Script)"
    assert "बहुत" in analysis["words"]
    assert "टेंशन" in analysis["words"]
    assert "aaj" in [w.lower() for w in analysis["words"]]


def test_text_immutability(processor):
    """Verify that text analysis does not mutate or alter the original input string."""
    original = "  Original Text With Extra Spaces   "
    analysis = processor.analyze_text(original)
    assert analysis["original_text"] == original


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda c: c["name"])
def test_invalid_and_empty_inputs(processor, case):
    """Verify ValueError on empty/whitespace and TypeError on non-string inputs."""
    with pytest.raises(case["expected_err"]):
        processor.tokenize(case["input"])

    with pytest.raises(case["expected_err"]):
        processor.analyze_text(case["input"])

    with pytest.raises(case["expected_err"]):
        processor.validate_input(case["input"])


# ---------------------------------------------------------------------------
# Standalone Execution Runner
# ---------------------------------------------------------------------------

def run_standalone_tests() -> bool:
    """Run all Phase 3 NLTK processor checks with formatted terminal output."""
    print("=" * 80)
    print("PHASE 3: NLTK Processor Verification")
    print("=" * 80)

    processor = NLTKProcessor()
    status = processor.get_status()
    print("\n[Step 1] Processor Status:")
    for k, v in status.items():
        print(f"  - {k}: {v}")

    print("\n[Step 2] Testing Multilingual Sentences (English, Hindi, Hinglish):")
    print("-" * 80)

    for case in SAMPLE_CASES:
        category = case["category"]
        text = case["text"]
        print(f"\nCategory: [{category}]")
        print(f"Input   : \"{text}\"")

        try:
            res = processor.analyze_text(text)
            print(f"  ✔ Script detected : {res['detected_script']} ({res['language_hint']})")
            print(f"  ✔ Word Count      : {res['word_count']} (Tokens: {res['token_count']})")
            print(f"  ✔ Char Count      : {res['char_count']} (No spaces: {res['char_count_no_spaces']})")
            print(f"  ✔ Sentences       : {res['sentence_count']}")
            print(f"  ✔ Avg Word Length : {res['avg_word_length']}")
            print(f"  ✔ Tokens          : {res['tokens']}")
            print(f"  ✔ Words           : {res['words']}")
            print(f"  ✔ Top Frequencies : {res['top_words'][:3]}")

            assert res["tokens"] == case["expected_tokens"]
            assert res["words"] == case["expected_words"]
            assert res["word_count"] == case["expected_word_count"]
            assert res["detected_script"] == case["expected_script"]
        except Exception as exc:
            print(f"  ❌ FAILED for {category}: {exc}")
            return False

    print("\n[Step 3] Testing Input Validation & Error Handling:")
    print("-" * 80)

    for case in INVALID_CASES:
        name = case["name"]
        val = case["input"]
        expected_err = case["expected_err"]
        try:
            processor.analyze_text(val)
            print(f"  ❌ Expected {expected_err.__name__} for '{name}', but call succeeded!")
            return False
        except expected_err as e:
            print(f"  ✔ Correctly caught {expected_err.__name__} for {name}: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected exception for {name}: {type(e).__name__}: {e}")
            return False

    print("\n" + "=" * 80)
    print("✔ All Phase 3 NLTK Processor tests passed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_standalone_tests()
    sys.exit(0 if success else 1)
