"""NLTK Processor Service.

This service performs linguistic text analysis, tokenization, word counting,
and text statistics for English, Hindi (Devanagari), and Romanized Hindi/Hinglish
inputs using NLTK.
"""

from typing import Dict, Any, List
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist


class NLTKProcessor:
    """Wrapper class for NLTK text processing, tokenization, and linguistic statistics."""

    # Regex matching words across Latin, numerals, and Devanagari characters (preserving
    # combining vowel matras and signs: \u0900-\u0963, \u0966-\u097F) while isolating
    # punctuation symbols (ASCII punctuation, Devanagari danda \u0964, and double danda \u0965).
    MULTILINGUAL_TOKEN_PATTERN = r'[\w\u0900-\u0963\u0966-\u097F]+|[^\w\s]'

    def __init__(self):
        """Initialize NLTK processor with multilingual tokenizer and resource checks."""
        self.is_ready: bool = True
        self.tokenizer = RegexpTokenizer(self.MULTILINGUAL_TOKEN_PATTERN)
        self.punkt_available: bool = self._check_punkt_resource()

    @staticmethod
    def _check_punkt_resource() -> bool:
        """Check whether optional NLTK punkt or punkt_tab models are available locally.

        Avoids downloading models on every function call or failing if unavailable.
        """
        try:
            nltk.data.find("tokenizers/punkt_tab")
            return True
        except Exception:
            try:
                nltk.data.find("tokenizers/punkt")
                return True
            except Exception:
                return False

    def validate_input(self, text: Any) -> str:
        """Validate input text before processing.

        Args:
            text: Input text to validate.

        Returns:
            Cleaned string if valid.

        Raises:
            TypeError: If input is not a string.
            ValueError: If input is empty or contains only whitespace.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected input text to be a string, got {type(text).__name__}.")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or contain only whitespace.")

        return cleaned

    def detect_script(self, text: str) -> str:
        """Detect the dominant script used in the input string.

        Identifies Devanagari (Hindi), Latin (English / Hinglish), or mixed scripts.

        Args:
            text: Input string.

        Returns:
            One of: 'devanagari', 'latin', 'mixed', or 'other'.
        """
        has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
        has_latin = any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text)

        if has_devanagari and has_latin:
            return "mixed"
        if has_devanagari:
            return "devanagari"
        if has_latin:
            return "latin"
        return "other"

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into individual tokens using NLTK's RegexpTokenizer.

        Accurately preserves Devanagari words and matras while keeping Latin and
        Hinglish words intact without corrupting script boundaries.

        Args:
            text: Input string in English, Hindi, or Hinglish.

        Returns:
            List of string tokens (words and punctuation).
        """
        self.validate_input(text)
        return self.tokenizer.tokenize(text)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform linguistic statistics and token analysis on the input text.

        Calculates token counts, lexical word counts, character counts, average word
        lengths, sentence estimations, vocabulary frequency, and detected script.
        Does NOT alter or mutate the original text, ensuring unmodified input can be
        passed directly to downstream emotion classification models.

        Args:
            text: Input text.

        Returns:
            Dict containing detailed text metrics and linguistic statistics.
        """
        self.validate_input(text)
        tokens = self.tokenize(text)

        # Separate lexical words from standalone punctuation tokens
        # Excludes ASCII punctuation and Devanagari danda (\u0964) / double danda (\u0965)
        words = [
            t for t in tokens
            if any(c.isalnum() or ("\u0900" <= c <= "\u097F" and c not in "\u0964\u0965") for c in t)
        ]

        word_count = len(words)
        token_count = len(tokens)
        char_count = len(text)
        char_count_no_spaces = len(re.sub(r"\s+", "", text))

        # Sentence segmentation (handles English punctuation [.!?] as well as Devanagari danda [।])
        sents = [s.strip() for s in re.split(r"[\.\!\?\।\n]+", text) if s.strip()]
        sentence_count = max(1, len(sents))

        # Average word length
        avg_word_length = round(sum(len(w) for w in words) / word_count, 2) if word_count > 0 else 0.0

        # Frequency distribution using NLTK FreqDist (normalize Latin to lowercase for counting)
        normalized_words = [
            w.lower() if any("a" <= c <= "z" or "A" <= c <= "Z" for c in w) else w
            for w in words
        ]
        fdist = FreqDist(normalized_words)
        top_words = [{"word": word, "count": count} for word, count in fdist.most_common(5)]

        # Script and language hints
        script = self.detect_script(text)
        language_hints = {
            "devanagari": "Hindi (Devanagari)",
            "latin": "English / Hinglish (Roman Script)",
            "mixed": "Mixed (Devanagari & Roman Script)",
            "other": "Other / Symbols"
        }

        return {
            "original_text": text,
            "tokens": tokens,
            "token_count": token_count,
            "words": words,
            "word_count": word_count,
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "vocabulary_size": len(fdist),
            "top_words": top_words,
            "detected_script": script,
            "language_hint": language_hints.get(script, "Other")
        }

    def get_status(self) -> Dict[str, Any]:
        """Return readiness status and configuration of the NLTK processor."""
        return {
            "processor": "NLTK",
            "ready": self.is_ready,
            "phase": 3,
            "status": "ready",
            "tokenizer": "NLTK RegexpTokenizer (Multilingual)",
            "punkt_available": self.punkt_available,
            "supported_scripts": [
                "Latin (English / Hinglish)",
                "Devanagari (Hindi)"
            ]
        }
