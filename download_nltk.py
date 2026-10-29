#!/usr/bin/env python3
"""Download required NLTK data on Render build."""
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
