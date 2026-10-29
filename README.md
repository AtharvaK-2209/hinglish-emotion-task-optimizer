# EmoSync: Emotion-Aware Productivity Assistant

An intelligent productivity assistant designed to analyze emotional states from text inputs in **English**, **Hindi**, and **Hinglish**, providing contextual task scheduling and adaptive productivity recommendations. Understand your emotions. Find your flow.

---

## 📌 Project Overview & Scope

- **Target Inputs**: English, Hindi, and Hinglish natural language text.
- **Emotion Classification Ensemble**: Language-routed architecture deploying specialized models:
  - **English**: [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) (28 fine-grained labels including neutral; multi-label sigmoid head).
  - **Hindi (Devanagari)**: [`vashuag/HindiEmotion`](https://huggingface.co/vashuag/HindiEmotion) (7 Hindi emotion labels; IndicBERT architecture).
  - **Hinglish (Code-Mixed)**: [`amaan00z/emotion_xlmr3`](https://huggingface.co/amaan00z/emotion_xlmr3) (10 code-mixed labels; XLM-RoBERTa architecture).
- **Text Processing**: NLTK for linguistic preprocessing, script detection, and tokenization.
- **Backend Framework**: Python Flask with modular service architecture.
- **Frontend Interface**: Semantic HTML5, modern CSS3 (with responsive glassmorphism), and dynamic JavaScript.
- **Inference Runtime**: Hugging Face Transformers & PyTorch with automatic MPS/CUDA/CPU device detection.
- **Out of Scope**:
  - No facial emotion recognition or camera access.
  - No speech/audio analysis.
  - No spaCy or WordNet dependencies.
  - No external database required.

---

## 📁 Project Structure

```text
AI-Task-Optimizer-NLP/
│
├── app.py                            # Flask application factory and API routing
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and run guide
│
├── services/                         # Modular business logic services
│   ├── __init__.py
│   ├── emotion_classifier.py        # Hugging Face transformer inference wrapper
│   ├── nltk_processor.py            # NLTK tokenization and text analysis
│   └── recommendation_engine.py     # Emotion-to-task recommendation engine
│
├── templates/                        # Frontend templates
│   └── index.html                   # Main web UI
│
├── static/                           # Static assets
│   ├── css/
│   │   └── style.css                # Application stylesheet
│   └── js/
│       └── app.js                   # Client-side JavaScript
│
└── tests/                            # Automated test suite
    ├── __init__.py
    ├── test_api.py                  # Phase 4 Flask API & integration tests (mocked model)
    ├── test_app.py                  # Unit and integration tests
    ├── test_emotion_model.py        # Phase 2 pretrained model verification
    ├── test_ensemble_router.py       # Language-routed ensemble & GoEmotions tests
    ├── test_nltk_processor.py       # Phase 3 NLTK text processing tests
    └── test_recommendation_engine.py# Phase 4 recommendation engine tests
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python**: 3.9+ (Python 3.10 to 3.13 supported)
- **pip**: Latest package manager

### 2. Virtual Environment Setup (Recommended)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. NLTK Setup & Optional Resources
The text processing module uses NLTK's `RegexpTokenizer` and `FreqDist` with custom Unicode boundary rules supporting English, Hindi (Devanagari), and Hinglish (Roman script) out of the box with zero runtime downloads required.

If you optionally wish to install NLTK's sentence tokenization resources (`punkt` / `punkt_tab`), run:
```bash
python3 -m nltk.downloader punkt punkt_tab
```
> [!NOTE]
> The processor automatically checks for the presence of optional NLTK models and falls back seamlessly to rule-based segmentation if they are absent, preventing runtime errors or blocking downloads.

---

## 🏃 Running the Application

### Start the Flask Development Server
```bash
python3 app.py
```
By default, the server runs on [http://localhost:5000](http://localhost:5000).

### API Endpoints

#### 1. `GET /`
Renders the interactive web dashboard.

#### 2. `GET /api/health`
Returns backend health and service readiness.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "system_status": "Operational",
  "project": "EmoSync — Emotion-Aware Productivity Assistant",
  "application": "EmoSync — Emotion-Aware Productivity Assistant",
  "services": {
    "emotion_classifier": {
      "configured_labels": ["anger", "fear", "disgust", "sadness", "disapproval", "surprise", "admiration", "joy", "love", "neutral"],
      "device": "mps",
      "loaded": true,
      "max_text_length": 1000,
      "model_name": "amaan00z/emotion_xlmr3",
      "phase": 2,
      "status": "ready"
    },
    "nltk_processor": {
      "phase": 3,
      "processor": "NLTK",
      "punkt_available": false,
      "ready": true,
      "status": "ready",
      "supported_scripts": ["Latin (English / Hinglish)", "Devanagari (Hindi)"],
      "tokenizer": "NLTK RegexpTokenizer (Multilingual)"
    },
    "recommendation_engine": {
      "engine": "RecommendationEngine",
      "fallback_available": true,
      "phase": 4,
      "ready": true,
      "status": "ready",
      "supported_labels": ["anger", "fear", "disgust", "sadness", "disapproval", "surprise", "admiration", "joy", "love", "neutral"]
    }
  },
  "status": "healthy"
}
```

#### 3. `POST /api/analyze`
Analyzes multilingual text for emotion, linguistic metrics, and task recommendations.
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "Yaar, aaj bahut tension ho raha hai."
}
```
- **Response `200 OK`**:
```json
{
  "status": "success",
  "data": {
    "original_text": "Yaar, aaj bahut tension ho raha hai.",
    "emotion": {
      "predicted_label": "sadness",
      "raw_label": "sadness",
      "score": 0.995,
      "ranked_scores": [
        {"label": "sadness", "score": 0.995},
        {"label": "fear", "score": 0.0042},
        {"label": "surprise", "score": 0.0002},
        {"label": "love", "score": 0.0002},
        {"label": "disapproval", "score": 0.0001}
      ],
      "model_id": "amaan00z/emotion_xlmr3",
      "device": "mps"
    },
    "text_statistics": {
      "tokens": ["Yaar", ",", "aaj", "bahut", "tension", "ho", "raha", "hai", "."],
      "token_count": 9,
      "words": ["Yaar", "aaj", "bahut", "tension", "ho", "raha", "hai"],
      "word_count": 7,
      "char_count": 36,
      "char_count_no_spaces": 30,
      "sentence_count": 1,
      "avg_word_length": 4.0,
      "vocabulary_size": 7,
      "top_words": [
        {"word": "yaar", "count": 1},
        {"word": "aaj", "count": 1},
        {"word": "bahut", "count": 1}
      ],
      "detected_script": "latin",
      "language_hint": "English / Hinglish (Roman Script)"
    },
    "recommendations": {
      "emotion": "sadness",
      "is_supported": true,
      "primary_strategy": "Gentle Pacing & Low-Friction Steps",
      "summary": "Lower energy or potential fatigue. Focus on reducing cognitive load and securing small wins.",
      "suggestions": [
        {
          "title": "Break Tasks into Micro-Steps",
          "action": "Divide larger deliverables into small 10-15 minute increments to lower resistance.",
          "category": "Pacing"
        },
        {
          "title": "Shift to Low-Stakes Activities",
          "action": "Focus on routine maintenance, documentation formatting, or light administrative tasks.",
          "category": "Task Selection"
        },
        {
          "title": "Take a Restorative Pause",
          "action": "Hydrate, step away from screens for a few minutes, or take a short walk.",
          "category": "Wellness"
        }
      ]
    }
  }
}
```

---

## 🧪 Running Tests

### 1. Fast Routine Test Suite (All Modules Mocked for Offline Execution)
Run the unit test suite verifying application routes, service separation, offline input validation, NLTK metrics, and recommendation mapping:
```bash
pytest tests/test_api.py tests/test_recommendation_engine.py tests/test_nltk_processor.py tests/test_app.py -v
```

### 2. Standalone Emotion Classification Verification (Phase 2 Real Model)
Verify real Hugging Face transformer model inference across English, Hindi (Devanagari), Hinglish, indirect emotion, neutral statements, and edge cases:
```bash
python3 tests/test_emotion_model.py
```
> [!NOTE]
> **Hardware Acceleration**: Automatically utilizes Apple Silicon MPS (`mps`) or NVIDIA CUDA (`cuda`) if available, falling back seamlessly to `cpu`.

### 3. Full Project Test Suite
```bash
pytest -v
```

---

## 📌 Limitations & Known Considerations

1. **Pretrained Model Scope**: The classifier model (`amaan00z/emotion_xlmr3`) was trained on social text and conversational Hinglish/English datasets. Highly technical phrasing, subtle sarcasm, or indirect metaphors may classify as `neutral` or reflect the dominant sentiment words present.
2. **Initial Download**: The transformer weights (~1.1 GB) are downloaded from Hugging Face Hub on the very first run and cached locally by Hugging Face `transformers` for offline reuse.
3. **Non-Clinical Design**: Task recommendations are rule-based productivity heuristics intended to assist personal task organization. They do not diagnose mental health conditions or clinically measure stress.

---

## 📋 Development Roadmap

- [x] **Phase 1**: Workspace scaffolding, modular architecture, minimal Flask server, frontend layout, and initial test suite.
- [x] **Phase 2**: Pretrained Hinglish transformer model (`amaan00z/emotion_xlmr3`) integration, PyTorch inference pipeline, validation, and standalone testing.
- [x] **Phase 3**: NLTK linguistic processing, multilingual tokenization (English, Hindi Devanagari, Hinglish), text statistics, validation, and focused test suite.
- [x] **Phase 4**: Emotion-to-task recommendation engine, Flask `/api/analyze` route wiring, dynamic frontend interaction, and end-to-end testing.
