# AI-Powered Task Optimizer: Text-Based Hinglish Emotion Analysis

An academic AI mini-project designed to analyze emotional states from text inputs in **English**, **Hindi**, and **Hinglish**, providing contextual task scheduling and productivity recommendations.

---

## 📌 Project Overview & Scope

- **Target Inputs**: English, Hindi, and Hinglish natural language text.
- **Emotion Classification Model**: Hugging Face pretrained transformer [`amaan00z/emotion_xlmr3`](https://huggingface.co/amaan00z/emotion_xlmr3).
- **Text Processing**: NLTK for linguistic preprocessing and tokenization.
- **Backend Framework**: Python Flask with modular architecture.
- **Frontend Interface**: Semantic HTML5, CSS3, and JavaScript.
- **Inference Runtime**: Hugging Face Transformers & PyTorch (integrated in Phase 2).
- **Out of Scope (Initial Version)**:
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
    └── test_app.py                  # Unit and integration tests
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

---

## 🏃 Running the Application

### Start the Flask Development Server
```bash
python3 app.py
```
By default, the server runs on [http://localhost:5000](http://localhost:5000).

### API Endpoints
- `GET /` — Renders web dashboard.
- `GET /api/health` — Returns JSON health status and service modules state.

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## 📋 Development Roadmap

- [x] **Phase 1**: Workspace scaffolding, modular architecture, minimal Flask server, frontend layout, and initial test suite.
- [ ] **Phase 2**: Transformer model (`amaan00z/emotion_xlmr3`) integration, PyTorch inference pipeline, and NLTK tokenization.
- [ ] **Phase 3**: Recommendation heuristics mapping emotions to productivity workflows, frontend interaction, and end-to-end evaluation.
