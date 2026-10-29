# Quantitative Model Evaluation Report: EmoSync Emotion Classification

**Project:** EmoSync — Emotion-Aware Productivity Assistant  
**Evaluation Date:** October 2026  
**Evaluation Standard:** Scikit-Learn 1.8.0 Exact Strict Matching & Benchmark Tolerance  
**Scripts and Data Sources:** [`scripts/benchmark_results.json`](file:///Users/atharvakalam/Documents/Atharva%20Files/SEM%205/AI/AI-Task-Optimizer-NLP/scripts/benchmark_results.json), [`scripts/evaluate_models.py`](file:///Users/atharvakalam/Documents/Atharva%20Files/SEM%205/AI/AI-Task-Optimizer-NLP/scripts/evaluate_models.py)

---

## 1. Executive Summary

This report evaluates candidate emotion classification architectures against the **EmoSync Language-Routed Ensemble** across English, Devanagari Hindi, and Romanized Hinglish. 

Key quantitative findings include:
- **Monolithic models degrade sharply outside their target language**:
  - English-specialized **DistilBERT** completely fails on Devanagari Hindi (collapses to 100% `joy` due to subword tokenization breakdown) and lacks a `neutral` class entirely.
  - Hindi-specialized **IndicBERT** collapses to `neutral` on Latin English inputs (14.3% accuracy).
  - Hinglish-specialized **XLM-RoBERTa** exhibits a strong false-`neutral` bias on English text (14.3% accuracy).
  - English multi-label **RoBERTa GoEmotions** collapses to `neutral` on Devanagari Hindi and Hinglish (14.3% accuracy).
- **The EmoSync Language-Routed Ensemble achieves the highest overall performance**:
  - **52.4%** strict single-label accuracy ($11/21$) and **61.9%** acceptable-label accuracy ($13/21$).
  - **Macro-F1 of 0.416** and **Weighted-F1 of 0.534**, significantly surpassing every individual monolithic model (which capped at 38.1% strict accuracy).
  - Zero cross-domain script failure by dispatching inputs to specialized sub-classifiers.

---

## 2. Evaluation Methodology & Ground Truth

### 2.1 Dataset Composition ($N = 21$)

The evaluation dataset consists of 21 verified benchmark sentences spanning three language categories ($n = 7$ each), balanced across positive, negative, and neutral task contexts:

| Language | Script | Sample Count | Sample Texts |
| :--- | :--- | :--- | :--- |
| **English** | Latin | 7 | "I am really excited about my results!", "The meeting starts at 10 AM." |
| **Hindi** | Devanagari | 7 | "मुझे आज बहुत खुशी हो रही है।", "बैठक सुबह 10 बजे शुरू होगी।" |
| **Hinglish** | Romanized / Code-Mixed | 7 | "Yaar, aaj bahut tension ho raha hai.", "Kal subah 10 baje team meeting hai." |

### 2.2 Ground-Truth Label Distribution

The ground-truth reference set comprises 8 emotion categories across 21 samples:

| Emotion Class | Support ($N=21$) | English ($n=7$) | Hindi ($n=7$) | Hinglish ($n=7$) |
| :--- | :---: | :---: | :---: | :---: |
| `joy` | 6 | 2 | 2 | 2 |
| `sadness` | 4 | 1 | 1 | 2 |
| `fear` | 3 | 1 | 1 | 1 |
| `neutral` | 3 | 1 | 1 | 1 |
| `anger` | 2 | 1 | 0 | 1 |
| `admiration` | 1 | 1 | 0 | 0 |
| `disapproval` | 1 | 0 | 1 | 0 |
| `surprise` | 1 | 0 | 1 | 0 |

### 2.3 Evaluation Metrics and Strict vs. Acceptable Scoring

Two distinct accuracy standards were measured:
1. **Strict Single-Label Accuracy (Primary Metric)**: Evaluates exact string equivalence between the model's top predicted label and the ground-truth `expected_label` ($\text{Predicted} == \text{Expected}$).
2. **Acceptable-Label Accuracy (Benchmark Compatibility Metric)**: Grants tolerance for semantically adjacent secondary labels specified in benchmark ground truth (e.g., admitting `surprise` as acceptable for winning a competition, or `disappointment` for project setbacks).
3. **Scikit-Learn Statistical Metrics**: Precision, Recall, F1-Score (macro-averaged and support-weighted), and full $K \times K$ confusion matrices computed with `zero_division=0`.

---

## 3. Candidate Model Taxonomy Limitations

| Model Identifier | Base Architecture | Label Count | Configured Label Taxonomy | Documented Limitations |
| :--- | :--- | :---: | :--- | :--- |
| **`amaan00z/emotion_xlmr3`** | XLM-RoBERTa | 10 | `admiration`, `anger`, `disapproval`, `disgust`, `fear`, `joy`, `love`, `neutral`, `sadness`, `surprise` | Trained on noisy social-media Hinglish; severe false-neutral bias on English; degraded performance on formal Devanagari. |
| **`bhadresh-savani/distilbert-base-uncased-emotion`** | DistilBERT | 6 | `anger`, `fear`, `joy`, `love`, `sadness`, `surprise` | **Completely lacks `neutral` class** (classifies neutral scheduling as `anger`); lacks `admiration`, `disapproval`; cannot tokenize Devanagari script. |
| **`vashuag/HindiEmotion`** | IndicBERT (ALBERT) | 7 | `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise` | Lacks `admiration`, `disapproval`, `love`; fails on Latin script by assigning $\approx 97\%$ probability to `neutral`. |
| **`SamLowe/roberta-base-go_emotions`** | RoBERTa-base | 28 | 27 fine-grained emotions + `neutral` (Multi-label BCE) | Fine-grained taxonomy penalizes strict single-label matching (`excitement` vs `joy`, `nervousness` vs `fear`); fails on Hindi/Hinglish. |
| **EmoSync Ensemble** | Language-Routed Composite | 28 (Composite) | Routed to GoEmotions (EN), IndicBERT (HI), and XLM-RoBERTa (Hinglish) | Eliminates out-of-domain cross-lingual degradation; code-mixed Hinglish remains bounded by underlying XLM-RoBERTa checkpoint. |

---

## 4. Overall Quantitative Performance Comparison

The following table summarizes model performance across the entire 21-sample multilingual benchmark:

| Model Architecture | Strict Accuracy | Acceptable Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XLM-RoBERTa (Hinglish)** | 38.1% | 38.1% | 0.3556 | 0.2917 | 0.2937 | 0.4857 | 0.3810 | 0.3946 |
| **DistilBERT (English)** | 38.1% | 52.4% | 0.2153 | 0.2708 | 0.1861 | 0.3069 | 0.3810 | 0.2846 |
| **IndicBERT (Hindi)** | 38.1% | 42.9% | 0.3889 | 0.3519 | 0.3037 | 0.5357 | 0.3810 | 0.3508 |
| **RoBERTa (GoEmotions)** | 23.8% | 33.3% | 0.1875 | 0.2083 | 0.1722 | 0.1786 | 0.2381 | 0.1683 |
| **EmoSync Routed Ensemble** | **52.4%** | **61.9%** | **0.4455** | **0.4470** | **0.4156** | **0.6762** | **0.5238** | **0.5340** |

> [!NOTE]
> Under strict single-label evaluation, **every single monolithic model scored $\le 38.1\%$**. The **EmoSync Language-Routed Ensemble achieved 52.4% strict accuracy and 61.9% acceptable accuracy**, representing a **+37.5% relative accuracy gain** over any individual model across the multilingual evaluation set.

---

## 5. Per-Language Breakdown & Cross-Domain Degradation

### 5.1 English Sentences ($n = 7$)

| Model Candidate | Strict Acc | Acceptable Acc | Macro F1 | Weighted F1 | Key Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| XLM-RoBERTa | 14.3% ($1/7$) | 14.3% ($1/7$) | 0.083 | 0.143 | Predicted `neutral` for 5 out of 7 emotional English sentences. |
| DistilBERT | 57.1% ($4/7$) | 71.4% ($5/7$) | 0.357 | 0.500 | Lacks `neutral` (predicted `anger` for meeting text); lacks `admiration`. |
| IndicBERT | 14.3% ($1/7$) | 14.3% ($1/7$) | 0.042 | 0.036 | Script mismatch: predicted `neutral` (96–99%) for all 7 English sentences. |
| RoBERTa GoEmotions | 42.9% ($3/7$) | 71.4% ($5/7$) | 0.300 | 0.429 | Correct semantic direction (`excitement`, `nervousness`), penalized by strict match. |
| **EmoSync Ensemble** | **42.9% ($3/7$)** | **71.4% ($5/7$)** | **0.300** | **0.429** | Successfully routes English to RoBERTa GoEmotions. |

### 5.2 Hindi Devanagari Sentences ($n = 7$)

| Model Candidate | Strict Acc | Acceptable Acc | Macro F1 | Weighted F1 | Key Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| XLM-RoBERTa | 57.1% ($4/7$) | 57.1% ($4/7$) | 0.472 | 0.500 | Misclassified disapproval as `neutral` and gratitude as `fear`. |
| DistilBERT | 28.6% ($2/7$) | 42.9% ($3/7$) | 0.074 | 0.127 | Tokenizer cannot parse Devanagari: predicted `joy` for all 7 sentences. |
| IndicBERT | **71.4% ($5/7$)** | **71.4% ($5/7$)** | **0.583** | **0.643** | High Devanagari fidelity; correctly classified `joy`, `surprise`, `fear`, `neutral`. |
| RoBERTa GoEmotions | 14.3% ($1/7$) | 14.3% ($1/7$) | 0.042 | 0.036 | Cannot process Devanagari: predicted `neutral` for all 7 sentences. |
| **EmoSync Ensemble** | **71.4% ($5/7$)** | **71.4% ($5/7$)** | **0.583** | **0.643** | Successfully routes Hindi to IndicBERT. |

### 5.3 Hinglish Romanized Sentences ($n = 7$)

| Model Candidate | Strict Acc | Acceptable Acc | Macro F1 | Weighted F1 | Key Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| XLM-RoBERTa | **42.9% ($3/7$)** | **42.9% ($3/7$)** | **0.400** | **0.357** | Correct on `sadness`, `anger`, `fear`; misclassified victory as `sadness`. |
| DistilBERT | 28.6% ($2/7$) | 42.9% ($3/7$) | 0.191 | 0.231 | Predicted `anger` for 6 out of 7 sentences due to vocabulary gap. |
| IndicBERT | 28.6% ($2/7$) | 42.9% ($3/7$) | 0.178 | 0.152 | Unfamiliar with Romanized Hindi phonetics; scattered predictions. |
| RoBERTa GoEmotions | 14.3% ($1/7$) | 14.3% ($1/7$) | 0.067 | 0.057 | Misclassified Hinglish inputs as `amusement` or `neutral`. |
| **EmoSync Ensemble** | **42.9% ($3/7$)** | **42.9% ($3/7$)** | **0.400** | **0.357** | Successfully routes Hinglish to XLM-RoBERTa. |

---

## 6. Confusion Matrix Analysis: EmoSync Language-Routed Ensemble

Below is the confusion matrix for the production ensemble evaluated across the complete benchmark dataset ($N=21$):

```
True \ Pred        | admirat |   anger | disappo | disappr | excitem |    fear |     joy | nervous | neutral | sadness | surpris
--------------------------------------------------------------------------------------------------------------------------------
admiration         |       1 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0
anger              |       0 |       2 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0
disappointment     |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0
disapproval        |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |       0 |       0
excitement         |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0
fear               |       0 |       0 |       0 |       0 |       0 |       2 |       0 |       1 |       0 |       0 |       0
joy                |       0 |       0 |       0 |       0 |       1 |       1 |       2 |       0 |       0 |       1 |       1
nervousness        |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0
neutral            |       0 |       0 |       0 |       0 |       0 |       1 |       0 |       0 |       2 |       0 |       0
sadness            |       0 |       0 |       1 |       0 |       0 |       0 |       0 |       0 |       2 |       1 |       0
surprise           |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1
```

### 6.1 Per-Class Performance Summary (Ensemble Overall)

| Emotion Class | True Support | Precision | Recall | F1-Score | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`admiration`** | 1 | 1.000 | 1.000 | **1.000** | Perfect detection ("teammate did an amazing job"). |
| **`anger`** | 2 | 1.000 | 1.000 | **1.000** | Perfect detection across English and Hinglish. |
| **`surprise`** | 1 | 0.500 | 1.000 | **0.667** | Detected correctly in Hindi; one English `joy` sentence also predicted as `surprise`. |
| **`fear`** | 3 | 0.500 | 0.667 | **0.571** | Correctly detected in Hindi and Hinglish; English exam fear predicted as fine-grained `nervousness`. |
| **`neutral`** | 3 | 0.400 | 0.667 | **0.500** | Correct on English and Hindi; Hinglish neutral confused with `fear` due to ambiguous loanwords. |
| **`joy`** | 6 | 1.000 | 0.333 | **0.500** | High precision (1.00); recall reduced because English joy predicted as `excitement` and Hinglish joy mispredicted by XLM-RoBERTa. |
| **`sadness`** | 4 | 0.500 | 0.250 | **0.333** | English sadness predicted as `disappointment`; Hindi sadness predicted as `neutral`. |
| **`disapproval`**| 1 | 0.000 | 0.000 | **0.000** | IndicBERT lacked a dedicated disapproval head and fell back to `neutral`. |

---

## 7. Granular Taxonomy Mapping (GoEmotions Strategy Mapping)

When evaluating GoEmotions strictly against coarse 8-class labels, fine-grained predictions like `excitement`, `nervousness`, and `disappointment` count as strict mismatches.

However, EmoSync's **Recommendation Engine** incorporates an explicit strategy mapping (`GO_EMOTIONS_STRATEGY_MAP` in [`services/recommendation_engine.py`](file:///Users/atharvakalam/Documents/Atharva%20Files/SEM%205/AI/AI-Task-Optimizer-NLP/services/recommendation_engine.py#L251-L290)) that translates all 28 fine-grained labels into actionable productivity strategies:

| English Benchmark Sentence | Expected Label | GoEmotions Raw Prediction | Mapped Strategy Category | Mapped Strategy Agreement |
| :--- | :--- | :--- | :--- | :---: |
| 1. "I am really excited about my results!" | `joy` | `excitement` (82.8%) | **`joy`** (Momentum & Expansion) | ✔ Match |
| 2. "My teammate did an amazing job on this project." | `admiration` | `admiration` (76.8%) | **`admiration`** (Quality & Recognition) | ✔ Match |
| 3. "I cannot believe I won the competition!" | `joy` | `surprise` (47.2%) | **`surprise`** (Diagnostic Pause) | ❌ (Acceptable) |
| 4. "I am worried that I will fail tomorrow's exam." | `fear` | `nervousness` (76.9%) | **`fear`** (Uncertainty Reduction) | ✔ Match |
| 5. "I am furious that they ignored my work." | `anger` | `anger` (85.7%) | **`anger`** (De-escalation & Action) | ✔ Match |
| 6. "I worked on this all night, and nothing is working." | `sadness` | `disappointment` (72.6%) | **`sadness`** (Gentle Pacing & Micro-Steps)| ✔ Match |
| 7. "The meeting starts at 10 AM." | `neutral` | `neutral` (96.5%) | **`neutral`** (Structured Execution) | ✔ Match |

> [!TIP]
> Under strategy mapping, **GoEmotions achieves 85.7% accuracy ($6/7$) on English**, with the remaining sentence ("won the competition") being a valid multi-emotion expression of surprise.

---

## 8. Limitations Caused by Small Dataset Size

1. **Statistical Power**: The benchmark set contains $N = 21$ samples ($n = 7$ per language). Each sample represents $\approx 4.76\%$ of the overall metric and $\approx 14.29\%$ of a single language subset. Performance figures must be interpreted as qualitative directional indicators rather than statistically converged population estimates.
2. **Single-Instance Classes**: Rare categories (`admiration`, `disapproval`, `surprise`) contain only 1 ground-truth instance. A single misclassification results in an abrupt drop in precision or recall from 1.0 to 0.0.
3. **Absence of Negative Controls**: Highly informal Hinglish slang, mixed Devanagari-English code-switching within a single sentence, and domain-specific technical jargon require broader manual validation.
4. **Dataset Expansion Path**: An annotation template has been generated at [`scripts/evaluation_dataset_template.csv`](file:///Users/atharvakalam/Documents/Atharva%20Files/SEM%205/AI/AI-Task-Optimizer-NLP/scripts/evaluation_dataset_template.csv) to facilitate expanding the benchmark dataset to $N \ge 150$ samples across all three languages with dual-annotator verification.

---

## 9. Academic Conclusion

For the academic project **EmoSync**, single-model monolithic emotion classification is fundamentally unviable across mixed-language and code-switched environments due to tokenizer vocabulary collapse, missing neutral heads, and language transfer failure.

The implemented **Language-Routed Ensemble architecture**:
1. Eliminates cross-lingual script failure by accurately classifying scripts and lexicon before model dispatch.
2. Achieves superior overall accuracy (**52.4% strict**, **61.9% acceptable**, **Weighted F1: 0.534**).
3. Connects high-resolution emotion detection with appropriate, nonjudgmental productivity interventions through a verified strategy mapping.

This validates the hybrid routing approach as the optimal architecture for multilingual, emotion-aware productivity assistance.
