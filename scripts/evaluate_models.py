#!/usr/bin/env python3
"""Quantitative Model Evaluation Script for EmoSync Emotion Classifiers.

Loads benchmark results from `scripts/benchmark_results.json` and calculates:
1. Strict Single-Label Accuracy & Acceptable-Label Accuracy
2. Macro-Averaged & Weighted Precision, Recall, and F1-Scores
3. Per-Class Precision, Recall, F1, and Support
4. Confusion Matrices (per model and language subset)
5. Comprehensive Scikit-Learn Classification Reports

Evaluates:
- `amaan00z/emotion_xlmr3` (Hinglish-specialized XLM-RoBERTa)
- `bhadresh-savani/distilbert-base-uncased-emotion` (English-specialized DistilBERT)
- `vashuag/HindiEmotion` (Hindi-specialized IndicBERT)
- `SamLowe/roberta-base-go_emotions` (English fine-grained RoBERTa)
- `EmoSync Language-Routed Ensemble` (Production multi-model architecture)

Subsets evaluated:
- Overall Combined (N=21)
- English (N=7)
- Hindi Devanagari (N=7)
- Hinglish Romanized (N=7)

Outputs:
- `scripts/evaluation_metrics.json`
- `scripts/evaluation_summary.csv`
- `scripts/confusion_matrices/` (CSV and text matrices)
- `scripts/evaluation_dataset_template.csv`

Usage:
    python3 scripts/evaluate_models.py
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    print("Error: scikit-learn is required. Install it using 'pip install scikit-learn'.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants and Taxonomy Metadata
# ---------------------------------------------------------------------------

MODEL_METADATA = {
    "amaan00z/emotion_xlmr3": {
        "name": "XLM-RoBERTa (Hinglish)",
        "short_name": "emotion_xlmr3",
        "intended_language": "Hinglish (Code-Mixed)",
        "label_count": 10,
        "taxonomy": [
            "admiration", "anger", "disapproval", "disgust", "fear",
            "joy", "love", "neutral", "sadness", "surprise"
        ],
        "taxonomy_limitations": "Trained on code-mixed Hinglish; lacks pure English and Devanagari Hindi representation; high false-neutral bias on English.",
    },
    "bhadresh-savani/distilbert-base-uncased-emotion": {
        "name": "DistilBERT (English)",
        "short_name": "distilbert_emotion",
        "intended_language": "English",
        "label_count": 6,
        "taxonomy": [
            "anger", "fear", "joy", "love", "sadness", "surprise"
        ],
        "taxonomy_limitations": "Completely lacks 'neutral' class; lacks 'admiration', 'disapproval', 'disgust'; cannot process Devanagari Hindi; collapses to 'joy' or 'anger' out-of-domain.",
    },
    "vashuag/HindiEmotion": {
        "name": "IndicBERT (Hindi Devanagari)",
        "short_name": "hindi_indicbert",
        "intended_language": "Hindi (Devanagari)",
        "label_count": 7,
        "taxonomy": [
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        ],
        "taxonomy_limitations": "Devanagari Hindi specialized; lacks 'admiration', 'disapproval', 'love'; fails on Latin/English scripts by collapsing to 'neutral'.",
    },
    "SamLowe/roberta-base-go_emotions": {
        "name": "RoBERTa (English GoEmotions)",
        "short_name": "roberta_go_emotions",
        "intended_language": "English",
        "label_count": 28,
        "taxonomy": [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "neutral", "optimism", "pride",
            "realization", "relief", "remorse", "sadness", "surprise"
        ],
        "taxonomy_limitations": "Fine-grained 28-class taxonomy causes label mismatch under strict single-label matching (e.g. 'excitement' vs 'joy', 'nervousness' vs 'fear'); collapses to 'neutral' on Hindi/Hinglish.",
    },
    "ensemble": {
        "name": "EmoSync Language-Routed Ensemble",
        "short_name": "emosync_ensemble",
        "intended_language": "Multilingual (Routed by script & lexicon)",
        "label_count": 28,
        "taxonomy": [
            "admiration", "anger", "disapproval", "disgust", "fear",
            "joy", "love", "neutral", "sadness", "surprise",
            "excitement", "nervousness", "disappointment"  # Observed subset
        ],
        "taxonomy_limitations": "Combines RoBERTa (English), IndicBERT (Hindi), and XLM-RoBERTa (Hinglish). Performance on code-mixed Hinglish remains constrained by XLM-RoBERTa baseline.",
    },
}

# GoEmotions to Core Emotion Strategy Mapping (from recommendation engine)
GO_EMOTIONS_MAPPING = {
    "excitement": "joy",
    "optimism": "joy",
    "pride": "joy",
    "amusement": "joy",
    "desire": "joy",
    "caring": "love",
    "gratitude": "admiration",
    "approval": "admiration",
    "relief": "neutral",
    "realization": "neutral",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "annoyance": "anger",
    "nervousness": "fear",
    "embarrassment": "fear",
    "curiosity": "surprise",
    "confusion": "surprise",
}


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def load_benchmark_data(json_path: str) -> List[Dict[str, Any]]:
    """Load benchmark dataset from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Benchmark results file not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("detailed_results", [])


def extract_predictions(
    detailed_results: List[Dict[str, Any]], model_key: str
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Extract ground truth, predicted top label, and acceptable labels for a model.

    For 'ensemble', dynamically routes English to GoEmotions, Hindi to HindiEmotion,
    and Hinglish to emotion_xlmr3.
    """
    y_true = []
    y_pred = []
    acceptable_list = []

    for item in detailed_results:
        lang = item["language"]
        exp = item["expected_label"]
        acc = item["acceptable_labels"]

        if model_key == "ensemble":
            if lang == "English":
                p = item["predictions"]["SamLowe/roberta-base-go_emotions"]["top_label"]
            elif lang == "Hindi":
                p = item["predictions"]["vashuag/HindiEmotion"]["top_label"]
            else:  # Hinglish
                p = item["predictions"]["amaan00z/emotion_xlmr3"]["top_label"]
        else:
            p = item["predictions"][model_key]["top_label"]

        y_true.append(exp)
        y_pred.append(p)
        acceptable_list.append(acc)

    return y_true, y_pred, acceptable_list


def evaluate_model_subset(
    y_true: List[str],
    y_pred: List[str],
    acceptable_list: List[List[str]],
    model_key: str,
    subset_name: str,
) -> Dict[str, Any]:
    """Compute complete scikit-learn evaluation metrics for a single model and subset."""
    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    # 1. Accuracy
    acc_strict = float(accuracy_score(y_true, y_pred))
    acc_acceptable = float(
        sum(1 for pred, acc in zip(y_pred, acceptable_list) if pred in acc) / n_samples
    )

    # 2. Consistent Label Set (union of true and predicted labels, alphabetically sorted)
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    ref_classes = sorted(list(set(y_true)))

    # 3. Macro & Weighted Metrics across all active classes
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_classes, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_classes, average="weighted", zero_division=0
    )

    # 4. Per-Class Metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=all_classes, average=None, zero_division=0
        )
    )

    per_class_dict = {}
    for idx, cls_name in enumerate(all_classes):
        per_class_dict[cls_name] = {
            "precision": round(float(prec_per_class[idx]), 4),
            "recall": round(float(rec_per_class[idx]), 4),
            "f1_score": round(float(f1_per_class[idx]), 4),
            "support": int(support_per_class[idx]),
            "in_reference_set": cls_name in ref_classes,
        }

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_dict = {
        "labels": all_classes,
        "matrix": cm.tolist(),
    }

    # 6. Detailed Scikit-Learn Classification Report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_classes,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=all_classes,
        zero_division=0,
    )

    return {
        "model_key": model_key,
        "model_name": MODEL_METADATA.get(model_key, {}).get("name", model_key),
        "subset": subset_name,
        "sample_count": n_samples,
        "accuracy_strict": round(acc_strict, 4),
        "accuracy_acceptable": round(acc_acceptable, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "active_classes": all_classes,
        "reference_classes": ref_classes,
        "per_class": per_class_dict,
        "confusion_matrix": cm_dict,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def format_confusion_matrix_table(cm_dict: Dict[str, Any]) -> str:
    """Format confusion matrix as a clear text table."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]
    max_label_len = max(max(len(l) for l in labels), 8)

    header = f"{'True \\ Pred':<{max_label_len+4}} | " + " | ".join(
        f"{l[:7]:>7}" for l in labels
    )
    separator = "-" * len(header)
    rows = [header, separator]

    for idx, true_label in enumerate(labels):
        row_vals = " | ".join(f"{matrix[idx][j]:>7}" for j in range(len(labels)))
        rows.append(f"{true_label:<{max_label_len+4}} | {row_vals}")

    return "\n".join(rows)


def save_confusion_matrices(
    all_metrics: Dict[str, Dict[str, Any]], output_dir: str
) -> None:
    """Save confusion matrices as CSV and text files."""
    os.makedirs(output_dir, exist_ok=True)

    for model_key, model_subsets in all_metrics.items():
        short_name = MODEL_METADATA.get(model_key, {}).get("short_name", model_key.replace("/", "_"))
        for subset_name, metrics in model_subsets.items():
            if not metrics:
                continue

            cm_dict = metrics["confusion_matrix"]
            labels = cm_dict["labels"]
            matrix = cm_dict["matrix"]

            # Save CSV
            csv_path = os.path.join(output_dir, f"cm_{short_name}_{subset_name.lower()}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["True\\Pred"] + labels)
                for idx, row in enumerate(matrix):
                    writer.writerow([labels[idx]] + row)

            # Save Text Table
            txt_path = os.path.join(output_dir, f"cm_{short_name}_{subset_name.lower()}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Confusion Matrix: {metrics['model_name']} [{subset_name}] (N={metrics['sample_count']})\n")
                f.write(f"Strict Accuracy: {metrics['accuracy_strict']*100:.1f}% | Acceptable: {metrics['accuracy_acceptable']*100:.1f}%\n\n")
                f.write(format_confusion_matrix_table(cm_dict))
                f.write("\n\nClassification Report:\n")
                f.write(metrics["classification_report_text"])


def save_summary_csv(
    all_metrics: Dict[str, Dict[str, Any]], csv_path: str
) -> None:
    """Save tabular evaluation summary across all models and subsets."""
    headers = [
        "Model",
        "Model_ID",
        "Language_Subset",
        "Sample_Count",
        "Strict_Accuracy",
        "Acceptable_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_Precision",
        "Weighted_Recall",
        "Weighted_F1",
        "Active_Classes_Count",
        "Taxonomy_Limitations",
    ]

    rows = []
    for model_key, subsets in all_metrics.items():
        meta = MODEL_METADATA.get(model_key, {})
        for subset_name, m in subsets.items():
            if not m:
                continue
            rows.append([
                meta.get("name", model_key),
                model_key,
                subset_name,
                m["sample_count"],
                f"{m['accuracy_strict']:.4f}",
                f"{m['accuracy_acceptable']:.4f}",
                f"{m['macro_precision']:.4f}",
                f"{m['macro_recall']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{m['weighted_precision']:.4f}",
                f"{m['weighted_recall']:.4f}",
                f"{m['weighted_f1']:.4f}",
                len(m["active_classes"]),
                meta.get("taxonomy_limitations", ""),
            ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_dataset_template(template_path: str) -> None:
    """Generate CSV template for future manual dataset expansion."""
    headers = ["text", "language", "true_label", "acceptable_synonyms", "domain_notes"]
    example_rows = [
        [
            "I finally solved the memory leak after four hours!",
            "English",
            "joy",
            "joy;relief",
            "Technical milestone celebration",
        ],
        [
            "मेरा प्रोजेक्ट आज समय पर पूरा हो गया।",
            "Hindi",
            "joy",
            "joy;gratitude",
            "Devanagari task completion",
        ],
        [
            "Bhai presentation achhi gayi sabne tareef ki.",
            "Hinglish",
            "joy",
            "joy;admiration",
            "Code-mixed positive peer feedback",
        ],
        [
            "The client changed all requirements at the last minute.",
            "English",
            "anger",
            "anger;disapproval",
            "Scope creep frustration",
        ],
        [
            "बहुत ज्यादा काम बचा है और नींद भी आ रही है।",
            "Hindi",
            "sadness",
            "sadness;fatigue",
            "Workload exhaustion",
        ],
        [
            "Kal deadline hai aur deploy fail ho gaya yaar.",
            "Hinglish",
            "fear",
            "fear;sadness",
            "Urgent blocker anxiety",
        ],
        [
            "Weekly team sync is moved to Thursday 3 PM.",
            "English",
            "neutral",
            "neutral",
            "Informational calendar update",
        ],
    ]

    with open(template_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(example_rows)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(root_dir, "scripts", "benchmark_results.json")
    output_dir = os.path.join(root_dir, "scripts")
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    metrics_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    summary_csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    template_csv_path = os.path.join(output_dir, "evaluation_dataset_template.csv")

    print("=" * 80)
    print("EMOSYNC QUANTITATIVE MODEL EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Loading benchmark dataset from: {json_path}")
    detailed_results = load_benchmark_data(json_path)
    total_n = len(detailed_results)
    print(f"Loaded {total_n} verified evaluation samples across 3 language subsets.\n")

    # Define models and subsets
    models_to_eval = [
        "amaan00z/emotion_xlmr3",
        "bhadresh-savani/distilbert-base-uncased-emotion",
        "vashuag/HindiEmotion",
        "SamLowe/roberta-base-go_emotions",
        "ensemble",
    ]

    subsets = {
        "Overall": detailed_results,
        "English": [s for s in detailed_results if s["language"] == "English"],
        "Hindi": [s for s in detailed_results if s["language"] == "Hindi"],
        "Hinglish": [s for s in detailed_results if s["language"] == "Hinglish"],
    }

    all_metrics: Dict[str, Dict[str, Any]] = {}

    for model_key in models_to_eval:
        all_metrics[model_key] = {}
        for subset_name, subset_items in subsets.items():
            y_true, y_pred, acceptable = extract_predictions(subset_items, model_key)
            metrics = evaluate_model_subset(
                y_true, y_pred, acceptable, model_key, subset_name
            )
            all_metrics[model_key][subset_name] = metrics

    # Print Terminal Summary Table
    print(f"{'Model Candidate':<34} | {'Subset':<8} | {'N':<3} | {'Strict Acc':<10} | {'Accept Acc':<10} | {'Macro F1':<8} | {'Weighted F1':<11}")
    print("-" * 96)
    for model_key in models_to_eval:
        meta = MODEL_METADATA.get(model_key, {})
        for subset_name in ["Overall", "English", "Hindi", "Hinglish"]:
            m = all_metrics[model_key][subset_name]
            print(
                f"{meta['name']:<34} | {subset_name:<8} | {m['sample_count']:<3} | "
                f"{m['accuracy_strict']*100:6.1f}%    | {m['accuracy_acceptable']*100:6.1f}%    | "
                f"{m['macro_f1']:6.3f}   | {m['weighted_f1']:7.3f}"
            )
        print("-" * 96)

    # Print Confusion Matrix Sample for Ensemble Overall
    ens_overall_cm = all_metrics["ensemble"]["Overall"]["confusion_matrix"]
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX: EmoSync Language-Routed Ensemble [Overall Dataset, N=21]")
    print("=" * 80)
    print(format_confusion_matrix_table(ens_overall_cm))

    # Save Output Files
    print("\nSaving evaluation artifacts...")

    # 1. JSON
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "project": "EmoSync — Emotion-Aware Productivity Assistant",
                    "total_samples": total_n,
                    "language_distribution": {
                        "English": len(subsets["English"]),
                        "Hindi": len(subsets["Hindi"]),
                        "Hinglish": len(subsets["Hinglish"]),
                    },
                    "models_evaluated": list(MODEL_METADATA.keys()),
                    "evaluation_standard": "Scikit-Learn 1.8.0 exact strict matching & benchmark acceptable tolerance",
                },
                "results": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"✔ Saved full metrics JSON: {metrics_json_path}")

    # 2. CSV Summary
    save_summary_csv(all_metrics, summary_csv_path)
    print(f"✔ Saved summary CSV: {summary_csv_path}")

    # 3. Confusion Matrices
    save_confusion_matrices(all_metrics, cm_dir)
    print(f"✔ Saved {len(models_to_eval) * len(subsets)} confusion matrices in: {cm_dir}/")

    # 4. Dataset Template
    generate_dataset_template(template_csv_path)
    print(f"✔ Saved dataset expansion template: {template_csv_path}")

    print("\nAll quantitative evaluations computed and saved successfully.")


if __name__ == "__main__":
    main()
