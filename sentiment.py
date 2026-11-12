import os
import re
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

# ------------------------------------------------------------------------------
# 1. Dependency Initialization & VADER Lexicon Setup
# ------------------------------------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


# ------------------------------------------------------------------------------
# 2. Text Preprocessing
# ------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Cleans raw text while preserving punctuation and casing needed by VADER."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------------
# 3. Approach A: VADER
# ------------------------------------------------------------------------------
def vader_classify(text: str) -> Tuple[str, float]:
    """Classifies text using NLTK VADER compound polarity thresholds."""
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", 0.0

    scores = sia.polarity_scores(cleaned)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, compound


# ------------------------------------------------------------------------------
# 4. Supervised Pipeline & Evaluation Suite
# ------------------------------------------------------------------------------
def run_evaluation_suite(data_path: str = "reviews.csv") -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset '{data_path}' not found!")

    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(clean_text)

    # 1. Plot & Save Dataset Label Distribution
    plt.figure(figsize=(6, 4))
    counts = df["label"].value_counts()
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    counts.plot(kind="bar", color=colors, edgecolor="black")
    plt.title("Dataset Class Distribution (reviews.csv)", fontsize=12, fontweight="bold")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150)
    plt.close()
    print(" Saved 'label_distribution.png'")

    # 2. Stratified 80/20 Train-Test Split (Exact same split for both)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # 3. Train Naive Bayes Pipeline
    nb_pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", lowercase=True),
        MultinomialNB(),
    )
    nb_pipeline.fit(X_train, y_train)

    # 4. Predictions on Unseen Test Split
    y_pred_nb = nb_pipeline.predict(X_test)
    y_pred_vader = [vader_classify(t)[0] for t in X_test]

    # Metrics
    labels = ["positive", "neutral", "negative"]
    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_vader = accuracy_score(y_test, y_pred_vader)

    rep_nb = classification_report(
        y_test, y_pred_nb, labels=labels, target_names=labels, zero_division=0
    )
    rep_vader = classification_report(
        y_test, y_pred_vader, labels=labels, target_names=labels, zero_division=0
    )

    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels)
    cm_vader = confusion_matrix(y_test, y_pred_vader, labels=labels)

    # 5. Plot & Save Dual Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    disp_vader = ConfusionMatrixDisplay(confusion_matrix=cm_vader, display_labels=labels)
    disp_vader.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"VADER Confusion Matrix\nAccuracy: {acc_vader * 100:.2f}%", fontweight="bold")

    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=labels)
    disp_nb.plot(ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title(f"Naive Bayes Confusion Matrix\nAccuracy: {acc_nb * 100:.2f}%", fontweight="bold")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()
    print(" Saved 'confusion_matrices.png'")

    # 6. Terminal Output & Export to results.txt
    output_lines = []
    output_lines.append("============================================================")
    output_lines.append("           TASK 5 MODEL EVALUATION SUMMARY                  ")
    output_lines.append("============================================================")
    output_lines.append(f"Total Dataset Rows: {len(df)}")
    output_lines.append(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    output_lines.append("-" * 60)
    output_lines.append(f"Approach A: VADER Test Accuracy        : {acc_vader * 100:.2f}%")
    output_lines.append(f"Approach B: Naive Bayes Test Accuracy  : {acc_nb * 100:.2f}%\n")
    output_lines.append("VADER Classification Report:")
    output_lines.append(rep_vader)
    output_lines.append("-" * 60)
    output_lines.append("Naive Bayes Classification Report:")
    output_lines.append(rep_nb)
    output_lines.append("============================================================")
    output_lines.append("                 TEST SET ERROR AUDIT                       ")
    output_lines.append("============================================================")

    for text, true_lbl, v_pred, nb_p in zip(X_test, y_test, y_pred_vader, y_pred_nb):
        if true_lbl != v_pred or true_lbl != nb_p:
            output_lines.append(f'Text: "{text}"')
            output_lines.append(f"  -> True: {true_lbl:<8} | VADER: {v_pred:<8} | Naive Bayes: {nb_p:<8}")

    results_text = "\n".join(output_lines)
    print("\n" + results_text)

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(results_text)
    print("\n Exported exact numerical metrics to 'results.txt'")


if __name__ == "__main__":
    run_evaluation_suite("reviews.csv")