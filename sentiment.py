import os
import re
from typing import Dict, Tuple
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

# ==============================================================================
# 1. Dependency Initialization & VADER Lexicon Setup
# ==============================================================================
# Download the VADER lexicon silently if not already present locally
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


# ==============================================================================
# 2. Text Preprocessing
# ==============================================================================
def clean_text(text: str) -> str:
    """
    Cleans raw text while preserving punctuation and capitalization nuances
    needed by VADER (e.g. '!' or ALL-CAPS emotional signals).
    """
    if not isinstance(text, str):
        return ""
    # Strip excessive internal spaces while keeping punctuation intact
    return re.sub(r"\s+", " ", text).strip()


# ==============================================================================
# 3. Approach A: VADER (Lexicon-Based Rule Engine)
# ==============================================================================
def vader_classify(text: str) -> Tuple[str, float]:
    """
    Classifies text using NLTK's VADER compound score:
      - score >= 0.05: positive
      - score <= -0.05: negative
      - otherwise: neutral
    Returns a tuple of (predicted_label, compound_score).
    """
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


# ==============================================================================
# 4. Approach B: TF-IDF + Multinomial Naive Bayes (Supervised ML)
# ==============================================================================
def train_naive_bayes_pipeline(data_path: str = "reviews.csv") -> Tuple[Pipeline, pd.DataFrame]:
    """
    Loads labeled dataset, validates balance, splits data 80/20 with stratification,
    and trains a TF-IDF + Multinomial Naive Bayes pipeline.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset '{data_path}' not found!")

    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(clean_text)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Scikit-learn pipeline: Stop-word filtering -> TF-IDF Vectorizer -> Naive Bayes
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", lowercase=True),
        MultinomialNB()
    )

    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    print("============================================================")
    print("  Dataset & Supervised Model Initialization                 ")
    print("============================================================")
    print(f"Total Dataset Rows: {len(df)}")
    print("Class Distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  - {label:<10}: {count} samples")
    print(f"\nTraining Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"Naive Bayes Accuracy -> Train: {train_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%\n")

    return pipeline, df


# ==============================================================================
# 5. Dual Prediction Comparator
# ==============================================================================
def predict_both(text: str, ml_pipeline: Pipeline) -> Dict[str, str]:
    """Runs text through both VADER and Naive Bayes, returning side-by-side predictions."""
    cleaned = clean_text(text)
    v_label, compound = vader_classify(cleaned)

    if not cleaned:
        nb_label = "neutral"
    else:
        nb_label = ml_pipeline.predict([cleaned])[0]

    return {
        "text": text,
        "vader_label": v_label,
        "vader_compound": compound,
        "nb_label": nb_label
    }


def main() -> None:
    # 1. Train supervised model
    nb_model, df = train_naive_bayes_pipeline("reviews.csv")

    # 2. Test cases: Positive, Negative, Neutral, Negations, Sarcasm, Edge Cases
    test_sentences = [
        "I love this, it's amazing!",
        "Terrible quality, waste of money.",
        "The box contains a charger and a cable.",
        "Not good at all, completely disappointing.",
        "Great, another delay just what I needed...",
        "Good but too expensive for the build quality.",
        "The package arrived on Monday morning.",
        "OK",
        ""
    ]

    print("============================================================")
    print("  Dual Model Predictions Comparison                         ")
    print("============================================================")
    print(f"{'Input Text':<42} | {'VADER':<10} | {'V-Compound':<10} | {'Naive Bayes':<11}")
    print("-" * 80)

    for sentence in test_sentences:
        res = predict_both(sentence, nb_model)
        display_text = sentence if len(sentence) <= 40 else sentence[:37] + "..."
        if not display_text:
            display_text = "<empty string>"
        print(f"{display_text:<42} | {res['vader_label']:<10} | {res['vader_compound']:<10.2f} | {res['nb_label']:<11}")

    print("-" * 80)


if __name__ == "__main__":
    main()