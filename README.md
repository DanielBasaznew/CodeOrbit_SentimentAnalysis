# CodeOrbit Sentiment Analysis Pipeline (Day 5 - Part 1)

A comparative dual-approach natural language processing pipeline evaluating rule-based sentiment classification (**VADER**) against supervised machine learning (**TF-IDF + Multinomial Naive Bayes**).

## System Architecture

### Approach A: VADER (Lexicon & Rule-Based)
- **Heuristic Valence Scoring:** Leverages a curated lexicon mapping tokens to emotional intensity.
- **Rule Modifiers:** Adjusts polarity based on punctuation, ALL-CAPS emphasis, and grammatical negations (e.g., "not good").
- **Decision Boundaries:**
  - Positive: $\ge 0.05$
  - Negative: $\le -0.05$
  - Neutral: between $-0.05$ and $0.05$

### Approach B: TF-IDF + Multinomial Naive Bayes (Supervised ML)
- **Feature Extraction:** `TfidfVectorizer` computes word importance scaled by document rarity, with stop-word filtering.
- **Probabilistic Classification:** `MultinomialNB` estimates posterior class probabilities under a naive feature independence assumption.
- **Stratified Split:** Data split 80/20 with `stratify=df['label']` and fixed random seed (`random_state=42`).

---

## Dataset Characteristics
- **Total Records:** 73 labeled text samples in `reviews.csv`.
- **Class Balance:**
  - Positive: 25 samples
  - Negative: 24 samples
  - Neutral: 24 samples

---

## How to Run

1. Activate virtual environment:
   ```bash
   source .venv/Scripts/activate  # On Windows Git Bash