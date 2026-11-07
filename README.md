# CodeOrbit Sentiment Analysis & Evaluation Pipeline (Day 6)

A comparative NLP benchmarking suite evaluating rule-based sentiment classification (**VADER**) against supervised machine learning (**TF-IDF + Multinomial Naive Bayes**).

## System Architecture

- **Approach A (VADER):** Rule-based emotional valence analyzer using punctuation, capitalization, and negation heuristics.
- **Approach B (TF-IDF + Naive Bayes):** Supervised scikit-learn pipeline using term-frequency inverse-document-frequency weighting and multinomial probability estimation.
- **Evaluation Suite:** Generates stratified confusion matrices, class distribution plots, classification reports, and automated error audits.

---

## Dataset & Distribution
- **Dataset Size:** 73 labeled reviews across positive, neutral, and negative targets.
- **Split:** Stratified 80% train (58 samples), 20% test (15 samples).

![Class Distribution](label_distribution.png)

---

## Empirical Benchmark Summary

| Metric | VADER | Naive Bayes |
|---|---|---|
| **Test Accuracy** | **80.00%** | **33.33%** |
| **Positive F1** | **0.75** | 0.36 |
| **Neutral F1** | **0.77** | 0.25 |
| **Negative F1** | **0.89** | 0.36 |
| **Macro Average F1** | **0.80** | 0.33 |

![Confusion Matrices](confusion_matrices.png)

For detailed per-class breakdowns, test sentence audits, and failure root-cause analysis, read the full [REPORT.md](REPORT.md).

---

## How to Run

1. Activate your virtual environment:
   ```bash
   source .venv/Scripts/activate  # On Windows Git Bash