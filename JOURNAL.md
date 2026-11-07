# Engineering Journal — Day 6: Evaluation Suite & Findings Report

## 1. What I Studied
- **Metric Limitations:** Why accuracy hides minority class collapse and why precision, recall, and F1-score are necessary for multi-class tasks.
- **Confusion Matrices:** Analyzing false positives and false negatives across a 3x3 contingency grid.
- **Data Scarcity in NLP:** Observed how statistical models (Naive Bayes) fail when evaluation vocabulary is missing from small training sets.

## 2. Core Notes
1. **Precision:** True Positives / (True Positives + False Positives); evaluates prediction reliability.
2. **Recall:** True Positives / (True Positives + False Negatives); evaluates class coverage.
3. **F1-Score:** The harmonic mean of precision and recall.
4. **Holdout Evaluation:** Evaluating both models on the identical stratified test split ensures fair, unskewed comparisons.
5. **Zero-Valence Heuristic:** VADER reliably classifies factual statements as neutral because factual descriptions lack emotional tokens.

## 3. What I Built
- Upgraded `sentiment.py` to:
  - Generate and export dataset balance charts (`label_distribution.png`).
  - Compute `accuracy_score`, `classification_report`, and `confusion_matrix`.
  - Render dual confusion matrices (`confusion_matrices.png`).
  - Output automated failure audits and export exact metrics to `results.txt`.
- Authored a one-page comparative findings document (`REPORT.md`).

## 4. Tests Run & Empirical Findings
- **VADER Accuracy:** 80.00% (12/15 correct). Flawless precision on positive (1.00) and negative (1.00), with 100% recall on neutral statements.
- **Naive Bayes Accuracy:** 33.33% (5/15 correct). Collapsed to random-chance performance due to vocabulary sparsity on a 58-row training split.
- **Key Discrepancies:**
  - VADER missed descriptive physical flaws lacking explicit emotional tokens (*"creak"* in *"Cheap plastic materials..."*).
  - Naive Bayes misclassified factual neutral statements (*"dimensions"*, *"meeting"*, *"shipment"*) as polar sentiments.

## 5. End-to-End Explanation (5 Lines)
1. `sentiment.py` loads `reviews.csv` and renders `label_distribution.png`.
2. The dataset is split into an 80/20 stratified holdout set using a fixed random seed.
3. Naive Bayes is trained on the training split, while VADER processes the test set using lexicon rules.
4. Precision, recall, F1, and confusion matrices are generated for both approaches and exported to `confusion_matrices.png` and `results.txt`.
5. Misclassified test sentences are audited and analyzed in `REPORT.md`.

## 6. What I Don't Understand Yet / Next Steps
- Implementing cross-validation (`cross_val_score`) to produce confidence intervals on small datasets.
- Exploring sub-word tokenization (Byte-Pair Encoding) to reduce out-of-vocabulary failures.

## 7. Tomorrow's Plan (Day 7: Task 4 - Recommendation System)
- Choose between a curated table of 15–20 movies/books or the MovieLens dataset.
- Build a content-based recommendation engine using cosine similarity.