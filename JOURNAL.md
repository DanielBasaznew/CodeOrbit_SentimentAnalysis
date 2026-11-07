# Engineering Journal — Day 5: Sentiment Analysis Pipeline (Part 1)

## 1. What I Studied
- **Rule-Based Lexicons (VADER):** How valence dictionaries and heuristic modifiers calculate normalized compound polarity scores within $[-1.0, 1.0]$.
- **TF-IDF Vectorization:** Balancing Term Frequency with Inverse Document Frequency to weight discriminative terms over ubiquitous vocabulary.
- **Multinomial Naive Bayes:** Supervised classification applying Bayes' rule under conditional word independence assumptions.
- **Stratification:** Splitting training and testing sets while preserving identical target class distributions.

## 2. Core Notes
1. **VADER Scoring:** Uses dictionary lookups and grammatical rules without requiring prior training.
2. **VADER Decision Thresholds:** Positive ($\ge 0.05$), Negative ($\le -0.05$), and Neutral (between $-0.05$ and $0.05$).
3. **TF-IDF Mechanism:** Suppresses high-frequency stop words and elevates informative terms specific to individual classes.
4. **Data Starvation in Naive Bayes:** A small training set (58 samples) leads to vocabulary sparsity, resulting in 100% training accuracy but random-guess baseline performance (33.33%) on unseen test sets.
5. **Contextual Limitations:** Sarcasm remains a systemic failure mode for both word-level statistical bags and fixed-lexicon rule sets.

## 3. What I Built
- Configured `CodeOrbit_SentimentAnalysis` with isolated `.venv` and frozen dependencies.
- Created `reviews.csv` with 73 balanced rows across positive, negative, and neutral classes.
- Implemented `sentiment.py` containing:
  - Text normalization preserving punctuation for VADER.
  - VADER compound scorer (`vader_classify`).
  - Scikit-learn pipeline chaining `TfidfVectorizer` and `MultinomialNB`.
  - Side-by-side inference comparator (`predict_both`).

## 4. Tests Run & Empirical Findings
- **Generalization Gap:** Naive Bayes attained 100% training accuracy vs 33.33% testing accuracy due to unseen test tokens.
- **Negation Verification:** `"Not good at all"` correctly resolved to negative in VADER due to proximity negation rules.
- **Sarcasm Failure:** `"Great, another delay..."` yielded false positive classifications across both models.
- **Mixed Clause Divergence:** `"Good but too expensive..."` produced model disagreement (VADER: positive `0.24` vs. Naive Bayes: negative).
- **Edge Conditions:** Handled empty string inputs and short tokens (`"OK"`) safely.

## 5. End-to-End Explanation (5 Lines)
1. `sentiment.py` ingests and balances `reviews.csv` across positive, negative, and neutral targets.
2. Text undergoes whitespace normalization while preserving punctuation.
3. Approach A runs `SentimentIntensityAnalyzer`, checking the compound value against the $[-0.05, 0.05]$ boundaries.
4. Approach B converts training strings to numerical sparse matrices via `TfidfVectorizer` and fits a `MultinomialNB` classifier.
5. The comparison routine iterates over novel test strings, printing predictions from both models.

## 6. What I Don't Understand Yet / Next Steps
- How n-grams (bigrams/trigrams) in `TfidfVectorizer(ngram_range=(1, 2))` improve negation handling in Naive Bayes.
- Methods to mitigate extreme overfitting on sparse corpora using Laplace smoothing or expanded training datasets.

## 7. Tomorrow's Plan (Day 6: Task 5, Part 2)
- Generate a confusion matrix, classification report, and calculate accuracy, precision, and recall metrics.
- Produce a structured one-page findings report evaluating VADER vs. Naive Bayes.