# Steps to build a Spam Email Classifier

This document describes step-by-step instructions to build a Spam Email Classifier using the dataset
located at:

`C:\Users\anny4\OneDrive\桌面\資安HW3\sms_spam_no_header.csv`

This project builds upon patterns from Chapter 3 of the Packt repository (Spam Email problem) and
expands preprocessing, visualization, and result delivery via CLI and Streamlit.

---

## Quick overview / contract

- Input: CSV file `sms_spam_no_header.csv` (assumed two columns: label and message; no header).
- Output: Trained model, evaluation metrics, visualizations, and optional CLI/Streamlit UI to classify new messages.
- Success criteria: working pipeline that loads data, trains at least one baseline model (e.g., MultinomialNB),
  produces evaluation metrics (precision, recall, F1, ROC-AUC), and exposes a simple UI for predictions.

If your CSV actually contains a header or different column ordering, adjust the loading step accordingly.

## Recommended project layout

Organize files like:

- data/
  - sms_spam_no_header.csv  # original dataset (gitignored if large)
- notebooks/                # EDA and visualization notebooks
- src/
  - data.py                 # loading & preprocessing helpers
  - features.py             # feature engineering
  - models.py               # training/evaluation functions
  - cli.py                  # simple CLI wrapper
  - app_streamlit.py        # Streamlit app for interactive demo
- tests/                    # pytest tests
- requirements.txt
- Step.md                   # this file

---

## Environment setup (PowerShell)

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
. \.venv\Scripts\Activate.ps1
```

2. Install required packages (example `requirements.txt` content):

```text
pandas
scikit-learn
numpy
nltk
matplotlib
seaborn
wordcloud
joblib
streamlit
```

Install them:

```powershell
pip install -r requirements.txt
```

3. If you plan to use Docker, create a `Dockerfile` and optional `docker-compose.yml` for reproducibility.

---

## Data loading & first checks

Because your file name ends with `_no_header.csv` we assume there is no header and two columns in order: label, message.

Example load (pandas):

```python
import pandas as pd

path = r"C:\Users\anny4\OneDrive\桌面\資安HW3\sms_spam_no_header.csv"
df = pd.read_csv(path, header=None, names=["label", "message"], encoding="utf-8", sep=",")
print(df.shape)
print(df['label'].value_counts())
df.head()
```

Checks to run and why:

- Confirm shape — expected ~ (n_samples, 2).
- Confirm label set — typically `ham` and `spam` (or `0/1`). Normalize if needed.
- Inspect message length distribution (very short vs very long messages).
- Check for missing values; drop or impute as needed.

---

## Exploratory data analysis (EDA)

Produce the following EDA outputs and save them under `notebooks/` or `outputs/figs/`:

- Class distribution bar chart.
- Top 20 most frequent words for `ham` and `spam` (use simple tokenization and stopword removal).
- WordClouds for `ham` and `spam`.
- Distribution of message lengths.

Tools: matplotlib/seaborn, wordcloud. Save PNGs for inclusion in reports.

---

## Preprocessing pipeline

Create a pipeline composed of deterministic, testable steps. Example pipeline:

1. Normalization
   - Lowercase text.
   - Strip leading/trailing whitespace.
   - Replace URLs and email addresses with tokens (e.g., `<URL>`, `<EMAIL>`).
2. Noise removal
   - Remove or replace numbers, excessive punctuation, and repeated characters (optional).
3. Tokenization
   - Use NLTK, SpaCy, or simple split. For fast baseline use sklearn's CountVectorizer tokenizer.
4. Stopwords
   - Remove standard English stopwords (NLTK) but be cautious: stopword removal can remove useful tokens in spam detection.
5. Lemmatization / Stemming (optional)
   - Lemmatize with NLTK WordNetLemmatizer or use stemming with PorterStemmer for a smaller vocabulary.
6. Feature creation
   - TF-IDF vectors (word-level) using sklearn's TfidfVectorizer.
   - Character n-grams (useful for obfuscated spam) via TfidfVectorizer(ngram_range=(3,5), analyzer='char').
   - Handcrafted features: message length, count of digits, count of punctuation, number of uppercase words, presence of typical spam tokens ("free", "win", "click").

Implement preprocessing as functions in `src/data.py` and wrap transformers with sklearn's `FunctionTransformer` or `Pipeline` where possible.

---

## Feature engineering

Example feature stack:

- Text features: TF-IDF (unigrams + bigrams), lowercase, min_df tuning.
- Char n-grams: useful for obfuscated words.
- Numeric features: message length, digit count, punctuation count.
- Combine using sklearn's `FeatureUnion` or `ColumnTransformer`.

Example: use `ColumnTransformer` to apply TfidfVectorizer to the `message` column and passthrough numeric features.

---

## Modeling

Start with baseline and iterate:

1. Baselines
   - Multinomial Naive Bayes (fast baseline for discrete features like counts/TF-IDF).
   - Logistic Regression (with L2 regularization, solver='liblinear' or 'saga').
2. Additional models
   - Linear SVM (LinearSVC), RandomForestClassifier, or GradientBoosting.
3. Cross-validation
   - Use StratifiedKFold (k=5 or 10) to preserve class balance.
4. Hyperparameter search
   - GridSearchCV or RandomizedSearchCV on a small grid for `alpha` (NB), `C` (LR/SVM), `n_estimators` and `max_depth` (RF).
   - Optimize for F1-score (or recall if you want to minimize false negatives).

Save the best model with `joblib.dump(best_estimator_, 'models/spam_classifier.joblib')`.

---

## Evaluation & metrics

Report the following:

- Confusion matrix.
- Accuracy, Precision, Recall, F1-score.
- Precision-Recall curve and ROC AUC.
- Per-class precision/recall (important when classes are imbalanced).
- Calibration curve (optional).

Example commands to compute metrics using sklearn:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('ROC AUC:', roc_auc_score(y_true, y_proba[:,1]))
```

For production-like evaluation use a held-out test set (train/validation/test split), or nested CV for unbiased hyperparameter selection.

---

## Visualization of results

- Confusion matrix heatmap (seaborn). Save as PNG.
- Precision-Recall and ROC curves.
- Feature importance / most informative features:
  - For linear models, the largest positive/negative coefficients for a class indicate important tokens.
  - For tree models, use feature importance attribute.
- For interpretability create a small table of sample messages with predicted/true label and model confidence.

---

## CLI & Streamlit view

CLI (simple): `src/cli.py`

- Use `argparse` to accept a message string or a path to a CSV file of messages.
- Load saved model and vectorizer, output predicted label and probabilities.

Example PowerShell invocation:

```powershell
python src\cli.py --message "Congratulations! You've won a free ticket"
```

Streamlit (optional interactive demo): `src/app_streamlit.py`

- Minimal Streamlit app that loads model and vectorizer and provides a text area to input a message and show prediction + probability and explanation (top tokens).

Run:

```powershell
streamlit run src\app_streamlit.py
```

---

## Reproducibility

- Pin package versions in `requirements.txt`.
- Set random seeds at the start of experiments:

```python
import numpy as np
import random
from sklearn.utils import check_random_state

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

- Save preprocessing artifacts (vectorizer, label encoder) alongside the model using `joblib`.
- Provide a `Dockerfile` and `docker-compose.yml` for a reproducible environment if needed.

---

## Testing

- Add unit tests for preprocessing functions (empty string, only punctuation, numbers-only).
- Add integration tests that train a tiny model on a subsample to ensure the training pipeline runs end-to-end.

Example pytest test idea:

```python
def test_tokenize_empty_message():
    assert tokenize("") == []
```

---

## Troubleshooting & common pitfalls

- If labels are not expected strings, map them: e.g., `df['label'] = df['label'].map({'ham':0, 'spam':1})`.
- Encoding errors when reading CSV: try `encoding='latin-1'` or detect encoding.
- Class imbalance: use class_weight='balanced' in models or oversample (SMOTE) for training.
- Overfitting: watch for very high train scores and low validation scores; use regularization and simpler models.

---

## Expected outputs / artifacts

- `models/spam_classifier.joblib` — saved best model and preprocessing pipeline.
- `outputs/figs/*` — saved visualizations (class distribution, wordclouds, ROC/PR, confusion matrix).
- `reports/metrics.md` — a short report listing evaluation metrics.

---

## Sample timeline (small project)

- Day 1: Setup, load data, EDA, and basic preprocessing.
- Day 2: Implement TF-IDF pipeline, baseline models, evaluation, and visualizations.
- Day 3: Hyperparameter tuning, CLI, Streamlit demo, and tests & README.

---

## References

- Packt Chapter 3 (Spam Email problem) patterns and dataset.
- scikit-learn documentation for text classification tutorials.

---

If you'd like, I can now:

- Create the skeleton `src/` modules with minimal code for loading, preprocessing, and training.
- Add a `requirements.txt`, `.gitignore` entry for data, and a simple `README.md`.
- Build a Streamlit demo scaffold.

Which of the follow-ups should I do next? 
