"""Model training, evaluation and persistence helpers.

This module provides spam classification model training and evaluation functionality.
"""
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def build_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int = 10000
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer with customizable parameters.

    Args:
        ngram_range: Range of n-gram sizes to consider
        min_df: Minimum document frequency for terms
        max_features: Maximum number of features to keep

    Returns:
        Configured TF-IDF vectorizer
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        stop_words="english"
    )


def evaluate_model(model, X_test, y_test) -> Dict[str, Union[str, float, np.ndarray]]:
    """Evaluate model performance with multiple metrics.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels

    Returns:
        Dict containing evaluation metrics and plots
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Basic metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "roc": {"fpr": fpr, "tpr": tpr, "auc": roc_auc},
        "pr": {"precision": precision, "recall": recall, "auc": pr_auc}
    }


def plot_evaluation(eval_metrics: Dict) -> Dict[str, plt.Figure]:
    """Create evaluation plots from metrics.

    Args:
        eval_metrics: Dictionary of evaluation metrics from evaluate_model()

    Returns:
        Dictionary of matplotlib figures
    """
    plots = {}
    
    # Confusion Matrix
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(
        eval_metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plots["confusion_matrix"] = fig_cm

    # ROC Curve
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(
        eval_metrics["roc"]["fpr"],
        eval_metrics["roc"]["tpr"],
        label=f'ROC (AUC = {eval_metrics["roc"]["auc"]:.2f})'
    )
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plots["roc_curve"] = fig_roc

    # Precision-Recall Curve
    fig_pr = plt.figure(figsize=(8, 6))
    plt.plot(
        eval_metrics["pr"]["recall"],
        eval_metrics["pr"]["precision"],
        label=f'P-R (AUC = {eval_metrics["pr"]["auc"]:.2f})'
    )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plots["pr_curve"] = fig_pr

    return plots


def train_multiple_models(
    messages,
    labels,
    random_state: int = 42,
    test_size: float = 0.2,
    cross_validate: bool = True
) -> Dict:
    """Train and evaluate multiple classifier models.
    
    Returns:
        Dict containing trained models, vectorizer, and comparative metrics
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    
    # Initialize vectorizer and transform data
    vec = build_vectorizer()
    X = vec.fit_transform(messages)
    
    # Split data
    if len(messages) < 10:
        X_train, y_train = X, labels
        X_test, y_test = X, labels
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size,
            stratify=labels, random_state=random_state
        )
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(
            solver='liblinear', class_weight='balanced',
            random_state=random_state, max_iter=1000
        ),
        'SVM': SVC(
            kernel='rbf', C=10.0, gamma='scale',
            probability=True, class_weight='balanced',
            random_state=random_state, max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
        'Naive Bayes': MultinomialNB()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        eval_metrics = evaluate_model(model, X_test, y_test)
        
        if cross_validate and len(messages) >= 10:
            cv_scores = cross_val_score(
                model, X, labels, cv=5,
                scoring="roc_auc"
            )
            eval_metrics["cross_val"] = {
                "scores": cv_scores,
                "mean": cv_scores.mean(),
                "std": cv_scores.std()
            }
        
        results[name] = {
            'model': model,
            'metrics': eval_metrics
        }
    
    return {
        'vectorizer': vec,
        'models': results
    }

def train_model(
    messages,
    labels,
    random_state: int = 42,
    test_size: float = 0.2,
    cross_validate: bool = True
) -> Tuple[object, object, Dict]:
    """Train and evaluate a spam classification model.

    Args:
        messages: List of text messages
        labels: List of labels (0 for ham, 1 for spam)
        random_state: Random seed for reproducibility
        test_size: Proportion of data to use for testing
        cross_validate: Whether to perform cross-validation

    Returns:
        Tuple of (vectorizer, model, evaluation_metrics)
    """
    vec = build_vectorizer()
    X = vec.fit_transform(messages)

    # For very small datasets, use all data
    if len(messages) < 10:
        X_train, y_train = X, labels
        X_test, y_test = X, labels
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size,
            stratify=labels, random_state=random_state
        )

    # Use SVM with optimized parameters for better convergence
    from sklearn.svm import SVC
    model = SVC(
        kernel='rbf',  # RBF kernel for non-linear data
        C=10.0,       # Regularization parameter
        gamma='scale', # Kernel coefficient
        probability=True,  # Enable probability estimates
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        max_iter=1000  # Increase max iterations for convergence
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    eval_metrics = evaluate_model(model, X_test, y_test)
    
    # Add cross-validation scores if requested
    if cross_validate and len(messages) >= 10:
        cv_scores = cross_val_score(
            model, X, labels, cv=5,
            scoring="roc_auc"
        )
        eval_metrics["cross_val"] = {
            "scores": cv_scores,
            "mean": cv_scores.mean(),
            "std": cv_scores.std()
        }

    return vec, model, eval_metrics


def save_pipeline(vectorizer, model, path: str = "models/spam_classifier.joblib") -> None:
    joblib.dump({"vectorizer": vectorizer, "model": model}, path)


def load_pipeline(path: str = "models/spam_classifier.joblib") -> dict:
    return joblib.load(path)
