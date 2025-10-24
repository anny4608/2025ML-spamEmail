"""Enhanced model training and evaluation with multiple classifiers."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib

from advanced_preprocessing import TextPreprocessor, extract_features

# Define scoring metrics
SCORING = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

class SpamClassifierFactory:
    """Factory for creating different spam classifiers."""
    
    @staticmethod
    def create_classifier(name: str) -> Tuple[str, Pipeline]:
        """Create a classifier pipeline by name."""
        
        # Text preprocessing pipeline
        text_pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', TfidfVectorizer(max_features=5000))
        ])
        
        # Feature extraction pipeline
        feature_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Create the main pipeline with the specified classifier
        if name == "naive_bayes":
            clf = MultinomialNB()
            feature_pipeline = None  # NB doesn't need scaled features
        elif name == "svm":
            clf = LinearSVC(random_state=42)
        elif name == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif name == "gradient_boosting":
            clf = GradientBoostingClassifier(random_state=42)
        elif name == "logistic_regression":
            clf = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {name}")
        
        # Create full pipeline
        if feature_pipeline:
            preprocessor = ColumnTransformer([
                ('text', text_pipeline, 'text'),
                ('features', feature_pipeline, ['length', 'word_count', 'contains_url', 
                                             'contains_number', 'contains_currency',
                                             'contains_exclamation', 'uppercase_ratio'])
            ])
        else:
            preprocessor = ColumnTransformer([
                ('text', text_pipeline, 'text')
            ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        return name, pipeline

def prepare_data(texts: list, labels: list) -> pd.DataFrame:
    """Prepare data for training by extracting features."""
    # Extract additional features
    features = [extract_features(text) for text in texts]
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['text'] = texts
    df['label'] = labels
    
    return df

def evaluate_models(texts: list, labels: list, cv=5) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple models using cross-validation."""
    # Prepare data
    data = prepare_data(texts, labels)
    
    # Models to evaluate
    models = [
        "naive_bayes",
        "svm",
        "random_forest",
        "gradient_boosting",
        "logistic_regression"
    ]
    
    results = {}
    for model_name in models:
        # Create and evaluate model
        name, pipeline = SpamClassifierFactory.create_classifier(model_name)
        cv_results = cross_validate(
            pipeline, data, data['label'],
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Aggregate results
        results[name] = {
            'accuracy': cv_results['test_accuracy'].mean(),
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'f1': cv_results['test_f1'].mean(),
            'std_accuracy': cv_results['test_accuracy'].std(),
            'std_f1': cv_results['test_f1'].std()
        }
    
    return results

def train_best_model(texts: list, labels: list, model_path: str) -> Tuple[Pipeline, Dict[str, float]]:
    """Train the best model on the full dataset."""
    # Prepare data
    data = prepare_data(texts, labels)
    
    # Evaluate models
    results = evaluate_models(texts, labels)
    
    # Find best model based on F1 score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"Best model: {best_model_name}")
    
    # Train best model on full dataset
    name, pipeline = SpamClassifierFactory.create_classifier(best_model_name)
    pipeline.fit(data, data['label'])
    
    # Save the model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    return pipeline, results[best_model_name]