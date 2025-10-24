import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import normalize_labels
from src.models import train_simple_model


def test_train_on_tiny_sample(tmp_path):
    """Test model training on a tiny dataset."""
    # tiny dataset
    df = pd.DataFrame({
        "label": ["ham", "spam", "ham", "spam"],
        "message": ["hello friend", "win money now!!!", "are you there?", "free prize click here"],
    })
    df2 = normalize_labels(df)
    vec, model = train_simple_model(df2["message"].tolist(), df2["label_num"].tolist(), random_state=0)
    # Smoke assertions
    assert vec is not None
    assert model is not None
    assert isinstance(model, LogisticRegression)
