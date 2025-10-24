"""Simple preprocessing utilities for SMS text.

Keep this module intentionally small and dependency-free so unit tests run fast.
"""
import re
from typing import Dict


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")


def preprocess_text(text: str) -> str:
    """Normalize a single text string.

    Steps:
    - convert to lowercase
    - replace URLs and emails with tokens
    - collapse repeated whitespace
    """
    if text is None:
        return ""
    s = str(text)
    s = s.lower()
    s = URL_RE.sub(" <URL> ", s)
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def basic_features(text: str) -> Dict[str, int]:
    """Compute a few lightweight numeric features from a message.

    Returns a dict with keys: len, digits, punctuation_count, upper_words
    """
    s = text or ""
    features = {
        "msg_len": len(s),
        "digits": sum(c.isdigit() for c in s),
        "punct": sum(1 for c in s if c in "!?,.;:"),
        "upper_words": sum(1 for w in s.split() if w.isupper()),
    }
    return features
