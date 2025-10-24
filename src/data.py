"""Data loading helpers for the spam classifier.

This module provides a small, robust CSV loader with sane defaults for the
`sms_spam_no_header.csv` dataset (assumed: label, message, no header).
"""
from typing import Tuple

import pandas as pd


import re

def preprocess_text(text: str) -> dict:
    """Apply various preprocessing steps to text data."""
    text = str(text)
    
    # Basic cleaning
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    text_whitespace = ' '.join(text_stripped.split())
    
    # Remove numbers
    text_no_numbers = re.sub(r'\d+', '', text_whitespace)
    
    # Mask contact information (phone numbers, emails, urls)
    text_contacts_masked = re.sub(r'\b\d{10,}\b', '[PHONE]', text_whitespace)
    text_contacts_masked = re.sub(r'\S+@\S+', '[EMAIL]', text_contacts_masked)
    text_contacts_masked = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                 '[URL]', text_contacts_masked)
    
    # Clean special characters
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_whitespace)
    
    return {
        'text_stripped': text_stripped,
        'text_lower': text_lower,
        'text_whitespace': text_whitespace,
        'text_numbers': text_no_numbers,
        'text_contacts_masked': text_contacts_masked,
        'text_clean': text_clean
    }

def load_sms_dataset(path: str) -> pd.DataFrame:
    """Load the SMS spam dataset.

    Args:
        path: Path to CSV file. Assumes no header and two columns: label, message.

    Returns:
        DataFrame with columns ['label', 'message'] and additional preprocessed text columns.
    """
    # Try common encodings if utf-8 fails
    try:
        df = pd.read_csv(path, header=None, names=["label", "message"], encoding="utf-8", sep=",")
    except Exception:
        df = pd.read_csv(path, header=None, names=["label", "message"], encoding="latin-1", sep=",")

    # Basic cleaning: strip whitespace
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    
    # Apply preprocessing to each message
    preprocessed = pd.DataFrame([preprocess_text(msg) for msg in df['message']])
    
    # Combine original and preprocessed columns
    df = pd.concat([df, preprocessed], axis=1)
    
    # Add column numbers for compatibility
    df['col_0'] = df['label']
    df['col_1'] = df['message']

    return df


def normalize_labels(df: pd.DataFrame, spam_label: str = "spam") -> pd.DataFrame:
    """Normalize label column to 0/1 where 1 indicates spam.

    Args:
        df: DataFrame with a `label` column.
        spam_label: Value used in dataset to mean spam (case-insensitive).

    Returns:
        DataFrame with an added `label_num` column (int 0/1).
    """
    df = df.copy()
    df["label_num"] = (df["label"].str.lower() == spam_label.lower()).astype(int)
    return df
