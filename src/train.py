"""Small training script to run the end-to-end pipeline and persist artifacts.

Usage:
    python src\train.py --data "C:\\path\\to\\sms_spam_no_header.csv" --model-out models/spam_classifier.joblib
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data import load_sms_dataset, normalize_labels
from models import build_vectorizer, train_simple_model, save_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to sms_spam_no_header.csv")
    p.add_argument("--model-out", default="models/spam_classifier.joblib", help="Output model path")
    args = p.parse_args()

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    df = load_sms_dataset(args.data)
    df = normalize_labels(df)

    messages = df["message"].tolist()
    labels = df["label_num"].tolist()

    vec, model = train_simple_model(messages, labels)
    save_pipeline(vec, model, path=args.model_out)

    print(f"Saved pipeline to {args.model_out}")


if __name__ == "__main__":
    main()
