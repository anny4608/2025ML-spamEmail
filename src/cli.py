"""Minimal CLI for classifying single messages or files of messages.

Usage examples:
    python src\cli.py --message "You won a prize"
    python src\cli.py --file data\new_messages.csv
"""
import argparse
import csv
import os
from typing import List

from sklearn.exceptions import NotFittedError

from src.models import load_pipeline


def predict_message(pipeline: dict, message: str) -> dict:
    vec = pipeline["vectorizer"]
    model = pipeline["model"]
    X = vec.transform([message])
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return {"pred": int(pred), "prob": prob.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--message", type=str, help="Single message to classify")
    p.add_argument("--file", type=str, help="CSV file with messages (one per line)")
    p.add_argument("--model", type=str, default="models/spam_classifier.joblib")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Train a model first using the training script.")
        return

    pipeline = load_pipeline(args.model)

    if args.message:
        out = predict_message(pipeline, args.message)
        print(f"prediction: {out}")
    elif args.file:
        with open(args.file, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                text = row[0] if row else ""
                out = predict_message(pipeline, text)
                print(text, "=>", out)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
