"""Enhanced training script with model comparison and evaluation."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_sms_dataset, normalize_labels
from advanced_preprocessing import clean_dataset
from enhanced_models import evaluate_models, train_best_model

def plot_model_comparison(results: dict, output_path: str):
    """Plot model comparison results."""
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    x = range(len(model_names))
    width = 0.2
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar([xi + i*width for xi in x], values, width, label=metric.capitalize())
    
    # Customize plot
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks([xi + width*1.5 for xi in x], model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train spam classifier with multiple models')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save model artifacts')
    args = parser.parse_args()
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("Loading dataset...")
    df = load_sms_dataset(args.data)
    df = normalize_labels(df)
    
    # Clean dataset
    print("Cleaning dataset...")
    texts, labels = clean_dataset(df['message'].tolist(), df['label_num'].tolist())
    print(f"Dataset size after cleaning: {len(texts)} messages")
    
    # Evaluate all models
    print("\nEvaluating models...")
    results = evaluate_models(texts, labels)
    
    # Print results
    print("\nModel Evaluation Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Plot comparison
    plot_path = model_dir / 'model_comparison.png'
    plot_model_comparison(results, str(plot_path))
    print(f"\nModel comparison plot saved to: {plot_path}")
    
    # Train and save best model
    print("\nTraining best model...")
    model_path = model_dir / 'best_spam_classifier.joblib'
    pipeline, best_metrics = train_best_model(texts, labels, str(model_path))
    print(f"Best model saved to: {model_path}")
    
    # Save results summary
    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv(model_dir / 'model_comparison.csv')
    print(f"Detailed results saved to: {model_dir / 'model_comparison.csv'}")

if __name__ == '__main__':
    main()