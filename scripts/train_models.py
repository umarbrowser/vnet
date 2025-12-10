"""
Train ML models for VANET misbehavior detection
Trains Random Forest, SVM, and DNN models on VeReMi dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.ml.models import create_classifier
from src.data.processor import BSMProcessor


def train_all_models(data_path: str = 'data/veremi/dataset.csv', 
                     output_dir: str = 'models'):
    """
    Train all ML models and evaluate performance
    """
    logger.info("=" * 60)
    logger.info("VANET Misbehavior Detection - Model Training")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    processor = BSMProcessor()
    bsm_data, labels = processor.load_veremi_dataset(data_path)
    
    # Extract features
    logger.info("Extracting features from BSMs...")
    X = processor.extract_features(bsm_data)
    y = labels
    
    # Filter out normal class (0) for binary classification, or keep all for multi-class
    # For this implementation, we'll do multi-class: 0=normal, 1-4=misbehavior types
    # Convert to binary: 0=normal, 1=misbehavior (any type)
    y_binary = (y > 0).astype(int)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y_binary)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Models to train
    models_config = [
        {'type': 'rf', 'name': 'Random Forest', 'params': {'n_estimators': 100, 'max_depth': 20}},
        {'type': 'svm', 'name': 'SVM', 'params': {'C': 1.0, 'kernel': 'rbf'}},
        {'type': 'dnn', 'name': 'Deep Neural Network', 'params': {'hidden_layers': [128, 64, 32]}}
    ]
    
    results = []
    confusion_matrices = []
    
    for model_config in models_config:
        logger.info("\n" + "=" * 60)
        logger.info(f"Training {model_config['name']}...")
        logger.info("=" * 60)
        
        # Create and train model
        model = create_classifier(model_config['type'], **model_config['params'])
        
        if model_config['type'] == 'dnn':
            metrics = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        else:
            metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        y_pred, confidence_scores = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'model': model_config['name'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append((model_config['name'], cm))
        
        # Print results
        logger.info(f"\n{model_config['name']} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Misbehavior']))
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_config['type']}_model.pkl")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(results)
    logger.info("\n" + results_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(output_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    # Plot confusion matrices
    plot_confusion_matrices(confusion_matrices, output_dir)
    
    # Select best model
    best_model_idx = results_df['accuracy'].idxmax()
    best_model = results_df.iloc[best_model_idx]
    logger.info(f"\nBest Model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")
    
    return results_df


def plot_confusion_matrices(confusion_matrices, output_dir):
    """Plot confusion matrices for all models"""
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, cm) in enumerate(confusion_matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Normal', 'Misbehavior'],
                   yticklabels=['Normal', 'Misbehavior'])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrices saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models for VANET misbehavior detection')
    parser.add_argument('--data', type=str, default='data/veremi/dataset.csv',
                       help='Path to VeReMi dataset')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    train_all_models(args.data, args.output)

