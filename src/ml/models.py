"""
Machine Learning Models for VANET Misbehavior Detection
Implements Random Forest, SVM, and Deep Neural Network classifiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from typing import Tuple, Dict, Any
from loguru import logger


class MisbehaviorClassifier:
    """Base class for misbehavior detection classifiers"""
    
    def __init__(self, model_type: str = 'dnn'):
        """
        Initialize classifier
        Args:
            model_type: 'rf' (Random Forest), 'svm', or 'dnn' (Deep Neural Network)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train the classifier
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict misbehavior
        Returns:
            (predictions, confidence_scores)
        """
        raise NotImplementedError
    
    def save(self, filepath: str):
        """Save trained model"""
        raise NotImplementedError
    
    def load(self, filepath: str):
        """Load trained model"""
        raise NotImplementedError


class RandomForestClassifierModel(MisbehaviorClassifier):
    """Random Forest classifier for misbehavior detection"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20):
        super().__init__('rf')
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train Random Forest model"""
        logger.info("Training Random Forest classifier...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        metrics = {'train_accuracy': train_score}
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            metrics['val_accuracy'] = val_score
        
        logger.info(f"Random Forest training complete. Accuracy: {train_score:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence_scores = np.max(probabilities, axis=1) * 10000  # Scale to 0-10000
        
        return predictions, confidence_scores.astype(int)
    
    def save(self, filepath: str):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and scaler"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class SVMClassifierModel(MisbehaviorClassifier):
    """Support Vector Machine classifier"""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf'):
        super().__init__('svm')
        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train SVM model"""
        logger.info("Training SVM classifier...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train_scaled, y_train)
        metrics = {'train_accuracy': train_score}
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            metrics['val_accuracy'] = val_score
        
        logger.info(f"SVM training complete. Accuracy: {train_score:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence_scores = np.max(probabilities, axis=1) * 10000
        
        return predictions, confidence_scores.astype(int)
    
    def save(self, filepath: str):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and scaler"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class DNNClassifierModel(MisbehaviorClassifier):
    """Deep Neural Network classifier"""
    
    def __init__(self, input_dim: int = None, hidden_layers: list = [128, 64, 32]):
        super().__init__('dnn')
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.history = None
    
    def _build_model(self, input_dim: int, num_classes: int):
        """Build DNN architecture"""
        model = keras.Sequential([
            layers.Dense(self.hidden_layers[0], activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
        ])
        
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> Dict[str, float]:
        """Train DNN model"""
        logger.info("Training Deep Neural Network...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        input_dim = X_train_scaled.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Build model
        self.model = self._build_model(input_dim, num_classes)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Train
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        train_loss, train_acc = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        metrics = {'train_accuracy': train_acc, 'train_loss': train_loss}
        
        if X_val is not None:
            val_loss, val_acc = self.model.evaluate(X_val_scaled, y_val, verbose=0)
            metrics['val_accuracy'] = val_acc
            metrics['val_loss'] = val_loss
        
        logger.info(f"DNN training complete. Accuracy: {train_acc:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidence_scores = (np.max(probabilities, axis=1) * 10000).astype(int)
        
        return predictions, confidence_scores
    
    def save(self, filepath: str):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # Save scaler separately
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        joblib.dump({
            'scaler': self.scaler,
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers
        }, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, filepath: str):
        """Load model and scaler"""
        # Load Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        data = joblib.load(scaler_path)
        self.scaler = data['scaler']
        self.input_dim = data.get('input_dim')
        self.hidden_layers = data.get('hidden_layers', [128, 64, 32])
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")


def create_classifier(model_type: str = 'dnn', **kwargs) -> MisbehaviorClassifier:
    """
    Factory function to create classifier
    Args:
        model_type: 'rf', 'svm', or 'dnn'
        **kwargs: Model-specific parameters
    """
    if model_type == 'rf':
        return RandomForestClassifierModel(**kwargs)
    elif model_type == 'svm':
        return SVMClassifierModel(**kwargs)
    elif model_type == 'dnn':
        return DNNClassifierModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

