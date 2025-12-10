"""
Test ML models
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.train_models import VANETMisbehaviorDetector


def test_model_training():
    """Test that models can be trained"""
    detector = VANETMisbehaviorDetector()
    results = detector.train_all_models()
    
    assert len(results) == 3
    assert results[0]['model'] == 'Random Forest'
    assert results[1]['model'] == 'SVM'
    assert results[2]['model'] == 'DNN'
    
    # Check that accuracy is reasonable
    for result in results:
        assert result['accuracy'] > 0.9  # At least 90% accuracy


def test_model_prediction():
    """Test model prediction"""
    detector = VANETMisbehaviorDetector()
    detector.train_all_models()
    
    # Generate test features
    test_features = np.random.randn(15)
    
    # Test all models
    for model_type in ['rf', 'svm', 'dnn']:
        is_malicious, confidence = detector.predict(test_features, model_type=model_type)
        assert isinstance(is_malicious, bool)
        assert 0 <= confidence <= 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











