"""
Main detection system that integrates ML models with blockchain
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger
import time

from ml.train_models import VANETMisbehaviorDetector
from src.blockchain_integration import BlockchainIntegration


class VANETDetectionSystem:
    """Complete VANET misbehavior detection system"""
    
    # Misbehavior type mapping
    MISBEHAVIOR_TYPES = {
        'sybil': 0,
        'falsification': 1,
        'replay': 2,
        'dos': 3
    }
    
    def __init__(
        self,
        ml_model_type: str = 'dnn',
        blockchain_config: str = "config/config.json"
    ):
        """
        Initialize detection system
        ml_model_type: 'rf', 'svm', or 'dnn'
        """
        self.ml_model_type = ml_model_type
        self.ml_detector = VANETMisbehaviorDetector()
        
        # Initialize blockchain (optional - allow failure)
        try:
            self.blockchain = BlockchainIntegration(blockchain_config)
        except Exception as e:
            logger.warning(f"Blockchain initialization failed: {e}. Continuing without blockchain.")
            self.blockchain = None
        
        # Load ML models
        self.load_ml_models()
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'blockchain_logs': 0,
            'total_latency': 0.0,
            'detection_history': []
        }
    
    def load_ml_models(self):
        """Load trained ML models"""
        import os
        import joblib
        
        models_dir = 'models'
        if not os.path.exists(models_dir):
            logger.warning("Models directory not found. Training models...")
            self.ml_detector.train_all_models()
        else:
            logger.info("Loading ML models...")
            try:
                self.ml_detector.scaler = joblib.load(f'{models_dir}/scaler.pkl')
                if self.ml_model_type == 'rf':
                    self.ml_detector.rf_model = joblib.load(f'{models_dir}/rf_model.pkl')
                    logger.info("✅ Random Forest model loaded (no TensorFlow needed)")
                elif self.ml_model_type == 'svm':
                    self.ml_detector.svm_model = joblib.load(f'{models_dir}/svm_model.pkl')
                    logger.info("✅ SVM model loaded (no TensorFlow needed)")
                elif self.ml_model_type == 'dnn':
                    # Load scikit-learn MLPClassifier
                    if os.path.exists(f'{models_dir}/dnn_model.pkl'):
                        self.ml_detector.dnn_model = joblib.load(f'{models_dir}/dnn_model.pkl')
                        self.ml_detector.dnn_backend = 'sklearn_mlp'
                        logger.info("✅ DNN model (scikit-learn MLPClassifier) loaded - no PyTorch/TensorFlow needed!")
                    else:
                        logger.error("❌ DNN model not found. Train models first.")
                        raise FileNotFoundError("DNN model not found")
                self.ml_detector.models_trained = True
                logger.info(f"ML model ready: {self.ml_model_type.upper()}")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                logger.info("Training new models (RF and SVM only - no TensorFlow)...")
                self.ml_detector.train_all_models()
    
    def extract_features(self, bsm_data: Dict) -> np.ndarray:
        """
        Extract features from Basic Safety Message (BSM)
        bsm_data should contain: position, speed, acceleration, heading, etc.
        """
        # Extract features (15 features as per training)
        features = np.array([
            bsm_data.get('msg_frequency', 1.0),
            bsm_data.get('pos_x_consistency', 0.5),
            bsm_data.get('pos_y_consistency', 0.5),
            bsm_data.get('pos_z_consistency', 0.5),
            bsm_data.get('pos_temporal_consistency', 0.5),
            bsm_data.get('speed_consistency', 0.4),
            bsm_data.get('accel_consistency', 0.4),
            bsm_data.get('heading_consistency', 0.4),
            bsm_data.get('signal_strength', 0.5),
            bsm_data.get('rssi_variance', 0.1),
            bsm_data.get('msg_timing', 0.5),
            bsm_data.get('neighbor_count', 5.0),
            bsm_data.get('route_deviation', 0.1),
            bsm_data.get('timestamp_anomaly', 0.0),
            bsm_data.get('payload_size', 100.0)
        ])
        
        return features
    
    def detect_misbehavior(
        self,
        vehicle_id: str,
        bsm_data: Dict,
        log_to_blockchain: bool = True
    ) -> Dict:
        """
        Detect misbehavior for a vehicle's BSM
        Returns detection result with confidence and blockchain log info
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(bsm_data)
        
        # ML prediction
        is_malicious, confidence = self.ml_detector.predict(
            features,
            model_type=self.ml_model_type
        )
        
        detection_time = time.time() - start_time
        
        result = {
            'vehicle_id': vehicle_id,
            'is_malicious': is_malicious,
            'confidence': confidence,
            'confidence_percent': confidence / 100.0,
            'detection_time_ms': detection_time * 1000,
            'blockchain_logged': False
        }
        
        # Determine misbehavior type (simplified - in production, use ML classification)
        if is_malicious:
            # Classify misbehavior type based on features
            if bsm_data.get('msg_frequency', 0) > 2.0:
                misbehavior_type = self.MISBEHAVIOR_TYPES['dos']
            elif bsm_data.get('pos_consistency', 1.0) < 0.3:
                misbehavior_type = self.MISBEHAVIOR_TYPES['falsification']
            elif bsm_data.get('timestamp_anomaly', 0) > 0.5:
                misbehavior_type = self.MISBEHAVIOR_TYPES['replay']
            else:
                misbehavior_type = self.MISBEHAVIOR_TYPES['sybil']
            
            result['misbehavior_type'] = misbehavior_type
            result['misbehavior_type_name'] = list(self.MISBEHAVIOR_TYPES.keys())[misbehavior_type]
            
            # Log to blockchain if enabled and blockchain is available
            if log_to_blockchain and self.blockchain and confidence > 5000:  # Only log if confidence > 50%
                try:
                    blockchain_result = self.blockchain.log_misbehavior(
                        vehicle_id=vehicle_id,
                        misbehavior_type=misbehavior_type,
                        confidence_score=confidence
                    )
                    
                    if blockchain_result.get('success'):
                        result['blockchain_logged'] = True
                        result['tx_hash'] = blockchain_result.get('tx_hash')
                        result['blockchain_latency'] = blockchain_result.get('latency_seconds')
                        result['gas_used'] = blockchain_result.get('gas_used')
                        result['cost_eth'] = blockchain_result.get('cost_eth')
                        
                        self.stats['blockchain_logs'] += 1
                        self.stats['total_latency'] += blockchain_result.get('latency_seconds', 0)
                except Exception as e:
                    logger.error(f"Failed to log to blockchain: {e}")
                    result['blockchain_error'] = str(e)
        
        # Update statistics
        self.stats['total_detections'] += 1
        self.stats['detection_history'].append(result)
        
        # Keep only last 1000 records
        if len(self.stats['detection_history']) > 1000:
            self.stats['detection_history'] = self.stats['detection_history'][-1000:]
        
        return result
    
    def process_batch(self, vehicle_bsms: List[Dict]) -> List[Dict]:
        """Process a batch of vehicle BSMs"""
        results = []
        for bsm in vehicle_bsms:
            result = self.detect_misbehavior(
                vehicle_id=bsm['vehicle_id'],
                bsm_data=bsm['data'],
                log_to_blockchain=True
            )
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        total = self.stats['total_detections']
        malicious = sum(1 for r in self.stats['detection_history'] if r.get('is_malicious'))
        
        avg_latency = 0.0
        if self.stats['blockchain_logs'] > 0:
            avg_latency = self.stats['total_latency'] / self.stats['blockchain_logs']
        
        return {
            'total_detections': total,
            'malicious_detections': malicious,
            'normal_detections': total - malicious,
            'blockchain_logs': self.stats['blockchain_logs'],
            'average_blockchain_latency': avg_latency,
            'detection_rate': malicious / total if total > 0 else 0.0
        }
    
    def get_vehicle_status(self, vehicle_id: str) -> Dict:
        """Get comprehensive status for a vehicle"""
        if not self.blockchain:
            return {
                'vehicle_id': vehicle_id,
                'trust_score': 0,
                'trust_percent': 0,
                'misbehavior_count': 0,
                'is_blacklisted': False,
                'blockchain_available': False
            }
        
        try:
            trust_score = self.blockchain.get_trust_score(vehicle_id)
            misbehavior_count = self.blockchain.get_misbehavior_count(vehicle_id)
            is_blacklisted = self.blockchain.is_blacklisted(vehicle_id)
            
            return {
                'vehicle_id': vehicle_id,
                'trust_score': trust_score,
                'trust_percent': trust_score / 100.0,
                'misbehavior_count': misbehavior_count,
                'is_blacklisted': is_blacklisted,
                'blockchain_available': True
            }
        except Exception as e:
            logger.error(f"Failed to get vehicle status: {e}")
            return {
                'vehicle_id': vehicle_id,
                'trust_score': 0,
                'trust_percent': 0,
                'misbehavior_count': 0,
                'is_blacklisted': False,
                'blockchain_available': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Test detection system
    system = VANETDetectionSystem(ml_model_type='dnn')
    
    # Simulate BSM data
    test_bsm = {
        'vehicle_id': 'VEH001',
        'data': {
            'msg_frequency': 2.5,  # High frequency (suspicious)
            'pos_x_consistency': 0.2,  # Low consistency
            'pos_y_consistency': 0.3,
            'pos_z_consistency': 0.4,
            'pos_temporal_consistency': 0.3,
            'speed_consistency': 0.2,
            'accel_consistency': 0.3,
            'heading_consistency': 0.4,
            'signal_strength': 0.6,
            'rssi_variance': 0.5,
            'msg_timing': 0.3,
            'neighbor_count': 10.0,
            'route_deviation': 0.8,
            'timestamp_anomaly': 0.2,
            'payload_size': 150.0
        }
    }
    
    result = system.detect_misbehavior(
        test_bsm['vehicle_id'],
        test_bsm['data'],
        log_to_blockchain=True
    )
    
    print("\nDetection Result:")
    print(f"  Vehicle: {result['vehicle_id']}")
    print(f"  Malicious: {result['is_malicious']}")
    print(f"  Confidence: {result['confidence_percent']:.2f}%")
    print(f"  Type: {result.get('misbehavior_type_name', 'N/A')}")
    print(f"  Blockchain Logged: {result['blockchain_logged']}")
    
    if result['blockchain_logged']:
        print(f"  TX Hash: {result.get('tx_hash', 'N/A')}")
        print(f"  Latency: {result.get('blockchain_latency', 0):.2f}s")
    
    stats = system.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Blockchain Logs: {stats['blockchain_logs']}")



