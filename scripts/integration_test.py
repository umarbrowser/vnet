"""
Integration test for the complete VANET misbehavior detection system
Tests ML detection + blockchain logging integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
from loguru import logger

from main import VANETMisbehaviorDetector


def test_integration():
    """Test complete integration"""
    logger.info("=" * 60)
    logger.info("Integration Test: VANET Misbehavior Detection System")
    logger.info("=" * 60)
    
    # Check if model exists
    model_path = 'models/dnn_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train models first: python scripts/train_models.py")
        return False
    
    # Initialize detector
    try:
        detector = VANETMisbehaviorDetector(model_type='dnn')
        logger.info("✓ Detector initialized")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return False
    
    # Test 1: Normal BSM
    logger.info("\n" + "-" * 60)
    logger.info("Test 1: Normal BSM (should not detect misbehavior)")
    logger.info("-" * 60)
    
    normal_bsm = pd.DataFrame([{
        'vehicle_id': 'VEH_0001',
        'timestamp': time.time(),
        'position_x': 100.0,
        'position_y': 200.0,
        'speed': 15.0,  # Normal speed
        'heading': 90.0
    }])
    
    result1 = detector.detect_misbehavior(normal_bsm)
    logger.info(f"Result: {result1}")
    
    if result1.get('detected', True):
        logger.warning("⚠️  False positive: Normal BSM detected as misbehavior")
    else:
        logger.info("✓ Correctly identified as normal")
    
    # Test 2: Suspicious BSM (misbehavior)
    logger.info("\n" + "-" * 60)
    logger.info("Test 2: Suspicious BSM (should detect misbehavior)")
    logger.info("-" * 60)
    
    suspicious_bsm = pd.DataFrame([{
        'vehicle_id': 'VEH_0002',
        'timestamp': time.time(),
        'position_x': 100.0,
        'position_y': 200.0,
        'speed': 150.0,  # Unrealistic speed (misbehavior)
        'heading': 90.0
    }])
    
    result2 = detector.detect_misbehavior(suspicious_bsm)
    logger.info(f"Result: {result2}")
    
    if result2.get('detected', False):
        logger.info(f"✓ Correctly detected misbehavior (confidence: {result2.get('confidence', 0):.2%})")
    else:
        logger.warning("⚠️  False negative: Suspicious BSM not detected")
    
    # Test 3: Blockchain logging (if available)
    if detector.blockchain_client:
        logger.info("\n" + "-" * 60)
        logger.info("Test 3: Blockchain Logging")
        logger.info("-" * 60)
        
        try:
            # Check balance
            balance = detector.blockchain_client.get_balance()
            logger.info(f"Account balance: {balance:.4f} ETH")
            
            if balance < 0.001:
                logger.warning("⚠️  Low balance. Skipping blockchain test.")
            else:
                # Try to log a test misbehavior
                test_result = detector.process_and_log(suspicious_bsm)
                
                if test_result.get('blockchain') and 'tx_hash' in test_result['blockchain']:
                    logger.info(f"✓ Successfully logged to blockchain")
                    logger.info(f"  TX Hash: {test_result['blockchain']['tx_hash']}")
                    logger.info(f"  Block: {test_result['blockchain']['block_number']}")
                    logger.info(f"  Gas Used: {test_result['blockchain']['gas_used']}")
                else:
                    logger.warning("⚠️  Blockchain logging failed or skipped")
        
        except Exception as e:
            logger.error(f"Blockchain test failed: {e}")
    else:
        logger.info("\n" + "-" * 60)
        logger.info("Test 3: Blockchain Logging (Skipped - not configured)")
        logger.info("-" * 60)
    
    # Test 4: Batch processing
    logger.info("\n" + "-" * 60)
    logger.info("Test 4: Batch Processing")
    logger.info("-" * 60)
    
    batch_data = []
    for i in range(10):
        batch_data.append({
            'vehicle_id': f'VEH_{i:04d}',
            'timestamp': time.time() + i * 0.1,
            'position_x': np.random.normal(0, 1000),
            'position_y': np.random.normal(0, 1000),
            'speed': np.random.uniform(0, 30),
            'heading': np.random.uniform(0, 360)
        })
    
    batch_df = pd.DataFrame(batch_data)
    batch_result = detector.detect_misbehavior(batch_df)
    
    if isinstance(batch_result, list):
        detected_count = sum(1 for r in batch_result if r.get('detected', False))
        logger.info(f"✓ Processed {len(batch_result)} BSMs")
        logger.info(f"  Detected misbehaviors: {detected_count}")
    else:
        logger.info(f"✓ Processed batch: {batch_result}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Integration Test Complete")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)

