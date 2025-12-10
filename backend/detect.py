#!/usr/bin/env python3
"""
Python script for ML detection - called from Node.js
"""

import sys
import json
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

from src.detection_system import VANETDetectionSystem

def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    vehicle_id = input_data.get('vehicle_id', f'VEH{int(__import__("time").time())}')
    bsm_data = input_data.get('bsm_data', {})
    model_type = input_data.get('model_type', 'dnn')
    log_to_blockchain = input_data.get('log_to_blockchain', True)
    
    try:
        # Initialize detection system (will cache if already initialized)
        system = VANETDetectionSystem(ml_model_type=model_type)
        
        # Run detection
        result = system.detect_misbehavior(
            vehicle_id=vehicle_id,
            bsm_data=bsm_data,
            log_to_blockchain=log_to_blockchain
        )
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'vehicle_id': vehicle_id,
            'is_malicious': False,
            'confidence': 0,
            'confidence_percent': 0
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()

