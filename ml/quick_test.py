"""
Quick test with small sample from dataset - generates visualizations
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection_system import VANETDetectionSystem
from ml.use_dataset import prepare_dataset_for_inference
from ml.visualize_results import create_visualizations
from loguru import logger

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv('data/veremi_dataset.csv', nrows=2000)  # Load only 2000 rows for quick test
logger.info(f"Loaded {len(df)} rows")

# Prepare for inference
vehicle_bsms = prepare_dataset_for_inference(df)

# Initialize system
logger.info("Initializing DNN model...")
system = VANETDetectionSystem(ml_model_type='dnn')

# Process vehicles
logger.info(f"Processing {len(vehicle_bsms)} vehicles...")
results = []

for i, bsm in enumerate(vehicle_bsms):
    result = system.detect_misbehavior(
        vehicle_id=bsm['vehicle_id'],
        bsm_data=bsm['data'],
        log_to_blockchain=True  # Enable blockchain
    )
    results.append(result)
    
    if (i + 1) % 100 == 0:
        logger.info(f"Processed {i+1}/{len(vehicle_bsms)} vehicles")

# Get statistics
stats = system.get_statistics()

# Save results
import json
import os
os.makedirs('results', exist_ok=True)
with open('results/inference_results.json', 'w') as f:
    json.dump({'statistics': stats, 'results': results}, f, indent=2, default=str)

logger.info(f"✅ Results saved. Generating visualizations...")

# Generate visualizations
create_visualizations({'statistics': stats, 'results': results})

logger.info("✅ Complete! Check results/visualizations/ for graphs")







