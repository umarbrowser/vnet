"""
Use dataset directly without training - for inference/evaluation
Loads pre-trained models and uses the dataset for predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.download_dataset import download_from_google_drive, load_dataset
from src.detection_system import VANETDetectionSystem


def prepare_dataset_for_inference(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare dataset for inference
    Converts DataFrame to list of BSM dictionaries
    """
    logger.info("Preparing dataset for inference...")
    
    # Expected feature columns (adjust based on your dataset)
    feature_columns = [
        'msg_frequency', 'pos_x_consistency', 'pos_y_consistency',
        'pos_z_consistency', 'pos_temporal_consistency', 'speed_consistency',
        'accel_consistency', 'heading_consistency', 'signal_strength',
        'rssi_variance', 'msg_timing', 'neighbor_count',
        'route_deviation', 'timestamp_anomaly', 'payload_size'
    ]
    
    # Check if dataset has required columns
    available_columns = df.columns.tolist()
    logger.info(f"Available columns: {available_columns}")
    
    # Try to map dataset columns to expected features
    # This is flexible - will work with different column names
    vehicle_bsms = []
    
    for idx, row in df.iterrows():
        # Extract vehicle ID (try common column names)
        vehicle_id = None
        for col in ['vehicle_id', 'vehicleId', 'id', 'vehicle', 'node_id']:
            if col in row:
                vehicle_id = str(row[col])
                break
        
        if vehicle_id is None:
            vehicle_id = f"VEH{idx+1:03d}"
        
        # Extract features
        bsm_data = {}
        
        # Map dataset columns to feature names
        column_mapping = {
            'msg_frequency': ['msg_frequency', 'frequency', 'msg_rate', 'message_frequency'],
            'pos_x_consistency': ['pos_x_consistency', 'x_consistency', 'pos_x'],
            'pos_y_consistency': ['pos_y_consistency', 'y_consistency', 'pos_y'],
            'pos_z_consistency': ['pos_z_consistency', 'z_consistency', 'pos_z'],
            'pos_temporal_consistency': ['pos_temporal_consistency', 'temporal_consistency', 'time_consistency'],
            'speed_consistency': ['speed_consistency', 'speed', 'velocity'],
            'accel_consistency': ['accel_consistency', 'acceleration', 'accel'],
            'heading_consistency': ['heading_consistency', 'heading', 'direction'],
            'signal_strength': ['signal_strength', 'rssi', 'signal'],
            'rssi_variance': ['rssi_variance', 'rssi_var', 'signal_variance'],
            'msg_timing': ['msg_timing', 'timing', 'time'],
            'neighbor_count': ['neighbor_count', 'neighbors', 'neighbor_num'],
            'route_deviation': ['route_deviation', 'deviation', 'route_diff'],
            'timestamp_anomaly': ['timestamp_anomaly', 'time_anomaly', 'ts_anomaly'],
            'payload_size': ['payload_size', 'size', 'payload']
        }
        
        # Extract features from row
        for feature_name, possible_columns in column_mapping.items():
            value = None
            for col in possible_columns:
                if col in row:
                    value = float(row[col])
                    break
            
            # Use default if not found
            if value is None:
                # Use default values based on feature type
                defaults = {
                    'msg_frequency': 1.0,
                    'pos_x_consistency': 0.5,
                    'pos_y_consistency': 0.5,
                    'pos_z_consistency': 0.5,
                    'pos_temporal_consistency': 0.5,
                    'speed_consistency': 0.4,
                    'accel_consistency': 0.4,
                    'heading_consistency': 0.4,
                    'signal_strength': 0.5,
                    'rssi_variance': 0.1,
                    'msg_timing': 0.5,
                    'neighbor_count': 5.0,
                    'route_deviation': 0.1,
                    'timestamp_anomaly': 0.0,
                    'payload_size': 100.0
                }
                value = defaults.get(feature_name, 0.5)
            
            bsm_data[feature_name] = value
        
        vehicle_bsms.append({
            'vehicle_id': vehicle_id,
            'data': bsm_data
        })
    
    logger.info(f"✅ Prepared {len(vehicle_bsms)} vehicle BSMs for inference")
    return vehicle_bsms


def run_inference_with_dataset(
    dataset_path: Optional[str] = None,
    model_type: str = 'dnn',
    google_drive_id: Optional[str] = None,
    log_to_blockchain: bool = True
):
    """
    Run inference using dataset without training
    
    Args:
        dataset_path: Path to local dataset file
        model_type: ML model to use ('rf', 'svm', 'dnn')
        google_drive_id: Google Drive file ID (if downloading)
        log_to_blockchain: Whether to log results to blockchain
    """
    logger.info("="*60)
    logger.info("VANET Misbehavior Detection - Dataset Inference")
    logger.info("="*60)
    
    # Prefer local dataset if available
    default_local_dataset = "data/veremi_synthetic.csv"
    
    # Download dataset if Google Drive ID provided and no local path
    if google_drive_id and not dataset_path and not os.path.exists(default_local_dataset):
        logger.info("Downloading dataset from Google Drive...")
        dataset_path = download_from_google_drive(
            google_drive_id,
            default_local_dataset
        )
        if not dataset_path:
            logger.error("Failed to download dataset")
            return
    
    # Use default local dataset path if not provided
    if not dataset_path:
        dataset_path = default_local_dataset
    
    # Load dataset
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Expected local dataset at data/veremi_synthetic.csv")
        return
    
    df = load_dataset(dataset_path)
    
    # Prepare dataset for inference
    vehicle_bsms = prepare_dataset_for_inference(df)
    
    # Sample dataset if too large (for faster processing and visualization)
    max_samples = 5000  # Process max 5k samples for faster processing
    if len(vehicle_bsms) > max_samples:
        logger.info(f"Dataset is large ({len(vehicle_bsms)} vehicles). Sampling {max_samples} for processing...")
        import random
        random.seed(42)
        vehicle_bsms = random.sample(vehicle_bsms, max_samples)
        logger.info(f"Processing {len(vehicle_bsms)} sampled vehicles")
    
    # Initialize detection system
    logger.info(f"Initializing detection system with {model_type.upper()} model...")
    system = VANETDetectionSystem(ml_model_type=model_type)
    
    # Run inference
    logger.info("\n" + "="*60)
    logger.info("Running Inference...")
    logger.info("="*60)
    
    results = []
    from tqdm import tqdm
    
    for i, bsm in enumerate(tqdm(vehicle_bsms, desc="Processing vehicles")):
        result = system.detect_misbehavior(
            vehicle_id=bsm['vehicle_id'],
            bsm_data=bsm['data'],
            log_to_blockchain=log_to_blockchain
        )
        
        results.append(result)
        
        # Print result every 100 vehicles
        if (i + 1) % 100 == 0 or result['is_malicious']:
            status = "🚨 MALICIOUS" if result['is_malicious'] else "✅ NORMAL"
            logger.info(
                f"  [{i+1}/{len(vehicle_bsms)}] {status} | Confidence: {result['confidence_percent']:.2f}% | "
                f"Type: {result.get('misbehavior_type_name', 'N/A')}"
            )
            
            if result['blockchain_logged']:
                logger.info(
                    f"    📝 Blockchain: TX {result.get('tx_hash', 'N/A')[:16]}... | "
                    f"Latency: {result.get('blockchain_latency', 0):.2f}s"
                )
    
    # Print statistics
    stats = system.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("INFERENCE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Vehicles Processed: {stats['total_detections']}")
    logger.info(f"Malicious Detections: {stats['malicious_detections']}")
    logger.info(f"Normal Detections: {stats['normal_detections']}")
    logger.info(f"Blockchain Logs: {stats['blockchain_logs']}")
    if stats['blockchain_logs'] > 0:
        logger.info(f"Average Blockchain Latency: {stats['average_blockchain_latency']:.2f}s")
    logger.info(f"Detection Rate: {stats['detection_rate']*100:.2f}%")
    
    # Save results
    results_path = "results/inference_results.json"
    os.makedirs("results", exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'results': results
        }, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to {results_path}")
    
    # Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("Generating Visualizations...")
    logger.info("="*60)
    
    try:
        from ml.visualize_results import create_visualizations
        create_visualizations({
            'statistics': stats,
            'results': results
        })
        logger.info("\n✅ Visualizations generated successfully!")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
        logger.info("You can generate visualizations later with:")
        logger.info("  python3 ml/visualize_results.py")
    
    return results, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Use dataset for inference without training')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/veremi_synthetic.csv',
        help='Path to dataset file (default: data/veremi_synthetic.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='dnn',
        choices=['rf', 'svm', 'dnn'],
        help='ML model to use (default: dnn - uses PyTorch or TensorFlow)'
    )
    parser.add_argument(
        '--google-drive-id',
        type=str,
        default='1jWt7YogasRuP_isbRXMhDGVoUbVy-uJ5',
        help='Google Drive file ID'
    )
    parser.add_argument(
        '--no-blockchain',
        action='store_true',
        help='Skip blockchain logging'
    )
    
    args = parser.parse_args()
    
    run_inference_with_dataset(
        dataset_path=args.dataset,
        model_type=args.model,
        google_drive_id=args.google_drive_id,
        log_to_blockchain=not args.no_blockchain
    )

