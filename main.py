"""
Main entry point for VANET Misbehavior Detection System
Run this to start the complete system
"""

import argparse
import time
import json
from loguru import logger
from src.detection_system import VANETDetectionSystem
import numpy as np


def generate_test_bsm(vehicle_id: str, is_malicious: bool = False) -> dict:
    """Generate test BSM data"""
    if is_malicious:
        # Malicious vehicle: high frequency, inconsistent data
        return {
            'vehicle_id': vehicle_id,
            'data': {
                'msg_frequency': np.random.uniform(2.5, 4.0),
                'pos_x_consistency': np.random.uniform(0.1, 0.3),
                'pos_y_consistency': np.random.uniform(0.1, 0.3),
                'pos_z_consistency': np.random.uniform(0.2, 0.4),
                'pos_temporal_consistency': np.random.uniform(0.1, 0.3),
                'speed_consistency': np.random.uniform(0.1, 0.3),
                'accel_consistency': np.random.uniform(0.1, 0.3),
                'heading_consistency': np.random.uniform(0.2, 0.4),
                'signal_strength': np.random.uniform(0.3, 0.5),
                'rssi_variance': np.random.uniform(0.5, 1.0),
                'msg_timing': np.random.uniform(0.2, 0.4),
                'neighbor_count': np.random.uniform(8, 15),
                'route_deviation': np.random.uniform(0.6, 1.0),
                'timestamp_anomaly': np.random.uniform(0.3, 0.7),
                'payload_size': np.random.uniform(150, 300)
            }
        }
    else:
        # Normal vehicle: consistent patterns
        return {
            'vehicle_id': vehicle_id,
            'data': {
                'msg_frequency': np.random.uniform(0.8, 1.5),
                'pos_x_consistency': np.random.uniform(0.6, 0.9),
                'pos_y_consistency': np.random.uniform(0.6, 0.9),
                'pos_z_consistency': np.random.uniform(0.7, 0.9),
                'pos_temporal_consistency': np.random.uniform(0.7, 0.9),
                'speed_consistency': np.random.uniform(0.6, 0.8),
                'accel_consistency': np.random.uniform(0.6, 0.8),
                'heading_consistency': np.random.uniform(0.7, 0.9),
                'signal_strength': np.random.uniform(0.6, 0.8),
                'rssi_variance': np.random.uniform(0.1, 0.3),
                'msg_timing': np.random.uniform(0.5, 0.7),
                'neighbor_count': np.random.uniform(3, 8),
                'route_deviation': np.random.uniform(0.0, 0.2),
                'timestamp_anomaly': np.random.uniform(0.0, 0.1),
                'payload_size': np.random.uniform(80, 120)
            }
        }


def run_demo(system: VANETDetectionSystem, num_vehicles: int = 10):
    """Run demonstration with simulated vehicles"""
    logger.info(f"Starting demo with {num_vehicles} vehicles...")
    
    # Generate test vehicles (70% normal, 30% malicious)
    vehicles = []
    for i in range(num_vehicles):
        vehicle_id = f"VEH{i+1:03d}"
        is_malicious = i >= int(num_vehicles * 0.7)
        vehicles.append(generate_test_bsm(vehicle_id, is_malicious))
    
    logger.info(f"Generated {num_vehicles} vehicles ({int(num_vehicles*0.7)} normal, {int(num_vehicles*0.3)} malicious)")
    
    # Process vehicles
    results = []
    for vehicle in vehicles:
        result = system.detect_misbehavior(
            vehicle_id=vehicle['vehicle_id'],
            bsm_data=vehicle['data'],
            log_to_blockchain=True
        )
        results.append(result)
        
        # Print result
        status = "🚨 MALICIOUS" if result['is_malicious'] else "✅ NORMAL"
        logger.info(
            f"{status} | Vehicle: {result['vehicle_id']} | "
            f"Confidence: {result['confidence_percent']:.2f}% | "
            f"Type: {result.get('misbehavior_type_name', 'N/A')}"
        )
        
        if result['blockchain_logged']:
            logger.info(
                f"  📝 Blockchain: TX {result.get('tx_hash', 'N/A')[:10]}... | "
                f"Latency: {result.get('blockchain_latency', 0):.2f}s | "
                f"Gas: {result.get('gas_used', 0)}"
            )
        
        time.sleep(0.5)  # Small delay between detections
    
    # Print statistics
    stats = system.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("DETECTION STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Detections: {stats['total_detections']}")
    logger.info(f"Malicious Detections: {stats['malicious_detections']}")
    logger.info(f"Normal Detections: {stats['normal_detections']}")
    logger.info(f"Blockchain Logs: {stats['blockchain_logs']}")
    logger.info(f"Average Blockchain Latency: {stats['average_blockchain_latency']:.2f}s")
    logger.info(f"Detection Rate: {stats['detection_rate']*100:.2f}%")
    
    # Show vehicle statuses
    logger.info("\n" + "="*60)
    logger.info("VEHICLE STATUS")
    logger.info("="*60)
    for vehicle in vehicles[:5]:  # Show first 5
        status = system.get_vehicle_status(vehicle['vehicle_id'])
        logger.info(
            f"Vehicle {status['vehicle_id']}: "
            f"Trust={status['trust_percent']:.2f}% | "
            f"Misbehaviors={status['misbehavior_count']} | "
            f"Blacklisted={status['is_blacklisted']}"
        )


def main():
    parser = argparse.ArgumentParser(description='VANET Misbehavior Detection System')
    parser.add_argument(
        '--model',
        type=str,
        default='dnn',
        choices=['rf', 'svm', 'dnn'],
        help='ML model to use (default: dnn - uses PyTorch or TensorFlow)'
    )
    parser.add_argument(
        '--vehicles',
        type=int,
        default=10,
        help='Number of test vehicles (default: 10)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.json',
        help='Configuration file path'
    )
    parser.add_argument(
        '--no-blockchain',
        action='store_true',
        help='Disable blockchain logging (for testing)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VANET Blockchain-Based Misbehavior Detection System")
    logger.info("="*60)
    logger.info(f"ML Model: {args.model.upper()}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Initialize system
        system = VANETDetectionSystem(
            ml_model_type=args.model,
            blockchain_config=args.config
        )
        
        # Run demo
        run_demo(system, num_vehicles=args.vehicles)
        
        logger.info("\n✅ System execution complete!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  System interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
