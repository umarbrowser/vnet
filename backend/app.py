"""
Flask backend for VANET Detection System Web Interface
Provides SSE streaming, API endpoints, and real-time data
"""

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import time
import threading
from datetime import datetime
from collections import deque
from typing import Dict, List, Set
import queue
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection_system import VANETDetectionSystem
from loguru import logger
import numpy as np

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Import and register training blueprint
try:
    from backend.training_api import training_bp
    app.register_blueprint(training_bp)
except ImportError as e:
    logger.warning(f"Training API not available: {e}")

# Global state
detection_system: VANETDetectionSystem = None
log_queue = queue.Queue()
connected_clients: Set[str] = set()
activity_logs = deque(maxlen=10000)  # Keep last 10k logs
detection_history = deque(maxlen=1000)  # Keep last 1k detections
stats_history = deque(maxlen=100)  # Keep last 100 stat snapshots

# Activity categories
ACTIVITY_CATEGORIES = {
    'info': {'color': '#3b82f6', 'icon': 'ℹ️'},
    'success': {'color': '#10b981', 'icon': '✅'},
    'warning': {'color': '#f59e0b', 'icon': '⚠️'},
    'error': {'color': '#ef4444', 'icon': '❌'},
    'detection': {'color': '#8b5cf6', 'icon': '🔍'},
    'blockchain': {'color': '#06b6d4', 'icon': '⛓️'},
    'system': {'color': '#6b7280', 'icon': '⚙️'}
}

def categorize_log(message: str) -> str:
    """Categorize log message based on content"""
    message_lower = message.lower()
    if 'malicious' in message_lower or 'detection' in message_lower:
        return 'detection'
    elif 'blockchain' in message_lower or 'tx' in message_lower or 'gas' in message_lower:
        return 'blockchain'
    elif 'error' in message_lower or 'failed' in message_lower or '❌' in message:
        return 'error'
    elif 'warning' in message_lower or '⚠️' in message:
        return 'warning'
    elif 'success' in message_lower or '✅' in message or 'complete' in message_lower:
        return 'success'
    elif 'system' in message_lower or 'initializ' in message_lower:
        return 'system'
    else:
        return 'info'

def broadcast_log(message: str, category: str = None, data: dict = None):
    """Broadcast log to all connected clients"""
    if category is None:
        category = categorize_log(message)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'category': category,
        'data': data or {}
    }
    
    activity_logs.append(log_entry)
    log_queue.put(log_entry)

def generate_sse_logs():
    """Generate SSE stream for logs"""
    client_id = id(threading.current_thread())
    connected_clients.add(client_id)
    
    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'client_id': str(client_id)})}\n\n"
        
        # Send recent logs
        for log in list(activity_logs)[-100:]:  # Last 100 logs
            yield f"data: {json.dumps({'type': 'log', **log})}\n\n"
        
        # Stream new logs
        while client_id in connected_clients:
            try:
                log_entry = log_queue.get(timeout=1)
                yield f"data: {json.dumps({'type': 'log', **log_entry})}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
    except GeneratorExit:
        pass
    finally:
        connected_clients.discard(client_id)

@app.route('/api/logs/stream')
def stream_logs():
    """SSE endpoint for log streaming"""
    return Response(
        generate_sse_logs(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/logs')
def get_logs():
    """Get recent logs"""
    limit = request.args.get('limit', 100, type=int)
    category = request.args.get('category', None)
    
    logs = list(activity_logs)
    if category:
        logs = [log for log in logs if log['category'] == category]
    
    return jsonify({
        'logs': logs[-limit:],
        'total': len(logs)
    })

@app.route('/api/stats')
def get_stats():
    """Get current system statistics"""
    try:
        if detection_system:
            stats = detection_system.get_statistics()
            
            # Add real-time metrics
            stats['timestamp'] = datetime.now().isoformat()
            stats['connected_clients'] = len(connected_clients)
            stats['total_logs'] = len(activity_logs)
            
            # Calculate rates
            if len(stats_history) > 1:
                prev_stats = stats_history[-1]
                time_diff = 1.0  # Assume 1 second between snapshots
                stats['detection_rate_per_sec'] = (stats['total_detections'] - prev_stats.get('total_detections', 0)) / time_diff
            
            stats_history.append(stats)
            return jsonify(stats)
        else:
            return jsonify({
                'total_detections': 0,
                'malicious_detections': 0,
                'normal_detections': 0,
                'blockchain_logs': 0,
                'average_blockchain_latency': 0.0,
                'detection_rate': 0.0,
                'timestamp': datetime.now().isoformat(),
                'connected_clients': len(connected_clients),
                'total_logs': len(activity_logs)
            })
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({
            'error': str(e),
            'total_detections': 0,
            'malicious_detections': 0,
            'normal_detections': 0,
            'blockchain_logs': 0,
            'average_blockchain_latency': 0.0,
            'detection_rate': 0.0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detections')
def get_detections():
    """Get recent detection history"""
    limit = request.args.get('limit', 100, type=int)
    malicious_only = request.args.get('malicious_only', 'false').lower() == 'true'
    
    detections = list(detection_history)
    if malicious_only:
        detections = [d for d in detections if d.get('is_malicious', False)]
    
    return jsonify({
        'detections': detections[-limit:],
        'total': len(detections)
    })

@app.route('/api/detections/chart')
def get_detections_chart():
    """Get detection data formatted for charts"""
    limit = request.args.get('limit', 50, type=int)
    detections = list(detection_history)[-limit:]
    
    # Time series data
    time_series = []
    malicious_count = 0
    normal_count = 0
    
    for i, det in enumerate(detections):
        timestamp = det.get('timestamp', datetime.now().isoformat())
        if det.get('is_malicious', False):
            malicious_count += 1
        else:
            normal_count += 1
        
        time_series.append({
            'time': timestamp,
            'index': i,
            'malicious': malicious_count,
            'normal': normal_count,
            'total': i + 1,
            'confidence': det.get('confidence_percent', 0)
        })
    
    # Confidence distribution
    confidences = [d.get('confidence_percent', 0) for d in detections]
    
    # Misbehavior types
    type_counts = {}
    for det in detections:
        if det.get('is_malicious', False):
            mtype = det.get('misbehavior_type_name', 'unknown')
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
    
    return jsonify({
        'timeSeries': time_series,
        'confidenceDistribution': confidences,
        'typeDistribution': type_counts,
        'summary': {
            'malicious': malicious_count,
            'normal': normal_count,
            'total': len(detections)
        }
    })

@app.route('/api/activity/summary')
def get_activity_summary():
    """Get activity summary by category"""
    category_counts = {}
    for log in activity_logs:
        cat = log['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return jsonify({
        'categories': category_counts,
        'total': len(activity_logs),
        'recent_activity': list(activity_logs)[-10:]
    })

@app.route('/api/system/info')
def get_system_info():
    """Get system information"""
    try:
        blockchain_status = 'disconnected'
        if detection_system and detection_system.blockchain:
            try:
                if detection_system.blockchain.w3 and detection_system.blockchain.w3.is_connected():
                    blockchain_status = 'connected'
            except:
                pass
        
        return jsonify({
            'system': {
                'name': 'VANET Misbehavior Detection System',
                'version': '1.0.0',
                'status': 'running' if detection_system else 'initializing',
                'blockchain_status': blockchain_status
            },
            'technology': {
                'backend': 'Flask + Python',
                'ml_models': ['Random Forest', 'SVM', 'Deep Neural Network'],
                'blockchain': 'Ethereum (Hardhat)',
                'frontend': 'React + Vite',
                'visualization': 'Recharts',
                'ml_framework': 'scikit-learn (MLPClassifier for DNN)'
            },
            'workflow': {
                'steps': [
                    'Vehicle sends Basic Safety Message (BSM)',
                    'ML model extracts features and predicts misbehavior',
                    'High-confidence detections logged to blockchain',
                    'Trust scores updated and vehicles blacklisted if needed',
                    'Real-time monitoring and visualization'
                ]
            },
            'model_info': {
                'current': detection_system.ml_model_type.upper() if detection_system else 'N/A',
                'available': ['RF', 'SVM', 'DNN']
            }
        })
    except Exception as e:
        logger.error(f"Error in get_system_info: {e}")
        return jsonify({
            'error': str(e),
            'system': {
                'name': 'VANET Misbehavior Detection System',
                'version': '1.0.0',
                'status': 'error',
                'blockchain_status': 'unknown'
            }
        }), 500

@app.route('/api/ml/training')
def get_training_results():
    """Get ML model training results"""
    try:
        import os
        import json
        
        training_file = 'results/training_results.json'
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                results = json.load(f)
        
            # Add feature importance if available
            feature_names = [
                'msg_frequency', 'pos_x_consistency', 'pos_y_consistency',
                'pos_z_consistency', 'pos_temporal_consistency', 'speed_consistency',
                'accel_consistency', 'heading_consistency', 'signal_strength',
                'rssi_variance', 'msg_timing', 'neighbor_count',
                'route_deviation', 'timestamp_anomaly', 'payload_size'
            ]
            
            # Try to get feature importance from Random Forest
            feature_importance = None
            if detection_system and detection_system.ml_detector.rf_model:
                try:
                    importances = detection_system.ml_detector.rf_model.feature_importances_
                    feature_importance = dict(zip(feature_names, importances.tolist()))
                except:
                    pass
            
            return jsonify({
                'results': results,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'models': [r['model'] for r in results],
                'metrics': {
                    'accuracy': [r['accuracy'] for r in results],
                    'precision': [r['precision'] for r in results],
                    'recall': [r['recall'] for r in results],
                    'f1_score': [r['f1_score'] for r in results]
                }
            })
        else:
            return jsonify({
                'error': 'Training results not found',
                'message': 'Run training first: python ml/train_models.py'
            }), 404
    except Exception as e:
        logger.error(f"Error in get_training_results: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to load training results'
        }), 500

@app.route('/api/ml/model-comparison')
def get_model_comparison():
    """Get detailed model comparison data"""
    try:
        import os
        import json
        
        training_file = 'results/training_results.json'
        if not os.path.exists(training_file):
            return jsonify({'error': 'Training results not found'}), 404
        
        with open(training_file, 'r') as f:
            results = json.load(f)
        
        comparison_data = []
        for result in results:
            cm = result['confusion_matrix']
            tn, fp = cm[0]
            fn, tp = cm[1]
            
            comparison_data.append({
            'model': result['model'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'sensitivity': result['recall'],  # Same as recall
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            })
        
        return jsonify({
        'comparison': comparison_data,
        'summary': {
            'best_accuracy': max(r['accuracy'] for r in results),
            'best_f1': max(r['f1_score'] for r in results),
            'best_model': max(results, key=lambda x: x['f1_score'])['model']
        }
    })
    except Exception as e:
        logger.error(f"Error in get_model_comparison: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to load model comparison data'
        }), 500

@app.route('/api/detect', methods=['POST'])
def run_detection():
    """Run detection on provided BSM data"""
    try:
        data = request.json or {}
        vehicle_id = data.get('vehicle_id', f'VEH{int(time.time())}')
        bsm_data = data.get('bsm_data', {})
        log_to_blockchain = data.get('log_to_blockchain', True)
        
        global detection_system
        if not detection_system:
            broadcast_log("Detection system not initialized, attempting to initialize...", category='warning')
            try:
                detection_system = VANETDetectionSystem(ml_model_type='dnn')
                broadcast_log("Detection system initialized successfully", category='success')
            except Exception as init_error:
                logger.error(f"Failed to initialize detection system: {init_error}")
                return jsonify({
                    'error': 'Detection system not initialized',
                    'details': str(init_error),
                    'message': 'Please wait for system initialization or check backend logs'
                }), 503
        
        # Validate BSM data
        required_features = [
            'msg_frequency', 'pos_x_consistency', 'pos_y_consistency',
            'pos_z_consistency', 'pos_temporal_consistency', 'speed_consistency',
            'accel_consistency', 'heading_consistency', 'signal_strength',
            'rssi_variance', 'msg_timing', 'neighbor_count',
            'route_deviation', 'timestamp_anomaly', 'payload_size'
        ]
        
        # Ensure all features are present with defaults
        for feature in required_features:
            if feature not in bsm_data:
                bsm_data[feature] = 0.5  # Default value
        
        # Run detection
        try:
            result = detection_system.detect_misbehavior(
                vehicle_id=vehicle_id,
                bsm_data=bsm_data,
                log_to_blockchain=log_to_blockchain
            )
        except Exception as detect_error:
            logger.error(f"Detection failed: {detect_error}", exc_info=True)
            return jsonify({
                'error': 'Detection failed',
                'details': str(detect_error),
                'message': 'Check that ML models are loaded correctly'
            }), 500
        
        result['timestamp'] = datetime.now().isoformat()
        detection_history.append(result)
        
        # Broadcast log
        status = "🚨 MALICIOUS" if result.get('is_malicious', False) else "✅ NORMAL"
        confidence = result.get('confidence_percent', 0)
        broadcast_log(
            f"{status} | Vehicle: {result.get('vehicle_id', vehicle_id)} | Confidence: {confidence:.2f}%",
            category='detection',
            data=result
        )
        
        if result.get('blockchain_logged'):
            broadcast_log(
                f"📝 Blockchain: TX {result.get('tx_hash', 'N/A')[:16]}... | Latency: {result.get('blockchain_latency', 0):.2f}s",
                category='blockchain',
                data=result
            )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'message': 'An error occurred during detection. Check backend logs for details.'
        }), 500

@app.route('/api/detect/batch', methods=['POST'])
def run_batch_detection():
    """Run batch detection"""
    try:
        global detection_system
        data = request.json or {}
        num_vehicles = data.get('num_vehicles', 10)
        malicious_ratio = data.get('malicious_ratio', 0.3)
        
        # Validate inputs
        if num_vehicles < 1 or num_vehicles > 1000:
            return jsonify({'error': 'Number of vehicles must be between 1 and 1000'}), 400
        
        if malicious_ratio < 0 or malicious_ratio > 1:
            return jsonify({'error': 'Malicious ratio must be between 0.0 and 1.0'}), 400
        
        if not detection_system:
            broadcast_log("Detection system not initialized, attempting to initialize...", category='warning')
            try:
                detection_system = VANETDetectionSystem(ml_model_type='dnn')
                broadcast_log("Detection system initialized successfully", category='success')
            except Exception as init_error:
                logger.error(f"Failed to initialize detection system: {init_error}")
                return jsonify({
                    'error': 'Detection system not initialized',
                    'details': str(init_error),
                    'message': 'Please wait for system initialization or check backend logs'
                }), 503
        
        broadcast_log(f"Starting batch detection with {num_vehicles} vehicles...", category='system')
        
        results = []
        for i in range(num_vehicles):
            vehicle_id = f"VEH{i+1:03d}"
            is_malicious = i >= int(num_vehicles * (1 - malicious_ratio))
            
            # Generate test BSM
            if is_malicious:
                bsm_data = {
                    'msg_frequency': float(np.random.uniform(2.5, 4.0)),
                    'pos_x_consistency': float(np.random.uniform(0.1, 0.3)),
                    'pos_y_consistency': float(np.random.uniform(0.1, 0.3)),
                    'pos_z_consistency': float(np.random.uniform(0.2, 0.4)),
                    'pos_temporal_consistency': float(np.random.uniform(0.1, 0.3)),
                    'speed_consistency': float(np.random.uniform(0.1, 0.3)),
                    'accel_consistency': float(np.random.uniform(0.1, 0.3)),
                    'heading_consistency': float(np.random.uniform(0.2, 0.4)),
                    'signal_strength': float(np.random.uniform(0.3, 0.5)),
                    'rssi_variance': float(np.random.uniform(0.5, 1.0)),
                    'msg_timing': float(np.random.uniform(0.2, 0.4)),
                    'neighbor_count': float(np.random.uniform(8, 15)),
                    'route_deviation': float(np.random.uniform(0.6, 1.0)),
                    'timestamp_anomaly': float(np.random.uniform(0.3, 0.7)),
                    'payload_size': float(np.random.uniform(150, 300))
                }
            else:
                bsm_data = {
                    'msg_frequency': float(np.random.uniform(0.8, 1.5)),
                    'pos_x_consistency': float(np.random.uniform(0.6, 0.9)),
                    'pos_y_consistency': float(np.random.uniform(0.6, 0.9)),
                    'pos_z_consistency': float(np.random.uniform(0.7, 0.9)),
                    'pos_temporal_consistency': float(np.random.uniform(0.7, 0.9)),
                    'speed_consistency': float(np.random.uniform(0.6, 0.8)),
                    'accel_consistency': float(np.random.uniform(0.6, 0.8)),
                    'heading_consistency': float(np.random.uniform(0.7, 0.9)),
                    'signal_strength': float(np.random.uniform(0.6, 0.8)),
                    'rssi_variance': float(np.random.uniform(0.1, 0.3)),
                    'msg_timing': float(np.random.uniform(0.5, 0.7)),
                    'neighbor_count': float(np.random.uniform(3, 8)),
                    'route_deviation': float(np.random.uniform(0.0, 0.2)),
                    'timestamp_anomaly': float(np.random.uniform(0.0, 0.1)),
                    'payload_size': float(np.random.uniform(80, 120))
                }
            
            try:
                result = detection_system.detect_misbehavior(
                    vehicle_id=vehicle_id,
                    bsm_data=bsm_data,
                    log_to_blockchain=True
                )
                result['timestamp'] = datetime.now().isoformat()
                detection_history.append(result)
                results.append(result)
                
                # Log progress every 10 vehicles
                if (i + 1) % 10 == 0:
                    broadcast_log(f"Processed {i + 1}/{num_vehicles} vehicles...", category='info')
            except Exception as detect_error:
                logger.error(f"Detection failed for {vehicle_id}: {detect_error}")
                # Continue with other vehicles
                continue
            
            # Small delay to avoid overwhelming the system
            if i < num_vehicles - 1:  # Don't delay after last vehicle
                time.sleep(0.05)
        
        if len(results) == 0:
            return jsonify({
                'error': 'No vehicles processed',
                'message': 'All vehicles failed detection. Check backend logs for details.'
            }), 500
        
        broadcast_log(f"Batch detection complete: {len(results)} vehicles processed", category='success')
        
        # Get statistics
        try:
            summary = detection_system.get_statistics()
        except Exception as stats_error:
            logger.error(f"Failed to get statistics: {stats_error}")
            malicious_count = sum(1 for r in results if r.get('is_malicious', False))
            summary = {
                'total_detections': len(results),
                'malicious_detections': malicious_count,
                'normal_detections': len(results) - malicious_count,
                'blockchain_logs': sum(1 for r in results if r.get('blockchain_logged', False)),
                'average_blockchain_latency': 0.0,
                'detection_rate': malicious_count / len(results) if len(results) > 0 else 0.0
            }
        
        return jsonify({
            'results': results,
            'summary': summary,
            'count': len(results),
            'processed': len(results),
            'requested': num_vehicles
        })
    except Exception as e:
        logger.error(f"Batch detection error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'message': 'An error occurred during batch detection. Check backend logs for details.'
        }), 500

@app.route('/api/vehicle/<vehicle_id>')
def get_vehicle_status(vehicle_id):
    """Get vehicle status"""
    if not detection_system:
        return jsonify({'error': 'Detection system not initialized'}), 500
    
    status = detection_system.get_vehicle_status(vehicle_id)
    return jsonify(status)

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def initialize_detection_system():
    """Initialize the detection system in background"""
    global detection_system
    try:
        broadcast_log("Initializing VANET Detection System...", category='system')
        
        # Try to initialize with blockchain, but allow it to fail gracefully
        try:
            detection_system = VANETDetectionSystem(ml_model_type='dnn')
            broadcast_log("✅ Detection system initialized successfully", category='success')
            if detection_system.blockchain.w3 and detection_system.blockchain.w3.is_connected():
                broadcast_log("✅ Blockchain connection active", category='success')
            else:
                broadcast_log("⚠️ Blockchain not connected - detection will work without blockchain logging", category='warning')
        except Exception as blockchain_error:
            # If blockchain fails, try without it
            broadcast_log(f"⚠️ Blockchain connection failed: {blockchain_error}", category='warning')
            broadcast_log("Initializing without blockchain (detection will still work)...", category='system')
            # Create a minimal system that works without blockchain
            from src.detection_system import VANETDetectionSystem
            detection_system = VANETDetectionSystem(ml_model_type='dnn')
            # Disable blockchain logging
            detection_system.blockchain = None
            broadcast_log("✅ Detection system initialized (blockchain disabled)", category='success')
            
    except Exception as e:
        broadcast_log(f"❌ Failed to initialize detection system: {e}", category='error')
        logger.error(f"Initialization error: {e}")
        # Create a minimal fallback
        try:
            from ml.train_models import VANETMisbehaviorDetector
            from src.detection_system import VANETDetectionSystem
            # Try with no blockchain config
            detection_system = VANETDetectionSystem(ml_model_type='dnn')
            detection_system.blockchain = None
            broadcast_log("✅ Detection system initialized in fallback mode", category='success')
        except:
            pass

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': str(e), 'type': type(e).__name__}), 500

if __name__ == '__main__':
    # Initialize detection system in background thread
    init_thread = threading.Thread(target=initialize_detection_system, daemon=True)
    init_thread.start()
    
    # Start Flask server
    broadcast_log("Starting Flask server on http://localhost:5000", category='system')
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

