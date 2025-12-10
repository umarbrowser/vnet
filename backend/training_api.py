"""
Training API endpoints for ML model training
"""

from flask import Blueprint, jsonify, request, Response
import subprocess
import threading
import queue
import os
import sys
import json
import time
from datetime import datetime
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

training_bp = Blueprint('training', __name__)

# Training log storage
training_logs = deque(maxlen=1000)
training_status = {
    'is_running': False,
    'progress': 0,
    'current_step': '',
    'error': None,
    'start_time': None,
    'end_time': None
}

def run_training(dataset_path='data/veremi_synthetic.csv'):
    """Run training in background and capture logs"""
    global training_status, training_logs
    
    training_status['is_running'] = True
    training_status['progress'] = 0
    training_status['error'] = None
    training_status['start_time'] = datetime.now().isoformat()
    training_status['current_step'] = 'Initializing...'
    training_logs.clear()
    
    def log_output(pipe, log_queue):
        """Capture output from subprocess"""
        for line in iter(pipe.readline, ''):
            if line:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'message': line.strip(),
                    'type': 'info'
                }
                log_queue.put(log_entry)
        pipe.close()
    
    try:
        # Run training script
        cmd = [sys.executable, 'ml/train_models.py', '--dataset', dataset_path]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Stream output
        log_queue = queue.Queue()
        thread = threading.Thread(target=log_output, args=(process.stdout, log_queue))
        thread.daemon = True
        thread.start()
        
        # Process logs
        while process.poll() is None:
            try:
                log_entry = log_queue.get(timeout=0.1)
                training_logs.append(log_entry)
                
                # Update status based on log content
                msg = log_entry['message'].lower()
                if 'training random forest' in msg:
                    training_status['current_step'] = 'Training Random Forest'
                    training_status['progress'] = 20
                elif 'training svm' in msg:
                    training_status['current_step'] = 'Training SVM'
                    training_status['progress'] = 40
                elif 'training dnn' in msg or 'training deep neural network' in msg:
                    training_status['current_step'] = 'Training Deep Neural Network'
                    training_status['progress'] = 60
                elif 'evaluate' in msg:
                    training_status['current_step'] = 'Evaluating models'
                    training_status['progress'] = 80
                elif 'complete' in msg or 'saved' in msg:
                    training_status['progress'] = 90
            except queue.Empty:
                continue
        
        # Get remaining logs
        while not log_queue.empty():
            log_entry = log_queue.get()
            training_logs.append(log_entry)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            training_status['progress'] = 100
            training_status['current_step'] = 'Training complete'
            training_status['end_time'] = datetime.now().isoformat()
        else:
            training_status['error'] = f'Training failed with return code {return_code}'
            training_status['end_time'] = datetime.now().isoformat()
            
    except Exception as e:
        training_status['error'] = str(e)
        training_status['end_time'] = datetime.now().isoformat()
        training_logs.append({
            'timestamp': datetime.now().isoformat(),
            'message': f'Error: {str(e)}',
            'type': 'error'
        })
    finally:
        training_status['is_running'] = False

@training_bp.route('/api/training/start', methods=['POST'])
def start_training():
    """Start ML model training"""
    global training_status
    
    if training_status['is_running']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json or {}
    dataset_path = data.get('dataset_path', 'data/veremi_synthetic.csv')
    
    # Start training in background thread
    thread = threading.Thread(target=run_training, args=(dataset_path,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Training started',
        'dataset': dataset_path
    })

@training_bp.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@training_bp.route('/api/training/logs')
def get_training_logs():
    """Get training logs"""
    limit = request.args.get('limit', 100, type=int)
    logs = list(training_logs)[-limit:]
    return jsonify({
        'logs': logs,
        'total': len(training_logs)
    })

@training_bp.route('/api/training/logs/stream')
def stream_training_logs():
    """SSE stream for training logs"""
    def generate():
        last_count = 0
        while True:
            current_count = len(training_logs)
            if current_count > last_count:
                # Send new logs
                new_logs = list(training_logs)[last_count:]
                for log in new_logs:
                    yield f"data: {json.dumps({'type': 'log', **log})}\n\n"
                last_count = current_count
            
            # Send status update
            yield f"data: {json.dumps({'type': 'status', **training_status})}\n\n"
            
            if not training_status['is_running'] and last_count == current_count:
                break
            
            time.sleep(0.5)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

