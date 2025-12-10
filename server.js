/**
 * Node.js Backend Server for VANET Detection System
 * Calls Python for ML processing, handles everything else in Node.js
 */

const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

// Global state
const connectedClients = new Set();
const activityLogs = [];
const detectionHistory = [];
const statsHistory = [];

// Activity categories
const ACTIVITY_CATEGORIES = {
  info: { color: '#3b82f6', icon: 'ℹ️' },
  success: { color: '#10b981', icon: '✅' },
  warning: { color: '#f59e0b', icon: '⚠️' },
  error: { color: '#ef4444', icon: '❌' },
  detection: { color: '#8b5cf6', icon: '🔍' },
  blockchain: { color: '#06b6d4', icon: '⛓️' },
  system: { color: '#6b7280', icon: '⚙️' }
};

function categorizeLog(message) {
  const msg = message.toLowerCase();
  if (msg.includes('malicious') || msg.includes('detection')) return 'detection';
  if (msg.includes('blockchain') || msg.includes('tx') || msg.includes('gas')) return 'blockchain';
  if (msg.includes('error') || msg.includes('failed') || msg.includes('❌')) return 'error';
  if (msg.includes('warning') || msg.includes('⚠️')) return 'warning';
  if (msg.includes('success') || msg.includes('✅') || msg.includes('complete')) return 'success';
  if (msg.includes('system') || msg.includes('initializ')) return 'system';
  return 'info';
}

function broadcastLog(message, category = null, data = {}) {
  if (!category) category = categorizeLog(message);
  
  const logEntry = {
    timestamp: new Date().toISOString(),
    message,
    category,
    data
  };
  
  activityLogs.push(logEntry);
  if (activityLogs.length > 10000) activityLogs.shift();
  
  // Broadcast to SSE clients
  connectedClients.forEach(client => {
    if (client.response && !client.response.closed) {
      try {
        client.response.write(`data: ${JSON.stringify({ type: 'log', ...logEntry })}\n\n`);
      } catch (e) {
        connectedClients.delete(client);
      }
    }
  });
}

// Call Python for ML operations
async function callPython(script, args = [], inputData = null) {
  return new Promise((resolve, reject) => {
    // Use virtual environment Python if available
    const venvPython = path.join(__dirname, 'venv', 'bin', 'python');
    const pythonPath = require('fs').existsSync(venvPython) ? venvPython : 
                      (process.platform === 'win32' ? 'python' : 'python3');
    const scriptPath = path.join(__dirname, script);
    
    const python = spawn(pythonPath, [scriptPath, ...args], {
      cwd: __dirname,
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
    
    let stdout = '';
    let stderr = '';
    
    if (inputData) {
      python.stdin.write(JSON.stringify(inputData));
      python.stdin.end();
    }
    
    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script failed: ${stderr || stdout}`));
      } else {
        try {
          const result = stdout.trim();
          // Try to parse as JSON if it looks like JSON
          if (result.startsWith('{') || result.startsWith('[')) {
            resolve(JSON.parse(result));
          } else {
            resolve(result);
          }
        } catch (e) {
          resolve(stdout.trim());
        }
      }
    });
    
    python.on('error', (error) => {
      reject(error);
    });
  });
}

// API Routes
app.get('/api/stats', (req, res) => {
  const stats = {
    total_detections: detectionHistory.length,
    malicious_detections: detectionHistory.filter(d => d.is_malicious).length,
    normal_detections: detectionHistory.filter(d => !d.is_malicious).length,
    blockchain_logs: detectionHistory.filter(d => d.blockchain_logged).length,
    average_blockchain_latency: 0.0,
    detection_rate: 0.0,
    timestamp: new Date().toISOString(),
    connected_clients: connectedClients.size,
    total_logs: activityLogs.length
  };
  
  if (stats.total_detections > 0) {
    stats.detection_rate = stats.malicious_detections / stats.total_detections;
  }
  
  res.json(stats);
});

app.get('/api/system/info', async (req, res) => {
  // Check blockchain status
  let blockchainInfo = {
    connected: false,
    chain_id: null,
    account_address: null,
    contract_address: null,
    network: 'local',
    network_url: 'http://127.0.0.1:8545',
    block_number: null,
    gas_price: null,
    total_records: 0
  };
  
  try {
    const blockchainStatus = await callPython('backend/blockchain_status.py', []);
    blockchainInfo = blockchainStatus;
  } catch (error) {
    console.error('Failed to check blockchain status:', error);
  }
  
  res.json({
    system: {
      name: 'VANET Misbehavior Detection System',
      version: '1.0.0',
      status: 'running',
      blockchain_status: blockchainInfo.connected ? 'connected' : 'disconnected'
    },
    blockchain: blockchainInfo,
    technology: {
      backend: 'Node.js + Express',
      ml_models: ['Random Forest', 'SVM', 'Deep Neural Network'],
      blockchain: 'Ethereum (Hardhat)',
      frontend: 'React + Vite',
      visualization: 'Recharts',
      ml_framework: 'scikit-learn (MLPClassifier for DNN)',
      ml_backend: 'Python (called from Node.js)'
    },
    workflow: {
      steps: [
        'Vehicle sends Basic Safety Message (BSM)',
        'ML model extracts features and predicts misbehavior',
        'High-confidence detections logged to blockchain',
        'Trust scores updated and vehicles blacklisted if needed',
        'Real-time monitoring and visualization'
      ]
    },
    model_info: {
      current: 'DNN',
      available: ['RF', 'SVM', 'DNN']
    }
  });
});

app.post('/api/detect', async (req, res) => {
  try {
    const { vehicle_id, bsm_data, log_to_blockchain = true } = req.body;
    
    if (!bsm_data) {
      return res.status(400).json({ error: 'BSM data is required' });
    }
    
    // Validate and fill missing features
    const requiredFeatures = [
      'msg_frequency', 'pos_x_consistency', 'pos_y_consistency',
      'pos_z_consistency', 'pos_temporal_consistency', 'speed_consistency',
      'accel_consistency', 'heading_consistency', 'signal_strength',
      'rssi_variance', 'msg_timing', 'neighbor_count',
      'route_deviation', 'timestamp_anomaly', 'payload_size'
    ];
    
    const completeBsm = { ...bsm_data };
    requiredFeatures.forEach(feature => {
      if (!(feature in completeBsm)) {
        completeBsm[feature] = 0.5;
      }
    });
    
    broadcastLog(`Running detection for vehicle ${vehicle_id || 'AUTO'}...`, 'detection');
    
    // Call Python detection script
    const result = await callPython('backend/detect.py', [], {
      vehicle_id: vehicle_id || `VEH${Date.now()}`,
      bsm_data: completeBsm,
      model_type: 'dnn',
      log_to_blockchain: log_to_blockchain
    });
    
    result.timestamp = new Date().toISOString();
    detectionHistory.push(result);
    if (detectionHistory.length > 1000) detectionHistory.shift();
    
    const status = result.is_malicious ? '🚨 MALICIOUS' : '✅ NORMAL';
    broadcastLog(
      `${status} | Vehicle: ${result.vehicle_id} | Confidence: ${result.confidence_percent?.toFixed(2) || 0}%`,
      'detection',
      result
    );
    
    // Broadcast blockchain log if detection was logged to blockchain
    if (result.blockchain_logged) {
      const txHash = result.tx_hash || 'N/A';
      const txHashShort = txHash.length > 16 ? `${txHash.substring(0, 16)}...` : txHash;
      const latency = result.blockchain_latency || 0;
      broadcastLog(
        `⛓️ Blockchain: TX ${txHashShort} | Latency: ${latency.toFixed(2)}s | Gas: ${result.gas_used || 'N/A'}`,
        'blockchain',
        result
      );
    }
    
    res.json(result);
  } catch (error) {
    console.error('Detection error:', error);
    broadcastLog(`❌ Detection failed: ${error.message}`, 'error');
    res.status(500).json({
      error: 'Detection failed',
      details: error.message,
      message: 'An error occurred during detection'
    });
  }
});

app.post('/api/detect/batch', async (req, res) => {
  try {
    const { num_vehicles = 10, malicious_ratio = 0.3 } = req.body;
    
    if (num_vehicles < 1 || num_vehicles > 1000) {
      return res.status(400).json({ error: 'Number of vehicles must be between 1 and 1000' });
    }
    
    broadcastLog(`Starting batch detection with ${num_vehicles} vehicles...`, 'system');
    
    const results = [];
    const total = num_vehicles;
    
    for (let i = 0; i < num_vehicles; i++) {
      const vehicleId = `VEH${String(i + 1).padStart(3, '0')}`;
      const isMalicious = i >= Math.floor(num_vehicles * (1 - malicious_ratio));
      
      // Generate BSM data
      let bsmData;
      if (isMalicious) {
        bsmData = {
          msg_frequency: 2.5 + Math.random() * 1.5,
          pos_x_consistency: 0.1 + Math.random() * 0.2,
          pos_y_consistency: 0.1 + Math.random() * 0.2,
          pos_z_consistency: 0.2 + Math.random() * 0.2,
          pos_temporal_consistency: 0.1 + Math.random() * 0.2,
          speed_consistency: 0.1 + Math.random() * 0.2,
          accel_consistency: 0.1 + Math.random() * 0.2,
          heading_consistency: 0.2 + Math.random() * 0.2,
          signal_strength: 0.3 + Math.random() * 0.2,
          rssi_variance: 0.5 + Math.random() * 0.5,
          msg_timing: 0.2 + Math.random() * 0.2,
          neighbor_count: 8 + Math.random() * 7,
          route_deviation: 0.6 + Math.random() * 0.4,
          timestamp_anomaly: 0.3 + Math.random() * 0.4,
          payload_size: 150 + Math.random() * 150
        };
      } else {
        bsmData = {
          msg_frequency: 0.8 + Math.random() * 0.7,
          pos_x_consistency: 0.6 + Math.random() * 0.3,
          pos_y_consistency: 0.6 + Math.random() * 0.3,
          pos_z_consistency: 0.7 + Math.random() * 0.2,
          pos_temporal_consistency: 0.7 + Math.random() * 0.2,
          speed_consistency: 0.6 + Math.random() * 0.2,
          accel_consistency: 0.6 + Math.random() * 0.2,
          heading_consistency: 0.7 + Math.random() * 0.2,
          signal_strength: 0.6 + Math.random() * 0.2,
          rssi_variance: 0.1 + Math.random() * 0.2,
          msg_timing: 0.5 + Math.random() * 0.2,
          neighbor_count: 3 + Math.random() * 5,
          route_deviation: Math.random() * 0.2,
          timestamp_anomaly: Math.random() * 0.1,
          payload_size: 80 + Math.random() * 40
        };
      }
      
      try {
        const result = await callPython('backend/detect.py', [], {
          vehicle_id: vehicleId,
          bsm_data: bsmData,
          model_type: 'dnn',
          log_to_blockchain: true
        });
        
        result.timestamp = new Date().toISOString();
        detectionHistory.push(result);
        results.push(result);
        
        // Broadcast blockchain log if logged
        if (result.blockchain_logged) {
          const txHash = result.tx_hash || 'N/A';
          const txHashShort = txHash.length > 16 ? `${txHash.substring(0, 16)}...` : txHash;
          const latency = result.blockchain_latency || 0;
          broadcastLog(
            `⛓️ Blockchain: TX ${txHashShort} | Vehicle: ${vehicleId} | Latency: ${latency.toFixed(2)}s`,
            'blockchain',
            result
          );
        }
        
        if ((i + 1) % 10 === 0) {
          broadcastLog(`Processed ${i + 1}/${num_vehicles} vehicles...`, 'info');
        }
      } catch (error) {
        console.error(`Error processing vehicle ${vehicleId}:`, error);
        continue;
      }
    }
    
    if (results.length === 0) {
      return res.status(500).json({
        error: 'No vehicles processed',
        message: 'All vehicles failed detection'
      });
    }
    
    const maliciousCount = results.filter(r => r.is_malicious).length;
    const blockchainLogged = results.filter(r => r.blockchain_logged);
    const blockchainLatencies = blockchainLogged
      .map(r => r.blockchain_latency)
      .filter(l => l && l > 0);
    const avgLatency = blockchainLatencies.length > 0
      ? blockchainLatencies.reduce((a, b) => a + b, 0) / blockchainLatencies.length
      : 0.0;
    
    const summary = {
      total_detections: results.length,
      malicious_detections: maliciousCount,
      normal_detections: results.length - maliciousCount,
      blockchain_logs: blockchainLogged.length,
      average_blockchain_latency: avgLatency,
      detection_rate: maliciousCount / results.length
    };
    
    broadcastLog(`Batch detection complete: ${results.length} vehicles processed`, 'success');
    
    res.json({
      results,
      summary,
      count: results.length,
      processed: results.length,
      requested: num_vehicles
    });
  } catch (error) {
    console.error('Batch detection error:', error);
    res.status(500).json({
      error: 'Batch detection failed',
      details: error.message
    });
  }
});

// SSE Logs endpoint
app.get('/api/logs/stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  
  const client = { response: res, id: Date.now() };
  connectedClients.add(client);
  
  // Send initial connection
  res.write(`data: ${JSON.stringify({ type: 'connected', client_id: client.id })}\n\n`);
  
  // Send recent logs
  activityLogs.slice(-100).forEach(log => {
    res.write(`data: ${JSON.stringify({ type: 'log', ...log })}\n\n`);
  });
  
  // Keep connection alive
  const heartbeat = setInterval(() => {
    if (res.closed) {
      clearInterval(heartbeat);
      connectedClients.delete(client);
    } else {
      res.write(`data: ${JSON.stringify({ type: 'heartbeat', timestamp: new Date().toISOString() })}\n\n`);
    }
  }, 30000);
  
  req.on('close', () => {
    clearInterval(heartbeat);
    connectedClients.delete(client);
  });
});

app.get('/api/logs', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const category = req.query.category;
  
  let logs = activityLogs;
  if (category) {
    logs = logs.filter(log => log.category === category);
  }
  
  res.json({
    logs: logs.slice(-limit),
    total: logs.length
  });
});

app.get('/api/detections', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const maliciousOnly = req.query.malicious_only === 'true';
  
  let detections = detectionHistory;
  if (maliciousOnly) {
    detections = detections.filter(d => d.is_malicious);
  }
  
  res.json({
    detections: detections.slice(-limit),
    total: detections.length
  });
});

app.get('/api/detections/chart', (req, res) => {
  const limit = parseInt(req.query.limit) || 50;
  const detections = detectionHistory.slice(-limit);
  
  const timeSeries = [];
  let maliciousCount = 0;
  let normalCount = 0;
  
  detections.forEach((det, index) => {
    if (det.is_malicious) maliciousCount++;
    else normalCount++;
    
    timeSeries.push({
      time: det.timestamp || new Date().toISOString(),
      index: index + 1,
      malicious: maliciousCount,
      normal: normalCount,
      total: index + 1,
      confidence: det.confidence_percent || 0
    });
  });
  
  const confidences = detections.map(d => d.confidence_percent || 0);
  const typeCounts = {};
  detections.forEach(d => {
    if (d.is_malicious) {
      const type = d.misbehavior_type_name || 'unknown';
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    }
  });
  
  res.json({
    timeSeries,
    confidenceDistribution: confidences,
    typeDistribution: typeCounts,
    summary: {
      malicious: maliciousCount,
      normal: normalCount,
      total: detections.length
    }
  });
});

app.get('/api/activity/summary', (req, res) => {
  const categoryCounts = {};
  activityLogs.forEach(log => {
    const cat = log.category;
    categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
  });
  
  res.json({
    categories: categoryCounts,
    total: activityLogs.length,
    recent_activity: activityLogs.slice(-10)
  });
});

// Training endpoints
app.post('/api/training/start', async (req, res) => {
  const { dataset_path = 'data/veremi_dataset.csv' } = req.body;
  
  broadcastLog(`Starting ML model training with dataset: ${dataset_path}`, 'system');
  
  // Use virtual environment Python if available
  const venvPython = path.join(__dirname, 'venv', 'bin', 'python');
  const pythonPath = require('fs').existsSync(venvPython) ? venvPython : 
                    (process.platform === 'win32' ? 'python' : 'python3');
  
  // Run training in background
  const trainingScript = spawn(pythonPath, ['ml/train_models.py', '--dataset', dataset_path], {
    cwd: __dirname,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
  
  trainingScript.stdout.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
      if (line.trim()) {
        // Categorize log lines
        const category = line.includes('ERROR') || line.includes('error') || line.includes('❌') ? 'error' :
                        line.includes('WARNING') || line.includes('warning') || line.includes('⚠️') ? 'warning' :
                        line.includes('✅') || line.includes('complete') ? 'success' :
                        line.includes('Training') || line.includes('model') ? 'system' : 'info';
        broadcastLog(line.trim(), category);
      }
    });
  });
  
  trainingScript.stderr.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
      if (line.trim()) {
        broadcastLog(line.trim(), 'error');
      }
    });
  });
  
  trainingScript.on('close', (code) => {
    if (code === 0) {
      broadcastLog('✅ Training completed successfully', 'success');
    } else {
      broadcastLog(`❌ Training failed with exit code ${code}`, 'error');
      broadcastLog('Check that virtual environment is activated and dependencies are installed', 'warning');
    }
  });
  
  trainingScript.on('error', (error) => {
    broadcastLog(`❌ Failed to start training: ${error.message}`, 'error');
    broadcastLog(`Python path used: ${pythonPath}`, 'info');
  });
  
  res.json({ status: 'started', message: 'Training started', dataset: dataset_path });
});

app.get('/api/training/status', (req, res) => {
  res.json({
    is_running: false,
    progress: 0,
    current_step: '',
    error: null
  });
});

app.get('/api/training/logs', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const trainingLogs = activityLogs.filter(log => 
    log.message.includes('Training') || 
    log.message.includes('model') ||
    log.category === 'system'
  );
  
  res.json({
    logs: trainingLogs.slice(-limit),
    total: trainingLogs.length
  });
});

// ML Training results
app.get('/api/ml/training', async (req, res) => {
  try {
    const trainingFile = path.join(__dirname, 'results/training_results.json');
    const data = await fs.readFile(trainingFile, 'utf8');
    const results = JSON.parse(data);
    
    res.json({
      results,
      feature_names: [
        'msg_frequency', 'pos_x_consistency', 'pos_y_consistency',
        'pos_z_consistency', 'pos_temporal_consistency', 'speed_consistency',
        'accel_consistency', 'heading_consistency', 'signal_strength',
        'rssi_variance', 'msg_timing', 'neighbor_count',
        'route_deviation', 'timestamp_anomaly', 'payload_size'
      ],
      models: results.map(r => r.model),
      metrics: {
        accuracy: results.map(r => r.accuracy),
        precision: results.map(r => r.precision),
        recall: results.map(r => r.recall),
        f1_score: results.map(r => r.f1_score)
      }
    });
  } catch (error) {
    res.status(404).json({
      error: 'Training results not found',
      message: 'Run training first: python ml/train_models.py'
    });
  }
});

app.get('/api/ml/model-comparison', async (req, res) => {
  try {
    const trainingFile = path.join(__dirname, 'results/training_results.json');
    const data = await fs.readFile(trainingFile, 'utf8');
    const results = JSON.parse(data);
    
    const comparison = results.map(result => {
      const cm = result.confusion_matrix;
      const tn = cm[0][0], fp = cm[0][1];
      const fn = cm[1][0], tp = cm[1][1];
      
      return {
        model: result.model,
        accuracy: result.accuracy,
        precision: result.precision,
        recall: result.recall,
        f1_score: result.f1_score,
        true_positives: tp,
        true_negatives: tn,
        false_positives: fp,
        false_negatives: fn,
        sensitivity: result.recall,
        specificity: tn / (tn + fp) || 0
      };
    });
    
    res.json({
      comparison,
      summary: {
        best_accuracy: Math.max(...results.map(r => r.accuracy)),
        best_f1: Math.max(...results.map(r => r.f1_score)),
        best_model: results.reduce((best, curr) => 
          curr.f1_score > best.f1_score ? curr : best
        ).model
      }
    });
  } catch (error) {
    res.status(404).json({ error: 'Training results not found' });
  }
});

// Blockchain endpoints
app.get('/api/blockchain/status', async (req, res) => {
  try {
    const status = await callPython('backend/blockchain_status.py', []);
    res.json(status);
  } catch (error) {
    res.status(500).json({
      connected: false,
      error: error.message
    });
  }
});

app.get('/api/blockchain/activity', (req, res) => {
  // Get blockchain-related logs
  const blockchainLogs = activityLogs.filter(log => 
    log.category === 'blockchain' || 
    log.message.includes('blockchain') ||
    log.message.includes('TX') ||
    log.message.includes('⛓️')
  );
  
  // Get blockchain detections
  const blockchainDetections = detectionHistory.filter(d => d.blockchain_logged);
  
  // Group by time for charts
  const timeSeries = [];
  const latencyData = [];
  const gasData = [];
  let cumulative = 0;
  
  blockchainDetections.forEach((det, index) => {
    cumulative++;
    timeSeries.push({
      time: det.timestamp || new Date().toISOString(),
      index: index + 1,
      count: cumulative,
      latency: det.blockchain_latency || 0,
      gas: det.gas_used || 0
    });
    
    if (det.blockchain_latency) {
      latencyData.push({
        index: index + 1,
        latency: det.blockchain_latency
      });
    }
    
    if (det.gas_used) {
      gasData.push({
        index: index + 1,
        gas: det.gas_used
      });
    }
  });
  
  // Misbehavior type distribution
  const typeCounts = {};
  blockchainDetections.forEach(d => {
    const type = d.misbehavior_type_name || 'Unknown';
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });
  
  res.json({
    logs: blockchainLogs.slice(-100),
    total_logs: blockchainLogs.length,
    detections: blockchainDetections.slice(-50),
    total_detections: blockchainDetections.length,
    transactions: blockchainDetections.map(d => ({
      vehicle_id: d.vehicle_id,
      tx_hash: d.tx_hash,
      timestamp: d.timestamp,
      misbehavior_type: d.misbehavior_type_name,
      confidence: d.confidence_percent,
      gas_used: d.gas_used,
      latency: d.blockchain_latency
    })),
    timeSeries: timeSeries.slice(-50),
    latencyData: latencyData.slice(-50),
    gasData: gasData.slice(-50),
    typeDistribution: Object.entries(typeCounts).map(([type, count]) => ({
      type,
      count
    })),
    averageLatency: latencyData.length > 0
      ? latencyData.reduce((sum, d) => sum + d.latency, 0) / latencyData.length
      : 0,
    totalGas: gasData.reduce((sum, d) => sum + d.gas, 0)
  });
});

// Serve static files
app.use(express.static(path.join(__dirname, 'frontend/dist')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/dist/index.html'));
});

// Initialize
broadcastLog('Starting Node.js server on http://localhost:5000', 'system');
broadcastLog('✅ Server initialized', 'success');

// Check blockchain status periodically
setInterval(async () => {
  try {
    const status = await callPython('backend/blockchain_status.py', []);
    if (status.connected) {
      if (status.block_number) {
        broadcastLog(
          `⛓️ Blockchain: Connected | Chain ID: ${status.chain_id} | Block: ${status.block_number} | Records: ${status.total_records}`,
          'blockchain'
        );
      }
    } else {
      broadcastLog(
        `⛓️ Blockchain: Disconnected | Network: ${status.network_url}`,
        'blockchain'
      );
    }
  } catch (error) {
    // Silently fail - don't spam logs
  }
}, 30000); // Check every 30 seconds

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  broadcastLog(`Server running on port ${PORT}`, 'system');
  
  // Initial blockchain check
  setTimeout(async () => {
    try {
      const status = await callPython('backend/blockchain_status.py', []);
      if (status.connected) {
        broadcastLog(
          `⛓️ Blockchain Connected! Account: ${status.account_address?.substring(0, 10)}... | Contract: ${status.contract_address?.substring(0, 10)}...`,
          'blockchain'
        );
      } else {
        broadcastLog(
          `⛓️ Blockchain Disconnected. Start Hardhat node: npx hardhat node`,
          'warning'
        );
      }
    } catch (error) {
      broadcastLog(
        `⛓️ Blockchain status check failed. Make sure Hardhat is running on port 8545`,
        'warning'
      );
    }
  }, 2000);
});

