import React, { useState } from 'react'
import axios from 'axios'
import './DetectionForm.css'

function DetectionForm() {
  const [formData, setFormData] = useState({
    vehicle_id: '',
    msg_frequency: '1.0',
    pos_x_consistency: '0.7',
    pos_y_consistency: '0.7',
    pos_z_consistency: '0.7',
    pos_temporal_consistency: '0.7',
    speed_consistency: '0.6',
    accel_consistency: '0.6',
    heading_consistency: '0.7',
    signal_strength: '0.6',
    rssi_variance: '0.2',
    msg_timing: '0.6',
    neighbor_count: '5',
    route_deviation: '0.1',
    timestamp_anomaly: '0.0',
    payload_size: '100'
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [batchMode, setBatchMode] = useState(false)
  const [batchConfig, setBatchConfig] = useState({
    num_vehicles: '1000',
    malicious_ratio: '0.3'
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleBatchConfigChange = (e) => {
    const { name, value } = e.target
    setBatchConfig(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      const bsm_data = {
        msg_frequency: parseFloat(formData.msg_frequency),
        pos_x_consistency: parseFloat(formData.pos_x_consistency),
        pos_y_consistency: parseFloat(formData.pos_y_consistency),
        pos_z_consistency: parseFloat(formData.pos_z_consistency),
        pos_temporal_consistency: parseFloat(formData.pos_temporal_consistency),
        speed_consistency: parseFloat(formData.speed_consistency),
        accel_consistency: parseFloat(formData.accel_consistency),
        heading_consistency: parseFloat(formData.heading_consistency),
        signal_strength: parseFloat(formData.signal_strength),
        rssi_variance: parseFloat(formData.rssi_variance),
        msg_timing: parseFloat(formData.msg_timing),
        neighbor_count: parseFloat(formData.neighbor_count),
        route_deviation: parseFloat(formData.route_deviation),
        timestamp_anomaly: parseFloat(formData.timestamp_anomaly),
        payload_size: parseFloat(formData.payload_size)
      }

      const response = await axios.post('/api/detect', {
        vehicle_id: formData.vehicle_id || undefined,
        bsm_data,
        log_to_blockchain: true
      })

      setResult(response.data)
    } catch (error) {
      setResult({
        error: error.response?.data?.error || error.message
      })
    } finally {
      setLoading(false)
    }
  }

  const handleBatchSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      const response = await axios.post('/api/detect/batch', {
        num_vehicles: parseInt(batchConfig.num_vehicles),
        malicious_ratio: parseFloat(batchConfig.malicious_ratio)
      })

      if (response.data.error) {
        setResult({
          error: response.data.error,
          message: response.data.message || 'Batch detection failed'
        })
      } else {
        setResult({
          batch: true,
          summary: response.data.summary,
          count: response.data.count || response.data.results?.length || 0,
          processed: response.data.processed,
          requested: response.data.requested
        })
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.response?.data?.message || error.message
      const errorDetails = error.response?.data?.details
      
      setResult({
        error: errorMessage,
        details: errorDetails,
        message: error.response?.data?.message || 'Failed to run batch detection'
      })
    } finally {
      setLoading(false)
    }
  }

  const fieldInfo = {
    msg_frequency: {
      description: 'Message frequency (messages per second)',
      example: 'Normal: 0.8-1.5, Malicious: 2.5-4.0',
      range: [0, 5]
    },
    pos_x_consistency: {
      description: 'Position X coordinate consistency (0-1)',
      example: 'Normal: 0.6-0.9, Malicious: 0.1-0.3',
      range: [0, 1]
    },
    pos_y_consistency: {
      description: 'Position Y coordinate consistency (0-1)',
      example: 'Normal: 0.6-0.9, Malicious: 0.1-0.3',
      range: [0, 1]
    },
    pos_z_consistency: {
      description: 'Position Z coordinate consistency (0-1)',
      example: 'Normal: 0.7-0.9, Malicious: 0.2-0.4',
      range: [0, 1]
    },
    pos_temporal_consistency: {
      description: 'Temporal position consistency (0-1)',
      example: 'Normal: 0.7-0.9, Malicious: 0.1-0.3',
      range: [0, 1]
    },
    speed_consistency: {
      description: 'Speed consistency across messages (0-1)',
      example: 'Normal: 0.6-0.8, Malicious: 0.1-0.3',
      range: [0, 1]
    },
    accel_consistency: {
      description: 'Acceleration consistency (0-1)',
      example: 'Normal: 0.6-0.8, Malicious: 0.1-0.3',
      range: [0, 1]
    },
    heading_consistency: {
      description: 'Heading/direction consistency (0-1)',
      example: 'Normal: 0.7-0.9, Malicious: 0.2-0.4',
      range: [0, 1]
    },
    signal_strength: {
      description: 'Signal strength (0-1)',
      example: 'Normal: 0.6-0.8, Malicious: 0.3-0.5',
      range: [0, 1]
    },
    rssi_variance: {
      description: 'RSSI variance (signal stability)',
      example: 'Normal: 0.1-0.3, Malicious: 0.5-1.0',
      range: [0, 1]
    },
    msg_timing: {
      description: 'Message timing consistency (0-1)',
      example: 'Normal: 0.5-0.7, Malicious: 0.2-0.4',
      range: [0, 1]
    },
    neighbor_count: {
      description: 'Number of neighboring vehicles',
      example: 'Normal: 3-8, Malicious: 8-15',
      range: [0, 20]
    },
    route_deviation: {
      description: 'Route deviation from expected path (0-1)',
      example: 'Normal: 0.0-0.2, Malicious: 0.6-1.0',
      range: [0, 1]
    },
    timestamp_anomaly: {
      description: 'Timestamp anomaly score (0-1)',
      example: 'Normal: 0.0-0.1, Malicious: 0.3-0.7',
      range: [0, 1]
    },
    payload_size: {
      description: 'Message payload size (bytes)',
      example: 'Normal: 80-120, Malicious: 150-300',
      range: [0, 500]
    }
  }

  return (
    <div className="detection-form">
      <div className="form-header">
        <h2 className="form-title">Vehicle Detection</h2>
        <p className="form-subtitle">Run misbehavior detection on vehicle BSM data</p>
      </div>

      <div className="form-mode-toggle">
        <button
          className={`mode-button ${!batchMode ? 'active' : ''}`}
          onClick={() => setBatchMode(false)}
        >
          Single Detection
        </button>
        <button
          className={`mode-button ${batchMode ? 'active' : ''}`}
          onClick={() => setBatchMode(true)}
        >
          Batch Detection
        </button>
      </div>

      {!batchMode ? (
        <form onSubmit={handleSubmit} className="detection-form-content">
          <div className="form-section">
            <h3 className="section-title">Vehicle Information</h3>
            <div className="form-group">
              <label htmlFor="vehicle_id" className="form-label">
                Vehicle ID (optional)
              </label>
              <input
                type="text"
                id="vehicle_id"
                name="vehicle_id"
                value={formData.vehicle_id}
                onChange={handleInputChange}
                className="form-input"
                placeholder="VEH001"
              />
              <p className="form-help">Leave empty to auto-generate</p>
            </div>
          </div>

          <div className="form-section">
            <h3 className="section-title">BSM Features</h3>
            <div className="form-grid">
              {Object.entries(formData).filter(([key]) => key !== 'vehicle_id').map(([key, value]) => (
                <div key={key} className="form-group">
                  <label htmlFor={key} className="form-label">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type="number"
                    id={key}
                    name={key}
                    value={value}
                    onChange={handleInputChange}
                    className="form-input"
                    step="0.01"
                    min={fieldInfo[key]?.range[0]}
                    max={fieldInfo[key]?.range[1]}
                  />
                  <p className="form-description">{fieldInfo[key]?.description}</p>
                  <p className="form-example">{fieldInfo[key]?.example}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Processing...' : 'Run Detection'}
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={handleBatchSubmit} className="detection-form-content">
          <div className="form-section">
            <h3 className="section-title">Batch Configuration</h3>
            <div className="form-group">
              <label htmlFor="num_vehicles" className="form-label">
                Number of Vehicles
              </label>
              <input
                type="number"
                id="num_vehicles"
                name="num_vehicles"
                value={batchConfig.num_vehicles}
                onChange={handleBatchConfigChange}
                className="form-input"
                min="1"
                max="1000"
              />
              <p className="form-description">Number of vehicles to simulate and detect (1 to 1000)</p>
            </div>
            <div className="form-group">
              <label htmlFor="malicious_ratio" className="form-label">
                Malicious Ratio
              </label>
              <input
                type="number"
                id="malicious_ratio"
                name="malicious_ratio"
                value={batchConfig.malicious_ratio}
                onChange={handleBatchConfigChange}
                className="form-input"
                step="0.1"
                min="0"
                max="1"
              />
              <p className="form-description">Ratio of malicious vehicles (0.0 to 1.0)</p>
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Processing...' : 'Run Batch Detection'}
            </button>
          </div>
        </form>
      )}

      {result && (
        <div className={`result-card ${result.error ? 'result-error' : result.is_malicious ? 'result-malicious' : 'result-normal'}`}>
          <h3 className="result-title">Detection Result</h3>
          {result.error ? (
            <div className="result-content">
              <p className="result-error-text">Error: {result.error}</p>
              {result.message && (
                <p className="result-error-details">{result.message}</p>
              )}
              {result.details && (
                <p className="result-error-details" style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                  Details: {result.details}
                </p>
              )}
            </div>
          ) : result.batch ? (
            <div className="result-content">
              <p className="result-success-text">✅ Batch detection complete!</p>
              <div className="result-stats">
                {result.requested && (
                  <div className="result-stat">
                    <span className="stat-label">Requested:</span>
                    <span className="stat-value">{result.requested}</span>
                  </div>
                )}
                <div className="result-stat">
                  <span className="stat-label">Vehicles Processed:</span>
                  <span className="stat-value">{result.count || result.processed || 0}</span>
                </div>
                {result.summary && (
                  <>
                    <div className="result-stat">
                      <span className="stat-label">Total Detections:</span>
                      <span className="stat-value">{result.summary.total_detections}</span>
                    </div>
                    <div className="result-stat">
                      <span className="stat-label">Malicious:</span>
                      <span className="stat-value">{result.summary.malicious_detections}</span>
                    </div>
                    <div className="result-stat">
                      <span className="stat-label">Normal:</span>
                      <span className="stat-value">{result.summary.normal_detections}</span>
                    </div>
                    <div className="result-stat">
                      <span className="stat-label">Detection Rate:</span>
                      <span className="stat-value">{(result.summary.detection_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="result-stat">
                      <span className="stat-label">Blockchain Logs:</span>
                      <span className="stat-value">{result.summary.blockchain_logs}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="result-content">
              <div className="result-status">
                <span className={`status-badge ${result.is_malicious ? 'badge-malicious' : 'badge-normal'}`}>
                  {result.is_malicious ? '🚨 MALICIOUS' : '✅ NORMAL'}
                </span>
              </div>
              <div className="result-details">
                <div className="result-detail">
                  <span className="detail-label">Vehicle ID:</span>
                  <span className="detail-value">{result.vehicle_id}</span>
                </div>
                <div className="result-detail">
                  <span className="detail-label">Confidence:</span>
                  <span className="detail-value">{result.confidence_percent.toFixed(2)}%</span>
                </div>
                {result.misbehavior_type_name && (
                  <div className="result-detail">
                    <span className="detail-label">Type:</span>
                    <span className="detail-value">{result.misbehavior_type_name}</span>
                  </div>
                )}
                {result.blockchain_logged && (
                  <>
                    <div className="result-detail">
                      <span className="detail-label">Blockchain TX:</span>
                      <span className="detail-value">{result.tx_hash?.substring(0, 16)}...</span>
                    </div>
                    <div className="result-detail">
                      <span className="detail-label">Latency:</span>
                      <span className="detail-value">{result.blockchain_latency?.toFixed(2)}s</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default DetectionForm

