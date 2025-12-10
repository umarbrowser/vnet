import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './TrainingViewer.css'

function TrainingViewer() {
  const [status, setStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [autoScroll, setAutoScroll] = useState(true)
  const [datasetPath, setDatasetPath] = useState('data/veremi_dataset.csv')
  const logsEndRef = useRef(null)
  const logsContainerRef = useRef(null)
  const eventSourceRef = useRef(null)

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    // Connect to SSE stream if training is running
    if (status?.is_running) {
      const eventSource = new EventSource('/api/training/logs/stream')
      eventSourceRef.current = eventSource

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        if (data.type === 'log') {
          setLogs(prev => [...prev, data].slice(-1000))
        } else if (data.type === 'status') {
          setStatus(data)
        }
      }

      eventSource.onerror = () => {
        eventSource.close()
      }

      return () => {
        eventSource.close()
      }
    } else {
      // Fetch logs if not streaming
      fetchLogs()
    }
  }, [status?.is_running])

  useEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      const container = logsContainerRef.current
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50
      if (isAtBottom) {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }, [logs, autoScroll])

  const fetchStatus = async () => {
    try {
      const response = await axios.get('/api/training/status')
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch training status:', error)
    }
  }

  const fetchLogs = async () => {
    try {
      const response = await axios.get('/api/training/logs?limit=500')
      setLogs(response.data.logs)
    } catch (error) {
      console.error('Failed to fetch training logs:', error)
    }
  }

  const startTraining = async () => {
    try {
      await axios.post('/api/training/start', { dataset_path: datasetPath })
      setLogs([])
      fetchStatus()
    } catch (error) {
      console.error('Failed to start training:', error)
      alert('Failed to start training: ' + (error.response?.data?.error || error.message))
    }
  }

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString()
  }

  return (
    <div className="training-viewer">
      <div className="training-header">
        <div>
          <h2 className="training-title">ML Model Training</h2>
          <p className="training-subtitle">Train machine learning models using your dataset</p>
        </div>
        <div className="training-controls">
          <div className="dataset-input-group">
            <label htmlFor="dataset-path" className="dataset-label">Dataset Path:</label>
            <input
              id="dataset-path"
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="dataset-input"
              disabled={status?.is_running}
              placeholder="data/veremi_dataset.csv"
            />
          </div>
          <button
            onClick={startTraining}
            disabled={status?.is_running}
            className="start-training-button"
          >
            {status?.is_running ? 'Training...' : 'Start Training'}
          </button>
        </div>
      </div>

      {status && (
        <div className="training-status-card">
          <div className="status-row">
            <div className="status-item">
              <span className="status-label">Status:</span>
              <span className={`status-value ${status.is_running ? 'running' : status.error ? 'error' : 'idle'}`}>
                {status.is_running ? 'Running' : status.error ? 'Error' : 'Idle'}
              </span>
            </div>
            {status.is_running && (
              <>
                <div className="status-item">
                  <span className="status-label">Progress:</span>
                  <span className="status-value">{status.progress}%</span>
                </div>
                <div className="status-item">
                  <span className="status-label">Step:</span>
                  <span className="status-value">{status.current_step}</span>
                </div>
              </>
            )}
            {status.start_time && (
              <div className="status-item">
                <span className="status-label">Started:</span>
                <span className="status-value">{formatTimestamp(status.start_time)}</span>
              </div>
            )}
          </div>
          {status.is_running && (
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${status.progress}%` }} />
            </div>
          )}
          {status.error && (
            <div className="error-message">
              <strong>Error:</strong> {status.error}
            </div>
          )}
        </div>
      )}

      <div className="training-logs-container" ref={logsContainerRef}>
        <div className="logs-header-bar">
          <h3 className="logs-title">Training Logs</h3>
          <label className="auto-scroll-toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
        </div>
        <div className="logs-content">
          {logs.length === 0 ? (
            <div className="logs-empty">
              <p>No training logs yet. Start training to see logs.</p>
            </div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="log-line">
                <span className="log-time">{formatTimestamp(log.timestamp)}</span>
                <span className={`log-message log-${log.type}`}>{log.message}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      <div className="training-info">
        <h3>Training Information</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>Models Trained</h4>
            <ul>
              <li>Random Forest (RF)</li>
              <li>Support Vector Machine (SVM)</li>
              <li>Deep Neural Network (DNN)</li>
            </ul>
          </div>
          <div className="info-card">
            <h4>Dataset Requirements</h4>
            <ul>
              <li>CSV format with header row</li>
              <li>15 feature columns</li>
              <li>1 label column (0 = normal, 1 = malicious)</li>
            </ul>
          </div>
          <div className="info-card">
            <h4>Training Process</h4>
            <ul>
              <li>Data loading and validation</li>
              <li>Train/test split (80/20)</li>
              <li>Feature scaling</li>
              <li>Model training</li>
              <li>Performance evaluation</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TrainingViewer


