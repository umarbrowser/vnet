import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ActivitySummary from './ActivitySummary'
import ProjectInfo from './ProjectInfo'
import './Dashboard.css'

function Dashboard() {
  const [systemInfo, setSystemInfo] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchSystemInfo()
    fetchStats()
    const interval = setInterval(fetchStats, 2000)
    return () => clearInterval(interval)
  }, [])

  const fetchSystemInfo = async () => {
    try {
      const response = await axios.get('/api/system/info')
      if (response.data && !response.data.error) {
        setSystemInfo(response.data)
      } else {
        console.warn('API returned error, using default data:', response.data)
      }
    } catch (error) {
      console.error('Failed to fetch system info:', error)
      // Use default data on error
    }
  }

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats')
      setStats(response.data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
      setLoading(false)
    }
  }

  if (loading && !systemInfo) {
    return <div className="dashboard-loading">Loading dashboard...</div>
  }

  // Default system info if API fails
  const defaultSystemInfo = {
    system: {
      name: 'VANET Misbehavior Detection System',
      version: '1.0.0',
      status: 'running',
      blockchain_status: 'disconnected'
    },
    technology: {
      backend: 'Flask + Python',
      ml_models: ['Random Forest', 'SVM', 'Deep Neural Network'],
      blockchain: 'Ethereum (Hardhat)',
      frontend: 'React + Vite',
      visualization: 'Recharts',
      ml_framework: 'scikit-learn (MLPClassifier for DNN)'
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
  }

  const displayInfo = systemInfo || defaultSystemInfo

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2 className="dashboard-title">System Overview</h2>
        <p className="dashboard-subtitle">Real-time monitoring and system information</p>
      </div>

      <div className="dashboard-grid">
        {/* System Information Card */}
        <div className="dashboard-card system-info-card">
          <h3 className="card-title">System Information</h3>
          {displayInfo && (
            <div className="system-info">
              <div className="info-item">
                <span className="info-label">Name:</span>
                <span className="info-value">{displayInfo.system.name}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Version:</span>
                <span className="info-value">{displayInfo.system.version}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Status:</span>
                <span className={`info-value status-${displayInfo.system.status}`}>
                  {displayInfo.system.status}
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Blockchain:</span>
                <span className={`info-value status-${displayInfo.system.blockchain_status || 'disconnected'}`}>
                  {displayInfo.system.blockchain_status || 'disconnected'}
                </span>
              </div>
              {displayInfo.blockchain && displayInfo.blockchain.connected && (
                <>
                  <div className="info-item">
                    <span className="info-label">Account:</span>
                    <span className="info-value" style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>
                      {displayInfo.blockchain.account_address || 'N/A'}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Contract:</span>
                    <span className="info-value" style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>
                      {displayInfo.blockchain.contract_address || 'N/A'}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Chain ID:</span>
                    <span className="info-value">{displayInfo.blockchain.chain_id || 'N/A'}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Block:</span>
                    <span className="info-value">{displayInfo.blockchain.block_number || 'N/A'}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Total Records:</span>
                    <span className="info-value">{displayInfo.blockchain.total_records || 0}</span>
                  </div>
                </>
              )}
              <div className="info-item">
                <span className="info-label">Model:</span>
                <span className="info-value">{displayInfo.model_info?.current || 'DNN'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Technology Stack Card */}
        <div className="dashboard-card tech-stack-card">
          <h3 className="card-title">Technology Stack</h3>
          {displayInfo && (
            <div className="tech-stack">
              <div className="tech-item">
                <span className="tech-label">Backend:</span>
                <span className="tech-value">{displayInfo.technology.backend}</span>
              </div>
              <div className="tech-item">
                <span className="tech-label">ML Models:</span>
                <span className="tech-value">{displayInfo.technology.ml_models.join(', ')}</span>
              </div>
              <div className="tech-item">
                <span className="tech-label">Blockchain:</span>
                <span className="tech-value">{displayInfo.technology.blockchain}</span>
              </div>
              <div className="tech-item">
                <span className="tech-label">Frontend:</span>
                <span className="tech-value">{displayInfo.technology.frontend}</span>
              </div>
              <div className="tech-item">
                <span className="tech-label">Visualization:</span>
                <span className="tech-value">{displayInfo.technology.visualization}</span>
              </div>
              <div className="tech-item">
                <span className="tech-label">ML Framework:</span>
                <span className="tech-value">{displayInfo.technology.ml_framework || 'scikit-learn'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Statistics Cards */}
        {stats && (
          <>
            <div className="dashboard-card stat-card">
              <div className="stat-icon">🔍</div>
              <div className="stat-content">
                <div className="stat-label">Total Detections</div>
                <div className="stat-value">{stats.total_detections.toLocaleString()}</div>
              </div>
            </div>

            <div className="dashboard-card stat-card stat-malicious">
              <div className="stat-icon">🚨</div>
              <div className="stat-content">
                <div className="stat-label">Malicious</div>
                <div className="stat-value">{stats.malicious_detections.toLocaleString()}</div>
              </div>
            </div>

            <div className="dashboard-card stat-card stat-normal">
              <div className="stat-icon">✅</div>
              <div className="stat-content">
                <div className="stat-label">Normal</div>
                <div className="stat-value">{stats.normal_detections.toLocaleString()}</div>
              </div>
            </div>

            <div className="dashboard-card stat-card stat-blockchain">
              <div className="stat-icon">⛓️</div>
              <div className="stat-content">
                <div className="stat-label">Blockchain Logs</div>
                <div className="stat-value">{stats.blockchain_logs.toLocaleString()}</div>
              </div>
            </div>

            <div className="dashboard-card stat-card">
              <div className="stat-icon">⚡</div>
              <div className="stat-content">
                <div className="stat-label">Avg Latency</div>
                <div className="stat-value">
                  {stats.average_blockchain_latency.toFixed(2)}s
                </div>
              </div>
            </div>

            <div className="dashboard-card stat-card">
              <div className="stat-icon">📊</div>
              <div className="stat-content">
                <div className="stat-label">Detection Rate</div>
                <div className="stat-value">
                  {(stats.detection_rate * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </>
        )}

        {/* Workflow Card */}
        <div className="dashboard-card workflow-card">
          <h3 className="card-title">System Workflow</h3>
          {displayInfo && displayInfo.workflow && (
            <div className="workflow">
              {displayInfo.workflow.steps && displayInfo.workflow.steps.map((step, index) => (
                <div key={index} className="workflow-step">
                  <div className="workflow-number">{index + 1}</div>
                  <div className="workflow-text">{step}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Activity Summary */}
      <ActivitySummary />

      {/* Project Information */}
      <ProjectInfo />
    </div>
  )
}

export default Dashboard

