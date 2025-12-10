import React, { useState, useEffect, useRef } from 'react'
import './LogsViewer.css'

function LogsViewer() {
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('all')
  const [autoScroll, setAutoScroll] = useState(true)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const logsEndRef = useRef(null)
  const logsContainerRef = useRef(null)
  const eventSourceRef = useRef(null)

  const categoryColors = {
    info: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    detection: '#8b5cf6',
    blockchain: '#06b6d4',
    system: '#6b7280'
  }

  const categoryIcons = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌',
    detection: '🔍',
    blockchain: '⛓️',
    system: '⚙️'
  }

  useEffect(() => {
    // Connect to SSE stream
    const eventSource = new EventSource('/api/logs/stream')
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'log') {
        setLogs(prev => {
          const newLogs = [...prev, data]
          // Keep only last 5000 logs
          return newLogs.slice(-5000)
        })
      }
    }

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error)
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (eventSource.readyState === EventSource.CLOSED) {
          eventSource.close()
          // Reconnect will happen automatically
        }
      }, 3000)
    }

    return () => {
      eventSource.close()
    }
  }, [])

  useEffect(() => {
    // Auto-scroll logic
    if (autoScroll && logsContainerRef.current) {
      const container = logsContainerRef.current
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50
      
      if (isAtBottom) {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }, [logs, autoScroll])

  useEffect(() => {
    // Check if user scrolled up
    const container = logsContainerRef.current
    if (!container) return

    const handleScroll = () => {
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50
      setAutoScroll(isAtBottom)
      setShowScrollButton(!isAtBottom)
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    setAutoScroll(true)
    setShowScrollButton(false)
  }

  const clearLogs = () => {
    setLogs([])
  }

  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(log => log.category === filter)

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString()
  }

  return (
    <div className="logs-viewer">
      <div className="logs-header">
        <div className="logs-header-content">
          <h2 className="logs-title">Logs & Terminal</h2>
          <p className="logs-subtitle">Real-time system activity and detection logs</p>
        </div>
        <div className="logs-controls">
          <div className="filter-group">
            <label htmlFor="log-filter" className="filter-label">Filter:</label>
            <select
              id="log-filter"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              <option value="info">Info</option>
              <option value="success">Success</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="detection">Detection</option>
              <option value="blockchain">Blockchain</option>
              <option value="system">System</option>
            </select>
          </div>
          <div className="auto-scroll-toggle">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="toggle-input"
              />
              <span className="toggle-text">Auto-scroll</span>
            </label>
          </div>
          <button onClick={clearLogs} className="clear-button">
            Clear
          </button>
        </div>
      </div>

      <div className="logs-container" ref={logsContainerRef}>
        {filteredLogs.length === 0 ? (
          <div className="logs-empty">
            <p>No logs to display</p>
            <p className="logs-empty-subtitle">Logs will appear here as the system processes detections</p>
          </div>
        ) : (
          <>
            {filteredLogs.map((log, index) => (
              <div
                key={index}
                className="log-entry"
                style={{ borderLeftColor: categoryColors[log.category] || '#6b7280' }}
              >
                <div className="log-timestamp">{formatTimestamp(log.timestamp)}</div>
                <div className="log-category">
                  <span className="log-icon">{categoryIcons[log.category] || '📝'}</span>
                  <span className="log-category-name">{log.category}</span>
                </div>
                <div className="log-message">{log.message}</div>
              </div>
            ))}
            <div ref={logsEndRef} />
          </>
        )}
      </div>

      {showScrollButton && (
        <button onClick={scrollToBottom} className="scroll-to-bottom-button">
          ↓ Scroll to Bottom
        </button>
      )}

      <div className="logs-footer">
        <div className="logs-stats">
          <span className="stat-item">Total: {logs.length.toLocaleString()}</span>
          <span className="stat-item">Filtered: {filteredLogs.length.toLocaleString()}</span>
          <span className="stat-item">
            {autoScroll ? '🟢 Auto-scroll ON' : '🔴 Auto-scroll OFF'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default LogsViewer




