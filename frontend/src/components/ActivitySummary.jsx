import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './ActivitySummary.css'

function ActivitySummary() {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchSummary()
    const interval = setInterval(fetchSummary, 3000)
    return () => clearInterval(interval)
  }, [])

  const fetchSummary = async () => {
    try {
      const response = await axios.get('/api/activity/summary')
      setSummary(response.data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch activity summary:', error)
      setLoading(false)
    }
  }

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

  if (loading) {
    return <div className="activity-summary-loading">Loading activity summary...</div>
  }

  if (!summary) {
    return null
  }

  return (
    <div className="activity-summary">
      <h3 className="activity-summary-title">Activity Summary</h3>
      <div className="activity-cards">
        {Object.entries(summary.categories).map(([category, count]) => (
          <div
            key={category}
            className="activity-card"
            style={{ borderLeftColor: categoryColors[category] || '#6b7280' }}
          >
            <div className="activity-icon">{categoryIcons[category] || '📝'}</div>
            <div className="activity-content">
              <div className="activity-label">{category.charAt(0).toUpperCase() + category.slice(1)}</div>
              <div className="activity-count">{count.toLocaleString()}</div>
            </div>
          </div>
        ))}
        <div className="activity-card activity-total">
          <div className="activity-icon">📊</div>
          <div className="activity-content">
            <div className="activity-label">Total</div>
            <div className="activity-count">{summary.total.toLocaleString()}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ActivitySummary




