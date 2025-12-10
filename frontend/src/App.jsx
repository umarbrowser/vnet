import React, { useState } from 'react'
import Dashboard from './components/Dashboard'
import DetectionForm from './components/DetectionForm'
import MLTraining from './components/MLTraining'
import TrainingViewer from './components/TrainingViewer'
import LogsViewer from './components/LogsViewer'
import Charts from './components/Charts'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'detect', label: 'Detection', icon: '🔍' },
    { id: 'train', label: 'Train Models', icon: '⚙️' },
    { id: 'ml-training', label: 'ML Analysis', icon: '🧠' },
    { id: 'logs', label: 'Logs & Terminal', icon: '📝' },
    { id: 'charts', label: 'Analytics', icon: '📈' }
  ]

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">VANET Misbehavior Detection System</h1>
          <nav className="app-nav">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`nav-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className="nav-icon">{tab.icon}</span>
                <span className="nav-label">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="app-main">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'detect' && <DetectionForm />}
        {activeTab === 'train' && <TrainingViewer />}
        {activeTab === 'ml-training' && <MLTraining />}
        {activeTab === 'logs' && <LogsViewer />}
        {activeTab === 'charts' && <Charts />}
      </main>
    </div>
  )
}

export default App

