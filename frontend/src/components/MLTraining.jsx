import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import './MLTraining.css'

function MLTraining() {
  const [trainingData, setTrainingData] = useState(null)
  const [comparisonData, setComparisonData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchTrainingData()
    fetchComparisonData()
  }, [])

  const fetchTrainingData = async () => {
    try {
      const response = await axios.get('/api/ml/training')
      setTrainingData(response.data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch training data:', error)
      setLoading(false)
    }
  }

  const fetchComparisonData = async () => {
    try {
      const response = await axios.get('/api/ml/model-comparison')
      setComparisonData(response.data)
    } catch (error) {
      console.error('Failed to fetch comparison data:', error)
    }
  }

  if (loading) {
    return <div className="ml-training-loading">Loading ML training data...</div>
  }

  if (!trainingData || trainingData.error) {
    return (
      <div className="ml-training-empty">
        <h2 className="ml-training-title">ML Training Dashboard</h2>
        <p>Training results not found. Please train models first.</p>
        <p className="ml-training-hint">Run: python ml/train_models.py</p>
      </div>
    )
  }

  const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444']

  // Prepare data for charts
  const metricsData = trainingData.results.map(r => ({
    model: r.model,
    Accuracy: (r.accuracy * 100).toFixed(2),
    Precision: (r.precision * 100).toFixed(2),
    Recall: (r.recall * 100).toFixed(2),
    'F1-Score': (r.f1_score * 100).toFixed(2)
  }))

  const comparisonChartData = comparisonData?.comparison || []

  // Feature importance data
  const featureImportanceData = trainingData.feature_importance
    ? Object.entries(trainingData.feature_importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, value]) => ({
          feature: name.replace(/_/g, ' '),
          importance: (value * 100).toFixed(2)
        }))
    : []

  // Confusion matrix data for each model
  const confusionMatrixData = trainingData.results.map(r => {
    const cm = r.confusion_matrix
    return {
      model: r.model,
      'True Negative': cm[0][0],
      'False Positive': cm[0][1],
      'False Negative': cm[1][0],
      'True Positive': cm[1][1]
    }
  })

  // Radar chart data
  const radarData = trainingData.results.map(r => ({
    model: r.model,
    Accuracy: r.accuracy * 100,
    Precision: r.precision * 100,
    Recall: r.recall * 100,
    'F1-Score': r.f1_score * 100
  }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p className="tooltip-label">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}%`}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="ml-training">
      <div className="ml-training-header">
        <h2 className="ml-training-title">ML Training & Model Performance</h2>
        <p className="ml-training-subtitle">Comprehensive analysis of machine learning models for VANET misbehavior detection</p>
      </div>

      {/* Model Performance Summary */}
      {comparisonData && (
        <div className="ml-summary-cards">
          <div className="ml-summary-card">
            <div className="ml-summary-label">Best Model</div>
            <div className="ml-summary-value">{comparisonData.summary.best_model}</div>
          </div>
          <div className="ml-summary-card">
            <div className="ml-summary-label">Best Accuracy</div>
            <div className="ml-summary-value">{(comparisonData.summary.best_accuracy * 100).toFixed(2)}%</div>
          </div>
          <div className="ml-summary-card">
            <div className="ml-summary-label">Best F1-Score</div>
            <div className="ml-summary-value">{(comparisonData.summary.best_f1 * 100).toFixed(2)}%</div>
          </div>
        </div>
      )}

      <div className="ml-training-grid">
        {/* Model Comparison Bar Chart */}
        <div className="ml-chart-card">
          <h3 className="ml-chart-title">Model Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="Accuracy" fill="#3b82f6" />
              <Bar dataKey="Precision" fill="#10b981" />
              <Bar dataKey="Recall" fill="#8b5cf6" />
              <Bar dataKey="F1-Score" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="ml-chart-card">
          <h3 className="ml-chart-title">Multi-Metric Performance Radar</h3>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#475569" />
              <PolarAngleAxis dataKey="model" tick={{ fill: '#cbd5e1' }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#cbd5e1' }} />
              <Radar name="Random Forest" dataKey="Accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
              <Radar name="SVM" dataKey="Precision" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
              <Radar name="DNN" dataKey="Recall" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
              <Tooltip />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance */}
        {featureImportanceData.length > 0 && (
          <div className="ml-chart-card">
            <h3 className="ml-chart-title">Top 10 Feature Importance (Random Forest)</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={featureImportanceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                <YAxis dataKey="feature" type="category" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} width={150} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="importance" fill="#06b6d4" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Confusion Matrix Visualization */}
        <div className="ml-chart-card">
          <h3 className="ml-chart-title">Confusion Matrix Breakdown</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={confusionMatrixData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="True Positive" stackId="a" fill="#10b981" />
              <Bar dataKey="True Negative" stackId="a" fill="#3b82f6" />
              <Bar dataKey="False Positive" stackId="b" fill="#f59e0b" />
              <Bar dataKey="False Negative" stackId="b" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Metrics Line Chart */}
        <div className="ml-chart-card">
          <h3 className="ml-chart-title">Metrics Comparison (Line)</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line type="monotone" dataKey="Accuracy" stroke="#3b82f6" strokeWidth={3} dot={{ r: 6 }} />
              <Line type="monotone" dataKey="Precision" stroke="#10b981" strokeWidth={3} dot={{ r: 6 }} />
              <Line type="monotone" dataKey="Recall" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 6 }} />
              <Line type="monotone" dataKey="F1-Score" stroke="#f59e0b" strokeWidth={3} dot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Sensitivity vs Specificity */}
        {comparisonData && (
          <div className="ml-chart-card">
            <h3 className="ml-chart-title">Sensitivity vs Specificity</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={comparisonChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="sensitivity" fill="#10b981" name="Sensitivity (Recall)" />
                <Bar dataKey="specificity" fill="#3b82f6" name="Specificity" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Detailed Metrics Table */}
        <div className="ml-chart-card ml-table-card">
          <h3 className="ml-chart-title">Detailed Performance Metrics</h3>
          <div className="ml-metrics-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  {comparisonData && <th>Sensitivity</th>}
                  {comparisonData && <th>Specificity</th>}
                </tr>
              </thead>
              <tbody>
                {trainingData.results.map((result, idx) => (
                  <tr key={idx}>
                    <td><strong>{result.model}</strong></td>
                    <td>{(result.accuracy * 100).toFixed(2)}%</td>
                    <td>{(result.precision * 100).toFixed(2)}%</td>
                    <td>{(result.recall * 100).toFixed(2)}%</td>
                    <td>{(result.f1_score * 100).toFixed(2)}%</td>
                    {comparisonData && (
                      <>
                        <td>{(comparisonChartData[idx]?.sensitivity * 100).toFixed(2)}%</td>
                        <td>{(comparisonChartData[idx]?.specificity * 100).toFixed(2)}%</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MLTraining

