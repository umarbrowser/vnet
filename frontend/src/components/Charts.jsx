import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import './Charts.css'

function Charts() {
  const [chartData, setChartData] = useState(null)
  const [stats, setStats] = useState(null)
  const [blockchainData, setBlockchainData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState('50')

  useEffect(() => {
    fetchChartData()
    fetchStats()
    fetchBlockchainData()
    const interval = setInterval(() => {
      fetchChartData()
      fetchStats()
      fetchBlockchainData()
    }, 2000)
    return () => clearInterval(interval)
  }, [timeRange])

  const fetchChartData = async () => {
    try {
      const response = await axios.get(`/api/detections/chart?limit=${timeRange}`)
      setChartData(response.data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch chart data:', error)
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const fetchBlockchainData = async () => {
    try {
      const response = await axios.get('/api/blockchain/activity')
      setBlockchainData(response.data)
    } catch (error) {
      console.error('Failed to fetch blockchain data:', error)
    }
  }

  if (loading) {
    return <div className="charts-loading">Loading charts...</div>
  }

  if (!chartData || !chartData.timeSeries || chartData.timeSeries.length === 0) {
    return (
      <div className="charts-empty">
        <h2 className="charts-title">Analytics Dashboard</h2>
        <p>No data available yet. Run some detections to see analytics.</p>
      </div>
    )
  }

  // Prepare data for charts
  const timeSeriesData = chartData.timeSeries.map((item, index) => ({
    index: index + 1,
    time: new Date(item.time).toLocaleTimeString(),
    malicious: item.malicious,
    normal: item.normal,
    total: item.total,
    confidence: item.confidence || 0
  }))

  // Confidence distribution data
  const confidenceRanges = [
    { range: '0-20%', count: 0 },
    { range: '20-40%', count: 0 },
    { range: '40-60%', count: 0 },
    { range: '60-80%', count: 0 },
    { range: '80-100%', count: 0 }
  ]

  chartData.confidenceDistribution.forEach(conf => {
    const range = Math.floor(conf / 20)
    if (range < 5) {
      confidenceRanges[range].count++
    }
  })

  // Misbehavior type distribution
  const typeData = Object.entries(chartData.typeDistribution).map(([type, count]) => ({
    type: type.charAt(0).toUpperCase() + type.slice(1),
    count
  }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p className="tooltip-label">{`Index: ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}`}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="charts">
      <div className="charts-header">
        <h2 className="charts-title">Analytics Dashboard</h2>
        <p className="charts-subtitle">Real-time detection analytics and visualizations</p>
        <div className="charts-controls">
          <label htmlFor="time-range" className="control-label">Time Range:</label>
          <select
            id="time-range"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="control-select"
          >
            <option value="25">Last 25</option>
            <option value="50">Last 50</option>
            <option value="100">Last 100</option>
            <option value="200">Last 200</option>
          </select>
        </div>
      </div>

      {stats && (
        <div className="charts-summary">
          <div className="summary-card">
            <div className="summary-label">Total Detections</div>
            <div className="summary-value">{stats.total_detections.toLocaleString()}</div>
          </div>
          <div className="summary-card">
            <div className="summary-label">Malicious</div>
            <div className="summary-value summary-malicious">{stats.malicious_detections.toLocaleString()}</div>
          </div>
          <div className="summary-card">
            <div className="summary-label">Normal</div>
            <div className="summary-value summary-normal">{stats.normal_detections.toLocaleString()}</div>
          </div>
          <div className="summary-card">
            <div className="summary-label">Detection Rate</div>
            <div className="summary-value">{(stats.detection_rate * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}

      <div className="charts-grid">
        {/* Time Series Line Chart */}
        <div className="chart-card">
          <h3 className="chart-card-title">Detection Time Series</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="index" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="malicious" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Malicious"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="normal" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Normal"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Area Chart */}
        <div className="chart-card">
          <h3 className="chart-card-title">Cumulative Detections</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="index" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="total" 
                stroke="#3b82f6" 
                fill="#3b82f6"
                fillOpacity={0.3}
                name="Total Detections"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence Distribution Bar Chart */}
        <div className="chart-card">
          <h3 className="chart-card-title">Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confidenceRanges}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="range" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" fill="#8b5cf6" name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Misbehavior Type Distribution */}
        {typeData.length > 0 && (
          <div className="chart-card">
            <h3 className="chart-card-title">Misbehavior Types</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={typeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis 
                  dataKey="type" 
                  stroke="#94a3b8"
                  tick={{ fill: '#cbd5e1' }}
                />
                <YAxis 
                  stroke="#94a3b8"
                  tick={{ fill: '#cbd5e1' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" fill="#06b6d4" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Confidence Over Time */}
        <div className="chart-card">
          <h3 className="chart-card-title">Confidence Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="index" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
                domain={[0, 100]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="confidence" 
                stroke="#f59e0b" 
                strokeWidth={2}
                name="Confidence %"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Detection Rate Trend */}
        <div className="chart-card">
          <h3 className="chart-card-title">Detection Rate Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="malicious" 
                stackId="1" 
                stroke="#ef4444" 
                fill="#ef4444" 
                fillOpacity={0.6}
                name="Malicious Rate"
              />
              <Area 
                type="monotone" 
                dataKey="normal" 
                stackId="1" 
                stroke="#10b981" 
                fill="#10b981" 
                fillOpacity={0.6}
                name="Normal Rate"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Misbehavior Type Pie Chart */}
        {typeData.length > 0 && (
          <div className="chart-card">
            <h3 className="chart-card-title">Misbehavior Type Distribution (Pie)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ type, percent }) => `${type}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Confidence vs Detection Scatter */}
        <div className="chart-card">
          <h3 className="chart-card-title">Confidence vs Detection Pattern</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="confidence" 
                name="Confidence" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <YAxis 
                dataKey="total" 
                name="Total Detections" 
                stroke="#94a3b8"
                tick={{ fill: '#cbd5e1' }}
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Detections" data={timeSeriesData} fill="#3b82f6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Composed Chart - Multiple Metrics */}
        <div className="chart-card">
          <h3 className="chart-card-title">Comprehensive Metrics Overview</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis yAxisId="left" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Area 
                yAxisId="left"
                type="monotone" 
                dataKey="total" 
                fill="#3b82f6" 
                fillOpacity={0.3}
                stroke="#3b82f6"
                name="Total Detections"
              />
              <Bar yAxisId="right" dataKey="malicious" fill="#ef4444" name="Malicious" />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="confidence" 
                stroke="#f59e0b" 
                strokeWidth={2}
                name="Confidence %"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly/Daily Aggregation (if we have enough data) */}
        {timeSeriesData.length > 20 && (
          <div className="chart-card">
            <h3 className="chart-card-title">Detection Velocity (Rate of Change)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeSeriesData.map((item, idx) => ({
                ...item,
                velocity: idx > 0 ? item.total - timeSeriesData[idx - 1].total : 0
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="velocity" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  name="Detection Velocity"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Blockchain Charts */}
        {blockchainData && blockchainData.total_detections > 0 && (
          <>
            <div className="chart-card">
              <h3 className="chart-card-title">⛓️ Blockchain Transactions Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={blockchainData.timeSeries || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    fill="#06b6d4" 
                    fillOpacity={0.6}
                    stroke="#06b6d4"
                    name="Cumulative Transactions"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-card-title">⛓️ Blockchain Transaction Latency</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={blockchainData.latencyData || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} label={{ value: 'Latency (s)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="latency" 
                    stroke="#06b6d4" 
                    strokeWidth={2}
                    name="Latency (seconds)"
                    dot={{ fill: '#06b6d4', r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-card-title">⛓️ Gas Usage per Transaction</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={blockchainData.gasData || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="index" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} label={{ value: 'Gas Used', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="gas" fill="#06b6d4" name="Gas Used" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {blockchainData.typeDistribution && blockchainData.typeDistribution.length > 0 && (
              <div className="chart-card">
                <h3 className="chart-card-title">⛓️ Misbehavior Types on Blockchain</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={blockchainData.typeDistribution}
                      dataKey="count"
                      nameKey="type"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label
                    >
                      {blockchainData.typeDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4']

export default Charts

