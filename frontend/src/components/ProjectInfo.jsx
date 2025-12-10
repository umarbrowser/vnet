import React, { useState } from 'react'
import './ProjectInfo.css'

function ProjectInfo() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="project-info">
      <button 
        className="info-toggle-button"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? '▼' : '▶'} Project Information
      </button>
      
      {isOpen && (
        <div className="info-content">
          <section className="info-section">
            <h3>Project Overview</h3>
            <p>
              This system implements a blockchain-based misbehavior detection framework for 
              Vehicular Ad Hoc Networks (VANETs). It combines machine learning models with 
              Ethereum smart contracts to provide real-time detection and immutable logging 
              of malicious vehicle behavior.
            </p>
          </section>

          <section className="info-section">
            <h3>System Architecture</h3>
            <div className="architecture-diagram">
              <div className="arch-layer">
                <strong>Frontend Layer</strong>
                <p>React-based web interface with real-time visualizations</p>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-layer">
                <strong>Backend API</strong>
                <p>Flask REST API with SSE streaming for real-time updates</p>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-layer">
                <strong>ML Detection Engine</strong>
                <p>Random Forest, SVM, and DNN models for anomaly detection</p>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-layer">
                <strong>Blockchain Layer</strong>
                <p>Ethereum smart contracts for immutable event logging</p>
              </div>
            </div>
          </section>

          <section className="info-section">
            <h3>Machine Learning Models</h3>
            <div className="model-grid">
              <div className="model-card">
                <h4>Random Forest</h4>
                <p>Ensemble method using multiple decision trees. Excellent for feature importance analysis.</p>
                <ul>
                  <li>Accuracy: ~98-100%</li>
                  <li>Fast training and inference</li>
                  <li>Handles non-linear relationships</li>
                </ul>
              </div>
              <div className="model-card">
                <h4>Support Vector Machine</h4>
                <p>Kernel-based classifier effective for high-dimensional data.</p>
                <ul>
                  <li>Accuracy: ~96-100%</li>
                  <li>Good generalization</li>
                  <li>Memory efficient</li>
                </ul>
              </div>
              <div className="model-card">
                <h4>Deep Neural Network</h4>
                <p>Multi-layer perceptron using scikit-learn MLPClassifier.</p>
                <ul>
                  <li>Accuracy: ~99-100%</li>
                  <li>Learns complex patterns</li>
                  <li>No PyTorch/TensorFlow required</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="info-section">
            <h3>Detection Features</h3>
            <p>The system analyzes 15 features from Basic Safety Messages (BSM):</p>
            <div className="features-grid">
              <div className="feature-group">
                <h4>Message Characteristics</h4>
                <ul>
                  <li>Message frequency</li>
                  <li>Message timing</li>
                  <li>Payload size</li>
                </ul>
              </div>
              <div className="feature-group">
                <h4>Position Consistency</h4>
                <ul>
                  <li>X, Y, Z coordinate consistency</li>
                  <li>Temporal position consistency</li>
                  <li>Route deviation</li>
                </ul>
              </div>
              <div className="feature-group">
                <h4>Motion Patterns</h4>
                <ul>
                  <li>Speed consistency</li>
                  <li>Acceleration consistency</li>
                  <li>Heading consistency</li>
                </ul>
              </div>
              <div className="feature-group">
                <h4>Network Metrics</h4>
                <ul>
                  <li>Signal strength</li>
                  <li>RSSI variance</li>
                  <li>Neighbor count</li>
                  <li>Timestamp anomalies</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="info-section">
            <h3>Blockchain Integration</h3>
            <p>
              High-confidence detections (confidence &gt; 50%) are logged to an Ethereum 
              smart contract, providing:
            </p>
            <ul>
              <li><strong>Immutability:</strong> Detection events cannot be altered</li>
              <li><strong>Transparency:</strong> All events are publicly verifiable</li>
              <li><strong>Trust Scores:</strong> Vehicles accumulate trust scores based on behavior</li>
              <li><strong>Automatic Blacklisting:</strong> Vehicles below threshold are automatically flagged</li>
            </ul>
          </section>

          <section className="info-section">
            <h3>Use Cases</h3>
            <ul>
              <li>Real-time monitoring of vehicle networks</li>
              <li>Detection of Sybil attacks, message falsification, replay attacks, and DoS</li>
              <li>Research and analysis of VANET security</li>
              <li>Development and testing of detection algorithms</li>
            </ul>
          </section>
        </div>
      )}
    </div>
  )
}

export default ProjectInfo




