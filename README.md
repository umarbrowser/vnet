# VANET Blockchain-Based Misbehavior Detection System

A comprehensive Vehicle Ad-Hoc Network (VANET) security system that combines Machine Learning anomaly detection with blockchain-based immutable logging. This system detects and records vehicle misbehavior in real-time using advanced ML models and Ethereum smart contracts.

## 🏗️ System Architecture

```
┌─────────────────┐
│   Vehicles      │  Send Basic Safety Messages (BSMs)
└────────┬────────┘
         │
┌────────▼────────┐
│   RSUs          │  Receive and preprocess messages
└────────┬────────┘
         │
┌────────▼────────┐
│  ML Detection   │  Anomaly detection (RF/SVM/DNN)
└────────┬────────┘
         │
┌────────▼────────┐
│  Smart Contract │  Log misbehavior to blockchain
└─────────────────┘
```

## 🛠️ Technology Stack

This project leverages a modern, production-ready technology stack across three main layers: **Backend (Python)**, **Blockchain (Ethereum/Solidity)**, and **Frontend (React)**.

---

## 📦 Backend Technologies (Python)

### Core Framework & Web Server

#### **Flask 3.0+** - Web Application Framework
- **Purpose**: Lightweight, flexible Python web framework for building RESTful APIs
- **Usage**: Powers the backend API server (`backend/app.py`) that handles HTTP requests, serves real-time data via Server-Sent Events (SSE), and manages the detection system
- **Features Used**:
  - RESTful API endpoints for detection, training, and monitoring
  - Server-Sent Events (SSE) for real-time data streaming to frontend
  - Blueprint architecture for modular API organization
  - CORS support for cross-origin requests

#### **Flask-CORS 4.0+** - Cross-Origin Resource Sharing
- **Purpose**: Enables secure cross-origin HTTP requests between frontend and backend
- **Usage**: Allows React frontend (running on different port) to communicate with Flask backend

### Machine Learning & Data Science

#### **NumPy 1.24+** - Numerical Computing
- **Purpose**: Fundamental library for numerical operations and array processing
- **Usage**: Core data structure for ML model inputs/outputs, feature extraction, and mathematical operations
- **Why**: Provides efficient multi-dimensional arrays and mathematical functions essential for ML computations

#### **Pandas 2.0+** - Data Analysis & Manipulation
- **Purpose**: Powerful data manipulation and analysis library
- **Usage**: 
  - Loading and preprocessing VeReMi dataset
  - Feature engineering from VANET message data
  - Data cleaning and transformation
  - Time-series analysis of detection patterns

#### **scikit-learn 1.3+** - Machine Learning Library
- **Purpose**: Comprehensive ML toolkit with pre-built algorithms
- **Algorithms Used**:
  - **Random Forest Classifier**: Ensemble method using multiple decision trees for robust misbehavior detection
  - **Support Vector Machine (SVM)**: Kernel-based classifier for high-dimensional feature spaces
  - **StandardScaler**: Feature normalization for consistent model training
  - **Cross-validation**: Model evaluation and hyperparameter tuning
- **Usage**: Primary ML framework for traditional machine learning models (`src/ml/models.py`)

#### **PyTorch 2.0+** - Deep Learning Framework
- **Purpose**: Modern deep learning framework with dynamic computation graphs
- **Usage**: 
  - Implements Deep Neural Network (DNN) classifier for misbehavior detection
  - Multi-layer perceptron architecture with dropout regularization
  - GPU acceleration support (when available)
  - Model serialization and loading
- **Why**: Provides state-of-the-art deep learning capabilities, achieving 99.9% accuracy in misbehavior detection

#### **TensorFlow (Optional)** - Alternative Deep Learning Framework
- **Purpose**: Alternative DNN implementation option
- **Usage**: Can be used instead of PyTorch for DNN models (install via `install_tensorflow_mac.sh`)
- **Note**: PyTorch is recommended for Mac systems due to easier installation

### Data Visualization & Analysis

#### **Matplotlib 3.7+** - Plotting Library
- **Purpose**: Comprehensive 2D plotting library for Python
- **Usage**: 
  - Training visualization (`ml/visualize_results.py`)
  - Model performance charts
  - Detection pattern analysis
  - Feature importance visualization

#### **Seaborn 0.12+** - Statistical Visualization
- **Purpose**: High-level interface for statistical graphics built on Matplotlib
- **Usage**: Enhanced visualizations for ML model comparisons, correlation matrices, and distribution plots

#### **SciPy 1.10+** - Scientific Computing
- **Purpose**: Library for scientific and technical computing
- **Usage**: Statistical functions, optimization algorithms, and signal processing utilities

### Blockchain Integration

#### **Web3.py 6.11+** - Ethereum Python Library
- **Purpose**: Python library for interacting with Ethereum blockchain
- **Usage**: 
  - Connects to Ethereum networks (local, Goerli, Sepolia)
  - Deploys and interacts with smart contracts (`src/blockchain/web3_client.py`)
  - Sends transactions to log misbehavior events
  - Reads blockchain state (trust scores, misbehavior counts)
  - Event filtering and monitoring
- **Features**:
  - Support for Proof-of-Authority (PoA) networks via middleware
  - Transaction signing and gas estimation
  - Contract ABI encoding/decoding

#### **eth-account 0.9+** - Ethereum Account Management
- **Purpose**: Ethereum account creation and transaction signing
- **Usage**: 
  - Private key management
  - Transaction signing for blockchain interactions
  - Address generation and validation

### Utilities & Development Tools

#### **Loguru 0.7+** - Advanced Logging
- **Purpose**: Modern, feature-rich logging library
- **Usage**: 
  - Structured logging throughout the application
  - Color-coded console output
  - File rotation and retention
  - Exception tracking and stack traces
- **Why**: Provides better developer experience than standard `logging` module

#### **python-dotenv 1.0+** - Environment Variable Management
- **Purpose**: Loads environment variables from `.env` files
- **Usage**: 
  - Secure storage of blockchain private keys
  - Network configuration (RPC URLs, contract addresses)
  - API keys and sensitive configuration

#### **PyYAML 6.0+** - YAML Parser
- **Purpose**: YAML file parsing and generation
- **Usage**: Configuration file management (`config/config.json`)

#### **tqdm 4.65+** - Progress Bars
- **Purpose**: Fast, extensible progress bars for loops
- **Usage**: Visual feedback during ML model training and data processing

#### **gdown 4.6+** - Google Drive Downloader
- **Purpose**: Download large files from Google Drive
- **Usage**: Automated VeReMi dataset download (`utils/download_dataset.py`)

#### **openpyxl 3.1+** - Excel File Support
- **Purpose**: Read/write Excel files (.xlsx)
- **Usage**: Processing VeReMi dataset files in Excel format

### Testing & Quality Assurance

#### **pytest 7.4+** - Testing Framework
- **Purpose**: Modern Python testing framework
- **Usage**: 
  - Unit tests for ML models (`test/test_ml.py`)
  - Integration tests (`scripts/integration_test.py`)
  - Test fixtures and parametrization

#### **pytest-cov 4.1+** - Code Coverage
- **Purpose**: Coverage plugin for pytest
- **Usage**: Measure test coverage and identify untested code paths

#### **Black 23.7+** - Code Formatter
- **Purpose**: Uncompromising Python code formatter
- **Usage**: Enforces consistent code style across the project

#### **Flake8 6.1+** - Linting Tool
- **Purpose**: Style guide enforcement (PEP 8)
- **Usage**: Static code analysis for code quality and style compliance

---

## ⛓️ Blockchain Technologies (Ethereum)

### Smart Contract Development

#### **Solidity 0.8.20** - Smart Contract Language
- **Purpose**: High-level, statically-typed programming language for Ethereum smart contracts
- **Usage**: 
  - `contracts/MisbehaviorDetection.sol` - Main smart contract
  - Defines data structures (MisbehaviorRecord, trust scores)
  - Implements business logic (logging, blacklisting, trust score updates)
  - Emits events for real-time monitoring
- **Features Used**:
  - Structs for complex data types
  - Mappings for efficient key-value storage
  - Enums for misbehavior types (Sybil, Falsification, Replay, DoS)
  - Events for off-chain monitoring
  - Access control and security patterns

#### **OpenZeppelin Contracts 5.0+** - Secure Smart Contract Library
- **Purpose**: Battle-tested, community-audited smart contract components
- **Usage**: 
  - Security patterns and best practices
  - Access control mechanisms
  - Reentrancy protection
- **Why**: Industry standard for secure smart contract development

### Development & Deployment Tools

#### **Hardhat 2.19+** - Ethereum Development Environment
- **Purpose**: Complete development environment for Ethereum smart contracts
- **Features**:
  - **Compilation**: Compiles Solidity contracts
  - **Testing**: Built-in testing framework with Chai assertions
  - **Deployment**: Automated contract deployment scripts
  - **Local Network**: Built-in Ethereum node for development
  - **Network Management**: Support for multiple networks (localhost, Goerli, Sepolia)
  - **Verification**: Contract verification on Etherscan
- **Configuration**: `hardhat.config.js` defines networks, compiler settings, and optimization

#### **@nomicfoundation/hardhat-toolbox 4.0+** - Hardhat Plugin Suite
- **Purpose**: Collection of essential Hardhat plugins
- **Includes**:
  - Hardhat-Ethers: Ethereum library integration
  - Hardhat-Chai-Matchers: Testing utilities
  - Hardhat-Etherscan: Contract verification
  - Hardhat-Gas-Reporter: Gas usage analysis

#### **@nomicfoundation/hardhat-verify 2.0+** - Contract Verification
- **Purpose**: Verify smart contracts on Etherscan
- **Usage**: Publishes source code for transparency and verification

### Blockchain Networks

#### **Ethereum Networks**
- **Localhost (Hardhat Network)**: 
  - Chain ID: 1337
  - Purpose: Local development and testing
  - Features: Instant block mining, zero gas costs
- **Goerli Testnet**:
  - Chain ID: 5
  - Purpose: Public testnet for integration testing
  - Features: Free test ETH, real network conditions
- **Sepolia Testnet**:
  - Chain ID: 11155111
  - Purpose: Modern Ethereum testnet
  - Features: Stable test environment

### Node.js Blockchain Libraries

#### **Web3.js 4.2+** - Ethereum JavaScript Library
- **Purpose**: JavaScript library for interacting with Ethereum blockchain
- **Usage**: 
  - Deployment scripts (`scripts/deploy.js`)
  - Contract interaction from Node.js
  - Transaction management
- **Note**: Python uses `web3.py`, Node.js uses `web3.js` - both serve the same purpose in their respective environments

#### **Express 4.21+** - Node.js Web Framework
- **Purpose**: Minimal web framework for Node.js
- **Usage**: 
  - Simple HTTP server (`server.js`)
  - API endpoints for blockchain status
  - Middleware for request handling

#### **CORS 2.8+** - Cross-Origin Resource Sharing
- **Purpose**: Enable CORS in Express server
- **Usage**: Allow cross-origin requests from frontend

#### **dotenv 16.3+** - Environment Variables (Node.js)
- **Purpose**: Load environment variables in Node.js
- **Usage**: Secure configuration management for blockchain credentials

---

## 🎨 Frontend Technologies (React)

### Core Framework

#### **React 18.2+** - UI Library
- **Purpose**: Declarative, component-based JavaScript library for building user interfaces
- **Usage**: 
  - Component architecture (`frontend/src/components/`)
  - State management with hooks (useState, useEffect)
  - Real-time data updates via API polling
  - Interactive dashboards and forms
- **Features Used**:
  - Functional components with hooks
  - Component composition
  - Event handling
  - Conditional rendering

#### **React DOM 18.2+** - React Renderer
- **Purpose**: React package for DOM rendering
- **Usage**: Renders React components to the browser DOM

### Build Tools & Development

#### **Vite 5.0+** - Next-Generation Frontend Build Tool
- **Purpose**: Lightning-fast build tool and development server
- **Features**:
  - **Hot Module Replacement (HMR)**: Instant updates during development
  - **Fast Builds**: Optimized production builds
  - **ES Modules**: Native ES module support
  - **Proxy Configuration**: API proxying for development
- **Configuration**: `frontend/vite.config.js` defines build settings and dev server

#### **@vitejs/plugin-react 4.2+** - React Plugin for Vite
- **Purpose**: Official React plugin for Vite
- **Usage**: Enables JSX transformation and React Fast Refresh

### Data Visualization

#### **Recharts 2.10+** - Composable Charting Library
- **Purpose**: Redefined chart library built on React and D3
- **Charts Used**:
  - **LineChart**: Time-series detection trends
  - **AreaChart**: Cumulative detection patterns
  - **BarChart**: Detection counts and gas usage
  - **PieChart**: Misbehavior type distribution
  - **ScatterChart**: Confidence vs detection patterns
  - **ComposedChart**: Multi-metric visualizations
  - **RadarChart**: Multi-metric performance comparison
- **Usage**: 
  - Real-time dashboard charts (`components/Charts.jsx`)
  - ML training visualization (`components/MLTraining.jsx`)
  - Responsive, interactive visualizations
- **Why**: React-native, highly customizable, and performant

### HTTP Client

#### **Axios 1.6+** - Promise-Based HTTP Client
- **Purpose**: HTTP client for making API requests
- **Usage**: 
  - Fetching detection data from Flask backend
  - Training API calls
  - Blockchain status queries
  - Error handling and request interceptors
- **Why**: Better than fetch API with automatic JSON parsing, request/response interceptors, and better error handling

### TypeScript Support (Development)

#### **@types/react 18.43+** - TypeScript Definitions
- **Purpose**: TypeScript type definitions for React
- **Usage**: Type safety during development (if using TypeScript)

#### **@types/react-dom 18.17+** - TypeScript Definitions
- **Purpose**: TypeScript type definitions for React DOM
- **Usage**: Type safety for DOM-related React APIs

---

## 📋 Prerequisites

### Required Software

- **Node.js >= 18.0.0** - JavaScript runtime for blockchain tools and frontend
- **Python >= 3.9** - Backend runtime and ML framework
- **npm** or **yarn** - Package manager for Node.js dependencies
- **pip** - Python package installer

### Optional Tools

- **MetaMask** - Browser extension for Ethereum wallet (for testnet interactions)
- **Ganache** - Local blockchain for development (alternative to Hardhat node)
- **Git** - Version control system

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
# Install Node.js dependencies (blockchain tools, frontend build tools)
npm install

# Install frontend dependencies
cd frontend && npm install && cd ..

# Install Python dependencies (ML libraries, Flask, Web3)
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Blockchain Configuration
PRIVATE_KEY=your_private_key_here
GOERLI_URL=https://goerli.infura.io/v3/YOUR_INFURA_KEY
SEPOLIA_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY

# Backend Configuration
FLASK_PORT=5001
VITE_PORT=3000
```

### 3. Start Local Blockchain

```bash
# Option 1: Use Hardhat's built-in network (Recommended)
npx hardhat node

# Option 2: Use Ganache GUI
# Download from https://trufflesuite.com/ganache/
```

### 4. Deploy Smart Contracts

```bash
# Deploy to local network
npx hardhat run scripts/deploy.js --network localhost

# Deploy to Goerli testnet
npx hardhat run scripts/deploy.js --network goerli

# Deploy to Sepolia testnet
npx hardhat run scripts/deploy.js --network sepolia
```

The deployment script will output the contract address. Update `config/config.json` with this address.

### 5. Train ML Models

```bash
# Train all models (Random Forest, SVM, DNN)
python ml/train_models.py

# This will:
# - Load VeReMi dataset
# - Preprocess features
# - Train three models
# - Evaluate performance
# - Save models to disk
```

### 6. Start the System

```bash
# Start Flask backend (Terminal 1)
cd backend
python app.py

# Start frontend dev server (Terminal 2)
cd frontend
npm run dev

# Or use the integrated start script
./start_web.sh
```

Access the application at `http://localhost:3000`

---

## 📊 Performance Metrics

| Metric | Value | Technology |
|--------|-------|------------|
| **ML Accuracy** | 99.9% | PyTorch DNN |
| **Precision** | 1.00 | scikit-learn |
| **Recall** | 0.99 | scikit-learn |
| **F1-Score** | 1.00 | scikit-learn |
| **Local Latency** | 1-3s | Hardhat Network |
| **Testnet Latency** | 10-15s | Goerli/Sepolia |
| **Gas per Transaction** | 45,000-70,000 | Solidity (optimized) |

---

## 🔧 Configuration

Edit `config/config.json` to configure:

- **Blockchain Settings**: Network URLs, contract addresses, gas limits
- **ML Parameters**: Model selection, confidence thresholds, feature weights
- **RSU Configuration**: Roadside unit settings, detection zones
- **Detection Thresholds**: Misbehavior classification criteria

---

## 📁 Project Structure

```
vanet-blockchain-ml/
├── backend/              # Flask API server
│   ├── app.py           # Main Flask application
│   ├── detect.py        # Detection endpoints
│   ├── training_api.py  # ML training API
│   └── blockchain_status.py  # Blockchain monitoring
├── contracts/           # Solidity smart contracts
│   └── MisbehaviorDetection.sol  # Main contract
├── frontend/            # React application
│   ├── src/
│   │   ├── components/  # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Charts.jsx
│   │   │   ├── DetectionForm.jsx
│   │   │   └── MLTraining.jsx
│   │   └── App.jsx      # Main app component
│   └── vite.config.js   # Vite configuration
├── ml/                  # Machine learning models
│   ├── train_models.py  # Model training script
│   ├── use_dataset.py   # Dataset utilities
│   └── visualize_results.py  # Visualization
├── scripts/             # Deployment & utility scripts
│   ├── deploy.js        # Hardhat deployment
│   └── integration_test.py  # System tests
├── src/                 # Core Python modules
│   ├── ml/
│   │   └── models.py    # ML model implementations
│   └── blockchain/
│       └── web3_client.py  # Blockchain client
├── test/                # Test files
│   ├── test_ml.py       # ML unit tests
│   └── MisbehaviorDetection.test.js  # Contract tests
├── config/              # Configuration files
│   └── config.json      # System configuration
├── data/                # VeReMi dataset
├── hardhat.config.js    # Hardhat configuration
├── package.json         # Node.js dependencies
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🧪 Testing

### Verify Setup

```bash
# Check all dependencies and configurations
python verify_setup.py
```

### Test Smart Contracts

```bash
# Run Hardhat tests
npx hardhat test

# Test specific contract
npx hardhat test test/MisbehaviorDetection.test.js
```

### Test ML Models

```bash
# Run pytest tests
pytest test/test_ml.py

# With coverage
pytest test/test_ml.py --cov=src/ml
```

### Integration Testing

```bash
# Run complete system test
./run_test.sh

# Or manually
python scripts/integration_test.py
```

---

## 📊 System Architecture Details

### Data Flow

1. **Vehicles** generate Basic Safety Messages (BSMs) with location, speed, and trajectory data
2. **RSUs (Roadside Units)** receive and preprocess BSMs, extracting features
3. **ML Detection System** analyzes features using trained models:
   - **Random Forest**: Ensemble of decision trees
   - **SVM**: Support Vector Machine with kernel functions
   - **DNN**: Deep Neural Network with multiple hidden layers
4. **Smart Contract** logs detected misbehavior to blockchain:
   - Immutable record storage
   - Trust score updates
   - Automatic blacklisting
   - Event emission for monitoring

### Technology Integration Points

- **Python ↔ Blockchain**: Web3.py connects Flask backend to Ethereum
- **Frontend ↔ Backend**: Axios HTTP client communicates with Flask API
- **Backend ↔ ML**: NumPy/Pandas arrays passed between Flask and ML models
- **Frontend ↔ Visualization**: Recharts renders data from Flask API responses

---

## 🎯 Key Features & Results

### Machine Learning Achievements

- **99.9% Detection Accuracy** using Deep Neural Network (PyTorch)
- **Real-time Processing** with optimized feature extraction
- **Multiple Model Support** (RF, SVM, DNN) for comparison
- **Confidence Scoring** for each detection

### Blockchain Benefits

- **Immutable Audit Trail** of all misbehavior events
- **Decentralized Trust** - no single point of failure
- **Transparent Logging** - all events publicly verifiable
- **Automatic Blacklisting** based on trust scores
- **Gas Optimization** - efficient contract design (45k-70k gas)

### System Performance

- **1-3s Latency** on local blockchain (Hardhat)
- **10-15s Latency** on public testnets (Goerli/Sepolia)
- **Real-time Dashboard** with live updates via SSE
- **Scalable Architecture** - handles high-volume BSM processing

---

## 🔒 Security Considerations

### Smart Contract Security

- **OpenZeppelin Contracts**: Industry-standard security patterns
- **Access Control**: Role-based permissions
- **Input Validation**: All parameters validated before execution
- **Gas Optimization**: Efficient storage patterns to minimize costs

### Backend Security

- **CORS Configuration**: Restricted cross-origin access
- **Input Validation**: All API inputs validated
- **Environment Variables**: Sensitive data stored securely
- **Error Handling**: Graceful error responses without exposing internals

### ML Model Security

- **Model Validation**: Cross-validation prevents overfitting
- **Feature Validation**: Input features checked before prediction
- **Confidence Thresholds**: Configurable detection sensitivity

---

## 🚀 Deployment Options

### Local Development

- Hardhat local network
- Flask development server
- Vite dev server with HMR

### Testnet Deployment

- Deploy contracts to Goerli or Sepolia
- Use testnet ETH for transactions
- Monitor via Etherscan

### Production Considerations

- Use production-grade Ethereum node (Infura, Alchemy)
- Deploy to mainnet (requires real ETH)
- Use production WSGI server (Gunicorn) for Flask
- Build optimized React bundle (`npm run build`)
- Set up monitoring and logging

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **VeReMi Dataset**: Vehicle misbehavior detection research dataset
- **Ethereum Foundation**: Blockchain infrastructure and tooling
- **OpenZeppelin**: Secure smart contract libraries
- **scikit-learn Community**: Machine learning algorithms
- **PyTorch Team**: Deep learning framework
- **React Team**: UI library and ecosystem

---

## 📞 Support & Documentation

For detailed documentation on specific technologies:

- **Flask**: https://flask.palletsprojects.com/
- **PyTorch**: https://pytorch.org/docs/
- **scikit-learn**: https://scikit-learn.org/stable/
- **Hardhat**: https://hardhat.org/docs
- **Solidity**: https://docs.soliditylang.org/
- **React**: https://react.dev/
- **Vite**: https://vitejs.dev/
- **Recharts**: https://recharts.org/
- **Web3.py**: https://web3py.readthedocs.io/

---

## 🔄 Version Information

- **Python**: 3.9+
- **Node.js**: 18.0+
- **Solidity**: 0.8.20
- **React**: 18.2+
- **Flask**: 3.0+
- **PyTorch**: 2.0+
- **Hardhat**: 2.19+

---

*Last Updated: 2024*
