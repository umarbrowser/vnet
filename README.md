
## 🏗️ Architecture

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

## 📋 Prerequisites

- Node.js >= 18.0.0
- Python >= 3.9
- MetaMask browser extension
- Ganache (for local blockchain)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Local Blockchain (Ganache)

```bash
# Install Ganache globally or use GUI
# Or use Hardhat's built-in network
npx hardhat node
```

### 3. Deploy Smart Contracts

```bash
# Deploy to local network
npx hardhat run scripts/deploy.js --network localhost

# Or deploy to testnet
npx hardhat run scripts/deploy.js --network goerli
```

### 4. Train ML Models

```bash
python ml/train_models.py
```

### 5. Run Detection System

```bash
# Start the integrated system
python main.py
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| ML Accuracy | 99.9% |
| Precision | 1.00 |
| Recall | 0.99 |
| F1-Score | 1.00 |
| Local Latency | 1-3s |
| Testnet Latency | 10-15s |
| Gas per Transaction | 45,000-70,000 |

## 🔧 Configuration

Edit `config/config.json` to configure:
- Blockchain network settings
- ML model parameters
- RSU configurations
- Detection thresholds

## 📁 Project Structure

```
vanet-blockchain-ml/
├── contracts/          # Solidity smart contracts
├── scripts/            # Deployment scripts
├── ml/                 # Machine learning models
├── src/                # Integration code
├── data/               # VeReMi dataset
├── config/             # Configuration files
└── tests/              # Test files
```

## 🧪 Testing

```bash
# Verify setup
python verify_setup.py

# Test smart contracts
npx hardhat test

# Test ML models
pytest test/test_ml.py

# Run complete system test
./run_test.sh
```

## 📊 System Architecture

```
┌─────────────┐
│  Vehicles   │  Generate Basic Safety Messages (BSMs)
└──────┬──────┘
       │
┌──────▼──────┐
│    RSUs    │  Receive and preprocess BSMs
└──────┬──────┘
       │
┌──────▼──────────────┐
│  ML Detection       │  Real-time anomaly detection
│  (RF/SVM/DNN)       │  Confidence scoring
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Blockchain         │  Immutable logging
│  Smart Contract     │  Trust score management
└─────────────────────┘
```

## 🎯 Key Results

- **99.9% Detection Accuracy** with Deep Neural Network
- **1-3s Latency** on local blockchain
- **10-15s Latency** on public testnets
- **45,000-70,000 Gas** per transaction
- **Automatic Blacklisting** of malicious vehicles
- **Immutable Audit Trail** of all events

## 📝 License

MIT License

## 🙏 Acknowledgments

- VeReMi dataset for VANET misbehavior detection
- Ethereum Foundation for blockchain infrastructure
