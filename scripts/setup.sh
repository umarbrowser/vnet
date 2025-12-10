#!/bin/bash

echo "Setting up VANET Blockchain ML Detection System..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js >= 18.0.0"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python >= 3.9"
    exit 1
fi

echo "Installing Node.js dependencies..."
npm install

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Compiling smart contracts..."
npx hardhat compile

echo "Creating necessary directories..."
mkdir -p data models results deployments artifacts cache

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start local blockchain: npx hardhat node"
echo "2. Deploy contracts: npm run deploy:local"
echo "3. Train ML models: python ml/train_models.py"
echo "4. Run system: python main.py"











