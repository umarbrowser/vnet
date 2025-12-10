#!/bin/bash

echo "=========================================="
echo "VANET Blockchain ML Detection System Test"
echo "=========================================="
echo ""

# Check if Hardhat node is running
if ! curl -s http://127.0.0.1:8545 > /dev/null 2>&1; then
    echo "⚠️  Hardhat node not running. Starting it in background..."
    npx hardhat node > /dev/null 2>&1 &
    sleep 5
    echo "✅ Hardhat node started"
fi

# Compile contracts
echo "📦 Compiling smart contracts..."
npx hardhat compile > /dev/null 2>&1

# Deploy contract
echo "🚀 Deploying smart contract..."
CONTRACT_OUTPUT=$(npx hardhat run scripts/deploy.js --network localhost 2>&1)
CONTRACT_ADDRESS=$(echo "$CONTRACT_OUTPUT" | grep "deployed to:" | awk '{print $NF}')

if [ -z "$CONTRACT_ADDRESS" ]; then
    echo "❌ Failed to deploy contract"
    exit 1
fi

echo "✅ Contract deployed: $CONTRACT_ADDRESS"

# Update config
echo "⚙️  Updating configuration..."
python3 << EOF
import json
import os

config_path = "config/config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['contract']['address'] = "$CONTRACT_ADDRESS"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Configuration updated")
else:
    print("⚠️  Config file not found")
EOF

# Train models if needed
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "🤖 Training ML models..."
    python3 ml/train_models.py
else
    echo "✅ ML models already trained"
fi

# Run system
echo ""
echo "🎯 Running detection system..."
echo ""
python3 main.py --model dnn --vehicles 5

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="











