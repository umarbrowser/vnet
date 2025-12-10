#!/bin/bash

echo "⛓️  Starting Blockchain for VANET Detection System"
echo "=================================================="
echo ""

# Check if Hardhat node is already running
if lsof -ti:8545 > /dev/null 2>&1; then
    echo "✅ Hardhat node is already running on port 8545"
    echo ""
    exit 0
fi

# Compile contracts
echo "📦 Compiling smart contracts..."
npx hardhat compile --quiet 2>/dev/null || npx hardhat compile

# Start Hardhat node
echo "🚀 Starting Hardhat node on http://127.0.0.1:8545..."
npx hardhat node > /tmp/hardhat.log 2>&1 &
HARDHAT_PID=$!
sleep 3

# Check if it started
if ps -p $HARDHAT_PID > /dev/null; then
    echo "✅ Hardhat node started (PID: $HARDHAT_PID)"
else
    echo "❌ Failed to start Hardhat node. Check /tmp/hardhat.log"
    exit 1
fi

# Check if contract is deployed
if [ ! -f "deployments/localhost.json" ]; then
    echo "📝 Deploying smart contract..."
    sleep 2
    npx hardhat run scripts/deploy.js --network localhost > /tmp/deploy.log 2>&1
    
    if [ $? -eq 0 ]; then
        CONTRACT_ADDRESS=$(grep "deployed to:" /tmp/deploy.log | awk '{print $NF}')
        echo "✅ Contract deployed: $CONTRACT_ADDRESS"
    else
        echo "⚠️  Contract deployment may have failed. Check /tmp/deploy.log"
    fi
else
    CONTRACT_ADDRESS=$(grep -o '"address": "[^"]*"' deployments/localhost.json | cut -d'"' -f4)
    echo "✅ Contract already deployed: $CONTRACT_ADDRESS"
fi

echo ""
echo "✅ Blockchain is ready!"
echo ""
echo "Contract Address: $CONTRACT_ADDRESS"
echo "Network: http://127.0.0.1:8545"
echo "Chain ID: 1337"
echo ""
echo "To stop: kill $HARDHAT_PID"
echo "Or run: lsof -ti:8545 | xargs kill"




