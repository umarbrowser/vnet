#!/bin/bash

# Start script for VANET Detection System Web Interface

echo "🚀 Starting VANET Detection System Web Interface"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Installing Flask dependencies..."
    pip install flask flask-cors
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Check if blockchain is running
echo "🔍 Checking blockchain connection..."
if ! curl -s http://127.0.0.1:8545 > /dev/null 2>&1; then
    echo "⚠️  Warning: Blockchain node not detected on port 8545"
    echo "   Please start blockchain in another terminal: npx hardhat node"
    echo ""
fi

# Start backend
echo "🔧 Starting Flask backend on http://localhost:5000..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! curl -s http://localhost:5000/api/stats > /dev/null 2>&1; then
    echo "❌ Backend failed to start. Check for errors above."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend started successfully!"
echo ""
echo "📊 Web Interface:"
echo "   Development: http://localhost:3000 (run 'cd frontend && npm run dev' in another terminal)"
echo "   Production:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the backend server"

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping backend...'; kill $BACKEND_PID 2>/dev/null; exit" INT TERM

wait $BACKEND_PID




