#!/bin/bash

# Stop all VANET Detection System services

echo "🛑 Stopping all services..."

# Read PIDs from file if exists
if [ -f .service_pids ]; then
    PIDS=$(cat .service_pids)
    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
            echo "✅ Stopped process $PID"
        fi
    done
    rm .service_pids
fi

# Kill by port (fallback)
lsof -ti:5000 | xargs kill -9 2>/dev/null && echo "✅ Stopped backend (port 5000)" || true
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "✅ Stopped frontend (port 3000)" || true
lsof -ti:8545 | xargs kill -9 2>/dev/null && echo "✅ Stopped blockchain (port 8545)" || true

# Kill by process name
pkill -f "hardhat node" 2>/dev/null && echo "✅ Stopped blockchain node" || true
pkill -f "node server.js" 2>/dev/null && echo "✅ Stopped backend server" || true
pkill -f "vite" 2>/dev/null && echo "✅ Stopped frontend server" || true

echo ""
echo "✅ All services stopped"




