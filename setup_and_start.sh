#!/bin/bash

# VANET Detection System - Complete Setup and Start Script for Mac
# This script sets up everything from scratch and starts the system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_header "VANET Misbehavior Detection System - Complete Setup"

# Step 1: Check and Install Prerequisites
print_header "Step 1: Checking Prerequisites"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    print_success "Homebrew installed"
else
    print_success "Homebrew is installed"
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Installing Node.js..."
    brew install node
    print_success "Node.js installed"
else
    NODE_VERSION=$(node --version)
    print_success "Node.js is installed ($NODE_VERSION)"
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    print_warning "Python 3 not found. Installing Python 3..."
    brew install python3
    print_success "Python 3 installed"
else
    PYTHON_VERSION=$(python3 --version)
    print_success "Python 3 is installed ($PYTHON_VERSION)"
fi

# Check for pip
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade
    print_success "pip3 installed"
else
    print_success "pip3 is installed"
fi

# Step 2: Setup Python Environment
print_header "Step 2: Setting Up Python Environment"

if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install Python dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
pip install numpy pandas scikit-learn matplotlib seaborn joblib loguru tqdm pyyaml python-dotenv web3 eth-account pytest pytest-cov gdown openpyxl flask flask-cors --quiet

print_success "Python dependencies installed"

# Step 3: Setup Node.js Dependencies
print_header "Step 3: Setting Up Node.js Dependencies"

if [ ! -d "node_modules" ]; then
    print_info "Installing Node.js dependencies..."
    npm install --silent
    print_success "Node.js dependencies installed"
else
    print_info "Checking Node.js dependencies..."
    npm install --silent
    print_success "Node.js dependencies up to date"
fi

# Step 4: Download Dataset
print_header "Step 4: Downloading Dataset"

DATASET_DIR="data"
DATASET_FILE="$DATASET_DIR/veremi_synthetic.csv"

if [ ! -f "$DATASET_FILE" ]; then
    print_info "Dataset not found. Checking for download script..."
    
    if [ -f "utils/download_dataset.py" ]; then
        print_info "Running dataset download script..."
        python utils/download_dataset.py
    else
        print_warning "Dataset download script not found. Generating synthetic dataset..."
        mkdir -p "$DATASET_DIR"
        
        # Generate synthetic dataset using Python
        python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from ml.train_models import VANETMisbehaviorDetector
import os

print("Generating synthetic dataset...")
detector = VANETMisbehaviorDetector()
df = detector.generate_synthetic_data(n_samples=10000)

os.makedirs('data', exist_ok=True)
df.to_csv('data/veremi_synthetic.csv', index=False)
print(f"✅ Dataset generated: {len(df)} samples saved to data/veremi_synthetic.csv")
PYTHON_SCRIPT
    fi
    
    if [ -f "$DATASET_FILE" ]; then
        print_success "Dataset ready: $DATASET_FILE"
    else
        print_error "Failed to create dataset"
        exit 1
    fi
else
    DATASET_SIZE=$(du -h "$DATASET_FILE" | cut -f1)
    print_success "Dataset already exists ($DATASET_SIZE)"
fi

# Step 5: Train ML Models
print_header "Step 5: Training ML Models"

MODELS_DIR="models"
if [ ! -f "$MODELS_DIR/dnn_model.pkl" ] || [ ! -f "$MODELS_DIR/rf_model.pkl" ] || [ ! -f "$MODELS_DIR/svm_model.pkl" ]; then
    print_info "ML models not found. Training models (this may take a few minutes)..."
    python ml/train_models.py --dataset "$DATASET_FILE"
    
    if [ -f "$MODELS_DIR/dnn_model.pkl" ]; then
        print_success "ML models trained successfully"
    else
        print_error "Failed to train models"
        exit 1
    fi
else
    print_success "ML models already trained"
fi

# Step 6: Compile Smart Contracts
print_header "Step 6: Compiling Smart Contracts"

if [ ! -d "artifacts" ] || [ -z "$(ls -A artifacts 2>/dev/null)" ]; then
    print_info "Compiling smart contracts..."
    npx hardhat compile --quiet
    print_success "Smart contracts compiled"
else
    print_success "Smart contracts already compiled"
fi

# Step 7: Start Services
print_header "Step 7: Starting Services"

# Kill any existing processes on ports
print_info "Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8545 | xargs kill -9 2>/dev/null || true
sleep 1

# Create logs directory
mkdir -p logs

# Start Blockchain Node (in background)
print_info "Starting Hardhat blockchain node..."
npx hardhat node > logs/blockchain.log 2>&1 &
BLOCKCHAIN_PID=$!
sleep 3

if ps -p $BLOCKCHAIN_PID > /dev/null; then
    print_success "Blockchain node started (PID: $BLOCKCHAIN_PID)"
else
    print_warning "Blockchain node may have failed to start (check logs/blockchain.log)"
fi

# Deploy contract if needed
if [ ! -f "deployments/localhost.json" ]; then
    print_info "Deploying smart contract..."
    sleep 2
    npx hardhat run scripts/deploy.js --network localhost > logs/deploy.log 2>&1 || print_warning "Contract deployment may have failed"
fi

# Start Backend Server (in background)
print_info "Starting Node.js backend server..."
node server.js > logs/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

if ps -p $BACKEND_PID > /dev/null; then
    print_success "Backend server started (PID: $BACKEND_PID)"
else
    print_error "Backend server failed to start (check logs/backend.log)"
    exit 1
fi

# Start Frontend (in background)
print_info "Starting frontend development server..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 3

if ps -p $FRONTEND_PID > /dev/null; then
    print_success "Frontend server started (PID: $FRONTEND_PID)"
else
    print_warning "Frontend server may have failed to start (check logs/frontend.log)"
fi

# Step 8: Verify Services
print_header "Step 8: Verifying Services"

sleep 2

# Check backend
if curl -s http://localhost:5000/api/stats > /dev/null 2>&1; then
    print_success "Backend API is responding"
else
    print_warning "Backend API not responding (may still be starting)"
fi

# Check frontend
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is responding"
else
    print_warning "Frontend not responding (may still be starting)"
fi

# Check blockchain
if curl -s http://localhost:8545 > /dev/null 2>&1; then
    print_success "Blockchain node is responding"
else
    print_warning "Blockchain node not responding (optional - system works without it)"
fi

# Final Summary
print_header "Setup Complete!"

echo -e "${GREEN}✅ All services are starting up!${NC}"
echo ""
echo -e "${BLUE}🌐 Access Points:${NC}"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:5000"
echo "   Blockchain: http://localhost:8545"
echo ""
echo -e "${BLUE}📊 Process IDs:${NC}"
echo "   Blockchain: $BLOCKCHAIN_PID"
echo "   Backend:    $BACKEND_PID"
echo "   Frontend:   $FRONTEND_PID"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo "   Blockchain: logs/blockchain.log"
echo "   Backend:    logs/backend.log"
echo "   Frontend:   logs/frontend.log"
echo ""
echo -e "${YELLOW}💡 To stop all services:${NC}"
echo "   ./stop_services.sh"
echo "   Or: kill $BLOCKCHAIN_PID $BACKEND_PID $FRONTEND_PID"
echo ""
echo -e "${GREEN}🎉 System is ready! Open http://localhost:3000 in your browser.${NC}"
echo ""

# Save PIDs for stop script
echo "$BLOCKCHAIN_PID $BACKEND_PID $FRONTEND_PID" > .service_pids

print_info "Waiting for services to fully start (10 seconds)..."
sleep 10

print_success "Setup and startup complete!"




