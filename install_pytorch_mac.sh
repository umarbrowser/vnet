#!/bin/bash

# PyTorch Installation Script for Mac
# PyTorch is easier to install than TensorFlow on Mac!

echo "=========================================="
echo "PyTorch Installation for Mac"
echo "=========================================="
echo ""

# Detect Mac architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" = "arm64" ]; then
    echo "🍎 Apple Silicon Mac (M1/M2/M3) detected"
    echo ""
    echo "Installing PyTorch for Apple Silicon..."
    echo ""
    
    # Install PyTorch with Metal support (GPU acceleration)
    pip3 install torch torchvision torchaudio
    
elif [ "$ARCH" = "x86_64" ]; then
    echo "💻 Intel Mac detected"
    echo ""
    echo "Installing PyTorch for Intel Mac..."
    echo ""
    
    # Install PyTorch for Intel
    pip3 install torch torchvision torchaudio
    
else
    echo "⚠️  Unknown architecture: $ARCH"
    echo "Installing PyTorch (should work on both)..."
    pip3 install torch torchvision torchaudio
fi

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ MPS (Metal) available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PyTorch installation successful!"
    echo ""
    echo "You can now run: python3 ml/train_models.py"
    echo "DNN model will use PyTorch backend!"
else
    echo ""
    echo "❌ PyTorch installation failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Try: pip3 install --upgrade pip"
    echo "2. Try: pip3 install torch torchvision --no-cache-dir"
    echo "3. Check Python version: python3 --version (need 3.9-3.11)"
fi








