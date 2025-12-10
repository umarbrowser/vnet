#!/bin/bash

# TensorFlow Installation Script for Mac
# Detects Mac type and installs appropriate TensorFlow version

echo "=========================================="
echo "TensorFlow Installation for Mac"
echo "=========================================="
echo ""

# Detect Mac architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" = "arm64" ]; then
    echo "🍎 Apple Silicon Mac (M1/M2/M3) detected"
    echo ""
    echo "Installing TensorFlow for Apple Silicon..."
    echo ""
    
    # Install tensorflow-macos and tensorflow-metal for GPU acceleration
    pip3 install tensorflow-macos
    pip3 install tensorflow-metal
    
    echo ""
    echo "✅ TensorFlow installed for Apple Silicon"
    echo "   - tensorflow-macos: CPU and GPU support"
    echo "   - tensorflow-metal: Metal GPU acceleration"
    
elif [ "$ARCH" = "x86_64" ]; then
    echo "💻 Intel Mac detected"
    echo ""
    echo "Installing TensorFlow for Intel Mac..."
    echo ""
    
    # For Intel Mac, use regular tensorflow or tensorflow-macos
    # tensorflow-macos works on both architectures
    pip3 install tensorflow-macos
    
    # Alternative: Use regular tensorflow (may have AVX issues on older Macs)
    # pip3 install tensorflow==2.13.0
    
    echo ""
    echo "✅ TensorFlow installed for Intel Mac"
    echo "   If you get AVX errors, try: pip3 install tensorflow==2.10.0"
    
else
    echo "⚠️  Unknown architecture: $ARCH"
    echo "Installing tensorflow-macos (works on both architectures)..."
    pip3 install tensorflow-macos
fi

echo ""
echo "Verifying installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TensorFlow installation successful!"
    echo ""
    echo "You can now run: python3 ml/train_models.py"
else
    echo ""
    echo "❌ TensorFlow installation failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Try: pip3 install --upgrade pip"
    echo "2. Try: pip3 install tensorflow==2.10.0 (older version without AVX)"
    echo "3. Check Python version: python3 --version (need 3.9-3.11)"
fi










