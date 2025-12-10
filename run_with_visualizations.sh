#!/bin/bash

# Run dataset inference with blockchain and generate visualizations

cd /Users/flexisaf/Documents/another
source venv/bin/activate

echo "=========================================="
echo "VANET Dataset Processing with Blockchain"
echo "=========================================="
echo ""

# Check if blockchain is running
if ! curl -s http://127.0.0.1:8545 > /dev/null 2>&1; then
    echo "⚠️  Starting Hardhat node in background..."
    npx hardhat node > /dev/null 2>&1 &
    sleep 5
    echo "✅ Blockchain started"
fi

echo "Processing dataset with DNN model..."
echo "Dataset: data/veremi_dataset.csv"
echo "Model: DNN (scikit-learn MLPClassifier)"
echo "Blockchain: Enabled"
echo ""

# Run inference
python3 ml/use_dataset.py --dataset data/veremi_dataset.csv --model dnn

echo ""
echo "=========================================="
echo "Generating Visualizations..."
echo "=========================================="

# Generate visualizations if results exist
if [ -f "results/inference_results.json" ]; then
    python3 ml/visualize_results.py
    echo ""
    echo "✅ Visualizations saved to: results/visualizations/"
    echo ""
    echo "Generated graphs:"
    echo "  1. Detection Distribution Pie Chart"
    echo "  2. Confidence Score Distribution"
    echo "  3. Misbehavior Types Distribution"
    echo "  4. Performance Dashboard"
    echo "  5. Time Series Analysis"
    echo "  6. Confidence Heatmap"
else
    echo "⚠️  Results file not found. Run inference first."
fi

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="







