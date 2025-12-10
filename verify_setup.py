#!/usr/bin/env python3
"""
Verification script to check if all components are set up correctly
"""

import os
import sys
import json
import subprocess

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"⚠️  {description} missing: {dirpath} (will be created)")
        return False

def check_node_modules():
    """Check if Node.js dependencies are installed"""
    if os.path.exists("node_modules"):
        print("✅ Node.js dependencies installed")
        return True
    else:
        print("❌ Node.js dependencies not installed. Run: npm install")
        return False

def check_python_packages():
    """Check if Python packages are installed"""
    try:
        import numpy
        import pandas
        import sklearn
        import tensorflow
        import web3
        print("✅ Python dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Python dependency missing: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def check_contract_compiled():
    """Check if contracts are compiled"""
    if os.path.exists("artifacts/contracts/MisbehaviorDetection.sol"):
        print("✅ Smart contracts compiled")
        return True
    else:
        print("⚠️  Smart contracts not compiled. Run: npx hardhat compile")
        return False

def check_models_trained():
    """Check if ML models are trained"""
    models_dir = "models"
    required_files = ["rf_model.pkl", "svm_model.pkl", "dnn_model.h5", "scaler.pkl"]
    
    if not os.path.exists(models_dir):
        print("⚠️  ML models not trained. Run: python ml/train_models.py")
        return False
    
    missing = [f for f in required_files if not os.path.exists(f"{models_dir}/{f}")]
    if missing:
        print(f"⚠️  Some models missing: {', '.join(missing)}")
        print("   Run: python ml/train_models.py")
        return False
    else:
        print("✅ ML models trained")
        return True

def check_config():
    """Check configuration file"""
    if os.path.exists("config/config.json"):
        with open("config/config.json", 'r') as f:
            config = json.load(f)
        
        if config.get('contract', {}).get('address'):
            print("✅ Configuration file has contract address")
        else:
            print("⚠️  Configuration file missing contract address")
            print("   Deploy contract first: npx hardhat run scripts/deploy.js --network localhost")
        
        return True
    else:
        print("❌ Configuration file missing: config/config.json")
        return False

def main():
    print("="*60)
    print("VANET Blockchain ML Detection System - Setup Verification")
    print("="*60)
    print()
    
    checks = []
    
    # File structure
    print("📁 File Structure:")
    checks.append(check_file_exists("contracts/MisbehaviorDetection.sol", "Smart contract"))
    checks.append(check_file_exists("ml/train_models.py", "ML training script"))
    checks.append(check_file_exists("src/detection_system.py", "Detection system"))
    checks.append(check_file_exists("main.py", "Main entry point"))
    checks.append(check_file_exists("hardhat.config.js", "Hardhat config"))
    checks.append(check_file_exists("package.json", "Package.json"))
    checks.append(check_file_exists("requirements.txt", "Requirements.txt"))
    print()
    
    # Dependencies
    print("📦 Dependencies:")
    checks.append(check_node_modules())
    checks.append(check_python_packages())
    print()
    
    # Build artifacts
    print("🔨 Build Artifacts:")
    checks.append(check_contract_compiled())
    checks.append(check_models_trained())
    print()
    
    # Configuration
    print("⚙️  Configuration:")
    checks.append(check_config())
    print()
    
    # Summary
    print("="*60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print()
        print("🚀 System is ready to run!")
        print("   Next steps:")
        print("   1. Start blockchain: npx hardhat node")
        print("   2. Deploy contract: npx hardhat run scripts/deploy.js --network localhost")
        print("   3. Update config/config.json with contract address")
        print("   4. Run system: python main.py")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print()
        print("Please fix the issues above before running the system.")
    
    print("="*60)

if __name__ == "__main__":
    main()











