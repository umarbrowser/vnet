"""
Setup script to create necessary directories and prepare environment
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/veremi',
        'models',
        'logs',
        'artifacts/contracts/MisbehaviorDetection.sol',
        'cache',
        'test'
    ]
    
    print("Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\n✓ All directories created")


def create_env_file():
    """Create .env file from example if it doesn't exist"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print("\n.env file already exists. Skipping...")
        return
    
    if env_example.exists():
        print("\nCreating .env file from .env.example...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("✓ .env file created")
        print("⚠️  Please update .env with your actual keys and configuration")
    else:
        print("\n⚠️  .env.example not found. Creating basic .env file...")
        basic_env = """# Blockchain Configuration
PRIVATE_KEY=your_private_key_here_without_0x_prefix
INFURA_API_KEY=your_infura_api_key
NETWORK=localhost
CONTRACT_ADDRESS=

# ML Configuration
MODEL_TYPE=dnn
MODEL_PATH=models/best_model.pkl
"""
        with open(env_file, 'w') as f:
            f.write(basic_env)
        print("✓ Basic .env file created")
        print("⚠️  Please update .env with your actual configuration")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_python = ['numpy', 'pandas', 'sklearn', 'tensorflow', 'web3']
    missing = []
    
    for package in required_python:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing Python packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    # Check Node.js dependencies
    node_modules = Path('node_modules')
    if not node_modules.exists():
        print("\n⚠️  Node.js dependencies not installed")
        print("   Install with: npm install")
        return False
    else:
        print("\n  ✓ Node.js dependencies installed")
    
    return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("VANET Misbehavior Detection - Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing. Please install them first.")
        print("\nNext steps:")
        print("  1. pip install -r requirements.txt")
        print("  2. npm install")
        print("  3. Update .env file with your configuration")
        print("  4. python scripts/train_models.py")
        print("  5. npm run deploy:local")
        print("  6. python main.py --simulate")
    else:
        print("\n✓ Setup complete! You can now:")
        print("  1. Train models: python scripts/train_models.py")
        print("  2. Deploy contract: npm run deploy:local")
        print("  3. Run system: python main.py --simulate")


if __name__ == '__main__':
    main()

