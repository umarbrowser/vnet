#!/usr/bin/env python3
"""
Python script to check blockchain status - called from Node.js
"""

import sys
import json
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

def main():
    try:
        from src.blockchain_integration import BlockchainIntegration
        
        # Try to connect to blockchain
        try:
            blockchain = BlockchainIntegration()
            
            # Get blockchain info
            status = {
                'connected': True,
                'chain_id': blockchain.w3.eth.chain_id if blockchain.w3 else None,
                'account_address': blockchain.account.address if blockchain.account else None,
                'contract_address': blockchain.contract_address if blockchain.contract_address else None,
                'network': blockchain.config.get('network', {}).get('active', 'local'),
                'network_url': blockchain.config.get('network', {}).get(blockchain.config.get('network', {}).get('active', 'local'), 'http://127.0.0.1:8545'),
                'block_number': blockchain.w3.eth.block_number if blockchain.w3 else None,
                'gas_price': str(blockchain.w3.eth.gas_price) if blockchain.w3 else None,
                'total_records': blockchain.get_total_records() if blockchain.contract else 0
            }
            
            print(json.dumps(status))
            
        except Exception as e:
            # Blockchain not available
            status = {
                'connected': False,
                'error': str(e),
                'chain_id': None,
                'account_address': None,
                'contract_address': None,
                'network': 'local',
                'network_url': 'http://127.0.0.1:8545',
                'block_number': None,
                'gas_price': None,
                'total_records': 0
            }
            print(json.dumps(status))
            
    except Exception as e:
        error_result = {
            'connected': False,
            'error': str(e),
            'chain_id': None,
            'account_address': None,
            'contract_address': None,
            'network': 'local',
            'network_url': 'http://127.0.0.1:8545',
            'block_number': None,
            'gas_price': None,
            'total_records': 0
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()




