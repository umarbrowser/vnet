"""
Blockchain integration layer for logging misbehavior events
Uses Web3.py to interact with Ethereum smart contracts
"""

import os
import json
from typing import Optional, Dict, List
from web3 import Web3
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    # For newer web3 versions
    from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
from eth_account import Account
from loguru import logger
import time


class BlockchainIntegration:
    """Handle blockchain interactions for misbehavior logging"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize blockchain connection"""
        self.config = self.load_config(config_path)
        self.w3 = None
        self.contract = None
        self.account = None
        self.contract_address = None
        self.connect()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "network": {
                    "local": "http://127.0.0.1:8545",
                    "goerli": "",
                    "sepolia": ""
                },
                "contract": {
                    "address": "",
                    "abi_path": "artifacts/contracts/MisbehaviorDetection.sol/MisbehaviorDetection.json"
                },
                "account": {
                    "private_key": os.getenv("PRIVATE_KEY", "")
                }
            }
    
    def connect(self):
        """Connect to blockchain network"""
        network = self.config.get('network', {}).get('active', 'local')
        network_url = self.config['network'].get(network, "http://127.0.0.1:8545")
        
        logger.info(f"Connecting to {network} network: {network_url}")
        
        self.w3 = Web3(Web3.HTTPProvider(network_url))
        
        # Add PoA middleware for testnets
        if network in ['goerli', 'sepolia']:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {network} network")
        
        logger.info(f"Connected! Chain ID: {self.w3.eth.chain_id}")
        
        # Load account
        private_key = self.config['account'].get('private_key')
        if private_key:
            self.account = Account.from_key(private_key)
            logger.info(f"Account loaded: {self.account.address}")
        else:
            logger.warning("No private key provided. Using default account.")
            # For local networks, use first account from Hardhat
            if network == 'local' or network == 'localhost':
                # Hardhat default private key for first account
                default_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                self.account = Account.from_key(default_key)
                logger.info(f"Using default Hardhat account: {self.account.address}")
        
        # Load contract
        self.load_contract()
    
    def load_contract(self):
        """Load deployed contract"""
        # Try to load from deployments
        network = self.config.get('network', {}).get('active', 'local')
        deployment_path = f"deployments/{network}.json"
        
        if os.path.exists(deployment_path):
            with open(deployment_path, 'r') as f:
                deployment = json.load(f)
                self.contract_address = deployment.get('address')
        
        # Fallback to config
        if not self.contract_address:
            self.contract_address = self.config['contract'].get('address')
        
        if not self.contract_address:
            raise ValueError("Contract address not found. Deploy contract first.")
        
        # Load ABI
        abi_path = self.config['contract'].get('abi_path')
        if os.path.exists(abi_path):
            with open(abi_path, 'r') as f:
                contract_json = json.load(f)
                abi = contract_json.get('abi', [])
        else:
            # Minimal ABI for testing
            abi = [
                {
                    "inputs": [
                        {"internalType": "string", "name": "_vehicleId", "type": "string"},
                        {"internalType": "uint8", "name": "_misbehaviorType", "type": "uint8"},
                        {"internalType": "uint256", "name": "_confidenceScore", "type": "uint256"}
                    ],
                    "name": "logMisbehavior",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=abi
        )
        
        logger.info(f"Contract loaded: {self.contract_address}")
    
    def log_misbehavior(
        self,
        vehicle_id: str,
        misbehavior_type: int,
        confidence_score: int
    ) -> Dict:
        """
        Log misbehavior event to blockchain
        misbehavior_type: 0=Sybil, 1=Falsification, 2=Replay, 3=DoS
        confidence_score: 0-10000 (representing 0.00-100.00%)
        """
        if not self.account:
            raise ValueError("No account configured")
        
        try:
            # Build transaction
            transaction = self.contract.functions.logMisbehavior(
                vehicle_id,
                misbehavior_type,
                confidence_score
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 300000,  # Increased gas limit
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            # Get private key - Account.key is a HexBytes object
            private_key = self.account.key.hex() if hasattr(self.account.key, 'hex') else bytes(self.account.key).hex()
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction,
                private_key=private_key
            )
            
            # Send transaction
            start_time = time.time()
            # Handle different web3 versions
            if hasattr(signed_txn, 'rawTransaction'):
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            elif hasattr(signed_txn, 'raw_transaction'):
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            else:
                # For newer web3 versions, use the signed transaction directly
                tx_hash = self.w3.eth.send_transaction(signed_txn)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            latency = time.time() - start_time
            
            # Get gas used
            gas_used = receipt['gasUsed']
            gas_price = transaction['gasPrice']
            cost_wei = gas_used * gas_price
            cost_eth = Web3.from_wei(cost_wei, 'ether')
            
            logger.info(f"Misbehavior logged: Vehicle {vehicle_id}, Type {misbehavior_type}")
            logger.info(f"Transaction: {tx_hash.hex()}")
            logger.info(f"Latency: {latency:.2f}s, Gas: {gas_used}, Cost: {cost_eth} ETH")
            
            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'block_number': receipt['blockNumber'],
                'gas_used': gas_used,
                'cost_eth': float(cost_eth),
                'latency_seconds': latency
            }
        
        except Exception as e:
            logger.error(f"Failed to log misbehavior: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_trust_score(self, vehicle_id: str) -> int:
        """Get trust score for a vehicle"""
        try:
            score = self.contract.functions.getTrustScore(vehicle_id).call()
            return score
        except Exception as e:
            logger.error(f"Failed to get trust score: {e}")
            return 10000  # Default trust score
    
    def get_misbehavior_count(self, vehicle_id: str) -> int:
        """Get misbehavior count for a vehicle"""
        try:
            count = self.contract.functions.getMisbehaviorCount(vehicle_id).call()
            return count
        except Exception as e:
            logger.error(f"Failed to get misbehavior count: {e}")
            return 0
    
    def is_blacklisted(self, vehicle_id: str) -> bool:
        """Check if vehicle is blacklisted"""
        try:
            return self.contract.functions.isBlacklisted(vehicle_id).call()
        except Exception as e:
            logger.error(f"Failed to check blacklist: {e}")
            return False
    
    def get_total_records(self) -> int:
        """Get total number of misbehavior records"""
        try:
            return self.contract.functions.getTotalRecords().call()
        except Exception as e:
            logger.error(f"Failed to get total records: {e}")
            return 0


if __name__ == "__main__":
    # Test blockchain integration
    integration = BlockchainIntegration()
    
    # Test logging
    result = integration.log_misbehavior(
        vehicle_id="VEH001",
        misbehavior_type=0,  # Sybil
        confidence_score=8500  # 85%
    )
    
    print(f"Logging result: {result}")

