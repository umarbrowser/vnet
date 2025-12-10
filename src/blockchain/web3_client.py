"""
Web3 Client for Blockchain Integration
Handles interaction with Ethereum smart contracts
"""

from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import os
from typing import Dict, Optional, List
from loguru import logger
from eth_account import Account


class Web3Client:
    """Client for interacting with MisbehaviorDetection smart contract"""
    
    # Misbehavior types matching Solidity enum
    MISBEHAVIOR_TYPES = {
        'Sybil': 0,
        'Falsification': 1,
        'Replay': 2,
        'DoS': 3
    }
    
    def __init__(self, network: str = 'localhost', contract_address: Optional[str] = None):
        """
        Initialize Web3 client
        Args:
            network: 'localhost', 'goerli', or 'sepolia'
            contract_address: Deployed contract address
        """
        self.network = network
        self.contract_address = contract_address
        self.w3 = None
        self.contract = None
        self.account = None
        
        self._connect()
        if contract_address:
            self._load_contract()
    
    def _connect(self):
        """Connect to Ethereum network"""
        if self.network == 'localhost':
            rpc_url = "http://127.0.0.1:8545"
        elif self.network == 'ganache':
            rpc_url = "http://127.0.0.1:7545"
        elif self.network == 'goerli':
            rpc_url = os.getenv('GOERLI_RPC_URL', f"https://goerli.infura.io/v3/{os.getenv('INFURA_API_KEY')}")
        elif self.network == 'sepolia':
            rpc_url = os.getenv('SEPOLIA_RPC_URL', f"https://sepolia.infura.io/v3/{os.getenv('INFURA_API_KEY')}")
        else:
            raise ValueError(f"Unknown network: {self.network}")
        
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add PoA middleware for testnets
        if self.network in ['goerli', 'sepolia']:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.network}")
        
        logger.info(f"Connected to {self.network} network")
        
        # Load account from private key
        private_key = os.getenv('PRIVATE_KEY')
        if private_key:
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            self.account = Account.from_key(private_key)
            logger.info(f"Account loaded: {self.account.address}")
    
    def _load_contract(self):
        """Load deployed contract"""
        if not self.contract_address:
            raise ValueError("Contract address not provided")
        
        # Load ABI from artifacts
        artifacts_path = os.path.join('artifacts', 'contracts', 'MisbehaviorDetection.sol', 'MisbehaviorDetection.json')
        
        if not os.path.exists(artifacts_path):
            logger.warning(f"Contract artifacts not found at {artifacts_path}")
            logger.warning("Please compile the contract first: npm run compile")
            return
        
        with open(artifacts_path, 'r') as f:
            contract_data = json.load(f)
            abi = contract_data['abi']
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=abi
        )
        
        logger.info(f"Contract loaded at {self.contract_address}")
    
    def log_misbehavior(self, vehicle_id: str, misbehavior_type: str, 
                       confidence_score: int, gas_limit: int = 200000) -> Dict:
        """
        Log misbehavior to blockchain
        Args:
            vehicle_id: Vehicle identifier
            misbehavior_type: 'Sybil', 'Falsification', 'Replay', or 'DoS'
            confidence_score: ML confidence (0-10000)
            gas_limit: Gas limit for transaction
        Returns:
            Transaction receipt
        """
        if not self.contract:
            raise ValueError("Contract not loaded. Deploy contract first.")
        
        if not self.account:
            raise ValueError("Account not loaded. Set PRIVATE_KEY in .env")
        
        # Convert misbehavior type to enum value
        misbehavior_enum = self.MISBEHAVIOR_TYPES.get(misbehavior_type)
        if misbehavior_enum is None:
            raise ValueError(f"Unknown misbehavior type: {misbehavior_type}")
        
        # Build transaction
        function = self.contract.functions.logMisbehavior(
            vehicle_id,
            misbehavior_enum,
            confidence_score
        )
        
        # Estimate gas
        try:
            gas_estimate = function.estimate_gas({'from': self.account.address})
            gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}. Using provided limit.")
        
        # Get nonce
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        # Build transaction
        transaction = function.build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign transaction
        signed_txn = self.account.sign_transaction(transaction)
        
        # Send transaction
        logger.info(f"Logging misbehavior: {vehicle_id}, {misbehavior_type}, confidence={confidence_score}")
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for receipt
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt.status == 1:
            logger.info(f"Transaction confirmed in block {receipt.blockNumber}")
            logger.info(f"Gas used: {receipt.gasUsed}")
        else:
            logger.error("Transaction failed!")
        
        return {
            'tx_hash': tx_hash.hex(),
            'block_number': receipt.blockNumber,
            'gas_used': receipt.gasUsed,
            'status': receipt.status
        }
    
    def get_trust_score(self, vehicle_id: str) -> int:
        """Get trust score for a vehicle"""
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        score = self.contract.functions.getTrustScore(vehicle_id).call()
        return score
    
    def get_vehicle_records(self, vehicle_id: str) -> List[Dict]:
        """Get all misbehavior records for a vehicle"""
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        records = self.contract.functions.getVehicleRecords(vehicle_id).call()
        
        # Convert to readable format
        result = []
        for record in records:
            result.append({
                'vehicle_id': record[0],
                'misbehavior_type': record[1],
                'timestamp': record[2],
                'confidence_score': record[3],
                'reporter': record[4],
                'is_verified': record[5]
            })
        
        return result
    
    def get_total_records(self) -> int:
        """Get total number of misbehavior records"""
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        return self.contract.functions.getTotalRecords().call()
    
    def get_latest_block(self) -> int:
        """Get latest block number"""
        return self.w3.eth.block_number
    
    def get_balance(self) -> float:
        """Get account balance in ETH"""
        if not self.account:
            return 0.0
        
        balance_wei = self.w3.eth.get_balance(self.account.address)
        balance_eth = Web3.from_wei(balance_wei, 'ether')
        return float(balance_eth)

