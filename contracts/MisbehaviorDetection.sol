// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title MisbehaviorDetection
 * @dev Smart contract for logging misbehavior events in VANETs
 * Records vehicle misbehavior with immutable blockchain storage
 */
contract MisbehaviorDetection {
    
    // Enum for misbehavior types
    enum MisbehaviorType {
        Sybil,
        Falsification,
        Replay,
        DoS
    }
    
    // Struct to store misbehavior record
    struct MisbehaviorRecord {
        string vehicleId;
        MisbehaviorType misbehaviorType;
        uint256 timestamp;
        uint256 confidenceScore; // ML confidence (0-10000, representing 0.00-100.00%)
        address reporter; // RSU or vehicle that reported
        bool isValid;
    }
    
    // Mapping from vehicle ID to trust score (0-10000)
    mapping(string => uint256) public trustScores;
    
    // Mapping from vehicle ID to misbehavior count
    mapping(string => uint256) public misbehaviorCounts;
    
    // Array of all misbehavior records
    MisbehaviorRecord[] public misbehaviorRecords;
    
    // Mapping from vehicle ID to array of record indices
    mapping(string => uint256[]) public vehicleRecords;
    
    // Events for real-time monitoring
    event MisbehaviorDetected(
        string indexed vehicleId,
        MisbehaviorType misbehaviorType,
        uint256 timestamp,
        uint256 confidenceScore,
        address reporter
    );
    
    event TrustScoreUpdated(
        string indexed vehicleId,
        uint256 oldScore,
        uint256 newScore
    );
    
    event VehicleBlacklisted(
        string indexed vehicleId,
        uint256 timestamp
    );
    
    // Constants
    uint256 public constant INITIAL_TRUST_SCORE = 10000; // 100.00%
    uint256 public constant MIN_TRUST_SCORE = 0;
    uint256 public constant MAX_TRUST_SCORE = 10000;
    uint256 public constant BLACKLIST_THRESHOLD = 2000; // 20.00%
    uint256 public constant PENALTY_PER_MISBEHAVIOR = 500; // 5.00%
    
    // Mapping to track blacklisted vehicles
    mapping(string => bool) public blacklistedVehicles;
    
    // Owner address (can be RSU or admin)
    address public owner;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Log a misbehavior event detected by ML model
     * @param _vehicleId Vehicle identifier
     * @param _misbehaviorType Type of misbehavior detected
     * @param _confidenceScore ML model confidence (0-10000)
     */
    function logMisbehavior(
        string memory _vehicleId,
        MisbehaviorType _misbehaviorType,
        uint256 _confidenceScore
    ) public returns (uint256) {
        require(_confidenceScore <= MAX_TRUST_SCORE, "Invalid confidence score");
        require(!blacklistedVehicles[_vehicleId], "Vehicle is blacklisted");
        
        // Create new record
        MisbehaviorRecord memory newRecord = MisbehaviorRecord({
            vehicleId: _vehicleId,
            misbehaviorType: _misbehaviorType,
            timestamp: block.timestamp,
            confidenceScore: _confidenceScore,
            reporter: msg.sender,
            isValid: true
        });
        
        // Store record
        uint256 recordIndex = misbehaviorRecords.length;
        misbehaviorRecords.push(newRecord);
        vehicleRecords[_vehicleId].push(recordIndex);
        
        // Update misbehavior count
        misbehaviorCounts[_vehicleId]++;
        
        // Update trust score (penalty based on confidence)
        uint256 penalty = (_confidenceScore * PENALTY_PER_MISBEHAVIOR) / MAX_TRUST_SCORE;
        uint256 currentTrust = trustScores[_vehicleId];
        if (currentTrust == 0) {
            currentTrust = INITIAL_TRUST_SCORE;
        }
        
        uint256 newTrustScore = currentTrust > penalty ? currentTrust - penalty : MIN_TRUST_SCORE;
        uint256 oldTrustScore = trustScores[_vehicleId];
        trustScores[_vehicleId] = newTrustScore;
        
        // Auto-blacklist if trust score drops below threshold
        if (newTrustScore < BLACKLIST_THRESHOLD && !blacklistedVehicles[_vehicleId]) {
            blacklistedVehicles[_vehicleId] = true;
            emit VehicleBlacklisted(_vehicleId, block.timestamp);
        }
        
        // Emit events
        emit MisbehaviorDetected(
            _vehicleId,
            _misbehaviorType,
            block.timestamp,
            _confidenceScore,
            msg.sender
        );
        
        emit TrustScoreUpdated(_vehicleId, oldTrustScore, newTrustScore);
        
        return recordIndex;
    }
    
    /**
     * @dev Get trust score for a vehicle
     * @param _vehicleId Vehicle identifier
     * @return Trust score (0-10000)
     */
    function getTrustScore(string memory _vehicleId) public view returns (uint256) {
        uint256 score = trustScores[_vehicleId];
        return score == 0 ? INITIAL_TRUST_SCORE : score;
        }
        
    /**
     * @dev Get misbehavior count for a vehicle
     * @param _vehicleId Vehicle identifier
     * @return Number of misbehavior records
     */
    function getMisbehaviorCount(string memory _vehicleId) public view returns (uint256) {
        return misbehaviorCounts[_vehicleId];
    }
    
    /**
     * @dev Get all records for a vehicle
     * @param _vehicleId Vehicle identifier
     * @return Array of record indices
     */
    function getVehicleRecords(string memory _vehicleId) public view returns (uint256[] memory) {
        return vehicleRecords[_vehicleId];
    }
    
    /**
     * @dev Get misbehavior record by index
     * @param _index Record index
     * @return MisbehaviorRecord struct
     */
    function getRecord(uint256 _index) public view returns (MisbehaviorRecord memory) {
        require(_index < misbehaviorRecords.length, "Record index out of bounds");
        return misbehaviorRecords[_index];
    }
    
    /**
     * @dev Get total number of records
     * @return Total count
     */
    function getTotalRecords() public view returns (uint256) {
        return misbehaviorRecords.length;
    }
    
    /**
     * @dev Check if vehicle is blacklisted
     * @param _vehicleId Vehicle identifier
     * @return True if blacklisted
     */
    function isBlacklisted(string memory _vehicleId) public view returns (bool) {
        return blacklistedVehicles[_vehicleId];
    }
    
    /**
     * @dev Manually blacklist a vehicle (owner only)
     * @param _vehicleId Vehicle identifier
     */
    function blacklistVehicle(string memory _vehicleId) public onlyOwner {
        blacklistedVehicles[_vehicleId] = true;
        emit VehicleBlacklisted(_vehicleId, block.timestamp);
    }
    
    /**
     * @dev Remove vehicle from blacklist (owner only)
     * @param _vehicleId Vehicle identifier
     */
    function removeFromBlacklist(string memory _vehicleId) public onlyOwner {
        blacklistedVehicles[_vehicleId] = false;
        trustScores[_vehicleId] = INITIAL_TRUST_SCORE;
    }
    
    /**
     * @dev Reset trust score for a vehicle (owner only)
     * @param _vehicleId Vehicle identifier
     */
    function resetTrustScore(string memory _vehicleId) public onlyOwner {
        trustScores[_vehicleId] = INITIAL_TRUST_SCORE;
        misbehaviorCounts[_vehicleId] = 0;
    }
}
