const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MisbehaviorDetection", function () {
  let misbehaviorDetection;
  let owner;
  let addr1;
  let addr2;

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();

    const MisbehaviorDetection = await ethers.getContractFactory("MisbehaviorDetection");
    misbehaviorDetection = await MisbehaviorDetection.deploy(7000); // 70% min confidence
    await misbehaviorDetection.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await misbehaviorDetection.owner()).to.equal(owner.address);
    });

    it("Should set the minimum confidence threshold", async function () {
      expect(await misbehaviorDetection.minConfidenceThreshold()).to.equal(7000);
    });
  });

  describe("Logging Misbehavior", function () {
    it("Should log misbehavior successfully", async function () {
      const vehicleId = "VEH_001";
      const misbehaviorType = 0; // Sybil
      const confidence = 8500; // 85%

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence)
      ).to.emit(misbehaviorDetection, "MisbehaviorDetected")
        .withArgs(vehicleId, misbehaviorType, anyValue, confidence, owner.address);

      const totalRecords = await misbehaviorDetection.getTotalRecords();
      expect(totalRecords).to.equal(1);
    });

    it("Should reject logging with confidence below threshold", async function () {
      const vehicleId = "VEH_002";
      const misbehaviorType = 0;
      const confidence = 5000; // 50% (below 70% threshold)

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence)
      ).to.be.revertedWith("Confidence below threshold");
    });

    it("Should reject invalid confidence score", async function () {
      const vehicleId = "VEH_003";
      const misbehaviorType = 0;
      const confidence = 15000; // > 10000

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence)
      ).to.be.revertedWith("Confidence score must be <= 10000");
    });
  });

  describe("Trust Scores", function () {
    it("Should update trust score after misbehavior", async function () {
      const vehicleId = "VEH_004";
      const misbehaviorType = 0; // Sybil
      const confidence = 9000; // 90%

      // Initial trust should be 0 (not set)
      let trustScore = await misbehaviorDetection.getTrustScore(vehicleId);
      expect(trustScore).to.equal(0);

      // Log misbehavior
      await misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence);

      // Trust score should be decreased
      trustScore = await misbehaviorDetection.getTrustScore(vehicleId);
      expect(trustScore).to.be.lessThan(10000);
      expect(trustScore).to.be.greaterThan(0);
    });

    it("Should emit TrustScoreUpdated event", async function () {
      const vehicleId = "VEH_005";
      const misbehaviorType = 0;
      const confidence = 8500;

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence)
      ).to.emit(misbehaviorDetection, "TrustScoreUpdated");
    });
  });

  describe("Record Retrieval", function () {
    it("Should retrieve misbehavior record", async function () {
      const vehicleId = "VEH_006";
      const misbehaviorType = 1; // Falsification
      const confidence = 8800;

      await misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidence);

      const recordId = 0;
      const record = await misbehaviorDetection.getRecord(recordId);

      expect(record.vehicleId).to.equal(vehicleId);
      expect(record.misbehaviorType).to.equal(misbehaviorType);
      expect(record.confidenceScore).to.equal(confidence);
    });

    it("Should get all records for a vehicle", async function () {
      const vehicleId = "VEH_007";

      // Log multiple misbehaviors
      await misbehaviorDetection.logMisbehavior(vehicleId, 0, 8500);
      await misbehaviorDetection.logMisbehavior(vehicleId, 1, 9000);

      const records = await misbehaviorDetection.getVehicleRecords(vehicleId);
      expect(records.length).to.equal(2);
    });
  });

  describe("Access Control", function () {
    it("Should allow only owner to verify records", async function () {
      const vehicleId = "VEH_008";
      await misbehaviorDetection.logMisbehavior(vehicleId, 0, 8500);

      await expect(
        misbehaviorDetection.connect(addr1).verifyRecord(0)
      ).to.be.revertedWith("Only owner can call this function");
    });

    it("Should allow owner to verify records", async function () {
      const vehicleId = "VEH_009";
      await misbehaviorDetection.logMisbehavior(vehicleId, 0, 8500);

      await misbehaviorDetection.verifyRecord(0);
      const record = await misbehaviorDetection.getRecord(0);
      expect(record.isVerified).to.be.true;
    });
  });

  describe("Configuration", function () {
    it("Should allow owner to update confidence threshold", async function () {
      await misbehaviorDetection.setMinConfidenceThreshold(8000);
      expect(await misbehaviorDetection.minConfidenceThreshold()).to.equal(8000);
    });

    it("Should allow owner to reset trust score", async function () {
      const vehicleId = "VEH_010";
      await misbehaviorDetection.logMisbehavior(vehicleId, 0, 8500);

      await misbehaviorDetection.resetTrustScore(vehicleId, 5000);
      const trustScore = await misbehaviorDetection.getTrustScore(vehicleId);
      expect(trustScore).to.equal(5000);
    });
  });
});

// Helper for anyValue matcher
function anyValue() {
  return true;
}

