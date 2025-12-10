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
    misbehaviorDetection = await MisbehaviorDetection.deploy();
    await misbehaviorDetection.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await misbehaviorDetection.owner()).to.equal(owner.address);
    });

    it("Should have initial trust score for new vehicles", async function () {
      const trustScore = await misbehaviorDetection.getTrustScore("VEH001");
      expect(trustScore).to.equal(10000); // 100.00%
    });
  });

  describe("Misbehavior Logging", function () {
    it("Should log misbehavior and update trust score", async function () {
      const vehicleId = "VEH001";
      const misbehaviorType = 0; // Sybil
      const confidenceScore = 8500; // 85%

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, misbehaviorType, confidenceScore)
      ).to.emit(misbehaviorDetection, "MisbehaviorDetected");

      const trustScore = await misbehaviorDetection.getTrustScore(vehicleId);
      expect(trustScore).to.be.below(10000);

      const misbehaviorCount = await misbehaviorDetection.getMisbehaviorCount(vehicleId);
      expect(misbehaviorCount).to.equal(1);
    });

    it("Should blacklist vehicle when trust score drops below threshold", async function () {
      const vehicleId = "VEH002";
      
      // Log multiple misbehaviors to drop trust score
      for (let i = 0; i < 20; i++) {
        await misbehaviorDetection.logMisbehavior(vehicleId, 0, 9000);
      }

      const isBlacklisted = await misbehaviorDetection.isBlacklisted(vehicleId);
      expect(isBlacklisted).to.be.true;
    });

    it("Should prevent logging for blacklisted vehicles", async function () {
      const vehicleId = "VEH003";
      
      // Blacklist vehicle
      await misbehaviorDetection.blacklistVehicle(vehicleId);

      await expect(
        misbehaviorDetection.logMisbehavior(vehicleId, 0, 5000)
      ).to.be.revertedWith("Vehicle is blacklisted");
    });
  });

  describe("Trust Score Management", function () {
    it("Should allow owner to reset trust score", async function () {
      const vehicleId = "VEH004";
      
      await misbehaviorDetection.logMisbehavior(vehicleId, 0, 8000);
      await misbehaviorDetection.resetTrustScore(vehicleId);

      const trustScore = await misbehaviorDetection.getTrustScore(vehicleId);
      expect(trustScore).to.equal(10000);
    });

    it("Should allow owner to remove from blacklist", async function () {
      const vehicleId = "VEH005";
      
      await misbehaviorDetection.blacklistVehicle(vehicleId);
      await misbehaviorDetection.removeFromBlacklist(vehicleId);

      const isBlacklisted = await misbehaviorDetection.isBlacklisted(vehicleId);
      expect(isBlacklisted).to.be.false;
    });
  });

  describe("Record Retrieval", function () {
    it("Should retrieve misbehavior records", async function () {
      const vehicleId = "VEH006";
      
      await misbehaviorDetection.logMisbehavior(vehicleId, 1, 7500);
      
      const records = await misbehaviorDetection.getVehicleRecords(vehicleId);
      expect(records.length).to.equal(1);

      const record = await misbehaviorDetection.getRecord(records[0]);
      expect(record.vehicleId).to.equal(vehicleId);
      expect(record.misbehaviorType).to.equal(1);
    });

    it("Should return total record count", async function () {
      await misbehaviorDetection.logMisbehavior("VEH007", 0, 6000);
      await misbehaviorDetection.logMisbehavior("VEH008", 2, 7000);

      const totalRecords = await misbehaviorDetection.getTotalRecords();
      expect(totalRecords).to.equal(2);
    });
  });
});











