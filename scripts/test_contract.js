const hre = require("hardhat");

async function main() {
  console.log("Testing MisbehaviorDetection Contract\n");

  // Load deployment info
  const fs = require("fs");
  const path = require("path");
  const deploymentPath = path.join(__dirname, "..", "deployment.json");

  if (!fs.existsSync(deploymentPath)) {
    console.error("❌ deployment.json not found. Please deploy contract first:");
    console.error("   npm run deploy:local");
    return;
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf8"));
  console.log(`Network: ${deployment.network}`);
  console.log(`Contract: ${deployment.contractAddress}\n`);

  // Get contract instance
  const MisbehaviorDetection = await hre.ethers.getContractFactory("MisbehaviorDetection");
  const contract = await MisbehaviorDetection.attach(deployment.contractAddress);

  // Get signer
  const [signer] = await hre.ethers.getSigners();
  console.log(`Testing with account: ${signer.address}\n`);

  console.log("=".repeat(60));
  console.log("Test 1: Contract Information");
  console.log("=".repeat(60));

  const owner = await contract.owner();
  const minConfidence = await contract.minConfidenceThreshold();
  const totalRecords = await contract.getTotalRecords();

  console.log(`✓ Owner: ${owner}`);
  console.log(`✓ Min Confidence Threshold: ${minConfidence.toString()}`);
  console.log(`✓ Total Records: ${totalRecords.toString()}`);

  console.log("\n" + "=".repeat(60));
  console.log("Test 2: Log Misbehavior");
  console.log("=".repeat(60));

  const testVehicleId = "VEH_TEST_001";
  const misbehaviorType = 0; // Sybil
  const confidence = 8500; // 85%

  console.log(`Logging misbehavior:`);
  console.log(`  Vehicle ID: ${testVehicleId}`);
  console.log(`  Type: Sybil (${misbehaviorType})`);
  console.log(`  Confidence: ${confidence} (85%)`);

  const tx = await contract.logMisbehavior(testVehicleId, misbehaviorType, confidence);
  console.log(`\n⏳ Transaction sent: ${tx.hash}`);
  
  const receipt = await tx.wait();
  console.log(`✓ Transaction confirmed in block ${receipt.blockNumber}`);
  console.log(`✓ Gas used: ${receipt.gasUsed.toString()}`);

  // Get the record
  const recordId = await contract.getTotalRecords() - 1n;
  const record = await contract.getRecord(recordId);
  
  console.log(`\n✓ Record retrieved:`);
  console.log(`  Vehicle ID: ${record.vehicleId}`);
  console.log(`  Type: ${record.misbehaviorType}`);
  console.log(`  Timestamp: ${record.timestamp.toString()}`);
  console.log(`  Confidence: ${record.confidenceScore.toString()}`);
  console.log(`  Reporter: ${record.reporter}`);

  console.log("\n" + "=".repeat(60));
  console.log("Test 3: Trust Score");
  console.log("=".repeat(60));

  const trustScore = await contract.getTrustScore(testVehicleId);
  console.log(`✓ Trust score for ${testVehicleId}: ${trustScore.toString()} (${Number(trustScore)/100}%)`);

  console.log("\n" + "=".repeat(60));
  console.log("Test 4: Vehicle Records");
  console.log("=".repeat(60));

  const vehicleRecords = await contract.getVehicleRecords(testVehicleId);
  console.log(`✓ Found ${vehicleRecords.length} record(s) for ${testVehicleId}`);

  console.log("\n" + "=".repeat(60));
  console.log("Test 5: Multiple Misbehaviors");
  console.log("=".repeat(60));

  // Log different types
  const misbehaviors = [
    { type: 1, name: "Falsification", vehicle: "VEH_TEST_002" },
    { type: 2, name: "Replay", vehicle: "VEH_TEST_003" },
    { type: 3, name: "DoS", vehicle: "VEH_TEST_004" }
  ];

  for (const mb of misbehaviors) {
    const tx2 = await contract.logMisbehavior(mb.vehicle, mb.type, 9000);
    await tx2.wait();
    console.log(`✓ Logged ${mb.name} for ${mb.vehicle}`);
  }

  const finalTotal = await contract.getTotalRecords();
  console.log(`\n✓ Total records: ${finalTotal.toString()}`);

  console.log("\n" + "=".repeat(60));
  console.log("✅ All Tests Passed!");
  console.log("=".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Test failed:", error);
    process.exit(1);
  });

