const hre = require("hardhat");

async function main() {
  console.log("Deploying MisbehaviorDetection contract...");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", (await deployer.provider.getBalance(deployer.address)).toString());
  
  const MisbehaviorDetection = await hre.ethers.getContractFactory("MisbehaviorDetection");
  const misbehaviorDetection = await MisbehaviorDetection.deploy();

  await misbehaviorDetection.waitForDeployment();

  const address = await misbehaviorDetection.getAddress();
  console.log("MisbehaviorDetection deployed to:", address);
  console.log("Network:", hre.network.name);

  // Verify deployment
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("Waiting for block confirmations...");
    await misbehaviorDetection.deploymentTransaction().wait(5);
    
    try {
      await hre.run("verify:verify", {
        address: address,
        constructorArguments: [],
      });
      console.log("Contract verified!");
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  }

  // Save deployment info
  const fs = require("fs");
  const deploymentInfo = {
    network: hre.network.name,
    address: address,
    deployer: deployer.address,
    timestamp: new Date().toISOString()
  };

  const deploymentsDir = "./deployments";
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  fs.writeFileSync(
    `${deploymentsDir}/${hre.network.name}.json`,
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\nDeployment complete!");
  console.log("Contract address saved to:", `${deploymentsDir}/${hre.network.name}.json`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
