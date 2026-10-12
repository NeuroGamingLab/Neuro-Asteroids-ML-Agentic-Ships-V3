#!/bin/bash
#
# Destruction script for Asteroids Game on AWS
# Safely destroys all AWS resources
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${RED}${NC}"
echo -e "${RED}                                                        ${NC}"
echo -e "${RED}       ASTEROIDS GAME - AWS DESTRUCTION              ${NC}"
echo -e "${RED}                                                        ${NC}"
echo -e "${RED}${NC}"
echo ""

# Check if Terraform is initialized
if [ ! -d ".terraform" ]; then
    echo -e "${YELLOW}  Terraform not initialized. Nothing to destroy.${NC}"
    exit 0
fi

# Show what will be destroyed
echo -e "${BLUE} Resources that will be destroyed:${NC}"
terraform plan -destroy -out=tfplan-destroy 2>/dev/null || {
    echo -e "${YELLOW}  Could not create destroy plan. Proceeding anyway...${NC}"
}

echo ""
echo -e "${RED}  WARNING: This will destroy ALL AWS resources created by Terraform!${NC}"
echo -e "${YELLOW}This includes:${NC}"
echo "  - EC2 Instance"
echo "  - Security Group"
echo "  - Elastic IP"
echo "  - CloudWatch Log Group"
echo ""
read -p "Type 'destroy' to confirm: " confirm

if [ "$confirm" != "destroy" ]; then
    echo -e "${GREEN} Destruction cancelled.${NC}"
    exit 0
fi

# Destroy resources
echo -e "${BLUE}  Destroying resources...${NC}"
terraform destroy -auto-approve

# Clean up local files
echo -e "${BLUE} Cleaning up local files...${NC}"
read -p "Remove Terraform state files? (yes/no): " clean_state

if [ "$clean_state" == "yes" ]; then
    rm -rf .terraform/
    rm -f terraform.tfstate terraform.tfstate.backup .terraform.lock.hcl tfplan tfplan-destroy
    echo -e "${GREEN} Local files cleaned${NC}"
else
    echo -e "${YELLOW}⏭  Keeping state files${NC}"
fi

# Optional: Remove SSH key
echo ""
read -p "Remove SSH key from AWS? (yes/no): " remove_key

if [ "$remove_key" == "yes" ]; then
    KEY_NAME="asteroids-game-key"
    echo -e "${BLUE}  Removing SSH key from AWS...${NC}"
    aws ec2 delete-key-pair --key-name "$KEY_NAME" --region $(grep aws_region terraform.tfvars | cut -d'"' -f2) 2>/dev/null || {
        echo -e "${YELLOW}  Key may not exist or already deleted${NC}"
    }
    echo -e "${GREEN} SSH key removed${NC}"
fi

echo ""
echo -e "${GREEN}${NC}"
echo -e "${GREEN}                                                        ${NC}"
echo -e "${GREEN}       DESTRUCTION COMPLETE!                          ${NC}"
echo -e "${GREEN}                                                        ${NC}"
echo -e "${GREEN}${NC}"
echo ""
echo -e "${BLUE} All AWS resources have been destroyed.${NC}"
echo -e "${BLUE} Verify in AWS Console that no resources remain.${NC}"
echo ""

