#!/bin/bash
#
# Deployment script for Asteroids Game on AWS
# Automates the entire deployment process
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

echo -e "${BLUE}${NC}"
echo -e "${BLUE}                                                        ${NC}"
echo -e "${BLUE}       ASTEROIDS GAME - AWS DEPLOYMENT                ${NC}"
echo -e "${BLUE}                                                        ${NC}"
echo -e "${BLUE}${NC}"
echo ""

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED} Terraform is not installed.${NC}"
    echo "Install it with: brew install terraform"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED} AWS CLI is not configured.${NC}"
    echo "Run: aws configure"
    exit 1
fi

# Function to create SSH key if it doesn't exist
create_ssh_key() {
    KEY_NAME="asteroids-game-key"
    KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
    
    if [ ! -f "$KEY_PATH" ]; then
        echo -e "${YELLOW} Creating SSH key pair...${NC}"
        aws ec2 create-key-pair \
            --key-name "$KEY_NAME" \
            --query 'KeyMaterial' \
            --output text \
            --region $(grep aws_region terraform.tfvars | cut -d'"' -f2) > "$KEY_PATH"
        chmod 400 "$KEY_PATH"
        echo -e "${GREEN} SSH key created: $KEY_PATH${NC}"
    else
        echo -e "${GREEN} SSH key already exists: $KEY_PATH${NC}"
    fi
}

# Initialize Terraform
init_terraform() {
    echo -e "${BLUE} Initializing Terraform...${NC}"
    terraform init
    echo -e "${GREEN} Terraform initialized${NC}"
}

# Plan deployment
plan_deployment() {
    echo -e "${BLUE} Planning deployment...${NC}"
    terraform plan -out=tfplan
    echo -e "${GREEN} Plan created${NC}"
}

# Apply deployment
apply_deployment() {
    echo -e "${YELLOW}  This will create AWS resources and may incur costs.${NC}"
    read -p "Continue with deployment? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo -e "${RED}Deployment cancelled.${NC}"
        exit 0
    fi
    
    echo -e "${BLUE} Deploying infrastructure...${NC}"
    terraform apply tfplan
    echo -e "${GREEN} Infrastructure deployed${NC}"
}

# Get instance IP
get_instance_ip() {
    INSTANCE_IP=$(terraform output -raw dashboard_public_ip 2>/dev/null || echo "")
    if [ -z "$INSTANCE_IP" ]; then
        echo -e "${RED} Could not get instance IP${NC}"
        return 1
    fi
    echo "$INSTANCE_IP"
}

# Wait for instance to be ready
wait_for_instance() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        return 1
    fi
    
    echo -e "${BLUE} Waiting for instance to be ready...${NC}"
    for i in {1..30}; do
        if ssh -i "$HOME/.ssh/asteroids-game-key.pem" \
            -o StrictHostKeyChecking=no \
            -o ConnectTimeout=5 \
            ec2-user@$INSTANCE_IP "echo 'ready'" &> /dev/null; then
            echo -e "${GREEN} Instance is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    echo -e "${YELLOW}  Instance may still be initializing${NC}"
}

# Upload game files
upload_game_files() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        return 1
    fi
    
    echo -e "${YELLOW} Upload game files? (yes/no): ${NC}"
    read -p "" upload_confirm
    
    if [ "$upload_confirm" != "yes" ]; then
        echo -e "${YELLOW}⏭  Skipping file upload${NC}"
        echo "You can upload files manually later using:"
        echo "  scp -i ~/.ssh/asteroids-game-key.pem -r app.py game.js entities.js ai_ship.js enemy_ship.js enemy_coordination.js ec2-user@$INSTANCE_IP:/opt/asteroids-game/app/"
        return 0
    fi
    
    # Find project root (two levels up from terraform/aws)
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    
    echo -e "${BLUE} Uploading game files...${NC}"
    
    # Upload required files
    scp -i "$HOME/.ssh/asteroids-game-key.pem" \
        -o StrictHostKeyChecking=no \
        "$PROJECT_ROOT/app.py" \
        "$PROJECT_ROOT/game.js" \
        "$PROJECT_ROOT/entities.js" \
        "$PROJECT_ROOT/ai_ship.js" \
        "$PROJECT_ROOT/enemy_ship.js" \
        "$PROJECT_ROOT/enemy_coordination.js" \
        ec2-user@$INSTANCE_IP:/opt/asteroids-game/app/ 2>/dev/null || {
        echo -e "${YELLOW}  Some files may not exist, continuing...${NC}"
    }
    
    # Upload optional files if they exist
    [ -f "$PROJECT_ROOT/index.html" ] && \
        scp -i "$HOME/.ssh/asteroids-game-key.pem" \
            -o StrictHostKeyChecking=no \
            "$PROJECT_ROOT/index.html" \
            ec2-user@$INSTANCE_IP:/opt/asteroids-game/app/ 2>/dev/null || true
    
    [ -f "$PROJECT_ROOT/style.css" ] && \
        scp -i "$HOME/.ssh/asteroids-game-key.pem" \
            -o StrictHostKeyChecking=no \
            "$PROJECT_ROOT/style.css" \
            ec2-user@$INSTANCE_IP:/opt/asteroids-game/app/ 2>/dev/null || true
    
    [ -f "$PROJECT_ROOT/requirements.txt" ] && \
        scp -i "$HOME/.ssh/asteroids-game-key.pem" \
            -o StrictHostKeyChecking=no \
            "$PROJECT_ROOT/requirements.txt" \
            ec2-user@$INSTANCE_IP:/opt/asteroids-game/app/ 2>/dev/null || true
    
    echo -e "${GREEN} Files uploaded${NC}"
}

# Start game service
start_game_service() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        return 1
    fi
    
    echo -e "${BLUE} Starting game service...${NC}"
    ssh -i "$HOME/.ssh/asteroids-game-key.pem" \
        -o StrictHostKeyChecking=no \
        ec2-user@$INSTANCE_IP \
        "sudo systemctl start asteroids-game && sudo systemctl status asteroids-game --no-pager"
    
    echo -e "${GREEN} Game service started${NC}"
}

# Verify deployment
verify_deployment() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        return 1
    fi
    
    STREAMLIT_PORT=$(grep streamlit_port terraform.tfvars | cut -d'=' -f2 | tr -d ' "')
    
    echo -e "${BLUE} Verifying deployment...${NC}"
    sleep 10  # Wait for service to start
    
    if curl -s -o /dev/null -w "%{http_code}" "http://${INSTANCE_IP}:${STREAMLIT_PORT}" | grep -q "200"; then
        echo -e "${GREEN} Game is accessible!${NC}"
        return 0
    else
        echo -e "${YELLOW}  Game may still be starting. Check logs with:${NC}"
        echo "  ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@$INSTANCE_IP 'sudo journalctl -u asteroids-game -f'"
        return 1
    fi
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        deploy)
            create_ssh_key
            init_terraform
            plan_deployment
            apply_deployment
            wait_for_instance
            upload_game_files
            start_game_service
            verify_deployment
            
            INSTANCE_IP=$(get_instance_ip)
            STREAMLIT_PORT=$(grep streamlit_port terraform.tfvars | cut -d'=' -f2 | tr -d ' "')
            
            echo ""
            echo -e "${GREEN}${NC}"
            echo -e "${GREEN}                                                        ${NC}"
            echo -e "${GREEN}       DEPLOYMENT COMPLETE!                           ${NC}"
            echo -e "${GREEN}                                                        ${NC}"
            echo -e "${GREEN}${NC}"
            echo ""
            echo -e "${BLUE} Game URL:${NC} http://${INSTANCE_IP}:${STREAMLIT_PORT}"
            echo -e "${BLUE} Public IP:${NC} ${INSTANCE_IP}"
            echo ""
            echo -e "${YELLOW} Useful commands:${NC}"
            echo "  SSH: ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@${INSTANCE_IP}"
            echo "  Logs: ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@${INSTANCE_IP} 'sudo journalctl -u asteroids-game -f'"
            echo "  Status: ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@${INSTANCE_IP} 'sudo systemctl status asteroids-game'"
            echo ""
            echo -e "${YELLOW} To destroy:${NC} ./destroy.sh"
            ;;
        *)
            echo "Usage: $0 [deploy]"
            exit 1
            ;;
    esac
}

main "$@"

