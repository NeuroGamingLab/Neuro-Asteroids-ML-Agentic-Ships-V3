#!/bin/bash
#
# User Data Script for Asteroids Arcade Game
# This script runs on first boot and sets up the entire game environment
#

set -e  # Exit on error

# Logging setup
LOG_FILE="/var/log/asteroids-game-setup.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=========================================="
echo "Starting Asteroids Game Setup"
echo "Time: $(date)"
echo "=========================================="

# Update system packages
echo "[1/6] Updating system packages..."
dnf update -y

# Install required system packages
echo "[2/6] Installing system dependencies..."
dnf install -y \
    python3.11 \
    python3.11-pip \
    git \
    htop \
    vim \
    wget \
    curl

# Set Python 3.11 as default
alternatives --set python3 /usr/bin/python3.11 || true

# Upgrade pip
echo "[3/6] Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install Streamlit (only dependency needed for the game)
echo "[4/6] Installing Streamlit..."
python3 -m pip install streamlit>=1.28.0

# Create application directory
echo "[5/6] Setting up application directory..."
APP_DIR="/opt/asteroids-game"
mkdir -p $APP_DIR/app
cd $APP_DIR

# Clone or prepare for manual file upload
echo "[6/6] Preparing application directory..."
if [ ! -z "${github_repo}" ] && [ "${github_repo}" != "none" ]; then
    echo "Cloning repository from: ${github_repo}"
    git clone ${github_repo} $APP_DIR/app
    cd $APP_DIR/app
    
    # Install Python dependencies if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies from requirements.txt..."
        python3 -m pip install -r requirements.txt
    fi
else
    # Create directory structure for manual deployment
    echo "Directory ready for manual file upload..."
    echo "Upload your game files to: $APP_DIR/app/"
    echo "Required files: app.py, game.js, entities.js, ai_ship.js, enemy_ship.js, enemy_coordination.js, and any other game assets"
fi

# Create systemd service for auto-start and management
echo "Creating systemd service..."
cat > /etc/systemd/system/asteroids-game.service << SERVICE_EOF
[Unit]
Description=Asteroids Arcade Game
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_DIR/app
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$APP_DIR/app"
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port ${streamlit_port} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Set proper permissions
chown -R ec2-user:ec2-user $APP_DIR

# Enable the service (will start once files are uploaded)
systemctl daemon-reload
systemctl enable asteroids-game.service

# Create status file
cat > $APP_DIR/SETUP_STATUS.txt << STATUS_EOF
Setup Status: READY
Time: $(date)

Asteroids Arcade Game

Next Steps:
1. Upload your game files to: $APP_DIR/app/
   Required files:
   - app.py (Streamlit application)
   - game.js (Game logic)
   - entities.js, ai_ship.js, enemy_ship.js, enemy_coordination.js (Game logic, extracted from game.js)
   - index.html (if used)
   - style.css (if used)
   - requirements.txt (if additional dependencies needed)

2. If using requirements.txt, install dependencies:
   cd $APP_DIR/app
   python3 -m pip install -r requirements.txt

3. Start the game service:
   sudo systemctl start asteroids-game

4. Check status:
   sudo systemctl status asteroids-game

5. View logs:
   sudo journalctl -u asteroids-game -f

Dashboard will be available at: http://<public-ip>:${streamlit_port}
STATUS_EOF

# Create helpful aliases
cat >> /home/ec2-user/.bashrc << ALIAS_EOF

# Asteroids Game Aliases
alias game-status='sudo systemctl status asteroids-game'
alias game-start='sudo systemctl start asteroids-game'
alias game-stop='sudo systemctl stop asteroids-game'
alias game-restart='sudo systemctl restart asteroids-game'
alias game-logs='sudo journalctl -u asteroids-game -f'
alias game-app='cd $APP_DIR/app'
ALIAS_EOF

# Create info message for login
cat > /etc/motd << MOTD_EOF

                                                                
       ASTEROIDS ARCADE GAME                                  
                                                                
     Instance is ready for game deployment                     
                                                                


 Game Commands:
  game-status   - Check service status
  game-start    - Start the game
  game-stop     - Stop the game
  game-restart  - Restart the game
  game-logs     - View live logs
  game-app      - Navigate to app directory

 Application Directory: $APP_DIR/app/
 Setup Status: cat $APP_DIR/SETUP_STATUS.txt

  To deploy your game:
  1. Upload files to $APP_DIR/app/
  2. Run: game-start
  3. Access: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):${streamlit_port}

MOTD_EOF

echo "=========================================="
echo " Setup Complete!"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Instance is ready. Next steps:"
echo "1. Upload game files to: $APP_DIR/app/"
echo "2. Start game: sudo systemctl start asteroids-game"
echo "3. Check status: sudo systemctl status asteroids-game"
echo ""
echo "For full details: cat $APP_DIR/SETUP_STATUS.txt"

