# AWS Configuration
aws_region = "ca-central-1"

# Instance Configuration
instance_type    = "t3.micro" # t3.micro is sufficient for the game (~$0.013/hour)
root_volume_size = 30         # Minimum 30GB required by AMI

# Network Configuration
#
# Both variables are required (no default in variables.tf) and Terraform
# rejects 0.0.0.0/0 for either one -- the SSH port and the dashboard (which
# has no login of its own) must not be opened to the whole internet. Replace
# YOUR_IP below with your own address (e.g. "203.0.113.4/32"; check
# https://checkip.amazonaws.com) before running terraform apply.
allowed_ssh_cidr       = ["YOUR_IP/32"]
allowed_dashboard_cidr = ["YOUR_IP/32"]

# Application Configuration
streamlit_port = 8501   # Default Streamlit port
github_repo    = "none" # Set to your GitHub repo URL if using Git deployment, or "none" for manual upload

# Resource Tags
owner_tag = "Asteroids-Game"

# Monitoring
enable_detailed_monitoring = false # Set to true for detailed monitoring (costs extra)
log_retention_days         = 7     # Keep logs for 7 days

# SSH Access
key_pair_name = "asteroids-game-key"  # EC2 Key Pair name for SSH access

