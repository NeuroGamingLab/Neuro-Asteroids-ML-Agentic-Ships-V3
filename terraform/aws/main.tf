terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Asteroids-Game"
      Environment = "Demo"
      ManagedBy   = "Terraform"
      Owner       = var.owner_tag
    }
  }
}

# Data source to get latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Data source to get default VPC
data "aws_vpc" "default" {
  default = true
}

# Data source to get default subnet
data "aws_subnet" "default" {
  vpc_id            = data.aws_vpc.default.id
  availability_zone = "${var.aws_region}a"
  default_for_az    = true
}

# Security Group for Asteroids Game
resource "aws_security_group" "asteroids_game" {
  name        = "asteroids-game-sg"
  description = "Security group for Asteroids Arcade Game"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    description = "SSH from allowed CIDR blocks"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr
  }

  # Streamlit dashboard access
  ingress {
    description = "Streamlit dashboard from allowed CIDR blocks"
    from_port   = var.streamlit_port
    to_port     = var.streamlit_port
    protocol    = "tcp"
    cidr_blocks = var.allowed_dashboard_cidr
  }

  # HTTP access (optional)
  ingress {
    description = "HTTP access"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_dashboard_cidr
  }

  # HTTPS access (optional)
  ingress {
    description = "HTTPS access"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_dashboard_cidr
  }

  # All outbound traffic
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "asteroids-game-sg"
  }
}

# Elastic IP for stable public IP
resource "aws_eip" "game_ip" {
  domain = "vpc"
  tags = {
    Name = "asteroids-game-ip"
  }
}

# EC2 Instance for Asteroids Game
resource "aws_instance" "asteroids_game" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.asteroids_game.id]
  subnet_id              = data.aws_subnet.default.id

  # Root volume configuration
  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size
    encrypted   = true
  }

  # User data script to set up the game
  user_data = templatefile("${path.module}/user_data.sh", {
    streamlit_port = var.streamlit_port
    github_repo    = var.github_repo
  })

  # Enable detailed monitoring
  monitoring = var.enable_detailed_monitoring

  # Instance metadata options
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = {
    Name = "asteroids-game"
  }
}

# Associate Elastic IP with instance
resource "aws_eip_association" "game_ip_assoc" {
  instance_id   = aws_instance.asteroids_game.id
  allocation_id = aws_eip.game_ip.id
}

# CloudWatch Log Group for game logs
resource "aws_cloudwatch_log_group" "asteroids_game" {
  name              = "/aws/ec2/asteroids-game"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "asteroids-game-logs"
  }
}

