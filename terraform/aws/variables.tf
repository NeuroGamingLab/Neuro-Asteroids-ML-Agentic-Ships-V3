variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "ca-central-1"
}

variable "instance_type" {
  description = "EC2 instance type for the Asteroids game"
  type        = string
  default     = "t3.micro"

  validation {
    condition     = can(regex("^t3\\.(micro|small|medium|large|xlarge)", var.instance_type))
    error_message = "Instance type must be a valid t3 instance type (micro, small, medium, large, xlarge)."
  }
}

variable "root_volume_size" {
  description = "Size of the root EBS volume in GB"
  type        = number
  default     = 20

  validation {
    condition     = var.root_volume_size >= 20 && var.root_volume_size <= 100
    error_message = "Root volume size must be between 20 and 100 GB."
  }
}

variable "allowed_ssh_cidr" {
  description = "CIDR blocks allowed to SSH into the instance. Set this to your own IP (e.g. [\"203.0.113.4/32\"]); there is deliberately no default."
  type        = list(string)

  validation {
    condition     = length(var.allowed_ssh_cidr) > 0
    error_message = "At least one SSH CIDR block must be specified."
  }

  # Defaulting SSH to 0.0.0.0/0 exposes port 22 to the whole internet. Requiring
  # an explicit value makes opening it up a deliberate choice rather than what
  # happens when nobody looks at this file.
  validation {
    condition     = !contains(var.allowed_ssh_cidr, "0.0.0.0/0")
    error_message = "allowed_ssh_cidr must not be 0.0.0.0/0. Restrict SSH to a known address range."
  }
}

variable "allowed_dashboard_cidr" {
  description = "CIDR blocks allowed to access the Streamlit dashboard. The dashboard has no authentication, so anyone who can reach it has full control of the app."
  type        = list(string)

  validation {
    condition     = length(var.allowed_dashboard_cidr) > 0
    error_message = "At least one dashboard CIDR block must be specified."
  }

  # Same reasoning as allowed_ssh_cidr: the dashboard has no login of its own,
  # so defaulting it open is handing out full control of the app to anyone on
  # the internet.
  validation {
    condition     = !contains(var.allowed_dashboard_cidr, "0.0.0.0/0")
    error_message = "allowed_dashboard_cidr must not be 0.0.0.0/0. The dashboard has no authentication -- restrict it to a known address range."
  }
}

variable "streamlit_port" {
  description = "Port for Streamlit dashboard"
  type        = number
  default     = 8501

  validation {
    condition     = var.streamlit_port > 0 && var.streamlit_port <= 65535
    error_message = "Streamlit port must be between 1 and 65535."
  }
}

variable "github_repo" {
  description = "GitHub repository URL to clone (optional, set to 'none' to skip)"
  type        = string
  default     = "none"
}

variable "owner_tag" {
  description = "Owner tag for resources"
  type        = string
  default     = "Asteroids-Game"
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring (costs extra)"
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 7

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "key_pair_name" {
  description = "Name of the EC2 Key Pair for SSH access"
  type        = string
  default     = "asteroids-game-key"
}

