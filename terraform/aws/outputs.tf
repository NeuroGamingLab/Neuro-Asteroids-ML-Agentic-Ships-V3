output "dashboard_url" {
  description = "URL to access the Asteroids game dashboard"
  value       = "http://${aws_eip.game_ip.public_ip}:${var.streamlit_port}"
}

output "dashboard_public_ip" {
  description = "Public IP address of the Asteroids game instance"
  value       = aws_eip.game_ip.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.asteroids_game.id
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.asteroids_game.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@${aws_eip.game_ip.public_ip}"
}

output "deployment_info" {
  description = "Deployment information"
  value       = <<-EOT
     Asteroids Game Deployment Complete!
    
     Dashboard URL: http://${aws_eip.game_ip.public_ip}:${var.streamlit_port}
     Public IP: ${aws_eip.game_ip.public_ip}
      Instance ID: ${aws_instance.asteroids_game.id}
    
     Next Steps:
    1. Upload game files to: /opt/asteroids-game/app/
       Required files:
       - app.py
       - game.js
       - index.html (if used)
       - style.css (if used)
       - requirements.txt
    
    2. Start the game service:
       ssh -i ~/.ssh/asteroids-game-key.pem ec2-user@${aws_eip.game_ip.public_ip}
       sudo systemctl start asteroids-game
    
    3. Check status:
       sudo systemctl status asteroids-game
    
    4. View logs:
       sudo journalctl -u asteroids-game -f
    
    5. Access game:
       Open http://${aws_eip.game_ip.public_ip}:${var.streamlit_port} in your browser
    
     Estimated Cost: ~$0.013/hour (t3.micro)
    
     To destroy:
       cd terraform/aws
       terraform destroy
  EOT
}

