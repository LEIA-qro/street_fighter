output "instance_id" {
  description = "ID de la instancia EC2 de la madre."
  value       = aws_instance.madre.id
}

output "tailscale_hostname" {
  description = "La instancia se registra en tu tailnet como 'madre'; entra con: tailscale ssh madre"
  value       = "madre"
}

output "checkpoint_bucket" {
  description = "Bucket S3 de checkpoints (null si create_bucket=false)."
  value       = try(aws_s3_bucket.checkpoints[0].bucket, null)
}
