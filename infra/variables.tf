# Variables de 'la madre'. Las dos sensibles (tailscale_auth_key, wandb_api_key) van en
# terraform.tfvars (git-ignorado) o por -var; NUNCA en un archivo commiteado.

variable "tailscale_auth_key" {
  description = "Auth key de Tailscale (settings/keys). Recomendado: reusable + ephemeral."
  type        = string
  sensitive   = true
}

variable "wandb_api_key" {
  description = "API key de Weights & Biases. Vacio = el coordinador corre sin W&B."
  type        = string
  sensitive   = true
  default     = ""
}

variable "owner" {
  description = "Quien levanto esta infra (tag Owner). Tu nombre o usuario de GitHub."
  type        = string
}

variable "instance_type" {
  description = "Tipo de instancia. t3.small alcanza: el coordinador solo agrega vectores y sirve HTTP."
  type        = string
  default     = "t3.small"
}

variable "region" {
  description = "Region AWS (cuenta de educacion)."
  type        = string
  default     = "us-west-2"
}

variable "arch" {
  description = "Arquitectura de la AMI. Debe corresponder al instance_type: t3.* = amd64, t4g.* = arm64."
  type        = string
  default     = "amd64"

  validation {
    condition     = contains(["amd64", "arm64"], var.arch)
    error_message = "arch debe ser 'amd64' o 'arm64'."
  }
}

variable "repo_url" {
  description = "URL del repo a clonar. La default (SSH) requiere deploy key en la VPS; ver README para la alternativa https con token."
  type        = string
  default     = "git@github.com:LEIA-qro/street_fighter.git"
}

variable "repo_branch" {
  description = "Rama a clonar."
  type        = string
  default     = "stage0-metrics-and-semantics"
}

variable "create_bucket" {
  description = "Crear bucket S3 para checkpoints (con force_destroy) + rol IAM minimo para usarlo."
  type        = bool
  default     = true
}

variable "wandb_project" {
  description = "Proyecto de W&B donde el coordinador loguea metricas."
  type        = string
  default     = "leia-sf2-es"
}
