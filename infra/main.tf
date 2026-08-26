# 'La madre' — coordinador de Evolution Strategies para leia-sf2-es.
#
# Una sola instancia EC2 chica en la cuenta de EDUCACION (perfil awsedu). Los workers
# (las maquinas del equipo) se conectan a ella por Tailscale; la instancia NO expone
# ningun puerto a internet. Todo esta pensado para crearse y destruirse sin dejar rastro:
#   - estado LOCAL a proposito (sin backend remoto): esta infra es desechable, no hay
#     nada que compartir ni que sobrevivir a un `terraform destroy`. Un backend S3/Dynamo
#     dejaria justamente los residuos que queremos evitar.
#   - cero recursos de red nuevos: VPC/subred default via data sources.
#   - bucket con force_destroy para que `destroy` no se atore con objetos dentro.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  # default_tags cae sobre TODO recurso del provider: aunque alguien agregue un recurso
  # y olvide taggearlo, queda etiquetado y es rastreable/borrable por Project.
  default_tags {
    tags = {
      Project   = "leia-sf2-es"
      ManagedBy = "terraform"
      Owner     = var.owner
    }
  }
}

# --- Red: solo data sources; no creamos VPC, subredes ni gateways -----------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# --- AMI: Ubuntu 24.04 LTS (Noble) oficial de Canonical --------------------------------

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-${var.arch}-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- Security group: SIN reglas de entrada ---------------------------------------------
# Tailscale solo necesita salida (hace hole-punching / DERP hacia afuera) y el SSH va por
# `tailscale ssh`, no por el puerto 22 publico. Un SG sin ingress = superficie cero.

resource "aws_security_group" "madre" {
  name_prefix = "leia-madre-"
  description = "leia-sf2-es coordinator: no inbound (Tailscale-only), all egress"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description      = "all egress (tailscale, apt, git, pip, wandb, s3)"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "leia-madre"
  }
}

# --- Bucket opcional de checkpoints ----------------------------------------------------
# force_destroy: el punto de este bucket es poder tirarlo con `terraform destroy` aunque
# tenga checkpoints dentro. Si un checkpoint importa, bajarlo antes de destruir.

resource "aws_s3_bucket" "checkpoints" {
  count = var.create_bucket ? 1 : 0

  # bucket_prefix + sufijo aleatorio de AWS: los nombres S3 son globales y asi dos
  # companeros pueden levantar su propia madre sin chocar.
  bucket_prefix = "leia-sf2-es-ckpt-"
  force_destroy = true

  tags = {
    Name = "leia-sf2-es-checkpoints"
  }
}

# --- IAM minimo: la instancia solo puede tocar SU bucket -------------------------------

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "madre" {
  count = var.create_bucket ? 1 : 0

  name_prefix        = "leia-madre-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json

  tags = {
    Name = "leia-madre"
  }
}

resource "aws_iam_role_policy" "madre_s3" {
  count = var.create_bucket ? 1 : 0

  name_prefix = "leia-madre-s3-"
  role        = aws_iam_role.madre[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListOwnBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.checkpoints[0].arn]
      },
      {
        Sid      = "RwOwnObjects"
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject"]
        Resource = ["${aws_s3_bucket.checkpoints[0].arn}/*"]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "madre" {
  count = var.create_bucket ? 1 : 0

  name_prefix = "leia-madre-"
  role        = aws_iam_role.madre[0].name

  tags = {
    Name = "leia-madre"
  }
}

# --- La instancia ----------------------------------------------------------------------

resource "aws_instance" "madre" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  subnet_id              = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids = [aws_security_group.madre.id]

  # try(): con create_bucket=false el profile[0] no existe y la instancia va sin rol.
  iam_instance_profile = try(aws_iam_instance_profile.madre[0].name, null)

  root_block_device {
    volume_size = 20
    volume_type = "gp3"

    # default_tags NO se propaga al volumen raiz; se etiqueta explicito para que ningun
    # EBS quede huerfano de Project al filtrar en consola.
    tags = {
      Name      = "leia-madre-root"
      Project   = "leia-sf2-es"
      ManagedBy = "terraform"
      Owner     = var.owner
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2: evita que un proceso cualquiera lea credenciales del rol
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    tailscale_auth_key = var.tailscale_auth_key
    repo_url           = var.repo_url
    repo_branch        = var.repo_branch
    wandb_project      = var.wandb_project
    # Las piezas opcionales se resuelven aqui para que el template quede en bash plano:
    # linea Environment= solo si hay API key; flag --s3-bucket solo si hay bucket.
    wandb_env_line = var.wandb_api_key == "" ? "" : "Environment=WANDB_API_KEY=${var.wandb_api_key}"
    s3_flag        = var.create_bucket ? " --s3-bucket ${try(aws_s3_bucket.checkpoints[0].bucket, "")}" : ""
  })

  # Cambiar el user_data recrea la instancia: es la forma correcta de "re-provisionar"
  # algo desechable (no hay config drift que perseguir).
  user_data_replace_on_change = true

  tags = {
    Name = "leia-madre"
  }
}
