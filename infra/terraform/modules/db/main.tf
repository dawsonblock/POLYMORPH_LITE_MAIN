variable "identifier" {}
variable "vpc_id" {}
variable "subnet_ids" { type = list(string) }
variable "db_name" {}
variable "db_username" {}
variable "db_password" {}
variable "instance_class" {}

module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.3.0"

  identifier = var.identifier

  engine            = "postgres"
  engine_version    = "14"
  instance_class    = var.instance_class
  allocated_storage = 20

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  iam_database_authentication_enabled = true

  vpc_security_group_ids = [aws_security_group.db.id]

  maintenance_window = "Mon:00:00-Mon:03:00"
  backup_window      = "03:00-06:00"

  # DB subnet group
  create_db_subnet_group = true
  subnet_ids             = var.subnet_ids

  # DB parameter group
  family = "postgres14"

  # DB option group
  major_engine_version = "14"

  deletion_protection = false # For dev; enable in prod

  tags = {
    Environment = "prod"
    Project     = "polymorph-lite"
    Terraform   = "true"
  }
}

resource "aws_security_group" "db" {
  name        = "${var.identifier}-sg"
  description = "Allow inbound traffic from VPC"
  vpc_id      = var.vpc_id

  ingress {
    description = "PostgreSQL from VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"] # Should match VPC CIDR
  }
}

output "db_endpoint" {
  value = module.db.db_instance_endpoint
}
