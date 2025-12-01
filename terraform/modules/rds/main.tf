variable "vpc_id" {}
variable "subnet_ids" { type = list(string) }
variable "db_name" {}

module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.0.0"

  identifier = "polymorph-db-prod"

  engine            = "postgres"
  engine_version    = "15.3"
  instance_class    = "db.t3.medium"
  allocated_storage = 20

  db_name  = var.db_name
  username = "polymorph_admin"
  port     = 5432

  iam_database_authentication_enabled = true

  vpc_security_group_ids = [] # Add SG here
  maintenance_window     = "Mon:00:00-Mon:03:00"
  backup_window          = "03:00-06:00"
  
  # Harden: Encryption and Backup
  storage_encrypted = true
  backup_retention_period = 30
  deletion_protection = true

  subnet_ids = var.subnet_ids
  family = "postgres15"
  major_engine_version = "15"
}
