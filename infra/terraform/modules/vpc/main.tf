variable "environment" {}
variable "region" {}
variable "vpc_cidr" {}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "polymorph-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = [cidrsubnet(var.vpc_cidr, 8, 1), cidrsubnet(var.vpc_cidr, 8, 2), cidrsubnet(var.vpc_cidr, 8, 3)]
  public_subnets  = [cidrsubnet(var.vpc_cidr, 8, 101), cidrsubnet(var.vpc_cidr, 8, 102), cidrsubnet(var.vpc_cidr, 8, 103)]
  database_subnets = [cidrsubnet(var.vpc_cidr, 8, 201), cidrsubnet(var.vpc_cidr, 8, 202), cidrsubnet(var.vpc_cidr, 8, 203)]

  enable_nat_gateway = true
  single_nat_gateway = true # Save costs in dev
  enable_vpn_gateway = false

  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Environment = var.environment
    Project     = "polymorph-lite"
    Terraform   = "true"
  }
}

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "private_subnets" {
  value = module.vpc.private_subnets
}

output "public_subnets" {
  value = module.vpc.public_subnets
}

output "database_subnets" {
  value = module.vpc.database_subnets
}
