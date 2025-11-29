terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  backend "s3" {
    # Placeholder for remote state configuration
    # bucket = "polymorph-terraform-state"
    # key    = "prod/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.region
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

module "vpc" {
  source = "./modules/vpc"
  
  environment = var.environment
  region      = var.region
  vpc_cidr    = var.vpc_cidr
}

module "eks" {
  source = "./modules/eks"

  cluster_name    = "${var.project_name}-${var.environment}"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  instance_types  = var.eks_instance_types
}

module "db" {
  source = "./modules/db"

  identifier     = "${var.project_name}-${var.environment}-db"
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.database_subnets
  db_name        = var.db_name
  db_username    = var.db_username
  db_password    = var.db_password # In prod, use Secrets Manager!
  instance_class = var.db_instance_class
}
