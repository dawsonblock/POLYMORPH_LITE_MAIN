terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket = "polymorph-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.9"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Project = "POLYMORPH-LITE"
      Environment = "Production"
      ManagedBy = "Terraform"
    }
  }
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "polymorph-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = true
}

module "eks" {
  source = "./modules/eks"
  cluster_name = "polymorph-prod"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
}

module "rds" {
  source = "./modules/rds"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  db_name = "polymorph_prod"
}

module "monitoring" {
  source = "./modules/monitoring"
  cluster_name = module.eks.cluster_name
}
