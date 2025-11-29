variable "cluster_name" {}
variable "vpc_id" {}
variable "subnet_ids" { type = list(string) }
variable "instance_types" { type = list(string) }

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  cluster_endpoint_public_access  = true

  vpc_id                   = var.vpc_id
  subnet_ids               = var.subnet_ids
  control_plane_subnet_ids = var.subnet_ids

  # EKS Managed Node Group(s)
  eks_managed_node_groups = {
    general = {
      min_size     = 1
      max_size     = 3
      desired_size = 2

      instance_types = var.instance_types
      capacity_type  = "ON_DEMAND"
    }
  }

  # Cluster access entry
  enable_cluster_creator_admin_permissions = true

  tags = {
    Environment = "prod"
    Project     = "polymorph-lite"
    Terraform   = "true"
  }
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  value = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  value = module.eks.cluster_certificate_authority_data
}

output "cluster_name" {
  value = module.eks.cluster_name
}
