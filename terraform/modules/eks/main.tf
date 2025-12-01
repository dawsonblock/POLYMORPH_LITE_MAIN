variable "cluster_name" {}
variable "vpc_id" {}
variable "subnet_ids" { type = list(string) }

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.27"

  vpc_id     = var.vpc_id
  subnet_ids = var.subnet_ids

  cluster_endpoint_public_access  = true
  
  # Harden: Enable control plane logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  eks_managed_node_groups = {
    general = {
      min_size     = 2
      max_size     = 5
      desired_size = 2

      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
    }
    gpu_nodes = {
      min_size     = 0
      max_size     = 2
      desired_size = 0 # Scale up for AI training
      
      instance_types = ["g4dn.xlarge"]
      taints = {
        dedicated = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

output "cluster_name" {
  value = module.eks.cluster_name
}
